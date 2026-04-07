# 학습 전략 및 Loss 함수 보고서

> **범위**: `train/`, `loss/`, `main.py`, `configs/` 디렉토리의 학습 파이프라인 전체를 다룬다.

---

## 1. 학습 전략 개요: 2-Phase 커리큘럼

본 프로젝트는 **2단계 커리큘럼 사전학습**(Curriculum Pre-training)을 채택한다. 단일 채널의 시간적 패턴을 먼저 학습한 뒤, 다변량 관계를 확장 학습하는 구조이다.

### 1.1 Phase 1: Channel-Independent (CI) 사전학습

- **목표**: 각 variate(채널)를 독립적으로 처리하여 **단일 신호의 시간적 표현**을 학습한다.
- **collate_mode**: `"ci"` — 채널을 개별 시퀀스로 분리하여 배치를 구성한다.
- **Loss 구성**: MPM(Masked Patch Modeling) + Next-Patch Prediction
  - `alpha=1.0`, `beta=1.0`, `gamma=0.0`, `delta=0.0`
  - Cross-modal 및 Contrastive loss는 비활성 (단일 채널이므로 의미 없음)
- **Masking**: 랜덤 패치 마스킹만 사용 (`variate_mask_prob=0.0`)
- **학습 설정**: batch_size=16, lr=1e-3, 70 epochs, warmup=5 epochs

### 1.2 Phase 2: Any-Variate (Cross-Modal) 학습

- **목표**: Phase 1에서 학습된 단일-신호 표현을 기반으로 **다변량 간 상호작용**(cross-modal relationship)을 학습한다.
- **collate_mode**: `"any_variate"` — 같은 세션의 여러 채널을 하나의 시퀀스로 패킹한다.
- **Loss 구성**: MPM + Next-Pred + Cross-Modal + Contrastive (전체 활성)
  - `alpha=0.7`, `beta=0.3`, `gamma=1.0`, `delta=0.1`
- **Masking**: 랜덤 패치 마스킹 + variate-level 마스킹 (`variate_mask_prob=0.3`)
  - 30% 확률로 랜덤 variate 전체를 마스킹하여 **가상 센싱**(Virtual Sensing) 학습
- **Phase 1 Checkpoint**: `--resume` 옵션으로 로드하거나 `outputs/phase1_ci/`에서 자동 탐색
- **학습 설정**: batch_size=4, lr=1e-4, 30 epochs, warmup=3 epochs

### 1.3 Phase 전환 요약

| 항목 | Phase 1 (CI) | Phase 2 (Any-Variate) |
|------|:---:|:---:|
| `collate_mode` | `"ci"` | `"any_variate"` |
| `batch_size` | 16~128 | 4~8 |
| `lr` | 1e-3 | 1e-4 |
| `n_epochs` | 70 | 30 |
| `alpha` / `beta` | 1.0 / 1.0 | 0.7 / 0.3 |
| `gamma` (cross-modal) | 0.0 | 1.0 |
| `delta` (contrastive) | 0.0 | 0.1 |
| `variate_mask_prob` | 0.0 | 0.3 |
| Checkpoint 경로 | `outputs/phase1_ci/` | `outputs/phase2_any_variate/` |

---

## 2. train/ 상세

### 2.1 `train/train_utils.py` — 공유 유틸리티

학습 스크립트 전체가 공유하는 핵심 함수와 클래스를 정의한다.

#### 2.1.1 `TrainConfig` (dataclass)

학습에 필요한 모든 하이퍼파라미터를 하나의 dataclass로 통합 관리한다.

**주요 필드**:

| 카테고리 | 필드 | 기본값 | 설명 |
|----------|------|--------|------|
| 모델 | `model_config` | `ModelConfig()` | 모델 아키텍처 설정 |
| 데이터 | `processed_dir` | `"datasets/processed"` | 전처리 데이터 경로 |
| | `window_seconds` | 30.0 | 슬라이딩 윈도우 길이(초) |
| | `crop_ratio_min/max` | 0.0 | Random crop 비율 범위 |
| 학습 | `batch_size` | 16 | 배치 크기 |
| | `lr` | 1e-3 | 학습률 |
| | `n_epochs` | 70 | 총 에폭 수 |
| | `warmup_epochs` | 5 | Linear warmup 에폭 수 |
| | `gradient_clip` | 1.0 | Gradient clipping max norm |
| Loss | `alpha/beta/gamma/delta` | 1/0/0/0 | 복합 Loss 가중치 |
| | `eeg_loss_weight` | 0.05 | V2 EEG reconstruction 가중치 |
| | `aux_loss_weight` | 0.01 | MoE load balancing 가중치 |
| Masking | `mask_ratio` | 0.15 | 랜덤 마스킹 비율 |
| | `variate_mask_prob` | 0.0 | Variate-level 마스킹 확률 |
| Validation | `val_ratio` | 0.2 | Subject 단위 val 비율 |
| | `patience` | 10 | Early stopping patience |
| AMP | `use_amp` | False | Mixed precision 활성화 |
| 실행 | `dry_run` | False | 1 batch 검증 모드 |

**YAML 직렬화**: `to_yaml()`, `from_yaml()`, `from_yaml_with_overrides()` 메서드를 통해 config 파일 저장/로드 및 CLI 오버라이드를 지원한다.

#### 2.1.2 `train_one_epoch()` (V1)

1에폭 학습 루프를 수행하고 평균 loss dict를 반환한다.

**핵심 흐름**:
1. `model(batch, task="both", horizon=H)` — 단일 forward call (DDP 호환)
2. MoE `aux_loss` 수집 (encoder layer별 `ffn.aux_loss` 합산)
3. `create_patch_mask()` — 랜덤/variate-level 마스킹 생성
4. 원본 패치 추출: `(values - loc) / scale` 정규화 후 `(B, N, P)` reshape
5. `CombinedLoss` 호출 → `total + aux_loss_weight * aux_loss`로 최종 loss 계산
6. NaN/Inf 감지: 연속 10 batch NaN 시 에폭 조기 종료
7. Gradient clipping: `clip_grad_norm_(max_norm=1.0)` + gradient NaN 감지
8. AMP 지원: `GradScaler`가 있으면 `scale/unscale_/step/update` 워크플로우
9. GPU 텐서 누적: `.item()` CUDA sync 없이 GPU에서 직접 loss 누적

**Horizon 랜덤 샘플링**: 매 batch마다 `H = random.randint(1, max_horizon)`으로 랜덤 horizon을 선택한다.

**반환**: `{"total", "masked_loss", "next_loss", "cross_modal_loss", "contrastive_loss", "aux_loss"}`

#### 2.1.3 `train_one_epoch_v2()` (V2 전용)

V1과 동일한 구조에 **EEG stem-target reconstruction loss**가 추가된다.

**V1과의 차이**:
- EEG 패치(`eeg_mask`)를 masked reconstruction에서 분리 → `non_eeg_pred_mask = pred_mask & ~eeg_mask`
- EEG 전용 loss: `MSE(eeg_reconstructed[eeg_pred_mask], eeg_recon_target[eeg_pred_mask])`
  - `eeg_recon_target`은 CNN stem 출력의 `.detach()` (stop-gradient)
- 최종: `total = CombinedLoss(non-EEG) + eeg_loss_weight * eeg_loss + aux_loss_weight * aux_loss`

**반환**: V1 반환값 + `"eeg_loss"` 키 추가

#### 2.1.4 `validate()` / `validate_v2()`

- `@torch.no_grad()` 데코레이터로 gradient 계산 없이 loss만 집계
- DDP 환경에서 unwrapped 모델(`model.module`)로 forward하여 rank별 배치 수 불일치 데드락 방지
- `validate_v2()`는 EEG loss 별도 계산 포함

#### 2.1.5 `load_manifest_from_processed()`

`datasets/processed/*/manifest.json` 파일들을 읽어 `RecordingManifest` 목록을 반환한다. `signal_types` 필터링 및 `max_subjects` 제한을 지원한다.

#### 2.1.6 `split_manifest_by_subject()`

Subject(디렉토리) 단위로 train/val을 분할한다. 같은 subject의 모든 recording이 동일한 split에 들어가며, 기본 비율은 80/20이다.

#### 2.1.7 `create_scheduler()`

**Linear Warmup + Cosine Annealing** 스케줄러를 생성한다.

```
lr_lambda(epoch):
  if epoch < warmup:
    return (epoch + 1) / warmup            # 선형 증가
  progress = (epoch - warmup) / (total - warmup)
  return min_ratio + 0.5 * (1 - min_ratio) * (1 + cos(pi * progress))  # cosine 감쇠
```

- `[0, warmup_epochs)`: 0 → `lr`까지 선형 증가
- `[warmup_epochs, n_epochs)`: `lr` → `lr * min_lr_ratio`까지 cosine 감쇠 (기본 min_lr_ratio=0.1)

#### 2.1.8 기타 유틸리티

| 함수/클래스 | 설명 |
|-------------|------|
| `EarlyStopping` | patience/min_delta 기반 조기 종료. `step(val_loss)` 호출 시 중단 여부 반환 |
| `CSVLogger` | 에폭별 train/val loss, lr, 시간을 CSV 파일로 기록 |
| `save_training_checkpoint()` | `checkpoints/` 하위에 `checkpoint_{phase}_{epoch}_{tag}.pt` 저장 |
| `resolve_output_dir()` | `exp_name` 설정 시 `output_dir/exp_name/` 경로 반환 |
| `save_experiment_info()` | `experiment_info.txt` + `config.yaml` 저장 |
| `create_scaler()` | AMP + CUDA 환경에서 `GradScaler` 생성 |
| `setup_ddp()` / `cleanup_ddp()` | DDP 프로세스 그룹 초기화/종료 (NCCL) |
| `is_main_process()` | rank 0 여부 확인 (단일 GPU면 항상 True) |

### 2.2 `train/v1_1_channel_independency.py` — Phase 1 CI 사전학습

Phase 1 전용 학습 스크립트. V1 모델(`BiosignalFoundationModelV1`)을 사용한다.

**핵심 설정**:
- `collate_mode = "ci"`: 각 채널을 독립 시퀀스로 분리
- `gamma = 0.0`, `variate_mask_prob = 0.0`: cross-modal 관련 비활성
- `beta = 1.0`: next-patch prediction 활성, random horizon(1~max_horizon)

**DDP 지원**:
- `torchrun --nproc_per_node=N` 실행 시 자동 DDP 감지 (`LOCAL_RANK` 환경 변수)
- `DistributedSampler`로 데이터 분배, `find_unused_parameters=True`
- rank 0에서만 checkpoint, 시각화, CSV 로깅 수행

**시각화**:
- `viz_every` 에폭마다 validation 배치로 reconstruction + next-pred figure 생성
- `figures/recon/`, `figures/next_pred/` 하위에 저장

**Config 지원**: `--config` 옵션으로 YAML 파일 로드 가능 (Phase 1 고정값 자동 적용)

### 2.3 `train/v1_2_any_variate.py` — Phase 2 Any-Variate 학습

Phase 1 checkpoint를 로드하여 cross-modal 학습을 수행하는 Phase 2 전용 스크립트.

**핵심 설정**:
- `collate_mode = "any_variate"`: 세션 내 다변량을 하나의 시퀀스로 패킹
- `alpha=0.7, beta=0.3, gamma=1.0, delta=0.1`: 전체 Loss 활성
- `variate_mask_prob=0.3`: variate-level 마스킹 활성
- `contrastive_proj_dim=128`: contrastive projection head 활성

**Phase 1 Checkpoint 자동 탐색**: `--resume` 미지정 시 `outputs/phase1_ci/` 경로에서 `*_best.pt` → `*_final.pt` → `*.pt` 순서로 탐색한다.

### 2.4 V2 학습 스크립트

`train/v2_1_channel_independency.py`와 `train/v2_2_any_variate.py`는 V1과 동일한 구조이나 `BiosignalFoundationModelV2`를 사용하며, `train_one_epoch_v2()` / `validate_v2()`를 호출하여 EEG stem loss를 별도 계산한다.

---

## 3. Loss 함수 상세

### 3.1 `loss/masked_mse_loss.py` — MaskedPatchLoss

#### 3.1.1 `MaskedPatchLoss`

마스킹된 패치 위치에서만 MSE를 계산하는 Masked Patch Modeling 손실 함수.

**수식**:

```
L_mpm = (1 / |M|) * sum_{(b,n) in M} || reconstructed[b,n] - original[b,n] ||^2
```

- `M`: `pred_mask=True`인 패치 집합
- `reconstructed`, `original`: `(B, N, P)` 텐서 (B=배치, N=패치 수, P=패치 크기)
- 마스킹된 위치가 없으면 `0.0` 반환

**설계 의도**: 마스킹되지 않은 패치는 모델이 직접 관찰할 수 있으므로, 마스킹된 위치의 복원 능력만 평가한다. BERT-style self-supervised learning의 핵심 원리.

#### 3.1.2 `create_patch_mask()`

유효 패치에 대해 마스킹 대상을 생성하는 함수.

**파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `patch_mask` | `(B, N) bool` | 유효 패치 마스크 (True=유효) |
| `mask_ratio` | `float` | 랜덤 마스킹 비율 (기본 0.15) |
| `patch_variate_id` | `(B, N) | None` | 패치별 variate ID |
| `variate_mask_prob` | `float` | Variate-level 마스킹 확률 (기본 0.0) |

**동작 (Phase별)**:

1. **Phase 1** (`variate_mask_prob=0.0`): 유효 패치 중 `mask_ratio` 비율을 랜덤으로 선택하여 마스킹
2. **Phase 2** (`variate_mask_prob>0`): `variate_mask_prob` 확률로 랜덤 variate를 선택하여 해당 variate의 **모든 패치**를 마스킹 (Virtual Sensing). 나머지 경우 Phase 1과 동일하게 랜덤 마스킹.
   - 조건: 2개 이상의 variate가 존재해야 variate-level 마스킹 발동
   - `variate_id=0`(패딩)은 마스킹 대상에서 제외

### 3.2 `loss/next_prediction_loss.py` — NextPredictionLoss

동일 variate 내 시간 예측과 cross-modal 예측을 동시에 수행하는 손실 함수.

#### 3.2.1 Same-Variate Next-Patch Loss

**수식**:

```
L_next = (1/H) * (1/|V|) * sum_{(b,n) in V} || next_pred[b,n] - original[b,n+H] ||^2
```

- `H`: horizon (미래 패치 거리)
- `V`: 유효 쌍 조건을 만족하는 인덱스 집합
  - `patch_mask[b,n]` 및 `patch_mask[b,n+H]` 모두 True
  - `sample_id[b,n] == sample_id[b,n+H]` (같은 샘플)
  - `variate_id[b,n] == variate_id[b,n+H]` (같은 variate)
- `1/H` 가중치: horizon이 클수록 예측이 어려우므로 loss를 비례적으로 감소시켜 학습 안정성 확보

**설계 의도**: 시간적 dynamics를 학습한다. 과거 패치로부터 미래 패치의 파형을 예측하는 과정에서 신호의 시간적 구조(주기성, 추세 등)를 포착한다.

#### 3.2.2 Cross-Modal Loss

**수식**:

```
L_cross = gamma * (1/|C|) * sum_{(b,i,j) in C} || cross_pred[b,i] - original[b,j] ||^2
```

- `C`: cross-modal 쌍 조건을 만족하는 `(b, i, j)` 인덱스 집합
  - `sample_id[b,i] == sample_id[b,j]` (같은 샘플)
  - `time_id[b,i] == time_id[b,j]` (같은 시간대)
  - `variate_id[b,i] != variate_id[b,j]` (다른 variate)
  - 양쪽 모두 유효 패치이며 패딩(`variate_id > 0`)이 아님

**설계 의도**: 동일 시간대에서 다른 모달리티(예: ECG → ABP)의 파형을 예측하도록 학습하여, 모달리티 간 인과 관계(causality)를 포착한다. Phase 2에서만 활성화된다.

**파라미터**:
- `cross_modal_weight` (`gamma`): 0이면 비활성 (Phase 1), >0이면 활성 (Phase 2)

### 3.3 `loss/contrastive_loss.py` — CrossModalContrastiveLoss

InfoNCE 기반 cross-modal contrastive loss. 표현 공간에서 modality 간 정렬(alignment)을 학습한다.

**수식**:

```
L_contrastive = -(1/|A|) * sum_{i in A} log( sum_{j in P(i)} exp(sim(z_i, z_j)/tau) / sum_{k in N(i)} exp(sim(z_i, z_k)/tau) )
```

여기서:
- `z`: L2 정규화된 projected embeddings `(B, N, D)`
- `sim(z_i, z_j) = z_i^T z_j`: cosine similarity
- `tau`: temperature (learnable 또는 fixed)
- `A`: positive pair가 존재하는 유효 anchor 집합
- `P(i)`: anchor `i`의 positive 집합 — 같은 `(sample_id, time_id)`, 다른 `variate_id`
- `N(i)`: anchor `i`의 negative 집합 — 자기 자신을 제외한 모든 유효 패치

**파라미터**:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `temperature` | 0.07 | InfoNCE 초기 temperature |
| `learnable_temperature` | True | log-parameterized learnable temperature (CLIP 방식) |

**구현 세부사항**:
- Temperature는 `log(tau)`로 파라미터화하여 항상 양수를 보장하며, `[0.01, 1.0]`으로 클램핑하여 수치 안정성을 확보
- `logsumexp`를 사용하여 수치적으로 안정된 계산
- 단일 variate만 존재하는 batch row는 positive pair가 없으므로 자동 스킵
- 패딩(`variate_id=0`)은 유효 패치에서 제외

### 3.4 `loss/criterion.py` — CombinedLoss

모든 Loss를 하나로 통합하는 하이브리드 손실 함수.

**수식**:

```
L_total = alpha * L_mpm + beta * (L_next + gamma * L_cross) + delta * L_contrastive
```

**파라미터**:

| 파라미터 | Phase 1 | Phase 2 | 설명 |
|----------|:-------:|:-------:|------|
| `alpha` | 1.0 | 0.7 | Masked reconstruction 가중치 |
| `beta` | 1.0 | 0.3 | Next-patch prediction 가중치 |
| `gamma` | 0.0 | 1.0 | Cross-modal prediction 가중치 (beta 내부) |
| `delta` | 0.0 | 0.1 | Contrastive loss 가중치 |

**반환 dict**:

```python
{
    "total": torch.Tensor,           # 최종 통합 loss
    "masked_loss": torch.Tensor,     # MPM loss (스칼라)
    "next_loss": torch.Tensor,       # Same-variate next-pred loss
    "cross_modal_loss": torch.Tensor, # Cross-modal prediction loss
    "contrastive_loss": torch.Tensor, # InfoNCE contrastive loss
}
```

**최종 학습 loss** (train_one_epoch에서 계산):

```
# V1
loss = L_total + aux_loss_weight * MoE_aux_loss

# V2
loss = L_total + eeg_loss_weight * L_eeg + aux_loss_weight * MoE_aux_loss
```

**MaskedMSELoss**: 하위 호환용 레거시 클래스. `(batch, seq_len)` 단위의 단순 masked MSE를 계산한다.

---

## 4. Config 구조

### 4.1 YAML Config 파일

`configs/` 디렉토리에 Phase별 YAML 설정 파일이 제공된다.

| 파일 | 용도 |
|------|------|
| `configs/phase1.yaml` | Phase 1 CI 기본 설정 |
| `configs/phase2.yaml` | Phase 2 Any-Variate 기본 설정 |
| `configs/phase1_base.yaml` | Phase 1 CI Base 모델 설정 (d=256, L=8) |
| `configs/dry_run.yaml` | 로컬 검증용 최소 설정 |

**YAML 파일 구조 예시** (`configs/phase1.yaml`):

```yaml
# ── 모델 ──
model_config:
  d_model: 64
  num_layers: 2
  patch_size: 100
  use_glu: true
  use_rope: true
  max_horizon: 5
  contrastive_proj_dim: 0   # Phase 1은 비활성

# ── 데이터 ──
processed_dir: datasets/processed
signal_types: [0, 1, 2, 3, 4, 5, 6]
window_seconds: 30.0

# ── 학습 ──
batch_size: 16
lr: 1.0e-3
n_epochs: 70
warmup_epochs: 5
collate_mode: ci

# ── Loss 가중치 ──
alpha: 1.0          # masked reconstruction
beta: 1.0           # next-patch prediction
gamma: 0.0          # cross-modal (비활성)
delta: 0.0          # contrastive (비활성)

# ── Masking ──
variate_mask_prob: 0.0

# ── Validation ──
val_ratio: 0.2
patience: 10
```

### 4.2 CLI 사용법

#### `main.py` — 통합 진입점

```bash
# Phase 1 학습
python main.py --config configs/phase1.yaml

# Phase 2 학습 (Phase 1 checkpoint 자동 탐색)
python main.py --config configs/phase2.yaml

# CLI 오버라이드
python main.py --config configs/phase1.yaml --batch_size 32 --lr 5e-4 --device cuda:0

# Dry-run (1 batch, shape 검증)
python main.py --config configs/phase1.yaml --dry-run
```

#### Phase별 전용 스크립트 + DDP

```bash
# Phase 1 V1 (단일 GPU)
python -m train.v1_1_channel_independency --device cuda:0

# Phase 1 V1 (DDP, 2 GPU)
torchrun --nproc_per_node=2 -m train.v1_1_channel_independency

# Phase 2 V1 (Phase 1 checkpoint 지정)
python -m train.v1_2_any_variate --resume outputs/phase1_ci/checkpoints/checkpoint_phase1_ci_epoch069_best.pt

# DDP launcher
torchrun --nproc_per_node=2 launch_phase1.py --model_version v1 --d_model 128
```

### 4.3 `--dry-run` 모드

1 batch만 실행하고 종료하는 검증 모드. OOM, shape 불일치, NaN 등을 빠르게 확인할 수 있다.

- `n_epochs = 1`, `max_batches = 1`로 자동 설정
- Loss 값과 gradient norm을 출력

---

## 5. Curriculum Horizon 전략

Next-Patch Prediction에서 horizon `H`는 예측할 미래 패치의 거리를 의미한다.

### 5.1 Random Horizon (현재 구현)

매 batch마다 `H = random.randint(1, max_horizon)`으로 랜덤 horizon을 샘플링한다 (`max_horizon=5`).

- **Horizon 가중치**: `1/H`를 곱하여 먼 horizon일수록 loss 기여도를 줄인다
- 이유: 먼 미래 예측은 본질적으로 더 어려우므로, 균등한 gradient 기여를 위해 가중치를 반비례로 조정

### 5.2 Curriculum Horizon 계획 (Phase 1 이후 적용 예정)

Phase 1 학습 100 epoch 이후, horizon을 점진적으로 증가시키는 전략:

```
Stage 1: H = 1       (단기 예측만 학습)
Stage 2: H = 1~3     (중기 예측 추가)
Stage 3: H = 1~5     (장기 예측 추가)
```

**설계 의도**: 쉬운 예측(H=1)에서 시작하여 점진적으로 어려운 예측(H=5)을 추가함으로써, 안정적인 시간적 표현 학습을 유도한다. 이는 curriculum learning의 원리에 기반한다.

---

## 6. 학습 안정화 메커니즘

### 6.1 Gradient Clipping

`torch.nn.utils.clip_grad_norm_(max_norm=1.0)`을 적용하여 gradient explosion을 방지한다.

### 6.2 NaN/Inf 감지

- **Loss NaN**: 해당 batch를 스킵하고 경고 출력. 연속 10 batch NaN 시 에폭 조기 종료.
- **Gradient NaN**: optimizer update를 스킵하고 경고 출력.

### 6.3 Early Stopping

`EarlyStopping(patience=10, min_delta=1e-4)`: validation loss가 `patience` 에폭 연속으로 `min_delta` 이상 개선되지 않으면 학습을 중단한다.

### 6.4 Mixed Precision (AMP)

`--use_amp` 플래그로 활성화. `torch.amp.autocast` + `GradScaler`를 사용하여 메모리 절약 및 학습 속도 향상. CUDA 디바이스에서만 지원된다.

---

## 7. 로깅 및 체크포인트

### 7.1 CSV 로깅

`CSVLogger`가 `outputs/{exp_name}/training_log.csv`에 에폭별 메트릭을 기록한다.

**컬럼**: epoch, phase, train_total/masked/next/cross/contrastive/eeg/aux, val_total/masked/next/cross/contrastive/eeg/aux, lr, epoch_sec

### 7.2 체크포인트 저장

| 유형 | 파일명 패턴 | 조건 |
|------|-------------|------|
| Best | `checkpoint_{phase}_epoch{N}_best.pt` | val_loss(또는 train_loss) 최저 갱신 시 |
| Periodic | `checkpoint_{phase}_epoch{N}.pt` | `checkpoint_every` 에폭마다 |
| Final | `checkpoint_{phase}_epoch{N}_final.pt` | 학습 종료 시 |

### 7.3 시각화

- **Reconstruction figure**: 원본 vs 마스킹 vs 복원 파형 비교 (`figures/recon/`)
- **Next-pred figure**: 원본 vs 예측 파형 비교 (`figures/next_pred/`)
- `viz_every` 에폭마다 생성 (기본 5)
