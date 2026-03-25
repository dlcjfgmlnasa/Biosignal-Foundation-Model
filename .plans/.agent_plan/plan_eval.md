# Train & Eval 서브 플랜

> **지침**: 이 파일은 **Train & Eval (학습 및 평가)** 역할의 서브 에이전트 전용 계획표이다.
> 작업을 시작하기 전 이 파일을 읽고, 할당된 `[ ]` 작업을 수행한 뒤 성공하면 `[x]`로 상태를 업데이트하라.
> 작업 전 반드시 `.plans/master_plan.md`의 **데이터 규약** (섹션 2)을 확인하라.

---

## 담당 파일

| 파일 | 역할 |
|------|------|
| `train/train_utils.py` | 학습 공유 유틸리티 (`TrainConfig`, `train_one_epoch`, manifest 로딩, checkpoint 헬퍼) — `train/` 서브패키지로 이동 완료 |
| `train/1_channel_independency.py` | Phase 1: CI 사전학습 스크립트 |
| `train/2_any_variate.py` | Phase 2: Any-Variate + Cross-Modal 학습 스크립트 |
| `loss/criterion.py` | `CombinedLoss` (α*MPM + β*(NextPred + γ*CrossModal)), `MaskedMSELoss` |
| `loss/masked_mse_loss.py` | `MaskedPatchLoss`, `create_patch_mask` (랜덤 + variate-level) |
| `loss/next_prediction_loss.py` | `NextPredictionLoss` (same-variate + cross-modal) |
| `tests/test_integration.py` | 학습 루프 통합 테스트 (200 tests) |
| `eval/__init__.py` | 다운스트림 평가 패키지 |
| `eval/forecasting.py` | ICU forecasting 평가 |
| `eval/fewshot.py` | Prototypical Network few-shot 분류 |
| `eval/imputation.py` | 가상 센싱 imputation |
| `tests/test_eval.py` | 다운스트림 평가 테스트 |
| `main.py` | 학습 진입점 (미구현 — 현재 `1_*`, `2_*` 스크립트로 대체) |
| `test_main.py` | 단일 Phase 학습 데모 (레거시 참고용) |
| `test_main2.py` | 2-Phase 커리큘럼 학습 데모 (레거시 참고용) |
| `test_main3.py` | 패치 기반 MPM 학습 데모 (레거시 참고용) |

---

## 1. 기존 구현 현황 (완료)

### 1.1 Loss 모듈 (구현 완료)
- [x] `MaskedMSELoss` (`loss/criterion.py`) — point-level masked MSE (하위 호환)
- [x] `MaskedPatchLoss` (`loss/masked_mse_loss.py`) — patch-level masked MSE
- [x] `create_patch_mask()` (`loss/masked_mse_loss.py`) — 랜덤 + variate-level 마스킹 헬퍼
  - `variate_mask_prob > 0`: Phase 2에서 전체 variate 마스킹 (Virtual Sensing)
- [x] `NextPredictionLoss` (`loss/next_prediction_loss.py`) — same-variate + cross-modal prediction
  - `_same_variate_loss()`: 기존 horizon-shifted MSE
  - `_cross_modal_loss()`: 벡터화된 (sample_id, time_id) 매칭으로 다른 variate 간 예측
- [x] `CombinedLoss` (`loss/criterion.py`) — α*MPM + β*(NextPred + γ*CrossModal)
  - 반환: `{"total", "masked_loss", "next_loss", "cross_modal_loss"}`

### 1.2 학습 파이프라인 (구현 완료)
- [x] `TrainConfig` 데이터클래스 (`train_utils.py`) — 데이터/모델/학습/Loss/시스템 설정 통합
- [x] `train_one_epoch()` (`train_utils.py`) — gradient clipping, cross-modal loss, random horizon, 배치별 로깅
- [x] `load_manifest_from_processed()` — processed 디렉토리에서 manifest 자동 로딩
- [x] `set_seed()`, `resolve_device()` — 시드 고정 + auto device
- [x] `save_training_checkpoint()` — Phase별 best/periodic/final checkpoint 저장

### 1.3 2-Phase 커리큘럼 학습 스크립트
- [x] `1_channel_independency.py` — Phase 1: CI 사전학습
  - `collate_mode="ci"`, `batch_size=16`, `lr=1e-3`, `n_epochs=70`
  - `alpha=1.0, beta=1.0, gamma=0.0` (MPM + NextPred, cross-modal 비활성)
  - `max_horizon=5` (랜덤 horizon 샘플링)
  - CosineAnnealingLR 스케줄러
  - Best/periodic/final checkpoint 저장 → `outputs/phase1_ci/`
- [x] `2_any_variate.py` — Phase 2: Any-Variate + Cross-Modal
  - Phase 1 checkpoint 자동 탐색 또는 `--resume` 지정
  - `collate_mode="any_variate"`, `batch_size=4`, `lr=1e-4`, `n_epochs=30`
  - `alpha=0.7, beta=0.3, gamma=1.0` (MPM + NextPred + CrossModal)
  - `variate_mask_prob=0.3` (30% 확률로 전체 variate 마스킹)
  - 새 optimizer/scheduler (Phase 전환 시 리셋)
  - checkpoint → `outputs/phase2_any_variate/`

### 1.4 Checkpoint 시스템 (구현 완료)
- [x] `save_checkpoint()` (`model/checkpoint.py`) — model + optimizer + epoch + config + extra 저장
- [x] `load_checkpoint()` (`model/checkpoint.py`) — state_dict 로드 + 메타 반환
- [x] Best model 별도 저장 (`*_best.pt`)
- [x] Periodic checkpoint (`checkpoint_every` 에폭 간격)
- [x] `--resume` CLI 인자로 Phase 2에서 Phase 1 checkpoint 로드
- [x] Phase 전환 시 optimizer 리셋 (새 Phase마다 새 Adam + CosineAnnealingLR)

### 1.5 레거시 데모 스크립트 (참고용)
- [x] `test_main.py` — 단일 Phase 학습 루프 (point-level masking, `MaskedModelingModel`)
- [x] `test_main2.py` — 2-Phase 커리큘럼 (CI→any_variate, `CosineAnnealingLR`)
- [x] `test_main3.py` — 패치 기반 MPM 루프 (`BiosignalFoundationModel`, `CombinedLoss`)

---

## 2. Mixed Precision & 최적화

### 2.1 AMP (Automatic Mixed Precision)
- [ ] `torch.amp.autocast("cuda")` 적용
- [ ] `GradScaler` 사용 (loss scaling)
- [ ] AMP 호환성 검증 (RMSNorm, GQA, PackedScaler 모두 fp16/bf16 안전한지)

### 2.2 학습 안정화
- [x] Gradient clipping (`max_norm=1.0`) — `train_one_epoch()`에 구현
- [ ] Learning rate warmup (linear warmup → cosine decay)
  ```python
  # CosineAnnealingLR 대신 OneCycleLR 또는 커스텀 스케줄러
  warmup_epochs = max(1, n_epochs // 10)
  ```
- [ ] Loss spike 감지 및 자동 LR 감소
- [ ] NaN/Inf loss 감지 시 학습 중단

---

## 3. Validation & Early Stopping

### 3.1 Validation 루프
- [ ] `validate()` 함수 구현 (`train_utils.py`에 추가)
  ```python
  @torch.no_grad()
  def validate(model, dataloader, criterion, config, device):
      model.eval()
      # ... 동일한 loss 계산, backward 없이
      return avg_losses
  ```
- [ ] 데이터셋 분할: Train/Val split (subject 단위, 예: 80/20)
  - 같은 subject의 다른 session이 train/val에 섞이지 않도록
- [ ] 에폭마다 validation loss 기록

### 3.2 Early Stopping
- [ ] `EarlyStopping` 클래스 구현
  ```python
  class EarlyStopping:
      def __init__(self, patience: int = 10, min_delta: float = 1e-4):
          ...
      def step(self, val_loss: float) -> bool:  # True = stop
  ```
- [ ] `patience` 에폭 동안 개선 없으면 Phase 조기 종료

---

## 4. Loss 함수 확장

### 4.1 추가 Loss 구현
- [ ] `ContrastiveLoss` — 같은 sample/variate의 패치 vs 다른 것의 패치

---

## 5. 다운스트림 태스크 평가 파이프라인

> master_plan.md Phase 4 Section 5 참조. 분류 태스크에 nn.Linear 등 학습 가능 head 추가 엄격 금지.
> **[설계 결정 — Resampling]** 평가 데이터도 학습 데이터와 동일하게 `target_sampling_rate = 100Hz`로 resampling되어 있어야 한다.
> 평가 파이프라인은 별도의 resampling 로직 없이 processed 데이터를 그대로 사용한다.
> 평가 시 `PackCollate`는 단일 `patch_size`(기본값 128)로 호출한다.

### 5.1 PrototypicalClassifier (`eval/fewshot.py`)
비-파라미터 분류기 (nn.Module 아님, 학습 가능 파라미터 없음):
- [x] `_embed(model, batch)`: `model.extract_features()` → `patch_sample_id`/`patch_variate_id`로 샘플별 분리 → 유효 패치 mean pooling → `(n, d_model)`
- [x] `fit(model, support_batches, labels)`: 클래스별 평균 프로토타입 벡터 계산 → `self.prototypes: (num_classes, d_model)`
- [x] `predict(model, query_batch)`: cosine similarity → argmax → 클래스 할당
- [x] `evaluate(model, query_batches, labels)`: predict + Accuracy, Balanced Accuracy, Macro F1, Cohen's Kappa, Confusion Matrix 계산

### 5.2 evaluate_forecasting (`eval/forecasting.py`)
- [x] `model.generate(batch, n_steps)` → ground truth 비교
- [x] 메트릭: MSE, MAE, MAPE
- [x] denormalize 옵션 지원

### 5.3 evaluate_imputation (`eval/imputation.py`)
- [x] 마스킹 대상 variate를 지정하여 전체 마스킹 → `model.forward(task="masked")` → 재구성 품질 평가
- [x] 메트릭: MSE, MAE, Pearson r

### 5.4 공통 메트릭 헬퍼
- [x] `eval/_metrics.py` — `regression_metrics(pred, target)`: MSE, MAE, MAPE, Pearson r 일괄 계산 헬퍼

### 5.4 사전학습 (Pre-training) 메트릭
- [ ] `module/metrics.py` 생성
- [ ] Reconstruction MSE (전체 패치 / 마스킹된 패치 / 비마스킹 패치)
- [ ] Reconstruction SNR (Signal-to-Noise Ratio)

---

## 6. 로깅 & 시각화

### 6.1 로깅
- [ ] `logging` 모듈 설정 (console + file)
- [ ] 에폭별 기록: `train_loss`, `val_loss`, `lr`, `epoch_time`, `gpu_memory`
- [ ] CSV 로그 파일 (`outputs/training_log.csv`)

### 6.2 TensorBoard (선택)
- [ ] `torch.utils.tensorboard.SummaryWriter` 통합
- [ ] Loss curve, LR curve
- [ ] Gradient norm histogram
- [ ] Reconstruction 시각화 (original vs reconstructed waveform)

### 6.3 결과 시각화 스크립트
- [ ] `scripts/plot_loss.py` — Loss curve 플롯
- [ ] `scripts/plot_reconstruction.py` — 원본 vs 복원 신호 비교
- [ ] `scripts/plot_attention.py` — Attention weight heatmap
- [ ] `scripts/plot_embedding.py` — t-SNE/UMAP of encoded representations

---

## 7. 분산 학습 (DDP)

### 7.1 기본 DDP
- [ ] `torch.nn.parallel.DistributedDataParallel` 래핑
- [ ] `DistributedSampler` + `GroupedBatchSampler` 호환 (커스텀 분산 샘플러 필요)
- [ ] `torch.distributed.launch` 또는 `torchrun` 진입점
- [ ] Gradient synchronization 검증

### 7.2 멀티노드
- [ ] NCCL 백엔드 설정
- [ ] 환경변수 기반 설정 (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`)

---

## 8. 핵심 참조 코드

### 학습 실행 패턴
```bash
# Phase 1: Channel-Independent (CI)
python 1_channel_independency.py
# → outputs/phase1_ci/checkpoint_phase1_ci_epoch*_best.pt

# Phase 2: Any-Variate + Cross-Modal
python 2_any_variate.py
# 또는 직접 지정:
python 2_any_variate.py --resume outputs/phase1_ci/checkpoint_phase1_ci_epoch069_best.pt
```

### Phase별 설정 비교
| | Phase 1 (CI) | Phase 2 (Any-Variate) |
|---|---|---|
| `collate_mode` | `"ci"` | `"any_variate"` |
| `batch_size` | 16 | 4 |
| `lr` | 1e-3 | 1e-4 |
| `n_epochs` | 70 | 30 |
| `alpha/beta/gamma` | 1.0/1.0/0.0 | 0.7/0.3/1.0 |
| `variate_mask_prob` | 0.0 | 0.3 |
| `max_horizon` | 5 | 5 |
| cross-modal loss | 비활성 | 활성 |
| checkpoint | `outputs/phase1_ci/` | `outputs/phase2_any_variate/` |

### 모델 출력 활용 (Loss 계산)
```python
from loss.criterion import CombinedLoss
from loss.masked_mse_loss import create_patch_mask

out = model(batch, task="masked")

# 원본 패치 추출
P = model.patch_size
normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
B, L = normalized.shape
N = L // P
original_patches = normalized.reshape(B, N, P)

# 마스킹
pred_mask = create_patch_mask(
    out["patch_mask"], mask_ratio=0.15,
    patch_variate_id=out["patch_variate_id"],  # Phase 2
    variate_mask_prob=0.3,                      # Phase 2
)

# CombinedLoss
criterion = CombinedLoss(alpha=0.7, beta=0.3, gamma=1.0)
losses = criterion(
    reconstructed=out["reconstructed"],
    next_pred=next_pred,           # from model(batch, task="next_pred", horizon=H)
    original_patches=original_patches,
    pred_mask=pred_mask,
    patch_mask=out["patch_mask"],
    patch_sample_id=out["patch_sample_id"],
    patch_variate_id=out["patch_variate_id"],
    horizon=H,
    cross_pred=out["cross_pred"],  # cross-modal head 출력
    time_id=out["time_id"],        # cross-modal 페어링용
)
# losses: {"total", "masked_loss", "next_loss", "cross_modal_loss"}
```

### Checkpoint 패턴
```python
from model.checkpoint import save_checkpoint, load_checkpoint

# 저장
save_checkpoint(path, model, optimizer=optimizer, epoch=epoch,
                config=model_config, phase="phase1_ci", loss=best_loss)

# 복원 (Phase 2에서 Phase 1 checkpoint 로드)
model = BiosignalFoundationModel(...)
state = load_checkpoint(ckpt_path, model, device=device)
# state["epoch"], state["loss"], state["config"] 등 접근 가능
```
