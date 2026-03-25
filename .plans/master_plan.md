# Master Plan — Biosignal Foundation Model

> **목적**: 서브 에이전트 간 소통 단절 없이 프로젝트를 고도화하기 위한 단일 진실 원천(Single Source of Truth).
> 각 서브 에이전트는 자신의 `.plans/plan_*.md`를 읽고 작업하되, 이 파일의 **데이터 규약**과 **오케스트레이션 지침**을 반드시 준수한다.

---

## 1. 프로젝트 마일스톤

### Phase 0: 기반 정비 (Infrastructure)
- [x] 프로젝트 디렉토리 구조 확립 (`data/`, `module/`, `model/`, `tests/`)
- [x] Python 3.13 + PyTorch 2.10 환경 구성 (`.venv/`)
- [x] CI/CD: pytest 기반 테스트 프레임워크 (166 tests passing)
- [ ] `main.py`를 실제 학습 진입점으로 교체 (현재 빈 템플릿)

### Phase 1: 데이터 파이프라인 완성
- [x] `BiosignalDataset` — Lazy-loading + Sliding window + LRU cache
- [x] `PackCollate` — FFD bin-packing + patch_size 정렬
- [x] `GroupedBatchSampler` — 세션 기반 그루핑
- [x] `create_dataloader` — patch_size/stride 파라미터 지원
- [ ] 여러 데이터셋 파서 개발 (VitalDB, SHHS, MESA, PhysioNet 등)
- [ ] 데이터 증강 (time-shift, scaling, Gaussian noise)
- [ ] 대규모 데이터셋 프로파일링 (메모리, I/O 병목)

### Phase 2: 모델 아키텍처 고도화
- [x] `PatchEmbedding` — Non-overlapping / Overlapping patching
- [x] `MultiResolutionPatchEmbedding` — **사용하지 않음**: 강제 resampling + 단일 patch_size 전략으로 결정. 구현 코드는 보존하되 모델에서 사용하지 않음.
- [x] `BiosignalFoundationModel` — Scaler → Patch → Transformer → Head
- [x] `TransformerEncoder` — GQA, GLU FFN, RoPE, BinaryAttentionBias
- [x] `Spatial Positional Encoding` — signal_type + spatial_id 이중 임베딩
- [x] `Masked Patch Modeling` - BERT 스타일의 masked patch modeling (`loss/masked_mse_loss.py`)
- [x] `Next Token Prediction` - GPT 스타일의 next token modeling (`loss/next_prediction_loss.py`)
- [x]  2-Phase에서 서로 다른 신호간 `Masked Patch Modeling` 그리고 `Next Token Prediction` loss 구현 (cross-modal loss)
- [x] `CombinedLoss` — Masked Patch Modeling + Next-Patch Prediction + Cross-Modal 복합 loss (`loss/criterion.py`)
- [ ] Cross-Modal Reconstruction 생리학적 제약 적용 — Mechanism Group 기반 필터링
  - **Reconstruction (MSE)**: 같은 mechanism group 내로 제한 (ECG↔ABP↔PPG는 허용, ECG↔EEG는 차단)
  - **Contrastive (InfoNCE)**: 전체 cross signal_type 허용 (representation alignment 수준)
  - `spatial_map.py`에 `MECHANISM_GROUP` 매핑 추가: Cardiovascular(ECG,ABP,PPG)=0, Neural(EEG)=1, Muscular(EMG)=2, Respiratory(Resp)=3
  - `_cross_modal_loss`에서 signal_type의 mechanism group이 동일한 쌍만 매칭하도록 필터링
- [ ] MoE 라우팅 검증 및 학습 안정화
- [ ] Decoder 아키텍처 (forecasting head만 — 분류는 Prototypical Network)
- [ ] 모델 스케일링 실험 (d_model, num_layers, num_heads 탐색)

### Phase 3: 학습 파이프라인 구축
- [x] Checkpoint save/load 구현 (`model/checkpoint.py`)
- [x] 2-Phase 커리큘럼 학습 (CI → Any-Variate) 기능 개발 (`1_channel_independency.py`, `2_any_variate.py`)
- [ ] 학습 루프 완성 (`main.py`)
- [ ] Mixed Precision (AMP) 적용
- [ ] Gradient clipping, warmup scheduler
- [ ] Validation loop + Early stopping
- [ ] 분산 학습 (DDP) 지원

### Phase 4: Inference API 및 다운스트림
- [x] `extract_features(batch)` — 양방향 attention feature 추출
- [x] `forecast(batch, horizon, denormalize)` — 단일-step 예측 + denormalization
- [x] `generate(batch, n_steps, denormalize)` — autoregressive 다단계 생성
- [x] `_append_patch_to_batch()` — autoregressive용 PackedBatch append 헬퍼
- [ ] loss 수렴 확인
- [ ] 결과 시각화 (loss curve, attention map, reconstruction)

### Phase 5: 다운스트림 태스크 및 평가 규약 (Downstream Tasks)

#### 5.1 핵심 타겟 포트폴리오 (Top 4)
| # | 태스크 | 방식 | 입출력 | 평가 모듈 | 매핑 API |
|---|--------|------|--------|----------|---------|
| 1 | ICU Forecasting (혈역학적 위기 예측) | Next-Patch Prediction (Regression) | 10min ECG/ABP → 5-10min IOH 궤적 | eval/forecasting.py | model.generate() / forecast() |
| 2 | Anesthesia State Analysis (마취 심도) | Few-shot Classification/Regression | Raw EEG → Burst Suppression + BIS | eval/fewshot.py | extract_features() → PrototypicalClassifier |
| 3 | Virtual Sensing (가상 센싱) | Masked Reconstruction (Imputation) | PPG + [masked ABP] → ABP 복원 | eval/imputation.py | model.forward(task="masked") |
| 4 | Sleep Staging Benchmark | Prototypical Network (Few-shot) | Sleep-EDF → W/N1/N2/N3/REM | eval/fewshot.py | extract_features() → PrototypicalClassifier |

#### 5.2 분류 아키텍처 규약: Prototypical Network
- **[경고]** nn.Linear 등 태스크 종속 분류 head 추가 및 재학습 엄격히 금지
- 인코더 가중치 완전 동결 (feature extractor only)
- Support Set: 클래스당 1-5개 전형 샘플 → 평균 프로토타입 벡터
- Query Set → 코사인 유사도 → 최고 유사도 클래스 할당

#### 5.3 평가 파이프라인 파일 구조
- [x] eval/__init__.py
- [x] eval/forecasting.py — ICU forecasting 평가
- [x] eval/fewshot.py — PrototypicalClassifier + few-shot 분류 평가
- [x] eval/imputation.py — 마스킹 재구성 imputation 평가
- [x] eval/_metrics.py — 공통 회귀 메트릭 헬퍼 (MSE, MAE, MAPE, Pearson r)

#### 5.4 태스크별 메트릭
| 태스크 | 메트릭 |
|--------|--------|
| ICU Forecasting | MSE, MAE, MAPE |
| Anesthesia (분류) | Accuracy, Balanced Accuracy, Cohen's Kappa |
| Anesthesia (BIS) | Pearson r, MSE |
| Virtual Sensing | MSE, MAE, Pearson r |
| Sleep Staging | Accuracy, Macro F1, Cohen's Kappa, Confusion Matrix |

---

## 2. 데이터 규약 (Data Contract)

> 모든 서브 에이전트는 이 규약에 명시된 클래스, 텐서 형상, 필드명을 정확히 사용해야 한다.

### 2.1 디스크 데이터 포맷

> **[설계 결정]** 모든 신호는 전처리 단계에서 `target_sampling_rate`로 강제 resampling된 뒤 저장된다.
> 모델은 단일 고정 `patch_size`만 사용하며 `MultiResolutionPatchEmbedding`은 사용하지 않는다.
> **기본 target_sampling_rate: 100 Hz** (ECG, EEG, PPG, Resp 등 모든 signal_type 공통 적용).

```
datasets/
  raw/                          # 원본 데이터 (EDF 등)
  processed/                    # 변환된 데이터 (resampling 완료)
    {subject_id}/
      manifest.json             # 아래 스키마 참조
      {session_id}_{signal_type_key}.pt   # torch.Tensor (n_channels, n_timesteps) float32
```

**manifest.json 스키마:**
```json
{
  "subject_id": "SC_00",
  "source": "sleep-edf",
  "sessions": [
    {
      "session_id": "SC4001E0",
      "recordings": [
        {
          "signal_type": 2,
          "file": "SC4001E0_eeg.pt",
          "n_channels": 2,
          "sampling_rate": 100.0,
          "n_timesteps": 2880000,
          "spatial_ids": [22, 23]
        }
      ]
    }
  ]
}
```

> **[규약]** `sampling_rate` 필드는 반드시 `target_sampling_rate`와 동일해야 한다.
> 파서가 resampling을 수행했음을 보증하며, `BiosignalDataset`은 `sampling_rate` 값을 신뢰한다.

**signal_type 매핑 (정수 코드):**

| 코드 | 타입 | 설명 (Description) | 비고 |
|:---:|:---:|:---|:---|
| 0 | **ECG** | 심전도 (Electrocardiogram) | 심장 활동 측정 |
| 1 | **ABP** | 동맥혈압 (Arterial Blood Pressure) | 실시간 혈압 모니터링 |
| 2 | **EEG** | 뇌전도 (Electroencephalogram) | 뇌파 활동 측정 |
| 3 | **PPG** | 광전용적맥파 (Photoplethysmogram) | 맥박 및 산소포화도 측정 |
| 4 | **EMG** | 근전도 (Electromyogram) | 근육 수축 및 활동 측정 |
| 5 | **Resp** | 호흡 (Respiration) | 호흡 주기 및 패턴 측정 |

**spatial_id 매핑 (`data/spatial_map.py` 참조):**
signal_type(대분류) + spatial_id(소분류) 이중 인코딩. 각 signal_type 내 0 = Unknown.
전역 ID는 `get_global_spatial_id(signal_type, local_id)` 로 변환. 총 55개.

### 2.2 핵심 데이터 클래스

#### `data.dataset.RecordingManifest`
```python
@dataclass
class RecordingManifest:
    path: str           # .pt 파일 절대/상대 경로
    n_channels: int
    n_timesteps: int
    sampling_rate: float
    signal_type: int    # 위 signal_type 매핑 참조
    session_id: str
    spatial_ids: list[int] | None = None  # per-channel 로컬 spatial_id, len == n_channels
```

#### `data.dataset.BiosignalSample`
```python
@dataclass
class BiosignalSample:
    values: torch.Tensor    # (time,) float32 — 단일 채널 윈도우
    length: int             # 유효 길이 (= values.shape[0])
    channel_idx: int        # 레코딩 내 채널 인덱스
    recording_idx: int      # manifest 내 레코딩 인덱스
    sampling_rate: float    # Hz
    signal_type: int
    session_id: str
    win_start: int          # 윈도우 시작 sample index
    spatial_id: int = 0     # 로컬 spatial_id (0=Unknown)
```

#### `data.collate.PackedBatch`
```python
@dataclass
class PackedBatch:
    values: torch.Tensor            # (batch, max_length) float32
    sample_id: torch.Tensor         # (batch, max_length) long — 1-based, 0=padding
    variate_id: torch.Tensor        # (batch, max_length) long — 1-based, 0=padding
    lengths: torch.Tensor           # (total_variates,) long
    sampling_rates: torch.Tensor    # (total_variates,) float — 항상 target_sampling_rate
    signal_types: torch.Tensor      # (total_variates,) long
    spatial_ids: torch.Tensor       # (total_variates,) long — 전역 spatial_id
    padded_lengths: Optional[torch.Tensor]         # (total_variates,) long | None
    variate_patch_sizes: Optional[torch.Tensor]    # (total_variates,) long | None — 단일 patch_size 사용으로 항상 None
```

> **[설계 결정]** 모든 신호가 동일한 sampling_rate를 가지므로 `variate_patch_sizes`는 항상 `None`이다.
> `PackCollate`의 `patch_sizes` / `target_patch_duration_ms` 파라미터는 사용하지 않는다.
> 단일 `patch_size`(기본값 128 timesteps = 1.28s at 100Hz)로 통일한다.

### 2.3 텐서 형상 변환 흐름

```
[Dataset]
  BiosignalSample.values:  (time,)                    # 단일 채널, 1D

[Collate — PackCollate]
  PackedBatch.values:      (B, max_length)             # bin-packed 2D
  PackedBatch.sample_id:   (B, max_length)             # 1-based sample index
  PackedBatch.variate_id:  (B, max_length)             # 1-based variate index

[Scaler — PackedAbsMeanScaler]
  loc:                     (B, max_length, 1)           # per-variate 위치
  scale:                   (B, max_length, 1)           # per-variate 스케일
  normalized:              (B, max_length)               # (values - loc) / scale

[Patch Embedding — PatchEmbedding]
  embedded:                (B, N, d_model)               # N = max_length // patch_size
  patch_sample_id:         (B, N) long
  patch_variate_id:        (B, N) long
  time_id:                 (B, N) long                   # variate 내 0-based 패치 순서
  patch_mask:              (B, N) bool                   # True = 유효 패치

[Spatial Embedding — Dual Additive]
  sig_emb:                 (B, N, d_model)               # signal_type embedding
  spa_emb:                 (B, N, d_model)               # spatial_id embedding
  embedded += sig_emb + spa_emb

[Attention Mask]
  attn_mask:               (B, N, N) bool                # same-sample & valid

[Transformer Encoder]
  encoded:                 (B, N, d_model)

[Reconstruction Head]
  reconstructed:           (B, N, patch_size)

[Loss — Masked Patch Modeling]
  pred_mask:               (B, N) bool                   # True = 마스킹 대상
  target:                  (B, N, patch_size)             # 원본 정규화 패치
  loss:                    scalar                         # MSE on masked patches only
```

### 2.4 주요 하이퍼파라미터 기본값

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `d_model` | 64 | 트랜스포머 임베딩 차원 |
| `num_layers` | 2 | 인코더 레이어 수 |
| `target_sampling_rate` | 100 | 전처리 시 모든 신호를 resampling하는 목표 Hz |
| `patch_size` | 128 | 패치 시간 스텝 수 (= 1.28s at 100Hz, 단일 고정값) |
| `num_heads` | `d_model // 64` | 어텐션 헤드 수 |
| `max_length` | 50000 | 패킹 행 너비 |
| `window_seconds` | 30.0 | 슬라이딩 윈도우 길이 |
| `mask_ratio` | 0.15 | MPM 마스킹 비율 |
| `batch_size` (CI) | 16 | Phase 1 배치 크기 |
| `batch_size` (MV) | 4 | Phase 2 배치 크기 |
| `lr` (CI) | 1e-3 | Phase 1 학습률 |
| `lr` (MV) | 1e-4 | Phase 2 학습률 |
| `use_rope` | True | RoPE 사용 |
| `use_var_attn_bias` | True | 변량 간 어텐션 바이어스 |
| `use_glu` | True | Gated Linear Unit FFN |

### 2.5 Collate 모드

| 모드 | 그루핑 키 | 용도 |
|------|----------|------|
| `"ci"` | `(sample_index,)` — 고유 키 | 채널 독립 학습 (Phase 1) |
| `"any_variate"` | `(session_id, physical_time_ms)` | 다변량 학습 (Phase 2) |

---

## 3. 오케스트레이션 지침

### 3.1 서브 에이전트 역할 분담

| 에이전트 | 플랜 파일 | 담당 영역 | 주요 파일 |
|----------|----------|----------|----------|
| **Data Engineer** | `.plans/plan_data.md` | 전처리, Dataset, DataLoader, 파서 | `data/`, `data/parser/` |
| **Model Architect** | `.plans/plan_model.md` | 모델 설계, 모듈 구현, 아키텍처 실험 | `module/`, `model/` |
| **Train & Eval** | `.plans/plan_eval.md` | 학습 루프, Loss, 스케줄러, 메트릭, 시각화, 다운스트림 평가 | `main.py`, `test_main*.py`, `loss/`, `eval/` |

### 3.2 작업 순서 의존성

```
[Data Engineer]                    [Model Architect]
  데이터 파서 추가                    MoE 검증
  데이터 증강 구현                    Decoder head 설계
         ↓                                ↓
         └──────────┬──────────────────────┘
                    ↓
              [Train & Eval]
                학습 루프 완성
                Checkpoint 구현
                평가 파이프라인
```

### 3.3 교차 의존 규칙

1. **Data → Model 인터페이스**: `PackedBatch` 데이터클래스가 유일한 인터페이스.
   - Data Engineer가 `PackedBatch` 필드를 변경하면 반드시 이 문서의 2.2절을 업데이트.
   - Model Architect는 `PackedBatch`의 필드만 사용하여 모델 입력 구성.

2. **Model → Eval 인터페이스**: `BiosignalFoundationModel.forward()` 반환 dict가 유일한 인터페이스.
   - 반환 키: `encoded`, `reconstructed`(masked), `cross_pred`(masked), `next_pred`(next_pred), `loc`, `scale`, `patch_mask`, `patch_sample_id`, `patch_variate_id`, `time_id`
   - Train & Eval은 이 dict의 키만 참조하여 loss 계산.
   - Inference API: `extract_features()`, `forecast()`, `generate()` 메서드 제공.

3. **변경 전파 프로토콜**:
   - `PackedBatch` 필드 추가/변경 → `master_plan.md` 2.2절 갱신 → 관련 에이전트에 통보
   - 모델 출력 dict 키 변경 → `master_plan.md` 2.3절 갱신 → Train & Eval에 통보
   - signal_type 코드 추가 → `master_plan.md` 2.1절 갱신 → 모든 에이전트에 통보

### 3.4 코딩 규약

- 텐서 타입: `torch.Tensor`로 선언, 차원은 인라인 주석 `# (batch, seq_len, dim)`
- 모듈 패턴: `nn.Module` 서브클래스, typed `__init__` + typed `forward()`
- 테스트: `pytest`, 파일명 `tests/test_{module_name}.py`
- Lint: `ruff` (미설정 — 추후 도입)
- 커밋 전 `pytest tests/ -v` 전체 통과 필수

---

## 4. 파일 트리 (현재)

```
C:\Projects\Biosignal-Foundation-Model\
├── data/
│   ├── __init__.py              # exports: BiosignalDataset, RecordingManifest, ...
│   ├── dataset.py               # BiosignalDataset, BiosignalSample, RecordingManifest
│   ├── collate.py               # PackCollate, PackedBatch, _PackUnit
│   ├── dataloader.py            # create_dataloader()
│   ├── sampler.py               # GroupedBatchSampler
│   ├── spatial_map.py           # Spatial ID 매핑 테이블 (signal_type + spatial_id)
│   └── parser/
│       ├── __init__.py
│       └── sleep_edf.py         # Sleep-EDF EDF → .pt 변환
├── loss/
│   ├── __init__.py              # exports: MaskedPatchLoss, create_patch_mask, NextPredictionLoss, CombinedLoss
│   ├── masked_mse_loss.py       # MaskedPatchLoss + create_patch_mask (랜덤/variate-level)
│   ├── next_prediction_loss.py  # NextPredictionLoss (same-variate + cross-modal)
│   └── criterion.py             # CombinedLoss (α*MPM + β*(NextPred + γ*CrossModal)), MaskedMSELoss
├── module/
│   ├── __init__.py              # exports: PatchEmbedding, MultiResolutionPatchEmbedding, ...
│   ├── _util.py                 # safe_div()
│   ├── attention.py             # GroupedQueryAttention, MHA, MQA
│   ├── ffn.py                   # FeedForward, GatedLinearUnitFeedForward, MoEFeedForward
│   ├── norm.py                  # RMSNorm
│   ├── packed_scaler.py         # PackedScaler, PackedStdScaler, PackedAbsMeanScaler
│   ├── patch.py                 # PatchEmbedding, MultiResolutionPatchEmbedding
│   ├── transformer.py           # TransformerEncoderLayer, TransformerEncoder
│   └── position/
│       ├── __init__.py          # exports: AttentionBias, RotaryProjection, ...
│       ├── attn_bias.py         # AttentionBias, BinaryAttentionBias
│       └── attn_projection.py   # Projection, RotaryProjection, QueryKeyProjection
├── model/
│   ├── __init__.py              # exports: BiosignalFoundationModel, save/load_checkpoint
│   ├── biosignal_model.py       # BiosignalFoundationModel + inference API
│   └── checkpoint.py            # save_checkpoint, load_checkpoint
├── eval/
│   ├── __init__.py              # exports: PrototypicalClassifier, evaluate_*
│   ├── forecasting.py           # ICU forecasting 평가
│   ├── fewshot.py               # PrototypicalClassifier + few-shot 분류
│   └── imputation.py            # 가상 센싱 imputation 평가
├── tests/
│   ├── test_attention.py        # GQA, MHA, MQA 테스트
│   ├── test_data.py             # Dataset, Collate, DataLoader 테스트
│   ├── test_eval.py             # 다운스트림 평가 파이프라인 테스트
│   ├── test_ffn.py              # FFN 변형 테스트
│   ├── test_integration.py      # 엔드투엔드 통합 테스트
│   ├── test_packed_scaler.py    # Scaler 테스트
│   ├── test_patch.py            # Patch 토큰화 테스트
│   ├── test_position.py         # RoPE, BinaryAttentionBias 테스트
│   └── test_transformer.py      # Transformer 테스트
├── .plans/
│   ├── plan_data.md             # Data Engineer 서브 플랜
│   ├── plan_model.md            # Model Architect 서브 플랜
│   └── plan_eval.md             # Train & Eval 서브 플랜
├── train/
│   ├── __init__.py
│   ├── train_utils.py           # 학습 공유 유틸리티 (TrainConfig, train_one_epoch, etc.)
│   ├── 1_channel_independency.py  # Phase 1: CI 사전학습
│   └── 2_any_variate.py         # Phase 2: Any-Variate + Cross-Modal 학습
├── main.py                      # 학습 진입점 (미구현)
├── test_main.py                 # 단일 Phase 학습 데모
├── test_main2.py                # 2-Phase 커리큘럼 데모
├── test_main3.py                # 패치 기반 MPM 데모
├── master_plan.md               # 이 파일
└── CLAUDE.md                    # 코딩 규약
```

---
# 5. 해봐야될 것들
- [] 학습이 제대로 진행이 안되면 `ConNext` 그리고 `NeuroNet` 스타일처럼 생체신호에 1D-CNN을 붙여서 학습을 진행