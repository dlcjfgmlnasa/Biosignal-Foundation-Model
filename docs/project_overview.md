# Biosignal Foundation Model - 프로젝트 전체 개요

> **최종 갱신**: 2026-03-27

---

## 1. 프로젝트 개요

### 1.1 목적

수술 중 모니터링(Intraoperative Monitoring) 환경에서 수집되는 다양한 생체신호를 통합적으로 이해하는 **사전학습(Pre-trained) Foundation Model**을 구축한다. 단일 모델이 심전도(ECG), 혈압(ABP), 뇌파(EEG), 광전용적맥파(PPG), 근전도(EMG), 호흡(Resp) 등 이질적인 생체신호를 함께 학습하여, 다운스트림 임상 태스크에 범용적으로 활용할 수 있는 표현(representation)을 생성하는 것이 핵심 목표이다.

### 1.2 타겟 신호 (7종)

| 코드 | 신호 타입 | 설명 |
|:---:|:---:|:---|
| 0 | **ECG** | 심전도 (Electrocardiogram) — 심장 활동 측정 |
| 1 | **ABP** | 동맥혈압 (Arterial Blood Pressure) — 실시간 혈압 모니터링 |
| 2 | **EEG** | 뇌전도 (Electroencephalogram) — 뇌파 활동 측정 |
| 3 | **PPG** | 광전용적맥파 (Photoplethysmogram) — 맥박 및 산소포화도 |
| 4 | **EMG** | 근전도 (Electromyogram) — 근육 수축 및 활동 |
| 5 | **Resp** | 호흡 (Respiration) — 호흡 주기 및 패턴 |
| (6) | *(예비)* | 향후 확장용 |

### 1.3 주요 데이터 소스

- **VitalDB**: 수술 중 다채널 생체신호 데이터베이스 (Primary). 7종 신호 동시 수집.
- **Sleep-EDF**: 수면 다원검사 EEG 데이터 (보조 학습 및 Sleep Staging 벤치마크용).

---

## 2. 코드베이스 구조

```
C:\Projects\Biosignal-Foundation-Model\
├── data/                # 데이터 로딩 및 전처리 파이프라인
│   ├── dataset.py       # BiosignalDataset, BiosignalSample, RecordingManifest
│   ├── collate.py       # PackCollate, PackedBatch (FFD bin-packing)
│   ├── dataloader.py    # create_dataloader()
│   ├── sampler.py       # GroupedBatchSampler
│   ├── spatial_map.py   # signal_type + spatial_id 이중 인코딩 (12 global IDs)
│   └── parser/          # 원본 데이터 → 전처리 변환 (sleep_edf.py, vitaldb.py)
├── module/              # 재사용 가능한 신경망 빌딩 블록
│   ├── norm.py          # RMSNorm
│   ├── attention.py     # GQA, MHA, MQA
│   ├── ffn.py           # FeedForward, GLU FFN, MoE FFN
│   ├── patch.py         # PatchEmbedding
│   ├── packed_scaler.py # PackedStdScaler, PackedAbsMeanScaler
│   ├── transformer.py   # TransformerEncoderLayer, TransformerEncoder
│   ├── cnn_stem.py      # Conv1dStem, ModalityCNNStem (V2용)
│   └── position/        # RotaryProjection (RoPE), BinaryAttentionBias
├── model/               # 고수준 모델 정의
│   ├── biosignal_model.py  # BiosignalFoundationModel + Inference API
│   ├── v1.py            # V1: 모든 신호 raw patch 복원
│   ├── v2.py            # V2: EEG=CNN stem 복원, 나머지=raw 복원
│   ├── config.py        # ModelConfig (YAML 직렬화)
│   └── checkpoint.py    # save/load checkpoint
├── loss/                # 손실 함수
│   ├── masked_mse_loss.py      # MaskedPatchLoss, create_patch_mask
│   ├── next_prediction_loss.py # NextPredictionLoss
│   ├── contrastive_loss.py     # CrossModalContrastiveLoss (InfoNCE)
│   └── criterion.py            # CombinedLoss (복합 손실)
├── train/               # 학습 스크립트 및 유틸리티
│   ├── train_utils.py   # TrainConfig, train_one_epoch(), 공유 헬퍼
│   ├── v1_1_channel_independency.py  # V1 Phase 1
│   ├── v1_2_any_variate.py           # V1 Phase 2
│   ├── v2_1_channel_independency.py  # V2 Phase 1
│   └── v2_2_any_variate.py           # V2 Phase 2
├── eval/                # 평가 모듈
│   ├── forecasting.py   # ICU forecasting 평가
│   ├── fewshot.py       # PrototypicalClassifier
│   └── imputation.py    # 가상 센싱 imputation 평가
├── downstream/          # 8개 Downstream Task 구현
│   ├── common/          # 공통 인프라 (data_utils, model_wrapper, eval_utils)
│   └── task1~8/         # 개별 태스크 디렉토리
├── tests/               # pytest 기반 테스트 (166 tests)
├── main.py              # 학습 진입점 (YAML config + CLI override + --dry-run)
└── .plans/              # 프로젝트 계획 파일
```

---

## 3. 학습 로드맵

### Phase 0: 기반 정비 (완료)

- Python 3.13 + PyTorch 2.10 환경 구성
- 프로젝트 디렉토리 구조 확립
- pytest 기반 테스트 프레임워크 (166 tests passing)

### Phase 1: Channel-Independent (CI) 사전학습

**목적**: 각 채널을 독립적으로 취급하여, 단일 신호의 시간적 패턴을 학습한다.

- **Collate 모드**: `"ci"` — 각 채널이 고유한 sample로 패킹
- **Loss**: Masked Patch Modeling (MPM) + Next-Patch Prediction
- **학습률**: 1e-3, 배치 크기 16
- **핵심 학습 내용**: 개별 신호의 형태학적 특성, 시간적 의존성

### Phase 2: Any-Variate 다변량 학습

**목적**: 서로 다른 신호 간의 관계를 학습한다 (Cross-Modal).

- **Collate 모드**: `"any_variate"` — 같은 시간대의 다른 채널을 하나의 sample로 패킹
- **Loss**: MPM + Next-Pred + Cross-Modal Reconstruction + Contrastive Loss
- **학습률**: 1e-4 (Phase 1 checkpoint에서 이어서 학습), 배치 크기 4
- **핵심 학습 내용**: ECG-ABP 간 혈역학적 관계, EEG-EMG 근신경 관계 등

### Phase 3: Downstream Evaluation

**목적**: 사전학습된 모델의 표현력을 8개 임상 태스크로 검증한다.

- Encoder 가중치 동결 (frozen)
- 분류/회귀 태스크: Lightweight Linear Probe만 학습
- 복원/생성 태스크: 사전학습 head 직접 평가 (fine-tuning 없음)

---

## 4. Downstream 평가 계획 (8개 Task)

| # | Task | 유형 | 방식 | 핵심 메트릭 |
|---|------|------|------|-----------|
| 1 | Hypotension Prediction | Binary Classification | Frozen encoder + Linear Probe | AUROC, AUPRC |
| 2 | 서맥/빈맥 감지 | 3-class Classification | Frozen encoder + Linear Probe | AUROC (macro), F1 |
| 3 | Imputation (채널 단위) | Reconstruction | Frozen encoder + 기존 head | MSE, Pearson r |
| 4 | Cross-modal 예측 (ECG->ABP) | Cross-modal Generation | Frozen encoder + cross_head | MSE, Pearson r |
| 5 | 시간 구간 복원 | Temporal Reconstruction | Frozen encoder + 기존 head | MSE, Pearson r |
| 6 | 이상 탐지 | Anomaly Detection | Frozen encoder, recon loss = score | AUROC, F1 |
| 7 | 마취 심도 추정 (BIS) | Regression | Frozen encoder + Linear Probe | MAE, Pearson r |
| 8 | Any->Any 예측 | Cross-modal Generation | Frozen encoder + cross_head | MSE, Pearson r |

**데이터 분할**: VitalDB 뒤쪽 케이스 (학습 미사용), subject 단위 train/test 70/30 분할.

**Encoder Freeze 규약**:
- Task 1, 2, 7: `model.requires_grad_(False)` 후 `nn.Linear` probe만 학습
- Task 3~6, 8: encoder + 기존 head 모두 동결, forward만 실행

---

## 5. 인프라

### 5.1 GPU 서버 스펙

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA L4 x 2 |
| CPU | 8 vCPU |
| RAM | 96 GB |
| 디스크 | 100 GB |
| 네트워크 | 원내망 (Public repo만 pull 가능) |

### 5.2 100GB 디스크 제약 대응 전략

VitalDB 전체 데이터는 100GB 디스크에 수용 불가하므로 다음 전략을 검토:

- **우선 학습**: 약 3,500명 분의 데이터를 우선 처리
- **Rolling Buffer**: 학습 완료된 데이터 삭제 후 새 데이터 로드
- **On-The-Fly 처리**: 전처리를 실시간으로 수행하여 디스크 사용 최소화
- **하이브리드**: 위 전략들의 조합

### 5.3 분산 학습

- `torchrun` 기반 DDP (Distributed Data Parallel) 지원
- `launch_phase1.py` / `launch_phase2.py`: `--model_version v1|v2` 선택 가능
- 2x L4 GPU 환경에서 DDP 학습 가능

---

## 6. 현재 진행 상황 (2026-03-27 기준)

### 6.1 워크스트림별 완료율

| 워크스트림 | 완료율 | 상태 요약 |
|-----------|--------|----------|
| **Data Pipeline** | ~47% | 핵심 파이프라인 완료. Parser(VitalDB, Sleep-EDF) 완료. 데이터 증강, 대규모 프로파일링 미완. |
| **Model Architecture** | ~69% | V1/V2 모델 완료. MoE Switch Transformer 교체 완료. EEG reconstruction 전략 개선, mechanism group 필터링 미완. |
| **Training & Eval** | ~66% | 2-Phase 커리큘럼 학습 스크립트 완료. AMP, Validation loop, CSV 로그 미완. |
| **Downstream** | ~10% | 8개 Task 디렉토리 구조 생성. 공통 인프라 및 개별 Task 구현 미완. |

### 6.2 주요 미완료 항목

**Critical**:
- V2 EEG reconstruction loss 검증 (CNN stem 복원 방식의 실제 학습 효과 확인 필요)
- V1 vs V2 비교 실험 (Phase 1 loss curve 기준 선택)
- Mixed Precision (AMP) 적용

**High Priority**:
- EEG Reconstruction 전략 개선 (raw 복원 한계 → 단계적 개선)
- Cross-Modal Reconstruction에 Mechanism Group 기반 필터링 적용
- Validation loop + Early stopping 구현
- Downstream 공통 인프라 구현

**Medium Priority**:
- 데이터 증강 (time-shift, scaling, Gaussian noise)
- 대규모 데이터셋 프로파일링 (메모리, I/O 병목)
- MoE 라우팅 검증 (expert 활용 분포 모니터링)
- 모델 스케일링 실험

---

## 7. 핵심 설계 결정

### 7.1 단일 Sampling Rate 통일 (100Hz)

모든 신호를 전처리 단계에서 **100Hz로 강제 리샘플링**한다. 이를 통해:
- 모델이 단일 고정 `patch_size`만 사용 가능
- `MultiResolutionPatchEmbedding` 불필요 (구현은 보존하되 사용하지 않음)
- 데이터 파이프라인 및 모델 구조 단순화

### 7.2 단일 Patch Size (100 timesteps = 1.0s)

- 초기 기본값 128에서 **100으로 변경** (100Hz에서 정확히 1초에 대응)
- `variate_patch_sizes`는 항상 `None` (모든 신호가 동일한 patch size 사용)

### 7.3 V1 vs V2 비교 실험 전략

- **V1**: 모든 신호를 raw patch로 복원. 단순하고 통일된 학습 목표.
- **V2**: EEG (signal_type=2)만 CNN stem 출력을 복원 (stop-gradient), 나머지는 raw 복원. EEG의 고주파/저진폭 특성에 맞춘 별도 처리.
- **채택 기준**: Phase 1 학습 후 loss curve 및 downstream few-shot 평가로 결정 예정.

### 7.4 2-Phase 커리큘럼 학습

1. **Phase 1 (CI)**: 채널을 독립적으로 학습 → 개별 신호의 시간적 패턴 습득
2. **Phase 2 (Any-Variate)**: 다변량 학습 → 신호 간 관계(cross-modal) 학습

이 순서는 쉬운 것에서 어려운 것으로의 커리큘럼을 따르며, Phase 1에서 안정적인 표현을 먼저 확보한 후 Phase 2에서 cross-modal 관계를 추가 학습한다.

### 7.5 Next-Prediction Curriculum Horizon

- Phase 1 학습 100 epoch 이후 적용
- H=1 -> H=3 -> H=5로 점진적으로 예측 horizon 증가
- 짧은 horizon에서 안정적 예측을 확보한 후 장기 예측으로 확장

### 7.6 Prototypical Network 기반 분류

- 분류 태스크에서 **nn.Linear 등 태스크 종속 head 추가 및 재학습 금지** (downstream linear probe 제외)
- Encoder 가중치 완전 동결
- Support Set 기반 프로토타입 벡터 생성 -> 코사인 유사도 분류

---

## 8. 데이터 규약 요약

### 8.1 디스크 포맷

```
datasets/processed/{subject_id}/
  manifest.json          # 세션/레코딩 메타데이터
  {session_id}_{signal_type_key}.pt   # (n_channels, n_timesteps) float32
```

### 8.2 핵심 데이터 클래스

| 클래스 | 위치 | 역할 |
|--------|------|------|
| `RecordingManifest` | `data/dataset.py` | 레코딩 메타데이터 (경로, 채널수, 샘플수, 신호 타입) |
| `BiosignalSample` | `data/dataset.py` | 단일 채널 윈도우 데이터 + 메타정보 |
| `PackedBatch` | `data/collate.py` | 모델 입력 — bin-packed 배치 (values, sample_id, variate_id, ...) |

### 8.3 텐서 형상 변환 흐름

```
BiosignalSample.values: (time,)
    ↓ PackCollate
PackedBatch.values: (B, max_length)
    ↓ Scaler
normalized: (B, max_length)
    ↓ PatchEmbedding
embedded: (B, N, d_model)       # N = max_length // patch_size
    ↓ Spatial Embedding (Dual Additive)
embedded += sig_emb + spa_emb
    ↓ TransformerEncoder
encoded: (B, N, d_model)
    ↓ Reconstruction Head
reconstructed: (B, N, patch_size)
```

### 8.4 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `d_model` | 64 | 트랜스포머 임베딩 차원 |
| `num_layers` | 2 | 인코더 레이어 수 |
| `patch_size` | 100 | 패치 시간 스텝 수 (1.0s at 100Hz) |
| `target_sampling_rate` | 100 Hz | 전처리 리샘플링 목표 |
| `window_seconds` | 30.0 | 슬라이딩 윈도우 길이 |
| `mask_ratio` | 0.15 | MPM 마스킹 비율 |
| `max_length` | 50,000 | 패킹 행 너비 |
| `use_rope` | True | Rotary Position Embedding |
| `use_glu` | True | Gated Linear Unit FFN |

---

## 9. 관련 문서

| 문서 | 경로 | 내용 |
|------|------|------|
| Master Plan | `.plans/master_plan.md` | 프로젝트 마일스톤, 데이터 규약, 오케스트레이션 지침 |
| Data Pipeline 보고서 | `docs/data_pipeline.md` | 데이터 파이프라인 상세 |
| Model Architecture 보고서 | `docs/model_architecture.md` | 모델 아키텍처 상세 |
| Training & Loss 보고서 | `docs/training_and_loss.md` | 학습 전략 및 Loss 함수 상세 |
| Data Engineer 플랜 | `.plans/.agent_plan/plan_data.md` | 데이터 엔지니어 작업 계획 |
| Model Architect 플랜 | `.plans/.agent_plan/plan_model.md` | 모델 아키텍트 작업 계획 |
| Train & Eval 플랜 | `.plans/.agent_plan/plan_eval.md` | 학습/평가 작업 계획 |
| Downstream 플랜 | `.plans/.agent_plan/plan_downstream.md` | 다운스트림 평가 작업 계획 |
