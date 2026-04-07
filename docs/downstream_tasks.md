# Downstream Tasks

## Overview

Biosignal Foundation Model의 사전학습된 표현력을 검증하기 위한 downstream evaluation 구성.
9개 task, 3가지 모델 API, 7종 신호, 3개 외부 데이터 소스를 커버한다.

### Task 목록

| # | Task | 유형 | 모델 API | 데이터 | External? |
|---|------|------|----------|--------|-----------|
| 1 | Hypotension Prediction | Binary Classification | `extract_features` | MIMIC-III | O |
| 2 | Arrhythmia Detection | 5-class Classification | `extract_features` | PTB-XL | O |
| 3 | BIS Prediction | Regression | `extract_features` | VitalDB | X (유일한 공개 데이터) |
| 4 | Anomaly Detection | Zero-shot scoring | `forward(masked)` | MIMIC-III | O |
| 5 | Any-to-Any Cross-modal | Zero-shot cross-modal | `forward(masked)` | MIMIC-III | O |
| 6 | Imputation | Temporal 결측 복원 | `forward(masked)` | MIMIC-III | O |
| 7 | Forecasting | Autoregressive 생성 | `generate()` | MIMIC-III | O |
| 8 | Ventilation Quality | 2-class Classification | `extract_features` | VitalDB | X |
| 9 | Mortality (예정) | Binary Classification | `extract_features` | MIMIC-III | O |

### 모델 API별 분류

- **`extract_features()`**: Encoder representation 추출 -> pooling -> LinearProbe 학습
  - Hypotension, Arrhythmia, BIS, Ventilation Quality, Mortality
- **`forward(task="masked")`**: Zero-shot masked reconstruction / cross-modal prediction
  - Anomaly, Any-to-Any, Imputation
- **`generate()`**: Autoregressive 다단계 미래 waveform 생성
  - Forecasting

### 외부 데이터 소스

| 데이터셋 | 국가 | 환경 | 사용 Task |
|----------|------|------|-----------|
| MIMIC-III Waveform Matched | 미국 | ICU | Hypotension, Anomaly, Any-to-Any, Imputation, Forecasting, Mortality |
| PTB-XL | 독일 | 외래 | Arrhythmia |
| CapnoBase | 캐나다 | 마취 | (RR Estimation 예정) |

---

## 디렉토리 구조

```
downstream/
├── data_utils.py       # VitalDB pilot 데이터 로딩, 윈도우 추출, 라벨링 유틸리티
├── model_wrapper.py    # DownstreamModelWrapper (encoder freeze/unfreeze, extract_features)
│                       # LinearProbe (LayerNorm -> Dropout -> Linear)
├── metrics.py          # AUROC, AUPRC, F1, MAE, Pearson r, Bland-Altman
├── viz.py              # ROC curve, Bland-Altman plot, Reconstruction plot
│
├── hypotension/
│   ├── prepare_data.py # MIMIC-III/VitalDB -> 미래 MAP<65 예측 데이터 생성
│   └── run.py          # LinearProbe 학습 + AUROC/AUPRC 평가
│
├── arrhythmia/
│   ├── prepare_data.py # PTB-XL 다운로드 + 공식 10-fold split + .pt 저장
│   └── run.py          # 5-class LinearProbe + confusion matrix + per-class AUROC
│
├── bis/
│   └── run.py          # VitalDB EEG+BIS inline 로딩 + regression probe + Bland-Altman
│
├── anomaly/
│   └── run.py          # Zero-shot reconstruction MSE scoring + AUROC
│
├── any_to_any/
│   ├── prepare_data.py # VitalDB/MIMIC-III 다채널 정렬 윈도우 추출
│   └── run.py          # 7개 시나리오 (ECG->ABP 등), reconstructed vs cross_pred 비교
│
├── imputation/
│   ├── prepare_data.py # 단일 신호 temporal 윈도우 추출
│   └── run.py          # 마스킹 구간 복원 MSE/MAE/Pearson r
│
├── forecasting/
│   ├── prepare_data.py # MIMIC-III context+target 윈도우 쌍 추출
│   └── run.py          # generate() -> 미래 waveform 비교
│
└── ventilation_quality/
    └── prepare_data.py # VitalDB CO2+AWP -> ETCO2 기반 2-class 라벨링
```

---

## Task 상세

### 1. Hypotension Prediction

미래 5~30분 후 MAP<65mmHg 발생 여부를 예측한다.

- **입력 모드**: ABP / ECG / PPG / ECG+PPG (4가지)
- **Label 소스**: ABP waveform의 미래 구간에서 MAP 계산
- **Horizon**: 5/10/15/20/25/30분 설정 가능
- **Window**: 30~300초
- **방식**: `extract_features()` -> mean pooling -> LinearProbe (BCEWithLogitsLoss)
- **평가**: AUROC, AUPRC, Sensitivity, Specificity
- **데이터**: MIMIC-III Waveform Matched Subset (open access)

```bash
# 데이터 준비
python -m downstream.hypotension.prepare_data \
    --source mimic3 --input-signals ecg --horizon-min 5 --n-cases 10

# 평가 (dummy)
python -m downstream.hypotension.run --dummy \
    --data-path outputs/downstream/hypotension/task1_hypotension_mimic3_ecg_h5min.pt

# 평가 (checkpoint)
python -m downstream.hypotension.run --checkpoint path/to/best.pt \
    --data-path outputs/downstream/hypotension/task1_hypotension_mimic3_ecg_h5min.pt
```

### 2. Arrhythmia Detection

PTB-XL 12-lead ECG의 5-class superclass 분류.

- **입력**: ECG Lead II (10초, 100Hz = 1,000 samples)
- **Classes**: NORM(0), MI(1), STTC(2), CD(3), HYP(4)
- **Split**: 공식 10-fold (train: fold 1-8, val: fold 9, test: fold 10)
- **방식**: `extract_features()` -> LinearProbe (CrossEntropyLoss)
- **평가**: AUROC (macro), F1 (macro/weighted), confusion matrix, per-class AUROC
- **데이터**: PTB-XL v1.0.3 (open access, 21,799 ECG)

```bash
# 데이터 준비 (다운로드 포함)
python -m downstream.arrhythmia.prepare_data --download --n-records 0

# 평가
python -m downstream.arrhythmia.run --dummy \
    --data-path outputs/downstream/arrhythmia/arrhythmia_ptbxl_II.pt
```

### 3. BIS Prediction

EEG waveform에서 BIS(Bispectral Index, 0~100) 수치를 regression.

- **입력**: EEG waveform (BIS/EEG1_WAV, 128Hz -> 100Hz)
- **출력**: BIS 연속값 (0=isoelectric, 40~60=적정 마취, 60~100=각성)
- **방식**: `extract_features()` -> LinearProbe (MSELoss)
- **평가**: MAE, RMSE, Pearson r, Bland-Altman, 3-bin accuracy
- **데이터**: VitalDB (internal - EEG+BIS 동시 공개 데이터가 VitalDB 유일)

```bash
python -m downstream.bis.run --dummy
python -m downstream.bis.run --checkpoint path/to/best.pt --n-cases 50
```

### 4. Anomaly Detection

Reconstruction MSE 기반 zero-shot anomaly scoring.

- **입력**: 임의 신호 (ECG, ABP, PPG 등)
- **방식**: `forward(task="masked")` -> reconstruction error = anomaly score
- **추가 학습 없음** (zero-shot)
- **평가**: AUROC, AUPRC, F1@optimal threshold
- **데이터**: MIMIC-III (ECG/ABP/PPG)

```bash
python -m downstream.anomaly.run --dummy --signal-type ecg
python -m downstream.anomaly.run --checkpoint path/to/best.pt --signal-type ecg
```

### 5. Any-to-Any Cross-modal Prediction

한 종류의 신호로 다른 종류의 신호 waveform을 zero-shot 복원.

- **시나리오**: ECG->ABP, PPG->ABP, ECG+PPG->ABP, ABP->ECG, ECG+ABP->PPG, ECG+PPG+CVP->ABP, ECG->EEG
- **방식**: `forward(task="masked")` -> `reconstructed` head + `cross_pred` head 비교
- **추가 학습 없음** (zero-shot)
- **평가**: MSE, MAE, Pearson r (normalized scale). ABP target 시 SBP/DBP 추출 가능 (AAMI/BHS)
- **데이터**: MIMIC-III (cardiovascular 시나리오)

```bash
python -m downstream.any_to_any.run --dummy
python -m downstream.any_to_any.run --checkpoint path/to/best.pt --n-cases 20
```

### 6. Imputation (Temporal)

같은 채널의 시간 구간 결측을 앞뒤 context로 복원.

- **입력**: 임의 신호 (마스킹된 시간 구간 포함)
- **방식**: `forward(task="masked")` -> `reconstructed`
- **추가 학습 없음** (zero-shot)
- **평가**: 마스킹 구간의 MSE, MAE, Pearson r
- **데이터**: MIMIC-III (ECG/ABP/PPG)

```bash
# 데이터 준비
python -m downstream.imputation.prepare_data --source mimic3 --signal-type ecg --n-cases 10

# 평가
python -m downstream.imputation.run --dummy
python -m downstream.imputation.run --checkpoint path/to/best.pt \
    --data-path outputs/downstream/imputation/imputation_mimic3_ecg.pt
```

### 7. Forecasting

과거 waveform으로 미래 waveform을 autoregressive 생성.

- **입력**: 과거 30초 waveform (context)
- **출력**: 미래 10초 waveform (target)
- **방식**: `model.generate(batch, n_steps)` — 1-step prediction -> append -> 반복
- **추가 학습 없음** (zero-shot)
- **평가**: MSE, MAE, Pearson r (generated vs target)
- **데이터**: MIMIC-III (ECG/ABP/PPG)

```bash
# 데이터 준비
python -m downstream.forecasting.prepare_data --signal-type ecg --n-cases 10

# 평가
python -m downstream.forecasting.run --dummy
python -m downstream.forecasting.run --checkpoint path/to/best.pt \
    --data-path outputs/downstream/forecasting/forecasting_mimic3_ecg_ctx30s_tgt10s.pt
```

### 8. Ventilation Quality

CO2+AWP waveform으로 환기 품질(과환기/정상) 2-class 분류.

- **입력**: CO2 + AWP (또는 CO2 단독 / AWP 단독)
- **Label**: ETCO2 numeric 기반 자동 라벨링 (<35mmHg = 과환기, >=35 = 정상)
- **방식**: `extract_features()` -> LinearProbe (BCEWithLogitsLoss)
- **평가**: AUROC, F1
- **데이터**: VitalDB (internal - CO2+AWP 동시 공개 데이터가 VitalDB 유일)

```bash
python -m downstream.ventilation_quality.prepare_data --n-cases 10 --visualize
```

### 9. Mortality Prediction (예정)

ICU 입실 초기 48시간 waveform으로 원내 사망 예측. MIMIC-III Matched Subset + Clinical DB 연결.

- **입력**: ECG + ABP + PPG (처음 48시간)
- **출력**: In-hospital mortality (binary)
- **방식**: `extract_features()` -> window embeddings -> aggregation -> linear probe
- **데이터**: MIMIC-III Clinical DB (credentialed access 필요)
- **상태**: 대기 중

---

## 공통 인프라

### DownstreamModelWrapper (`downstream/model_wrapper.py`)

```python
from downstream.model_wrapper import DownstreamModelWrapper, LinearProbe

# Checkpoint 로드 + encoder freeze
wrapper = DownstreamModelWrapper("checkpoints/best.pt", model_version="v1")

# Feature 추출
features = wrapper.extract_features(batch, pool="mean")  # (B, d_model)
features = wrapper.extract_features(batch, pool="none")   # (B, N, d_model)

# Linear probe
probe = LinearProbe(wrapper.d_model, n_classes=5)
logits = probe(features)
```

### Variate별 Pooling

`pool="none"`으로 패치 레벨 feature를 받은 뒤, `patch_signal_types`로 variate별 분리 pooling 가능:

```python
out = model(batch, task="masked")
encoded = out["encoded"]                    # (B, N, d_model)
signal_types = out["patch_signal_types"]     # (B, N)

# ECG 패치만 pooling
ecg_mask = (signal_types == 0)
ecg_feat = (encoded * ecg_mask.unsqueeze(-1).float()).sum(1) / ecg_mask.sum(1, keepdim=True).clamp(min=1)

# PPG 패치만 pooling
ppg_mask = (signal_types == 3)
ppg_feat = (encoded * ppg_mask.unsqueeze(-1).float()).sum(1) / ppg_mask.sum(1, keepdim=True).clamp(min=1)

# Concat
feature = torch.cat([ecg_feat, ppg_feat], dim=-1)  # (B, d_model*2)
```

### MIMIC-III Waveform Parser (`data/parser/mimic3_waveform.py`)

PhysioNet MIMIC-III Waveform Matched Subset에서 ABP/ECG/PPG를 스트리밍으로 읽고 전처리:

```bash
# ABP manifest 스캔
python -m data.parser.mimic3_waveform scan --max-records 200

# 파싱 + 전처리 + 시각화
python -m data.parser.mimic3_waveform parse --n-cases 5 --visualize
```

- Open access (credentialed 불필요)
- 125Hz -> 100Hz 리샘플링
- VitalDB와 동일한 전처리 파이프라인 (range check, spike detection, filter)
- 시간 정렬된 다채널 로딩 지원 (segment 내 동시 존재 채널만)

### PTB-XL Parser (`data/parser/ptbxl.py`)

PTB-XL 12-lead ECG 로딩 + 5-class superclass 분류:

```python
from data.parser.ptbxl import load_ptbxl_split

splits = load_ptbxl_split("datasets/ptb-xl/1.0.3", lead="II", sampling_rate=100)
# {"train": (samples, labels), "val": (...), "test": (...)}
```

---

## Instance Normalization with Loc/Scale Injection

### 정규화

각 입력 시퀀스 내 동일 (sample_id, variate_id) 그룹별로 Z-score 정규화:

```
x_norm = (x - mu) / sigma
```

복원 손실은 전적으로 normalized scale에서 계산된다.

### Loc/Scale Injection

정규화로 소실되는 절대 스케일 정보를 모델 내부에 직접 주입:

```python
# model/biosignal_model.py
loc_emb = self.loc_proj(patch_loc)        # Linear(1, d_model)
scale_emb = self.scale_proj(patch_scale)  # Linear(1, d_model)
embedded = embedded + (loc_emb + scale_emb) * valid_token
```

이를 통해 `extract_features()`의 representation에 amplitude 정보가 내장된다.
Downstream classification/regression에서 별도 denormalize 없이 절대 스케일 의존 task
(Hypotension MAP<65, BIS 0~100 등)가 가능하다.

이 설계는 MOMENT(ICML 2024)의 한계("vertically shifted 시계열 구분 불가")를
구조적으로 해결한다.

---

## Block Masking

연속 패치 블록 단위로 마스킹하여 인접 패치 보간을 방지:

```yaml
block_mask: true
block_size_min: 2   # 최소 2패치 (2초) 연속 마스킹
block_size_max: 4   # 최대 4패치 (4초) 연속 마스킹
```

- 블록 배치 후 양옆 1패치 gap 확보 -> `block_size_max` 초과 방지
- Fallback 랜덤 패치도 기존 마스킹 인접 제외
- ECG/ABP/PPG: 2~7 심박 주기 마스킹 (적절)
- EEG: quasi-stationary segment (2~4초)와 일치
- CO2/AWP: 호흡 주기(3~5초) 대비 다소 짧을 수 있음
