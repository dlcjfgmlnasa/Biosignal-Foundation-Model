# Data Pipeline

## 데이터 소스

### Pretrain

| 소스 | 환경 | 환자 수 | 신호 |
|------|------|---------|------|
| **VitalDB** | 수술중 (SNUH) | 6,388 | ECG, ABP, EEG, PPG, CVP, CO2, AWP (7종) |
| **MIMIC-III Waveform** | ICU (Beth Israel) | 10,282 | ECG, ABP, PPG |

### Downstream External

| 소스 | 환경 | 환자 수 | 용도 | 접근 |
|------|------|---------|------|------|
| **MIMIC-III Waveform Matched** | ICU | 10,282 | Hypotension, Anomaly, Any-to-Any, Imputation, Forecasting | Open |
| **PTB-XL** | 외래 | 21,799 ECG | Arrhythmia (5-class) | Open |
| **CapnoBase** | 마취 | 42 | RR Estimation (예정) | Open |

## 전처리 파이프라인

```
Raw Signal -> Range Check -> Spike Detection -> NaN Segment Extraction
-> Median Filter -> Notch Filter -> Bandpass/Lowpass -> Resample (100Hz)
```

### 신호별 설정

| 신호 | Valid Range | Filter | Freq | Notch | Spike | Median |
|------|------------|--------|------|-------|-------|--------|
| ECG | -5~5 mV | Bandpass | 0.5-40Hz | 60Hz | O (10 MAD) | - |
| ABP | 20~300 mmHg | Lowpass | 0-15Hz | - | O (6 MAD) | 5 |
| EEG | -500~500 uV | Bandpass | 0.5-45Hz | 60Hz | O (10 MAD) | - |
| PPG | 0~2000 | Lowpass | 0-8Hz | 60Hz | O (6 MAD) | 5 |
| CVP | -5~40 mmHg | Lowpass | 0-10Hz | - | O (8 MAD) | - |
| CO2 | 0~100 mmHg | Lowpass | 0-5Hz | - | - | - |
| AWP | -20~80 hPa | Lowpass | 0-20Hz | - | - | - |

## 파서

### VitalDB (`data/parser/vitaldb.py`)
```bash
python -m data.parser.vitaldb --raw datasets/raw/vitaldb --out datasets/processed --workers 4
```

### MIMIC-III Waveform (`data/parser/mimic3_waveform.py`)
```bash
python -m data.parser.mimic3_waveform scan --max-records 200
python -m data.parser.mimic3_waveform parse --n-cases 5 --visualize
```
- wfdb 스트리밍 (로컬 다운로드 불필요)
- 다채널 시간 정렬 (segment 내 동시 채널만)

### PTB-XL (`data/parser/ptbxl.py`)
```bash
python -m downstream.arrhythmia.prepare_data --download --n-records 0
```
- 100Hz 직접 사용, 공식 10-fold split

## Data Collation (`data/collate.py`)

FFD bin-packing으로 가변 길이 배치:
- **CI 모드**: 단일 채널씩 (Phase 1)
- **Any-Variate 모드**: 다채널 동시 (Phase 2)
