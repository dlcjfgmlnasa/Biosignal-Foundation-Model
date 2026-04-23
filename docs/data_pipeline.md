# Data Pipeline

## 데이터 소스

### Pretrain

| 소스 | 환경 | 환자 수 | 신호 |
|------|------|---------|------|
| **VitalDB OR** | 수술중 (SNUH, vitaldb.net 공개) | ~6,388 | ECG, ABP, EEG, PPG, CVP, CO2, AWP, PAP (8종) |
| **K-MIMIC-MORTAL** | ICU (SNUH 내부) | ~수천~1만 | ECG, ABP, PPG, CVP, PAP, ICP |
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

### VitalDB OR (`data/parser/vitaldb.py`)

**구조**: Flat (파일 1개 = 환자 1명)
```
vitaldb/
├── 0001.vital     # subject_id = 파일명 숫자 → VDB_0001
├── 0002.vital
└── ...
```

**명령**:
```bash
python -m data.parser.vitaldb --raw datasets/raw/vitaldb --out datasets/processed --workers 4
```

`--subject-from-parent 0` (default) — 파일명에서 숫자 추출해 subject_id 생성.

### K-MIMIC-MORTAL (`data/parser/vitaldb.py` 재사용)

SNUH ICU 데이터셋. VitalDB와 같은 `.vital` 파일 포맷이지만 **디렉토리 구조가 다름**.

**구조**: 4-level 중첩 (`hadm_id/subject_id/icustay/file`)
```
K-MIMIC-MORTAL/1.0.0/VITALDB/    # 여기 'VITALDB/'는 K-MIMIC 내부 하위 폴더 이름
├── 398/                          # bucket = hadm_id (입원 번호)
├── 413/
│   └── 6929/                     # subject_id (환자 ID, 전역 unique)
│       └── CCU_295205260750/     # ICU stay (CCU/MICU/RICU/SICU + timestamp)
│           └── SICU1_..vital     # 시간 분할 recording
└── ...
```

**⚠ 주의**: 중간의 `VITALDB/`는 K-MIMIC이 waveform 저장용으로 쓴 하위 폴더 이름일 뿐, 공개 VitalDB 프로젝트와 무관.

**구조 검증 근거** (같은 subject_id가 여러 bucket에 존재하는 경우):
- 같은 subject `6929`가 6개 bucket (413/422/424/427/433/440)에 등장
- 각 bucket의 ICU stay timestamp가 2952-03 → 2952-05 → ... → 2953-10으로 **시간 순 분포**
- Bucket 422 내부에 CCU→MICU→RICU→CCU 연속 stay 존재 (한 입원 내 ICU 이동)
- → `bucket = hadm_id`, `4자리 = subject_id(전역 unique)` 확정

**명령** (`--subject-from-parent 2` 필수):
```bash
python -u -m data.parser.vitaldb \
  --raw /path/to/K-MIMIC-MORTAL/1.0.0/VITALDB/ \
  --out /path/to/processed/k_mimic/ \
  --subject-from-parent 2 \
  --workers 16 \
  --skip-manifest-full

# 전체 완료 후 manifest_full 1회만 생성
python -u -m data.parser.vitaldb \
  --rebuild-manifest-full \
  --out /path/to/processed/k_mimic/
```

**파싱 플래그 의미**:
- `--subject-from-parent 2`: `vital_path.parents[1].name`을 subject로 사용 → `6929` → `VDB_6929`
- `--skip-manifest-full`: 분할 실행 시 O(N²) bottleneck 회피 (매 iteration마다 2,488+ subject manifest 재읽기)
- `--rebuild-manifest-full`: 모든 파싱 완료 후 1회만 통합 manifest 생성

**주요 차이점** (VitalDB OR vs K-MIMIC):

| 항목 | VitalDB OR | K-MIMIC |
|---|---|---|
| 구조 | Flat | 4-level 중첩 |
| 환경 | 수술 (OR) | ICU |
| Track 접두사 | `SNUADC/*`, `Primus/*`, `Solar8000/*` | `SNUADCM/*` |
| subject 당 session | 1 | 여러 개 (재입원) |
| 가용 신호 | ECG, ABP, PPG, CVP, CO2, AWP, PAP | ECG, ABP, PPG, CVP, PAP, ICP (CO2/AWP 없음) |
| `--subject-from-parent` | 0 (default) | **2** |
| 파일명 패턴 | `{subject}_S0_{signal}_..._seg{i}_{j}.pt` | `{subject}_S_{digits}_{signal}_..._seg{i}_{j}.pt` |

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

---

## Storage Strategy (Shard Backend)

### 현재 구조 (Legacy 2-step)

```
.vital (raw)
   ↓ data/parser/vitaldb.py
*.pt 수만 개 (per-recording)        ← 중간 단계, 디스크 낭비
   ↓ scripts/build_shards.py
shard_*.pt (~1GB each)              ← 최종 (학습에 사용)

총 디스크: 원본 + per-recording.pt + shard.pt = ~3배 데이터 크기
```

이유: shard backend는 기존 코드 작성 후 IO interrupt 문제 발견 시 추가됨.
2-step으로 점진적 진화한 결과 → **기술 부채**.

### 사용 시 주의

- `data_dir` (manifest 위치)와 `shard_index_path` (shard 위치) **둘 다 필수**
- 학습 시 manifest가 카탈로그 역할, shard가 실제 텐서 데이터
- per-recording `.pt` 파일은 shard 빌드 후 **삭제 가능** (디스크 절약 ~50%)
  - `scripts/build_shards.py --delete-source` 옵션 활용 (향후 추가)
  - 단, 일부 디버깅/탐색 도구가 직접 .pt 접근하면 보존 필요

### 새 데이터셋 추가 시 권장 (1-pass 설계)

신규 데이터셋 (K-MIMIC 추가, 외부 dataset 등) 처리 시:

```
.vital (raw)
   ↓ scripts/parse_to_shard.py (1-pass)
shard_*.pt + manifest_full.jsonl    ← 즉시 최종 형태

총 디스크: 원본 + shard.pt = ~2배 데이터 크기 (기존 대비 33% 절약)
```

핵심 차이:
- 중간 per-recording .pt 안 만듦 (메모리 → 직접 shard write)
- preprocess + shard build 한 번에
- 시간/디스크 모두 절약

**구현 위치**: `scripts/parse_to_shard.py` (TODO — 새 데이터셋 추가 시 작성)

### Shard backend 사용 RAM 가이드

shard cache는 **per-worker** 값이라 RAM 사용 =
`num_workers × shard_cache_size × shard_size_GB × num_ranks (DDP)`.

| 설정 | RAM 사용 | 안전성 |
|------|---------|--------|
| 4 workers × 2 cache × 1GB × 2 ranks | 16GB | ✓ 192GB 서버 안전 |
| 8 workers × 12 cache × 1GB × 2 ranks | **192GB** | ✗ OOM kill 발생 (실측) |

**권장**: `num_workers ≤ 4-8`, `shard_cache_size = 2`. Locality sampler가
sequential read 보장 → cache_size 작아도 hit ~100%.

### 검증 도구

| 스크립트 | 용도 |
|---------|------|
| `scripts/build_manifest_full.py` | per-subject manifest.json → manifest_full.jsonl 통합 |
| `scripts/build_shards.py` | per-recording .pt → shard 패킹 (`--workers N` parallel) |
| `scripts/test_shard_load.py` | shard 형식 + 통합 throughput 측정 |
| `scripts/bench_dataloader_memory.py` | 학습 시작 전 RAM 사용량 측정 (OOM 사전 차단) |
