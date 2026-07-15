# Data Pipeline

## 데이터 소스

> **2026-04-27 정책**: 사전학습은 **K-MIMIC-MORTAL 단독**. VitalDB Open은 OR 도메인 holdout/finetuning 전용으로 분리. 자세한 근거: `.plans/master_plan.md` §0.

### Pretrain (사전학습 메인)

| 소스 | 환경 | Raw 크기 | .vital 파일 | 추정 토큰 | 신호 |
|------|------|---------|---|---|------|
| **K-MIMIC-MORTAL** ⭐ | ICU (SNUH 내부) | **880 GB** | **228,608** | **~13B** | ECG, ABP, PPG, CVP, PAP, ICP |

**디스크 streaming 필수** — 서버 디스크 100 GB << raw 880 GB.
Subject batch 단위 [download → parse → shard → delete raw] 파이프라인. 자세한 구현: `.plans/.agent_plan/plan_data.md` §3.

#### ⚠️ Cohort 성격 — Deceased subset 임을 명시 (2026-06-24 확인)

K-MIMIC-MORTAL 은 일반 K-MIMIC 이 아니라 **K-MIMIC에서 사망환자만 추려 만든 파생 subset** 이다 (데이터 제공자 확인). 출력 폴더 이름이 `k_mimic/` 라 deceased 코호트임이 드러나지 않으니 주의. raw source = `datasets/K-MIMIC-MORTAL/1.0.0/VITALDB/`.

**검증 수치** (`EMR/admissions.csv` 직접 집계):

| 지표 | 값 |
|------|----|
| 고유 subject | 8,138 |
| 사망 경험 subject | **8,138 (100%)** → subject 단위 deceased cohort |
| 총 admission | 10,846 |
| 사망 admission | 8,139 (~75%) |
| 생존 퇴원 admission | **~2,707 (~25%)** (Home 2,771 / Other hospital 268 / Etc 118 / Nursing home 72) |

**핵심**: 편향은 **"환자 레벨"(전원 종국 사망)** 이지 **"신호 레벨"이 아니다** — 약 25% admission 이 생존 퇴원이고, 사망 admission 도 임종 직전이 아닌 수일~수주 ICU 모니터링을 포함한다. 학습 신호 manifold 는 광범위한 ICU 생리를 커버. → **재학습 불필요가 default, downstream(MIMIC-III matched / VitalDB 생존자 포함)이 실증적 심판자.**

**Design rationale (limitation → 설계 의도 재프레이밍)**:
1. SSL 은 deteriorating 환자의 풍부한 병태생리 다양성에서 가장 많이 학습한다 (hard-example).
2. Downstream 거의 전부 acute-event / deterioration 예측이라, pretrain 분포가 task 분포에 정합한다.
3. 침습 modality(ICP/CVP/PAP)는 일반 코호트에 sparse 하나 high-acuity 코호트엔 dense → multimodal 커버리지 우위.
- caveat: 정상 baseline 의 상대적 과소표현. 단 25% 생존 admission + 사망 전 안정기로 완화됨.

**기관 수 — 미확정 (확인 필요)**: K-MIMIC 레지스트리 전체는 다기관(10기관·71 ICU)으로 알려짐. 그러나 (1) `EMR/admissions.csv` 의 `hospital_id` 가 **전부 NaN** 이고 (2) KHDP 카탈로그가 MORTAL 을 *"derived from **SNUH** ICU Registry"* 로 기술 → MORTAL subset 은 **SNUH 비중이 크거나 SNUH 전용일 가능성**. 단 `hospital_id` 가 빈 건 de-id 로 기관 ID 를 비운 것일 수도 있어 **단일기관 증명은 아님**.
- ⚠️ **핵심 구분**: 임상/EMR 레지스트리(다기관)와 **연속 파형(Vital Recorder) sub-DB**(SNUH 기술 기반, SNUH 중심일 개연성)는 **다른 layer**. 사전학습에 쓰는 건 후자.
- → **현재 데이터로는 "다기관" 입증 불가. 제공자(이철희) 확인 전까지 논문에 "다기관/멀티센터/10기관·71 ICU" 규모 주장 보류.** 668,437h/22,588 recordings 가 MORTAL subset 기준인지 전체 K-MIMIC 값인지도 함께 확인.

**일반 K-MIMIC 확보 경로**: 가명데이터라 연구자 개인 **IRB/DRB 심의 후** 제공팀이 플랫폼에 별도 적재 (다운로드 불가, 플랫폼 내 이용). 이형철 교수님께 IRB 진행 = 보험으로 추진 (리드타임 김, low-regret). 용도 = scratch 재학습이 아니라 **robustness ablation / continue-pretrain**. 서버 `datasets/` 엔 현재 K-MIMIC-MORTAL 만 존재.

### Holdout / Finetuning (사전학습 미사용)

| 소스 | 환경 | Raw 크기 | 환자 수 | 역할 | 신호 |
|------|------|---|---|---|------|
| **VitalDB Open** | 수술중 (SNUH, vitaldb.net 공개) | 95.4 GB | 6,388 | OR 도메인 holdout, intraop downstream finetuning | ECG, ABP, EEG, PPG, CVP, CO2, AWP, PAP (8종) |

### Downstream Tasks (확정 정의)

총 9개 task, 3개 카테고리.

#### Acute Event Detection
| Task | 데이터 소스 | 주요 신호 |
|------|------------|----------|
| Intraoperative Hypotension Prediction | VitalDB Open (OR) | ABP, ECG, PPG |
| Arrhythmia Classification | MIMIC-III-Ext-PPG (PPG 보유 subset) | ECG, PPG |
| Intracranial Hypertension | MIMIC-III Waveform Matched (ICP 보유 subset) | ICP, ABP, ECG |

#### Outcome Prediction
| Task | 데이터 소스 | 주요 신호 |
|------|------------|----------|
| Sepsis Prediction | MIMIC-III Waveform Matched + Clinical (Sepsis-3) | ECG, ABP, PPG |
| Mortality Prediction | MIMIC-III Waveform Matched + Clinical | ECG, ABP, PPG |
| Cardiac Arrest Prediction | MIMIC-III Waveform Matched + Clinical | ECG, ABP, PPG |
| Postoperative AKI Prediction | VitalDB Open (intraop waveform + clinical/lab, KDIGO Cr) | ABP, ECG, PPG, CVP |

#### Physiological Generation
| Task | 데이터 소스 | 주요 신호 |
|------|------------|----------|
| Cross-Modal Reconstruction | MIMIC-III + VitalDB (양쪽) | ECG ↔ ABP ↔ PPG (cross-channel) |
| Waveform Forecasting | MIMIC-III + VitalDB (양쪽) | ECG, ABP, PPG (intra-modal) |

> **데이터 소스 분리 원칙**:
> - **VitalDB Open**: 수술중(intraop) downstream (Hypotension, AKI) + Generation 양쪽 사용. OR-natural task의 자연스러운 benchmark.
> - **MIMIC-III Waveform Matched**: 외부 병원(BIDMC ICU) generalization 검증 + ICU 환경 task (ICH, Sepsis, Mortality, Cardiac Arrest) + Generation 양쪽 사용.
> - **MIMIC-III-Ext-PPG**: MIMIC-III Waveform 중 PPG 채널 보유 subset — Arrhythmia Classification 전용.
> - Generation tasks (Cross-Modal Reconstruction, Waveform Forecasting)는 MIMIC-III와 VitalDB **양쪽 모두**에서 평가 (도메인 일반화 측정).

### 보조 외부 데이터셋 (참고용, 메인 평가 X)

| 소스 | 환경 | 환자 수 | 용도 | 접근 |
|------|------|---------|------|------|
| **MIMIC Database v1.0.0** | BIDMC ICU (다른 병원) | 90+ | 외부 병원 generalization 검증 (보조) | Open (23.5GB) |
| **PTB-XL** | 외래 | 21,799 ECG | Arrhythmia 보조 검증 | Open |
| **CapnoBase** | 마취 | 42 | RR Estimation (예정) | Open |

### 합산하지 않는 이유 (VitalDB Open + K-MIMIC)

1. K-MIMIC 13B에 VitalDB 1.8B 추가 시 +14% (marginal gain 작음)
2. 양쪽 모두 SNUH 출처 → subject leakage 위험
3. VitalDB Open의 더 큰 가치는 **OR 도메인 generalization 측정**
4. Hypotension/Cuffless BP/BIS 등 OR-natural task의 자연스러운 benchmark 보존

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

### K-MIMIC Streaming 파이프라인 (Raw 880GB → 디스크 100GB)

K-MIMIC 사전학습용 raw가 880GB이지만 서버 디스크는 100GB.
**일괄 다운로드 절대 불가** → subject batch 단위 streaming 필수.

```
┌─ batch (subject 50-100명, ~5-10 GB raw) ──────────┐
│  1. download raw .vital (외부 NAS → /tmp)          │
│  2. parse + shard (1-pass, 메모리 → shard write)   │
│  3. shard 위치 확인 + manifest_full 갱신           │
│  4. raw + 임시 .pt delete (디스크 회수)            │
└────────────────────────────────────────────────────┘
            ↓ resume (manifest skip)
       다음 batch ...
```

**디스크 회전 안전 가이드**:
- batch 크기는 `disk_free > batch_raw + batch_shard + 20GB margin` 만족하도록 설정
- subject 50명 ≈ raw 5GB, shard 2.5GB → margin 92GB → 안전
- 모니터: cron으로 디스크 사용률 80% 초과 시 alert

**구현 위치**: `scripts/stream_parse_kmimic.py` (TODO — `.plans/.agent_plan/plan_data.md` §3)
**예상 시간**: 228,608 파일 / 8 workers × ~1초 = **8-15시간 (파싱)** + 다운로드 시간 → **2-4일**

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
