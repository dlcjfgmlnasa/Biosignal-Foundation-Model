# 외부 Foundation Model (FM) baselines

CARMEN 과 **같은 prepared 데이터·같은 5-fold·같은 평가** 위에서, 공개된 외부 생체신호
foundation model 을 **frozen feature extractor + linear probe** 로 붙여 직접 비교하기
위한 어댑터 모음입니다. 산출물(`preds_fold{f}.npz` + `fold{f}.json`)이 CARMEN·CNN
baseline 과 동일 스키마라 `python -m downstream.run_eval` 로 **구분 없이 집계**됩니다.

> ⚠ 이 디렉토리는 downstream 평가 전용입니다. CARMEN 본코드
> (`module/`, `model/`, `train/`, `data/collate.py`, downstream `run.py`)는 전혀
>건드리지 않습니다. 외부 레포·가중치는 런타임에 import/load 합니다.

## 지원 모델 & modality coverage

| Encoder      | modality  | native SR | 세그먼트 | 임베딩 | 가중치 | upstream |
|--------------|-----------|-----------|----------|--------|--------|----------|
| `ecgfounder` | ECG 전용  | 500Hz     | 10s      | 1024   | HF 공개 ✅ | [PKUDigitalHealth/ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) |
| `heartbeit`  | ECG 전용(이미지) | 500Hz | 10s | 768 | **접근 신청 필요** ⚠ | [arXiv 2212.14040](https://arxiv.org/abs/2212.14040) |
| `stmem`      | ECG 전용  | 250Hz     | 9s       | 768    | Google Drive 공개 ✅ | [vuno/ST-MEM](https://github.com/vuno/ST-MEM) |
| `papagei`    | PPG 전용  | 125Hz     | 10s      | 512    | Zenodo 공개 ✅ | [Nokia-Bell-Labs/papagei-foundation-model](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) |
| `pulseppg`   | PPG 전용  | 50Hz      | 60s      | 512*   | Zenodo 공개 ✅ | [maxxu05/pulseppg](https://github.com/maxxu05/pulseppg) |
| `anyppg`     | PPG 전용  | 125Hz     | 10s      | 512    | repo 공개 ✅ | [PKUDigitalHealth/AnyPPG](https://github.com/PKUDigitalHealth/AnyPPG) |
| `biot`       | 다채널    | 200Hz     | 10s      | 256    | repo/HF 공개 ✅ | [ycq091044/BIOT](https://github.com/ycq091044/BIOT) |
| `csfm`       | ECG+PPG 멀티모달 | 100Hz | 25s   | 768/1024 | **접근 신청 필요** ⚠ | [guxiao0822/Cardiac-Sensing-FM](https://github.com/guxiao0822/Cardiac-Sensing-FM) |

`*` Pulse-PPG 임베딩 차원은 로드 시 dummy forward 로 추론합니다.
CSFM(Gu et al., *Nat. Mach. Intell.* 2026)은 **CARMEN 과 SR·모달리티가 가장 가까운** 핵심
비교군입니다(둘 다 ECG+PPG, 100Hz → 리샘플 불필요). variant 는 `FM_CSFM_VARIANT`(tiny|base|large).

**task 적용 가능 여부** = task 신호 조합에 encoder 지원 modality 가 포함되는지로 결정됩니다.
ECG 전용 모델은 ecg 채널만, PPG 전용 모델은 ppg 채널만 사용하고, 나머지는 무시합니다.
비호환 task 는 실행 시 `[SKIP]` 후 종료(exit 0)합니다.

| task | 신호(on-disk) | ecgfounder / heartbeit / stmem | papagei / pulseppg / anyppg | csfm | biot |
|------|---------------|:--:|:--:|:--:|:--:|
| hypotension     | ecg_ppg_abp | ✅(ecg) | ✅(ppg) | ✅(ecg+ppg) | ✅(3ch) |
| cardiac_arrest  | ecg_ppg     | ✅(ecg) | ✅(ppg) | ✅(ecg+ppg) | ✅(2ch) |
| ich             | abp_icp_ecg | ✅(ecg) | ✖(ppg 없음) | ✅(ecg) | ✅(3ch) |
| desaturation    | co2_awp_resp_flow | ✖ | ✖ | ✖ | ✅ |

## 어댑터 동작

우리 prepared window 는 **100Hz, `(n, C, L)`** 인데 각 FM 은 native SR·입력길이·modality 가
제각각입니다. `encoders/base.py` 의 `FMEncoder` 가 공통 처리합니다:

1. **채널 선택** — encoder 가 지원하는 modality 채널만 선택(ECG/PPG 전용) 또는 전부(BIOT).
2. **세그먼트 분할** — 긴 window 를 `seg_sec` 단위로 잘라 최대 `--max-segments` 개(균등 샘플).
3. **리샘플** — 100Hz → native SR 선형보간.
4. **정규화** — 세그먼트별 per-channel z-norm(기본).
5. **forward + 세그먼트 평균** → window 당 임베딩 `(n, d)`.

이후 frozen 임베딩(1회 추출·캐시) 위에서 `LinearProbe`(LayerNorm→Dropout→Linear)를 학습하고,
CARMEN 과 동일하게 val Youden threshold·per-fold bootstrap CI 로 평가합니다.

## Setup (서버)

각 upstream 레포를 clone 하고 가중치를 받은 뒤, 레포 루트를 env(또는 `--third-party-root`)로
지정합니다. 레포 코드는 런타임에 `sys.path` 에 추가되어 import 됩니다.

```bash
# 예: ECGFounder
git clone https://github.com/PKUDigitalHealth/ECGFounder /repos/ECGFounder
# 가중치: https://huggingface.co/PKUDigitalHealth/ECGFounder → 1_lead_ECGFounder.pth
export FM_ECGFOUNDER_ROOT=/repos/ECGFounder

# 예: PaPaGei
git clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model /repos/papagei
# 가중치: Zenodo 13983110 → weights/papagei_s.pt
export FM_PAPAGEI_ROOT=/repos/papagei
pip install -r /repos/papagei/requirements.txt   # dotmap, pyPPG, torch_ecg 등

# 예: Pulse-PPG
git clone https://github.com/maxxu05/pulseppg /repos/pulseppg
bash /repos/pulseppg/download_model.sh           # Zenodo 17345536
export FM_PULSEPPG_ROOT=/repos/pulseppg

# 예: BIOT
git clone https://github.com/ycq091044/BIOT /repos/BIOT
# pretrained-models/*.ckpt (repo 동봉) 또는 HF braindecode/BIOT
export FM_BIOT_ROOT=/repos/BIOT

# 예: CSFM (가중치 Academic Access Agreement 필요 — xiao.gu@eng.ox.ac.uk)
git clone https://github.com/guxiao0822/Cardiac-Sensing-FM /repos/Cardiac-Sensing-FM
export FM_CSFM_ROOT=/repos/Cardiac-Sensing-FM
export FM_CSFM_VARIANT=base      # tiny|base|large (받은 가중치에 맞춰)

# 예: ST-MEM
git clone https://github.com/vuno/ST-MEM /repos/ST-MEM
# 가중치: repo README 의 Google Drive (encoder-only)
export FM_STMEM_ROOT=/repos/ST-MEM

# 예: AnyPPG
git clone https://github.com/PKUDigitalHealth/AnyPPG /repos/AnyPPG
# 가중치: repo 내 공개 load_anyppg/anyppg_ckpt.pth
export FM_ANYPPG_ROOT=/repos/AnyPPG

# 예: HeartBEiT (가중치 접근 신청 필요, HF BEiT 디렉토리)
pip install transformers
export FM_HEARTBEIT_ROOT=/repos/HeartBEiT   # (선택)
```

모델별 환경변수: `FM_ECGFOUNDER_ROOT`, `FM_STMEM_ROOT`, `FM_PAPAGEI_ROOT`, `FM_PULSEPPG_ROOT`,
`FM_ANYPPG_ROOT`, `FM_BIOT_ROOT`, `FM_CSFM_ROOT`(+`FM_CSFM_VARIANT`), `FM_HEARTBEIT_ROOT`.

## 실행

encoder 별 wrapper 가 적용 가능한 task 를 순회합니다(데이터 없으면 자동 SKIP):

```bash
FM_ECGFOUNDER_ROOT=/repos/ECGFounder WEIGHTS=/weights/1_lead_ECGFounder.pth \
  bash downstream/baselines/fm/bash/run_ecgfounder.sh

FM_PAPAGEI_ROOT=/repos/papagei WEIGHTS=/repos/papagei/weights/papagei_s.pt \
  bash downstream/baselines/fm/bash/run_papagei.sh

FM_PULSEPPG_ROOT=/repos/pulseppg WEIGHTS=/repos/pulseppg/model_weights/<ckpt>.pt \
  bash downstream/baselines/fm/bash/run_pulseppg.sh

FM_BIOT_ROOT=/repos/BIOT WEIGHTS=/repos/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt \
  bash downstream/baselines/fm/bash/run_biot.sh

FM_CSFM_ROOT=/repos/Cardiac-Sensing-FM FM_CSFM_VARIANT=base WEIGHTS=/weights/csfm_base.pth \
  bash downstream/baselines/fm/bash/run_csfm.sh

FM_STMEM_ROOT=/repos/ST-MEM WEIGHTS=/weights/st_mem_encoder.pth \
  bash downstream/baselines/fm/bash/run_stmem.sh

FM_ANYPPG_ROOT=/repos/AnyPPG WEIGHTS=/repos/AnyPPG/load_anyppg/anyppg_ckpt.pth \
  bash downstream/baselines/fm/bash/run_anyppg.sh
```

단일 (encoder, task) 직접 실행:

```bash
python -m downstream.baselines.fm.run_fm_probe \
    --encoder ecgfounder --weights /weights/1_lead_ECGFounder.pth \
    --third-party-root /repos/ECGFounder \
    --data-path .../cardiac_arrest/scope_arrest_ecg_ppg_w600s_h15min \
    --input-signals ecg ppg --task-name cardiac_arrest \
    --id-fields subject_ids case_ids \
    --n-folds 5 --fold 0 --epochs 100 --batch-size 256 \
    --out-dir .../result/main/cardiac_arrest_ecgfounder/ecg_ppg_w600s_h15min
```

## 집계 (CARMEN 과 동일)

```bash
python -m downstream.run_eval --tasks-root .../result/main/cardiac_arrest_ecgfounder/ecg_ppg_w600s_h15min
```

`preds_fold*.npz` 를 OOF concat → patient-level bootstrap 95% CI + 5-fold mean±SD 로
CARMEN·CNN baseline 과 같은 표에 넣습니다.

## 한계 / 주의

- **frozen linear-probe 전용**: 공정 비교(각 FM 논문의 표준 probing 관행)와 이식성을 위해
  LoRA/fine-tune 은 포함하지 않았습니다. CARMEN 의 Frozen 행과 대응됩니다.
- **binary classification 전용**: multiclass(arrhythmia)·generation·patient-level(mortality)
  task 는 아직 미지원(별도 head/aggregator 필요). 표준 per-(fold,split) chunk 이면서 binary 인
  window-task 만 지원합니다.
- **BIOT 도메인 이질**: BIOT 사전학습 채널 임베딩은 EEG montage 의미라, ECG/ABP/PPG 를 앞
  채널 슬롯에 매핑해 쓰는 것은 근사입니다(BIOT 의 cross-data 설계 취지에는 부합).
- **HeartBEiT 렌더링**: 저자 원본 ECG 이미지 렌더와 화소 단위로 동일하지 않은 경량
  line-raster 를 사용합니다. 가중치도 접근 신청이 필요합니다.
- **정규화**: 기본 per-segment z-norm 을 씁니다. 일부 모델(PaPaGei)은 자체 bandpass 전처리를
  쓰지만, 이식성·배치처리를 위해 통일했습니다.
- **API 검증 수준**: 각 encoder wrapper 상단 docstring 에 upstream 빌드/로드 방식을
  명시했습니다. 서버의 실제 레포 버전과 시그니처가 다르면 해당 wrapper 만 조정하세요.

### 검토했으나 미통합
- **ECG-FM** ([bowang-lab/ECG-FM](https://github.com/bowang-lab/ECG-FM), HF `wanglab/ecg-fm`):
  가중치는 공개(MIT)지만 ⑴ **12-lead 입력 전제**(feature extractor conv in_channels=12)라
  우리 단일-lead 모니터링 ECG 와 맞지 않고(단일 lead→12ch 복제는 오해 소지), ⑵ 로딩이
  `fairseq_signals` 프레임워크(오프라인 전처리→CLI 추론→memmap)에 강하게 결합돼 in-process
  `encode` 로 이식하기 무겁다. 향후 12-lead 코호트가 생기면 재검토.
- **QualityFM / GPT-PPG**: 공개 가중치 없음(QualityFM 은 저자 문의 필요) → 현재 제외.
- **SiamQuality**: 가중치 비공개 → 제외.
