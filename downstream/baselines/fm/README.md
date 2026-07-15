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
| `stmem`      | ECG 전용  | 250Hz     | 9s       | 768    | Google Drive 공개 ✅ | [vuno/ST-MEM](https://github.com/vuno/ST-MEM) |
| `papagei`    | PPG 전용  | 125Hz     | 10s      | 512    | Zenodo 공개 ✅ | [Nokia-Bell-Labs/papagei-foundation-model](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) |
| `pulseppg`   | PPG 전용  | 50Hz      | 60s      | 512*   | Zenodo 공개 ✅ | [maxxu05/pulseppg](https://github.com/maxxu05/pulseppg) |
| `anyppg`     | PPG 전용  | 125Hz     | 10s      | 512    | repo 공개 ✅ | [PKUDigitalHealth/AnyPPG](https://github.com/PKUDigitalHealth/AnyPPG) |
| `biot`       | 다채널    | 200Hz     | 10s      | 256    | repo/HF 공개 ✅ | [ycq091044/BIOT](https://github.com/ycq091044/BIOT) |

`*` Pulse-PPG 임베딩 차원은 로드 시 dummy forward 로 추론합니다.
CSFM·HeartBEiT 는 가중치 미확보로 제외했습니다(아래 "검토했으나 미통합" 참조).

**task 적용 가능 여부** = task 신호 조합에 encoder 지원 modality 가 포함되는지로 결정됩니다.
ECG 전용 모델은 ecg 채널만, PPG 전용 모델은 ppg 채널만 사용하고, 나머지는 무시합니다.
비호환 task 는 실행 시 `[SKIP]` 후 종료(exit 0)합니다.

| task | 신호(on-disk) | ecgfounder / stmem | papagei / pulseppg / anyppg | biot |
|------|---------------|:--:|:--:|:--:|
| hypotension     | ecg_ppg_abp | ✅(ecg) | ✅(ppg) | ✅(3ch) |
| cardiac_arrest  | ecg_ppg     | ✅(ecg) | ✅(ppg) | ✅(2ch) |
| ich             | abp_icp_ecg | ✅(ecg) | ✖(ppg 없음) | ✅(3ch) |
| desaturation    | co2_awp_resp_flow | ✖ | ✖ | ✅ |

## 어댑터 동작

우리 prepared window 는 **100Hz, `(n, C, L)`** 인데 각 FM 은 native SR·입력길이·modality 가
제각각입니다. `encoders/base.py` 의 `FMEncoder` 가 공통 처리합니다:

1. **채널 선택** — encoder 가 지원하는 modality 채널만 선택(ECG/PPG 전용) 또는 전부(BIOT).
2. **세그먼트 분할** — 긴 window 를 `seg_sec` 단위로 잘라 최대 `--max-segments` 개(균등 샘플).
3. **리샘플** — 100Hz → native SR 선형보간.
4. **전처리** — `_preprocess`. 기본은 세그먼트별 per-channel z-norm 이지만, upstream 전처리가
   명시된 모델은 `encoders/_dsp.py` 로 그것을 재현한다(아래).
5. **forward + 세그먼트 평균** → window 당 임베딩 `(n, d)`.

### upstream 전처리 재현 (`_preprocess` override)

| Encoder | upstream 전처리 | 근거 |
|---|---|---|
| `ecgfounder` | notch 50Hz(Q=30) → Butterworth N=4 bandpass [0.67,40] → median(0.4·fs) baseline 제거 → z-score | `util.filter_bandpass` + `dataset.py`. HF 모델카드 Notice 가 준수를 명시 요구 |
| `stmem` | highpass 0.67 → lowpass 40 (Butterworth SOS order 5, `sosfiltfilt`) → standardize | `util/transforms.py`, `configs/pretrain/st_mem.yaml` |
| `biot` | `x / (quantile(|x|, 0.95) + 1e-8)` — **평균 미차감**(DC 보존) | `utils.py` 전 데이터 로더 |
| `papagei` | z-score → pyPPG Chebyshev II bandpass(order4/rs20, [0.5,12]) + 0.02·fs 이동평균 | `preprocessing/ppg.py` → pyPPG `preproc.Preprocessing` |
| `pulseppg` | upstream 전처리 미재현(기본 z-norm) — forward 첫 층이 InstanceNorm 이라 스케일 불변 | — |

**샘플링레이트에 대해**: 우리 파서가 ECG 를 native SR 에서 bandpass (0.5, 40)Hz 로 자른 뒤 100Hz 로
내리고(`data/parser/vitaldb.py` `SIGNAL_CONFIGS`), ECGFounder·ST-MEM 역시 40Hz 에서 저역통과한다.
따라서 두 모델이 소비하는 대역은 100Hz(Nyquist 50Hz) 데이터 안에 **전부** 들어 있으며, native SR
prepared set 을 따로 만들 실익이 없다. 실측(합성 ECG): upstream 필터 적용 시 100Hz 경로와
native-SR 경로의 임베딩 cosine 이 ECGFounder **0.987**, ST-MEM **0.997** 로 수렴한다(필터 미적용 시
각각 0.73/0.94). upstream 의 40Hz 저역통과가 선형보간 imaging 아티팩트도 함께 제거하기 때문이다.
BIOT 만 0–100Hz 전체(101 STFT bin)를 소비하므로 상위 절반이 비지만, 이는 리샘플이 아니라
ECG/PPG/ABP 에 50Hz 초과 성분이 원래 없기 때문이다(>50Hz 파워 0.2% 미만).

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

# 예: ST-MEM
git clone https://github.com/vuno/ST-MEM /repos/ST-MEM
# 가중치: repo README 의 Google Drive (encoder-only)
export FM_STMEM_ROOT=/repos/ST-MEM

# 예: AnyPPG
git clone https://github.com/PKUDigitalHealth/AnyPPG /repos/AnyPPG
# 가중치: repo 내 공개 load_anyppg/anyppg_ckpt.pth
export FM_ANYPPG_ROOT=/repos/AnyPPG
```

모델별 환경변수: `FM_ECGFOUNDER_ROOT`, `FM_STMEM_ROOT`, `FM_PAPAGEI_ROOT`, `FM_PULSEPPG_ROOT`,
`FM_ANYPPG_ROOT`, `FM_BIOT_ROOT`.

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
- **ST-MEM 단일 lead**: 12-lead 모델이라 `num_leads=1` 로 생성하고 사전학습 lead 임베딩
  12개 중 **Lead II(index 1)** 하나만 로드합니다(`encoders/stmem.py` 의 `LEAD_INDEX`).
  num_leads=12 로 두고 단일 lead 를 넣으면 upstream `forward_encoding` 의 브로드캐스트로
  신호가 12배 복제되어 **에러 없이 잘못된 임베딩**이 나오므로 주의하세요.
- **정규화**: ECGFounder·ST-MEM·BIOT·PaPaGei 는 upstream 전처리를 재현합니다(위 표,
  `encoders/_dsp.py`). Pulse-PPG 만 미재현(InstanceNorm 자체 정규화).
- **ECGFounder lead 불일치**: 1-lead 체크포인트는 upstream `dataset.py` 상 **Lead I** 로 학습된
  것으로 보입니다. 우리 모니터링 ECG 는 Lead II 계열이라 전처리로 변환할 수 없는 불일치이며,
  논문에 한계로 명시해야 합니다.
- **ST-MEM single-lead**: upstream 은 12-lead 전용 config 만 제공합니다. `num_leads=1` + Lead II
  임베딩 선택은 우리 구현이며, 12-lead spatio-temporal 설계를 온전히 쓰지 못하는 한계가 있습니다.
- **세그먼트 커버리지**: `--max-segments` 기본은 **0 = window 전체 커버**(겹침 없이 연속)입니다.
  CARMEN 은 window 전체를 pool 하므로 부분 커버는 외부 FM 을 불리하게 만듭니다. frozen 추출은
  1회뿐이라 비용은 선형입니다. RAM/시간을 제한하려면 `MAX_SEGMENTS=<N>` 으로 상한을 두세요
  (그 경우 [0, L-seg] 구간 균등 샘플).
- **API 검증 수준**: 각 encoder wrapper 상단 docstring 에 upstream 빌드/로드 방식을
  명시했습니다. 서버의 실제 레포 버전과 시그니처가 다르면 해당 wrapper 만 조정하세요.

### 검토했으나 미통합
- **CSFM** ([guxiao0822/Cardiac-Sensing-FM](https://github.com/guxiao0822/Cardiac-Sensing-FM),
  Gu et al., *Nat. Mach. Intell.* 2026): CARMEN 과 SR·모달리티가 가장 가까운(둘 다 ECG+PPG, 100Hz)
  핵심 비교군이었으나 **가중치가 공개되지 않음**(레포 `pretrained/` 는 빈 `__init__.py`뿐, Academic
  Access Agreement 를 institutional email `xiao.gu@eng.ox.ac.uk` 로 신청해야 함). 확보 불가로 **제외**
  (2026-07-12). 어댑터 코드(`encoders/csfm.py`)와 러너는 삭제했으며 git 이력에 보존됨 — 향후 가중치를
  받으면 복원. 어댑터 API(`network.model.CSFM_model`, forward 시그니처, mlp_head, ts_pos_embedding)는
  삭제 전 upstream 과 대조해 일치 확인 완료.
- **HeartBEiT** ([akhilvaid/HeartBEiT](https://github.com/akhilvaid/HeartBEiT),
  npj Digital Medicine 2023, Mount Sinai): ECG 파형→이미지→BEiT. 가중치는 **Mount Sinai
  Intellectual Partners 와의 IRB-approved agreement** 로만 배포(공개 다운로드·HF gated·저자 이메일
  창구 없음) → CSFM 보다 확보가 더 어려워 **제외**(2026-07-12). 게다가 어댑터가
  `transformers.BeitModel.from_pretrained`(HF 디렉토리)를 가정하나 upstream 은 unilm/timm 커스텀
  `.pth` 라 확보하더라도 포맷 변환이 추가로 필요. 어댑터(`encoders/heartbeit.py`)·러너 삭제, git 이력 보존.
- **ECG-FM** ([bowang-lab/ECG-FM](https://github.com/bowang-lab/ECG-FM), HF `wanglab/ecg-fm`):
  가중치는 공개(MIT)지만 ⑴ **12-lead 입력 전제**(feature extractor conv in_channels=12)라
  우리 단일-lead 모니터링 ECG 와 맞지 않고(단일 lead→12ch 복제는 오해 소지), ⑵ 로딩이
  `fairseq_signals` 프레임워크(오프라인 전처리→CLI 추론→memmap)에 강하게 결합돼 in-process
  `encode` 로 이식하기 무겁다. 향후 12-lead 코호트가 생기면 재검토.
- **QualityFM / GPT-PPG**: 공개 가중치 없음(QualityFM 은 저자 문의 필요) → 현재 제외.
- **SiamQuality**: 가중치 비공개 → 제외.
