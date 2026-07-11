#!/bin/bash
# Intraoperative Hypoxemia Prediction — 데이터 준비 (VitalDB Open, window-level).
#
# 라벨: 미래 horizon 구간에 SpO2 < 92% 가 ≥60초 지속되면 positive.
#   임계 92% = Lundberg et al., Nat Biomed Eng 2018 (Prescience) 및 WHO 중재 수준.
#   Prescience 는 numeric AIMS 시계열로 5분 전 예측 — 우리는 raw waveform 으로 같은 문제를 푼다.
# 입력: parsed .pt 파형 (ECG/PPG/CO2/AWP — 심폐 4-modality). SpO2 numeric 은
#   라벨 전용이며 입력에 넣지 않는다 (넣으면 persistence baseline 이 압도 — SWIFT 계열).
#
# 시간 정렬: parser manifest 의 start_sample(dtstart 원점 100Hz) 로 SpO2 1Hz trend 를 인덱싱.
# Baseline 가드: 예측 시점에 이미 SpO2<92% 인 윈도우는 제외(기본) — 지속성 누수 차단.
#
# 코호트(inspect_cohort, 6,388 subject 전체, 2026-07-11 실측):
#   ecg+ppg = 4,248 / ecg+ppg+co2 = 3,735 / ecg+ppg+co2+awp = 3,296 subject
#   (1230s=window300+horizon900+stride30 연속 교집합 기준. --allow-tail-windows 시
#    330s 기준 각각 5,676 / 5,221 / 4,994 로 늘어난다.)
#   → canonical = 4-modality(ECG+PPG+CO2+AWP). 심혈관만(ECG+PPG)은 ablation 으로.
#
# 사용법:
#   bash downstream/acute_event/hypoxemia/bash/prepare_data.sh
#   (env override: DATA_DIR=... RAW_DIR=... OUT_DIR=... SIGNALS="ecg ppg" bash ...)

set -e

REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
UPDOWN_ROOT="${UPDOWN_ROOT:-/home/coder/workspace/updown}"
BIOFM_ROOT="${BIOFM_ROOT:-/home/coder/workspace/k-mimic-/bio_fm}"
DATA_DIR="${DATA_DIR:-${UPDOWN_ROOT}/parser/vitaldb}"
RAW_DIR="${RAW_DIR:-/home/coder/workspace/datasets/vitaldb_open/1.0.0}"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/hypoxemia}"

WINDOWS="${WINDOWS:-300}"              # canonical: 5 min input window
HORIZONS="${HORIZONS:-5 10 15}"        # Table 3(a) 공통 lead-time 격자
SIGNALS="${SIGNALS:-ecg ppg co2 awp}"  # 입력 파형 (라벨은 SpO2 numeric)
REQUIRED="${REQUIRED:-ecg ppg co2 awp}" # SIGNALS ⊆ REQUIRED 필수 (아니면 채널 소실)
N_FOLDS="${N_FOLDS:-5}"
WORKERS="${WORKERS:-16}"               # 네트워크 마운트 → SpO2 로딩 ThreadPool
MAX_SUBJECTS="${MAX_SUBJECTS:-}"       # 설정 시 dry-run (코호트 크기 확인용)

# ── 라벨 정의 변이 (VARIANT) ──
#   기본 SpO2<92%·60초 지속은 술중 prevalence 가 극히 낮다(5분 ~0.1%). 두 완화안:
#   anypoint92 : SpO2<92% any-point + 아티팩트 게이트(HR concordance).  ~1.5/3.5/6.2%
#                (Lundberg 2018 Prescience 정의. any-point 는 게이트 필수 — 라벨노이즈)
#   sus95      : SpO2<95%·60초 지속.                                      ~0.24/0.63/1.4%
#                (Park et al. PLoS One 2023 소아 임계)
#   sus92      : 원안 SpO2<92%·60초 (참고용, 극희귀)
# 결과 충돌 방지: 변이별 out-dir 접미사(_<VARIANT>). run.sh 도 동일 VARIANT 로 읽는다.
VARIANT="${VARIANT:-anypoint92}"
case "$VARIANT" in
  anypoint92) SPO2_THRESHOLD=92; SUSTAINED_SEC=1;  GATE_ARG="--artifact-gate" ;;
  sus95)      SPO2_THRESHOLD=95; SUSTAINED_SEC=60; GATE_ARG="" ;;
  sus92)      SPO2_THRESHOLD=92; SUSTAINED_SEC=60; GATE_ARG="" ;;
  *) echo "ERROR: unknown VARIANT '$VARIANT' (anypoint92|sus95|sus92)"; exit 1 ;;
esac
# env 로 직접 threshold/sustained override 도 허용 (VARIANT 프리셋 위에)
SPO2_THRESHOLD="${SPO2_THRESHOLD_OVERRIDE:-$SPO2_THRESHOLD}"
SUSTAINED_SEC="${SUSTAINED_SEC_OVERRIDE:-$SUSTAINED_SEC}"

OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/hypoxemia_${VARIANT}}"

EXTRA_ARGS="$GATE_ARG"
if [ -n "$MAX_SUBJECTS" ]; then
    # ⚠ 앞 N 명은 파싱 배치상 co2/awp 커버리지가 낮다(초기 subject 미포함). dry-run 은
    #   대표 표본이 되도록 --sample-seed 로 무작위 추출(기본 42). 전체 실행 시엔 무시됨.
    SAMPLE_SEED="${SAMPLE_SEED:-42}"
    EXTRA_ARGS="$EXTRA_ARGS --max-subjects $MAX_SUBJECTS --sample-seed $SAMPLE_SEED"
    OUT_DIR="${OUT_DIR}_dryrun"
fi

LABEL_DESC="SpO2 < ${SPO2_THRESHOLD}%"
if [ "$SUSTAINED_SEC" -le 1 ] 2>/dev/null; then LABEL_DESC="$LABEL_DESC any-point"; else LABEL_DESC="$LABEL_DESC sustained >= ${SUSTAINED_SEC}s"; fi

echo "============================================================"
echo "  Intraoperative Hypoxemia — Data Preparation (VitalDB Open)"
echo "  Variant:   $VARIANT"
echo "  Parsed:    $DATA_DIR"
echo "  Raw vital: $RAW_DIR"
echo "  Output:    $OUT_DIR"
echo "  Input:     $SIGNALS   (required: $REQUIRED)"
echo "  Window:    ${WINDOWS}s | Horizons: ${HORIZONS} min | Folds: $N_FOLDS"
echo "  Label:     $LABEL_DESC ${GATE_ARG:+(+artifact-gate)}"
if [ -n "$MAX_SUBJECTS" ]; then echo "  ⚠ DRY-RUN: max-subjects=$MAX_SUBJECTS"; fi
echo "============================================================"

# SIGNALS / REQUIRED / HORIZONS / EXTRA_ARGS 는 다중값이라 일부러 unquoted (word-split).
python -m downstream.acute_event.hypoxemia.prepare_data \
    --data-dir "$DATA_DIR" \
    --raw-dir "$RAW_DIR" \
    --input-signals $SIGNALS \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS \
    --horizon-mins $HORIZONS \
    --n-folds $N_FOLDS \
    --workers $WORKERS \
    --spo2-threshold $SPO2_THRESHOLD \
    --sustained-sec $SUSTAINED_SEC \
    --out-dir "$OUT_DIR" \
    $EXTRA_ARGS

SIG_TOKEN=$(echo "$SIGNALS" | tr ' ' '_')
echo -e "\n============================================================"
echo "  Done! Output prefix: $OUT_DIR/hypoxemia_${SIG_TOKEN}_w${WINDOWS}s_h{H}min"
echo "============================================================"
