#!/bin/bash
# Mechanical Ventilation Need Prediction (K-MIMIC primary)
# Label: ICU 입원 24h 내 invasive vent 시작 여부

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/k-mimic}"
COHORT_CSV="${COHORT_CSV:-$HOME/workspace/cohort_csv/vent_need_cohort.csv}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/vent_need}"
WINDOW_SEC="${WINDOW_SEC:-600}"
STRIDE_SEC="${STRIDE_SEC:-300}"
# Input: pre-vent 상태 → 호흡 신호 + 심혈관계
INPUT_SIGNALS="${INPUT_SIGNALS:-abp ecg ppg resp co2 awp}"
# Pre-vent input window: ICU 입원 후 first 6h 만 사용
PRE_VENT_HOURS="${PRE_VENT_HOURS:-6}"

echo "============================================================"
echo "  Mech Vent Need (K-MIMIC primary, 24h horizon)"
echo "  Data:           $DATA_DIR"
echo "  Cohort:         $COHORT_CSV"
echo "  Pre-vent hours: $PRE_VENT_HOURS h (input window source)"
echo "  Window/Strd:    ${WINDOW_SEC}s / ${STRIDE_SEC}s"
echo "  Inputs:         $INPUT_SIGNALS"
echo "============================================================"

python -m downstream.outcome.vent_need.prepare_data \
    --data-dir "$DATA_DIR" \
    --cohort-csv "$COHORT_CSV" \
    --input-signals $INPUT_SIGNALS \
    --window-sec $WINDOW_SEC \
    --stride-sec $STRIDE_SEC \
    --pre-vent-hours $PRE_VENT_HOURS \
    --out-dir "$OUT_DIR"
