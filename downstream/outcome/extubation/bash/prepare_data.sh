#!/bin/bash
# Extubation Failure Prediction (K-MIMIC primary)
# Cohort: BigQuery sql/01_extubation_cohort.sql 결과 CSV
# Label: extubation 후 48h or 72h 내 재삽관

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/k-mimic}"
COHORT_CSV="${COHORT_CSV:-$HOME/workspace/cohort_csv/extubation_cohort.csv}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/extubation}"
WINDOW_SEC="${WINDOW_SEC:-600}"
STRIDE_SEC="${STRIDE_SEC:-300}"
INPUT_SIGNALS="${INPUT_SIGNALS:-abp ecg ppg resp}"

echo "============================================================"
echo "  Extubation Failure (K-MIMIC primary)"
echo "  Data:        $DATA_DIR"
echo "  Cohort:      $COHORT_CSV"
echo "  Output:      $OUT_DIR"
echo "  Window/Strd: ${WINDOW_SEC}s / ${STRIDE_SEC}s"
echo "  Inputs:      $INPUT_SIGNALS"
echo "============================================================"

python -m downstream.outcome.extubation.prepare_data \
    --data-dir "$DATA_DIR" \
    --cohort-csv "$COHORT_CSV" \
    --input-signals $INPUT_SIGNALS \
    --window-sec $WINDOW_SEC \
    --stride-sec $STRIDE_SEC \
    --out-dir "$OUT_DIR"
