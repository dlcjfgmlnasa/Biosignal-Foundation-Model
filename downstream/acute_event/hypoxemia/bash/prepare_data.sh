#!/bin/bash
# Intraoperative Hypoxemia Prediction — paper canonical + 5-fold CV (Task #6)
# Paper Table 1 FINAL (2026-05-14): ECG + PPG + CO2 + AWP, window=5min, horizon=10min ahead
# Label: SpO2 < 90% sustained ≥ 1min (raw .vital → PLETH_SPO2 1Hz trend)
#
# 사용법:
#   bash downstream/acute_event/hypoxemia/bash/prepare_data.sh

set -e

REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
UPDOWN_ROOT="${UPDOWN_ROOT:-/home/coder/workspace/updown}"
BIOFM_ROOT="${BIOFM_ROOT:-${UPDOWN_ROOT}/bio_fm}"
DATA_DIR="${DATA_DIR:-${UPDOWN_ROOT}/parser/vitaldb}"
RAW_DIR="${RAW_DIR:-/home/coder/workspace/datasets/vitaldb_open/1.0.0}"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/hypoxemia}"
WINDOWS="${WINDOWS:-300}"           # canonical: 5 min input window
HORIZONS="${HORIZONS:-10}"           # canonical: 10 min ahead
N_FOLDS="${N_FOLDS:-5}"              # clinical AI 표준
REQUIRED="${REQUIRED:-ecg ppg co2 awp}"

echo "============================================================"
echo "  Intraop Hypoxemia (Task #6) — canonical + ${N_FOLDS}-fold CV"
echo "  Parsed:    $DATA_DIR"
echo "  Raw vital: $RAW_DIR"
echo "  Output:    $OUT_DIR"
echo "  Windows:   $WINDOWS"
echo "  Horizons:  $HORIZONS"
echo "  N folds:   $N_FOLDS"
echo "  Modality:  ECG + PPG + CO2 + AWP"
echo "============================================================"

python -m downstream.acute_event.hypoxemia.prepare_data \
    --data-dir "$DATA_DIR" \
    --raw-dir "$RAW_DIR" \
    --input-signals ecg ppg co2 awp \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS \
    --horizon-mins $HORIZONS \
    --n-folds $N_FOLDS \
    --out-dir "$OUT_DIR"

echo -e "\n============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
