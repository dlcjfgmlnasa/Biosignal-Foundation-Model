#!/bin/bash
# Intraoperative Hypoxemia Prediction — paper-fixed combo only (Task #6)
# Paper Table 1 (2026-05-13): PPG + ECG + ABP + CO2 (4 modalities)
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
WINDOWS="${WINDOWS:-60 180 300 600}"
HORIZONS="${HORIZONS:-5 10 15}"
REQUIRED="${REQUIRED:-ppg ecg abp co2}"

echo "============================================================"
echo "  Intraop Hypoxemia (Task #6) — fixed combo"
echo "  Parsed:    $DATA_DIR"
echo "  Raw vital: $RAW_DIR"
echo "  Output:    $OUT_DIR"
echo "  Windows:   $WINDOWS"
echo "  Horizons:  $HORIZONS"
echo "  Modality:  PPG + ECG + ABP + CO2"
echo "============================================================"

python -m downstream.acute_event.hypoxemia.prepare_data \
    --data-dir "$DATA_DIR" \
    --raw-dir "$RAW_DIR" \
    --input-signals ppg ecg abp co2 \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS \
    --horizon-mins $HORIZONS \
    --out-dir "$OUT_DIR"

echo -e "\n============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
