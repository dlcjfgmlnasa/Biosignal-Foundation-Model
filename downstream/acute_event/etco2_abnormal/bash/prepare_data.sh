#!/bin/bash
# Intraoperative EtCO2 Abnormal — paper-fixed combo only (Task #5)
# Paper Table 1 (2026-05-13): CO2 + ECG + ABP (3 modalities, PPG 제거)
# Label: EtCO2 < 35 또는 > 45 mmHg sustained ≥ 1min

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
RAW_DIR="${RAW_DIR:-/home/coder/workspace/datasets/vitaldb_open/1.0.0}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/etco2_abnormal}"
WINDOWS="${WINDOWS:-60 180 300 600}"
HORIZONS="${HORIZONS:-5 10 15}"
REQUIRED="${REQUIRED:-co2 ecg abp}"

echo "============================================================"
echo "  Intraop EtCO2 Abnormal (Task #5) — fixed combo"
echo "  Parsed:    $DATA_DIR"
echo "  Raw vital: $RAW_DIR"
echo "  Output:    $OUT_DIR"
echo "  Modality:  CO2 + ECG + ABP"
echo "============================================================"

python -m downstream.acute_event.etco2_abnormal.prepare_data \
    --data-dir "$DATA_DIR" \
    --raw-dir "$RAW_DIR" \
    --input-signals co2 ecg abp \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS \
    --horizon-mins $HORIZONS \
    --out-dir "$OUT_DIR"

echo -e "\nDone! Saved to: $OUT_DIR"
