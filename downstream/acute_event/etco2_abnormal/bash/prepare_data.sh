#!/bin/bash
# Intraoperative EtCO2 Abnormal — VitalDB primary 데이터 준비 sweep
# Label: EtCO2 < 35 또는 > 45 mmHg sustained ≥ 1min
# Input: CO2 wave (Primus) + ECG/ABP/PPG

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
RAW_DIR="${RAW_DIR:-/home/coder/workspace/datasets/vitaldb_open/1.0.0}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/etco2_abnormal}"
WINDOWS="${WINDOWS:-60 180 300 600}"
HORIZONS="${HORIZONS:-5 10 15}"
REQUIRED="${REQUIRED:-co2 ecg ppg abp}"

echo "============================================================"
echo "  Intraop EtCO2 Abnormal (VitalDB primary)"
echo "  Parsed:    $DATA_DIR"
echo "  Raw vital: $RAW_DIR"
echo "  Output:    $OUT_DIR"
echo "  Required:  $REQUIRED"
echo "============================================================"

run_combo() {
    local label="$1"
    local sigs="$2"
    echo -e "\n[$label] $sigs"
    python -m downstream.acute_event.etco2_abnormal.prepare_data \
        --data-dir "$DATA_DIR" \
        --raw-dir "$RAW_DIR" \
        --input-signals $sigs \
        --required-signals $REQUIRED \
        --window-secs $WINDOWS \
        --horizon-mins $HORIZONS \
        --out-dir "$OUT_DIR"
}

run_combo "1/4" "co2"
run_combo "2/4" "ecg"
run_combo "3/4" "ppg"
run_combo "4/4" "co2 ecg ppg abp"

echo -e "\nDone! Saved to: $OUT_DIR"
