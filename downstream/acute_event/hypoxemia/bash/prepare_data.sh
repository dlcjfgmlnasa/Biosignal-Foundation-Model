#!/bin/bash
# Intraoperative Hypoxemia Prediction — VitalDB primary 데이터 준비 sweep
# Paper Table S7 (Task 6) 정렬: 6 signal combos × 4 windows × 3 horizons
# Label: SpO2 < 90% sustained ≥ 1min (raw .vital → PLETH_SPO2 1Hz trend)
#
# 사용법:
#   bash downstream/acute_event/hypoxemia/bash/prepare_data.sh
#
# env override:
#   DATA_DIR=... RAW_DIR=... OUT_DIR=... bash ...

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
RAW_DIR="${RAW_DIR:-/home/coder/workspace/datasets/vitaldb_open/1.0.0}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/hypoxemia}"
WINDOWS="${WINDOWS:-60 180 300 600}"
HORIZONS="${HORIZONS:-5 10 15}"
# Paper full 6-modal combo 가능하려면 ecg ppg abp co2 awp resp 모두 보유 환자만 사용
REQUIRED="${REQUIRED:-ecg ppg abp co2 awp resp}"

echo "============================================================"
echo "  Intraop Hypoxemia (VitalDB primary) — Paper Table S7 Combos"
echo "  Parsed:    $DATA_DIR"
echo "  Raw vital: $RAW_DIR"
echo "  Output:    $OUT_DIR"
echo "  Windows:   $WINDOWS"
echo "  Horizons:  $HORIZONS"
echo "  Required:  $REQUIRED (paired comparison)"
echo "============================================================"

run_combo() {
    local label="$1"
    local sigs="$2"
    echo -e "\n[$label] $sigs"
    python -m downstream.acute_event.hypoxemia.prepare_data \
        --data-dir "$DATA_DIR" \
        --raw-dir "$RAW_DIR" \
        --input-signals $sigs \
        --required-signals $REQUIRED \
        --window-secs $WINDOWS \
        --horizon-mins $HORIZONS \
        --out-dir "$OUT_DIR"
}

# Paper Table S7: 6 combos
run_combo "1/6" "ppg"
run_combo "2/6" "ecg"
run_combo "3/6" "ppg ecg"
run_combo "4/6" "ppg abp"
run_combo "5/6" "ppg ecg abp"
run_combo "6/6" "ppg ecg abp co2 awp resp"  # full 6-modal cardiorespiratory

echo -e "\n============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
