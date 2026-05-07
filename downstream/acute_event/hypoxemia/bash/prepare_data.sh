#!/bin/bash
# Intraoperative Hypoxemia Prediction — VitalDB primary 데이터 준비 sweep
# VitalDB Open (SNUADC) 가용 신호: ECG, ABP, PPG, CO2, AWP (RESP 없음)
# → Paper Table S7 (5-modal) + 호흡 baseline = 8 signal combos × 4 windows × 3 horizons
# Label: SpO2 < 90% sustained ≥ 1min (raw .vital → PLETH_SPO2 1Hz trend)
#
# 사용법:
#   bash downstream/acute_event/hypoxemia/bash/prepare_data.sh
#
# env override:
#   DATA_DIR=... RAW_DIR=... OUT_DIR=... bash ...

set -e

REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
RAW_DIR="${RAW_DIR:-/home/coder/workspace/datasets/vitaldb_open/1.0.0}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/downstream/hypoxemia}"
WINDOWS="${WINDOWS:-60 180 300 600}"
HORIZONS="${HORIZONS:-5 10 15}"
# 5-modal full combo 가능하려면 ecg ppg abp co2 awp 모두 보유 환자만 사용
# (VitalDB Open 에는 RESP 없음 — K-MIMIC-MORTAL 전용 신호)
REQUIRED="${REQUIRED:-ecg ppg abp co2 awp}"

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

# Paper Table S7 (5) + 호흡 단독 baseline (2) + 5-modal full = 8 combos
run_combo "1/8" "ppg"
run_combo "2/8" "ecg"
run_combo "3/8" "ppg ecg"
run_combo "4/8" "ppg abp"
run_combo "5/8" "ppg ecg abp"
run_combo "6/8" "ppg ecg abp co2 awp"  # full 5-modal cardiorespiratory
run_combo "7/8" "co2"
run_combo "8/8" "awp"

echo -e "\n============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
