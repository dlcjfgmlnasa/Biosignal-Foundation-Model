#!/bin/bash
# Hypotension Prediction — paper-fixed combo only (Task #1)
# Paper Table 1 (2026-05-13): ECG + PPG + ABP (3 modalities)
#
# 사용법:
#   bash downstream/acute_event/hypotension/bash/prepare_data.sh

set -e

# env override 가능: DATA_DIR=... OUT_DIR=... bash prepare_data.sh
DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/hypotension}"
WINDOWS="${WINDOWS:-60 180 300 600}"
HORIZONS="${HORIZONS:-5 10 15}"
REQUIRED="${REQUIRED:-ecg ppg abp}"

echo "============================================================"
echo "  Hypotension Prediction (Task #1) — fixed combo"
echo "  Data:     $DATA_DIR"
echo "  Output:   $OUT_DIR"
echo "  Windows:  $WINDOWS"
echo "  Horizons: $HORIZONS"
echo "  Modality: ECG + PPG + ABP"
echo "============================================================"

python -m downstream.acute_event.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg ppg abp \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

echo -e "\n============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
