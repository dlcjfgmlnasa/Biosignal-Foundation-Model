#!/bin/bash
# Hypotension Prediction — paper canonical combo (Task #1)
# Paper Table 1 FINAL (2026-05-14): VitalDB, ECG + PPG + ABP, window=5min, horizon=15min ahead
#
# 사용법:
#   bash downstream/acute_event/hypotension/bash/prepare_data.sh
#
# Sweep ablation (appendix) 돌릴 때:
#   WINDOWS="60 180 300 600" HORIZONS="5 10 15" bash ...

set -e

# env override 가능: DATA_DIR=... OUT_DIR=... bash prepare_data.sh
DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/hypotension}"
WINDOWS="${WINDOWS:-300}"          # canonical: 5 min input window
HORIZONS="${HORIZONS:-15}"          # canonical: 15 min ahead
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
