#!/bin/bash
# Anomaly Detection — ECG Signal Quality 데이터 준비
# PhysioNet/CinC Challenge 2011 (12-lead ECG, 500Hz → 100Hz)
#
# 사용법:
#   bash downstream/anomaly_detection/bash/prepare_data.sh

set -e

RAW_DIR=/home/coder/workspace/updown/bio_fm/data/test/physionet2011
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/signal_quality

echo "============================================================"
echo "  Anomaly Detection — ECG Signal Quality Data Preparation"
echo "  Raw:    $RAW_DIR"
echo "  Output: $OUT_DIR"
echo "============================================================"

python -m data.parser.physionet2011 \
    --raw-dir $RAW_DIR \
    --out-dir $OUT_DIR

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "============================================================"
