#!/bin/bash
# Hypotension Prediction — 전체 데이터 준비 스크립트
# 6가지 신호 조합 × 4 windows × 3 horizons = 72개 데이터셋
#
# 사용법:
#   bash downstream/classification/hypotension/prepare_all.sh

set -e

DATA_DIR=/home/coder/workspace/updown/bio_fm/data/test/vitaldb
OUT_DIR=/home/coder/workspace/updown/bio_fm/outputs/downstream/hypotension
WINDOWS="30 60 300 600"
HORIZONS="5 10 15"

echo "============================================================"
echo "  Hypotension Prediction — Data Preparation Sweep"
echo "  Data:     $DATA_DIR"
echo "  Output:   $OUT_DIR"
echo "  Windows:  $WINDOWS"
echo "  Horizons: $HORIZONS"
echo "  Total:    6 signal combos x 12 (w,h) = 72 datasets"
echo "============================================================"

# 1. ABP only
echo -e "\n[1/6] ABP only"
python -m downstream.classification.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals abp \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

# 2. ECG only
echo -e "\n[2/6] ECG only"
python -m downstream.classification.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

# 3. PPG only
echo -e "\n[3/6] PPG only"
python -m downstream.classification.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ppg \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

# 4. ECG + PPG
echo -e "\n[4/6] ECG + PPG"
python -m downstream.classification.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg ppg \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

# 5. ECG + ABP
echo -e "\n[5/6] ECG + ABP"
python -m downstream.classification.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg abp \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

# 6. ECG + PPG + ABP
echo -e "\n[6/6] ECG + PPG + ABP"
python -m downstream.classification.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg ppg abp \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

echo -e "\n============================================================"
echo "  Done! All datasets saved to: $OUT_DIR"
echo "============================================================"
