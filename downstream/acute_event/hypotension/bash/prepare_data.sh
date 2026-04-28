#!/bin/bash
# Hypotension Prediction — 전체 데이터 준비 스크립트 (Paired Comparison)
# 모든 조합이 동일한 환자(ECG+PPG+ABP 모두 있는 환자)에서 추출
# 5가지 신호 조합 × 4 windows × 3 horizons = 60개 데이터셋
#
# 사용법:
#   bash downstream/acute_event/hypotension/bash/prepare_data.sh

set -e

DATA_DIR=/home/coder/workspace/updown/bio_fm/data/test/vitaldb
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/hypotension
WINDOWS="60 180 300 600"
HORIZONS="5 10 15"
REQUIRED="ecg ppg abp"

echo "============================================================"
echo "  Hypotension Prediction — Paired Comparison Data Preparation"
echo "  Data:     $DATA_DIR"
echo "  Output:   $OUT_DIR"
echo "  Windows:  $WINDOWS"
echo "  Horizons: $HORIZONS"
echo "  Required: $REQUIRED (same patients for all combos)"
echo "  Total:    5 signal combos x 12 (w,h) = 60 datasets"
echo "============================================================"

# 1. ABP only (same patients as ECG+PPG+ABP)
echo -e "\n[1/5] ABP only"
python -m downstream.acute_event.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals abp \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

# 2. ECG only (same patients)
echo -e "\n[2/5] ECG only"
python -m downstream.acute_event.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

# 3. PPG only (same patients)
echo -e "\n[3/5] PPG only"
python -m downstream.acute_event.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ppg \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

# 4. ECG + PPG (same patients)
echo -e "\n[4/5] ECG + PPG"
python -m downstream.acute_event.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg ppg \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

# 5. ECG + PPG + ABP (same patients)
echo -e "\n[5/5] ECG + PPG + ABP"
python -m downstream.acute_event.hypotension.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg ppg abp \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS --horizon-mins $HORIZONS --out-dir $OUT_DIR

echo -e "\n============================================================"
echo "  Done! All datasets saved to: $OUT_DIR"
echo "  All combos use the same patient pool (paired comparison)"
echo "============================================================"
