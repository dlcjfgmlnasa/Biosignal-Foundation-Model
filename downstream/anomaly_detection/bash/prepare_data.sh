#!/bin/bash
# ICU False Alarm Reduction — 데이터 준비 스크립트
# 3가지 신호 조합 × 4 윈도우 = 12개 데이터셋
#
# 사전 조건: data/parser/physionet2015.py로 파싱 완료
#
# 사용법:
#   bash downstream/anomaly_detection/bash/prepare_data.sh

set -e

DATA_DIR=/home/coder/workspace/updown/bio_fm/datasets/processed/anomaly_detection
OUT_DIR=/home/coder/workspace/updown/bio_fm/datasets/processed/anomaly_detection
WINDOWS="30 60 120 300"

echo "============================================================"
echo "  ICU False Alarm Reduction — Data Preparation"
echo "  Data:    $DATA_DIR"
echo "  Output:  $OUT_DIR"
echo "  Windows: $WINDOWS"
echo "============================================================"

# 1. ECG only
echo -e "\n[1/3] ECG only"
python -m downstream.anomaly_detection.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg \
    --window-secs $WINDOWS --out-dir $OUT_DIR

# 2. ECG + PPG
echo -e "\n[2/3] ECG + PPG"
python -m downstream.anomaly_detection.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg ppg \
    --window-secs $WINDOWS --out-dir $OUT_DIR

# 3. ECG + PPG + ABP
echo -e "\n[3/3] ECG + PPG + ABP"
python -m downstream.anomaly_detection.prepare_data \
    --data-dir $DATA_DIR --input-signals ecg ppg abp \
    --window-secs $WINDOWS --out-dir $OUT_DIR

echo -e "\n============================================================"
echo "  Done! All datasets saved to: $OUT_DIR"
echo "============================================================"
