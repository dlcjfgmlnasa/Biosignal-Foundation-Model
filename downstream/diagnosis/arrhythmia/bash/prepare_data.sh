#!/bin/bash
# Arrhythmia Classification — 데이터 준비 스크립트
# 1. MIMIC-III-Ext-PPG 파싱 (WFDB → .pt)
# 2. Arrhythmia 데이터셋 생성 (3가지 신호 조합)
#
# 사용법:
#   bash downstream/classification/arrhythmia/bash/prepare_data.sh

set -e

RAW_DIR=/home/coder/workspace/datasets/mimic-iii-ext-ppg/1.1.0
PARSED_DIR=/home/coder/workspace/updown/bio_fm/data/processed/mimic3_ext_ppg
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/arrhythmia

echo "============================================================"
echo "  Arrhythmia Classification — Data Preparation"
echo "  Raw:    $RAW_DIR"
echo "  Parsed: $PARSED_DIR"
echo "  Output: $OUT_DIR"
echo "  Classes: SR / AF / STACH / SBRAD / AFLT (5-class)"
echo "============================================================"

# 1. WFDB → .pt 파싱
echo -e "\n[1/2] Parsing MIMIC-III-Ext-PPG (WFDB → .pt)..."
python -m data.parser.mimic3_ext_ppg \
    --raw-dir $RAW_DIR \
    --out-dir $PARSED_DIR

# 2. Arrhythmia 데이터셋 생성
echo -e "\n[2/2] Generating arrhythmia datasets..."

# PPG only
echo -e "\n  [1/3] PPG only"
python -m downstream.diagnosis.arrhythmia.prepare_data \
    --data-dir $PARSED_DIR --input-signals ppg --out-dir $OUT_DIR

# ECG only
echo -e "\n  [2/3] ECG only"
python -m downstream.diagnosis.arrhythmia.prepare_data \
    --data-dir $PARSED_DIR --input-signals ecg --out-dir $OUT_DIR

# PPG + ECG
echo -e "\n  [3/3] PPG + ECG"
python -m downstream.diagnosis.arrhythmia.prepare_data \
    --data-dir $PARSED_DIR --input-signals ppg ecg --out-dir $OUT_DIR

echo -e "\n============================================================"
echo "  Done! Datasets saved to: $OUT_DIR"
echo "============================================================"
