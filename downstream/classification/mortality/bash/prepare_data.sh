#!/bin/bash
# ICU Mortality Prediction — 데이터 준비
# Mortality cohort CSV + MIMIC-III Waveform → mortality .pt
#
# 사전 조건:
#   1. BigQuery에서 mortality cohort SQL 실행 → CSV 다운로드
#   2. download_waveforms.py로 waveform 다운로드 완료
#
# 사용법:
#   bash downstream/classification/mortality/bash/prepare_data.sh

set -e

COHORT_CSV=/home/coder/workspace/updown/bio_fm/downstream/classification/mortality/bquxjob_6a8255f2_19d9042b214.csv
WAVEFORM_DIR=/home/coder/workspace/updown/bio_fm/data/raw/mimic3-waveform-mortality
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/mortality

WINDOW_SECS=(600)
STRIDE_SEC=300

echo "============================================================"
echo "  ICU Mortality Prediction — Data Preparation"
echo "  Cohort:   $COHORT_CSV"
echo "  Waveform: $WAVEFORM_DIR"
echo "  Output:   $OUT_DIR"
echo "============================================================"

for WIN in "${WINDOW_SECS[@]}"; do
    echo -e "\n[Window ${WIN}s, Stride ${STRIDE_SEC}s]"

    python -m downstream.classification.mortality.prepare_data \
        --cohort-csv "$COHORT_CSV" \
        --waveform-dir "$WAVEFORM_DIR" \
        --out-dir "$OUT_DIR" \
        --window-sec "$WIN" \
        --stride-sec "$STRIDE_SEC"
done

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "============================================================"
