#!/bin/bash
# Sepsis Prediction — 데이터 준비
#
# 사전 조건:
#   1. BigQuery sepsis3_cohort CSV 다운로드 완료
#   2. download_waveforms.py로 waveform 다운로드 완료
#
# 사용법:
#   bash downstream/classification/sepsis/bash/prepare_data.sh

set -e

REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
COHORT_CSV="${COHORT_CSV:-${REPO_ROOT}/downstream/outcome/sepsis/bquxjob_93e3c7c_19d8f609070.csv}"
WAVEFORM_DIR="${WAVEFORM_DIR:-${REPO_ROOT}/data/raw/mimic3-waveform-sepsis}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/downstream/sepsis}"

echo "============================================================"
echo "  Sepsis Prediction — Data Preparation"
echo "  Cohort:   $COHORT_CSV"
echo "  Waveform: $WAVEFORM_DIR"
echo "  Output:   $OUT_DIR"
echo "============================================================"

python -m downstream.outcome.sepsis.prepare_data \
    --cohort-csv "$COHORT_CSV" \
    --waveform-dir "$WAVEFORM_DIR" \
    --out-dir "$OUT_DIR" \
    --window-sec 600 \
    --stride-sec 300

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "============================================================"
