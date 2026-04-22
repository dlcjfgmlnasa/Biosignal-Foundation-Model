#!/bin/bash
# cardiac arrest prediction — 데이터 준비
#
# 사전 조건:
#   1. BigQuery cardiac_arrest_cohort CSV 다운로드 완료
#   2. download_waveforms.py로 waveform 다운로드 완료
#
# 사용법:
#   bash downstream/outcome/cardiac_arrest/bash/prepare_data.sh

set -e

COHORT_CSV=/home/coder/workspace/updown/bio_fm/downstream/outcome/cardiac_arrest/bquxjob_cardiac_arrest_TODO.csv
WAVEFORM_DIR=/home/coder/workspace/updown/bio_fm/data/raw/mimic3-waveform-cardiac-arrest
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/cardiac_arrest

echo "============================================================"
echo "  cardiac arrest prediction — Data Preparation"
echo "  Cohort:   $COHORT_CSV"
echo "  Waveform: $WAVEFORM_DIR"
echo "  Output:   $OUT_DIR"
echo "============================================================"

python -m downstream.outcome.cardiac_arrest.prepare_data \
    --cohort-csv "$COHORT_CSV" \
    --waveform-dir "$WAVEFORM_DIR" \
    --out-dir "$OUT_DIR" \
    --window-sec 600 \
    --stride-sec 300

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "============================================================"
