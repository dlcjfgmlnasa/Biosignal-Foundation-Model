#!/bin/bash
# Postoperative AKI Prediction — VitalDB 데이터 준비 스크립트
#
# 사용법:
#   bash downstream/outcome/aki/bash/prepare_data.sh
#
# env override:
#   DATA_DIR=... CLINICAL_CSV=... LAB_CSV=... OUT_DIR=... \
#     bash downstream/outcome/aki/bash/prepare_data.sh

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
CLINICAL_CSV="${CLINICAL_CSV:-/home/coder/workspace/updown/parser/clinical_data.csv}"
LAB_CSV="${LAB_CSV:-/home/coder/workspace/updown/parser/lab_data.csv}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/aki}"
WINDOW_SEC="${WINDOW_SEC:-600}"
STRIDE_SEC="${STRIDE_SEC:-300}"
LABEL_MODE="${LABEL_MODE:-binary}"
MAX_POSTOP_DAYS="${MAX_POSTOP_DAYS:-7}"
INPUT_SIGNALS="${INPUT_SIGNALS:-abp ecg ppg cvp}"

echo "============================================================"
echo "  Postop AKI — Data Preparation"
echo "  Data:        $DATA_DIR"
echo "  Clinical:    $CLINICAL_CSV"
echo "  Lab:         $LAB_CSV"
echo "  Output:      $OUT_DIR"
echo "  Window:      ${WINDOW_SEC}s, Stride: ${STRIDE_SEC}s"
echo "  Label mode:  $LABEL_MODE"
echo "  Postop win:  ${MAX_POSTOP_DAYS} days"
echo "  Inputs:      $INPUT_SIGNALS"
echo "============================================================"

python -m downstream.outcome.aki.prepare_data \
    --data-dir "$DATA_DIR" \
    --clinical-csv "$CLINICAL_CSV" \
    --lab-csv "$LAB_CSV" \
    --input-signals $INPUT_SIGNALS \
    --window-sec $WINDOW_SEC \
    --stride-sec $STRIDE_SEC \
    --label-mode $LABEL_MODE \
    --max-postop-days $MAX_POSTOP_DAYS \
    --out-dir "$OUT_DIR"

echo "============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
