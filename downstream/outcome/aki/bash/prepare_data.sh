#!/bin/bash
# Postoperative AKI Prediction — paper-fixed combo only (Task #9)
# Paper Table 1 (2026-05-13): ABP + ECG + PPG + CVP (4 modalities)
# Paper §Common preprocessing: window 600 s, stride 30 s
#
# 사용법:
#   bash downstream/outcome/aki/bash/prepare_data.sh
#
# env override:
#   DATA_DIR=... CLINICAL_CSV=... LAB_CSV=... OUT_DIR=... \
#     [WINDOW_SEC=60 STRIDE_SEC=30 LABEL_MODE=binary] \
#     bash downstream/outcome/aki/bash/prepare_data.sh

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
CLINICAL_CSV="${CLINICAL_CSV:-/home/coder/workspace/datasets/vitaldb_open/1.0.0/clinical_data.csv}"
LAB_CSV="${LAB_CSV:-/home/coder/workspace/datasets/vitaldb_open/1.0.0/lab_data.csv}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/aki}"
WINDOW_SEC="${WINDOW_SEC:-600}"
STRIDE_SEC="${STRIDE_SEC:-30}"
LABEL_MODE="${LABEL_MODE:-binary}"
MAX_POSTOP_DAYS="${MAX_POSTOP_DAYS:-7}"
REQUIRED="${REQUIRED:-abp ecg ppg cvp}"

echo "============================================================"
echo "  Postop AKI (Task #9) — fixed combo"
echo "  Data:        $DATA_DIR"
echo "  Clinical:    $CLINICAL_CSV"
echo "  Lab:         $LAB_CSV"
echo "  Output:      $OUT_DIR"
echo "  Window:      ${WINDOW_SEC}s, Stride: ${STRIDE_SEC}s"
echo "  Label mode:  $LABEL_MODE"
echo "  Postop win:  ${MAX_POSTOP_DAYS} days"
echo "  Modality:    ABP + ECG + PPG + CVP"
echo "============================================================"

python -m downstream.outcome.aki.prepare_data \
    --data-dir "$DATA_DIR" \
    --clinical-csv "$CLINICAL_CSV" \
    --lab-csv "$LAB_CSV" \
    --input-signals abp ecg ppg cvp \
    --required-signals $REQUIRED \
    --window-sec $WINDOW_SEC \
    --stride-sec $STRIDE_SEC \
    --label-mode $LABEL_MODE \
    --max-postop-days $MAX_POSTOP_DAYS \
    --out-dir "$OUT_DIR"

echo -e "\n============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
