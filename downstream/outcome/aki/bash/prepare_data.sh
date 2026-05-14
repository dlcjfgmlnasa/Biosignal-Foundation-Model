#!/bin/bash
# Postoperative AKI Prediction — paper canonical + 5-fold CV (Task #9)
# Paper Table 1 FINAL (2026-05-14): ABP + ECG + PPG (3 modalities, CVP 제거)
# Prediction setting: Whole surgery → KDIGO Stage ≥ 1 within 7d postop
#
# 사용법:
#   bash downstream/outcome/aki/bash/prepare_data.sh
#
# env override:
#   DATA_DIR=... CLINICAL_CSV=... LAB_CSV=... OUT_DIR=... \
#     [WINDOW_SEC=600 STRIDE_SEC=30 LABEL_MODE=binary N_FOLDS=5] \
#     bash downstream/outcome/aki/bash/prepare_data.sh

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
CLINICAL_CSV="${CLINICAL_CSV:-/home/coder/workspace/datasets/vitaldb_open/1.0.0/clinical_data.csv}"
LAB_CSV="${LAB_CSV:-/home/coder/workspace/datasets/vitaldb_open/1.0.0/lab_data.csv}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/aki}"
WINDOW_SEC="${WINDOW_SEC:-600}"      # 10min window (whole-surgery sliding)
STRIDE_SEC="${STRIDE_SEC:-30}"
LABEL_MODE="${LABEL_MODE:-binary}"
MAX_POSTOP_DAYS="${MAX_POSTOP_DAYS:-7}"
N_FOLDS="${N_FOLDS:-5}"              # clinical AI 표준
REQUIRED="${REQUIRED:-abp ecg ppg}"

echo "============================================================"
echo "  Postop AKI (Task #9) — canonical + ${N_FOLDS}-fold CV"
echo "  Data:        $DATA_DIR"
echo "  Clinical:    $CLINICAL_CSV"
echo "  Lab:         $LAB_CSV"
echo "  Output:      $OUT_DIR"
echo "  Window:      ${WINDOW_SEC}s, Stride: ${STRIDE_SEC}s"
echo "  Label mode:  $LABEL_MODE"
echo "  Postop win:  ${MAX_POSTOP_DAYS} days"
echo "  N folds:     $N_FOLDS"
echo "  Modality:    ABP + ECG + PPG"
echo "============================================================"

python -m downstream.outcome.aki.prepare_data \
    --data-dir "$DATA_DIR" \
    --clinical-csv "$CLINICAL_CSV" \
    --lab-csv "$LAB_CSV" \
    --input-signals abp ecg ppg \
    --required-signals $REQUIRED \
    --window-sec $WINDOW_SEC \
    --stride-sec $STRIDE_SEC \
    --label-mode $LABEL_MODE \
    --max-postop-days $MAX_POSTOP_DAYS \
    --n-folds $N_FOLDS \
    --out-dir "$OUT_DIR"

echo -e "\n============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
