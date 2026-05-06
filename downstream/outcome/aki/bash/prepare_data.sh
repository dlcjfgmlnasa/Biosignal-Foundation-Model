#!/bin/bash
# Postoperative AKI Prediction — 4-combo sweep (paired comparison)
# Paper Table S7 (Task 9) 정렬: ABP / ECG+PPG / ECG+ABP+PPG / ECG+ABP+PPG+CVP
# Paper §Common preprocessing: window 600 s, stride 30 s
# 모든 combo가 동일 환자 풀 (REQUIRED 보유 환자) 사용
#
# 사용법:
#   bash downstream/outcome/aki/bash/prepare_data.sh
#
# env override:
#   DATA_DIR=... CLINICAL_CSV=... LAB_CSV=... OUT_DIR=... \
#     [WINDOW_SEC=60 STRIDE_SEC=30 LABEL_MODE=binary] \
#     [REQUIRED="abp ecg ppg"] \
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
# 4번째 combo 가 +CVP 이므로 paper 정렬을 위해 cvp 도 required 로 두면 cohort 가
# 크게 줄어든다 (Postop_AKI 노트: CVP 보유율 낮음 + NaN-free 동시 600s 제약 시
# 환자 98.9% 유실). 4 combos 의 paired comparison 만 유지하고 CVP 는 input 으로만
# 사용하려면 REQUIRED 에서 cvp 를 제외하시고 (`abp ecg ppg`), 그러면 4번째 combo
# 만 cvp 보유 환자에서 추가 추출됨 (paired 깨짐). 둘 다 가능 — 본 default 는
# paper paired comparison 우선 (cvp 포함 동일 cohort).
REQUIRED="${REQUIRED:-abp ecg ppg cvp}"

echo "============================================================"
echo "  Postop AKI — Paired Comparison Data Preparation"
echo "  Data:        $DATA_DIR"
echo "  Clinical:    $CLINICAL_CSV"
echo "  Lab:         $LAB_CSV"
echo "  Output:      $OUT_DIR"
echo "  Window:      ${WINDOW_SEC}s, Stride: ${STRIDE_SEC}s"
echo "  Label mode:  $LABEL_MODE"
echo "  Postop win:  ${MAX_POSTOP_DAYS} days"
echo "  Required:    $REQUIRED (same patients for all combos)"
echo "  Total:       4 signal combos"
echo "============================================================"

run_combo() {
    local combo_label="$1"
    local input_signals="$2"
    echo -e "\n[$combo_label] $input_signals"
    python -m downstream.outcome.aki.prepare_data \
        --data-dir "$DATA_DIR" \
        --clinical-csv "$CLINICAL_CSV" \
        --lab-csv "$LAB_CSV" \
        --input-signals $input_signals \
        --required-signals $REQUIRED \
        --window-sec $WINDOW_SEC \
        --stride-sec $STRIDE_SEC \
        --label-mode $LABEL_MODE \
        --max-postop-days $MAX_POSTOP_DAYS \
        --out-dir "$OUT_DIR"
}

# Paper Table S7: ABP / ECG+PPG / ECG+ABP+PPG / ECG+ABP+PPG+CVP
run_combo "1/4" "abp"
run_combo "2/4" "ecg ppg"
run_combo "3/4" "ecg abp ppg"
run_combo "4/4" "ecg abp ppg cvp"

echo -e "\n============================================================"
echo "  Done! 4 datasets saved to: $OUT_DIR"
echo "  All combos use the same patient pool (paired comparison)"
echo "============================================================"
