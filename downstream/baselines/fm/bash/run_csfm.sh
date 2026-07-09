#!/bin/bash
# CSFM (Cardiac Sensing FM) frozen linear-probe baseline — 멀티모달 (ECG+PPG, 100Hz).
# CARMEN 과 가장 직접적으로 비교되는 cardiac multimodal FM.
# ⚠ 가중치는 Academic Access Agreement 필요 (xiao.gu@eng.ox.ac.uk).
# 사용법:
#   FM_CSFM_ROOT=/repos/Cardiac-Sensing-FM FM_CSFM_VARIANT=base \
#     WEIGHTS=/weights/csfm_base.pth bash downstream/baselines/fm/bash/run_csfm.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT=${DATA_ROOT:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream}
OUT_ROOT=${OUT_ROOT:-/home/coder/workspace/k-mimic-/bio_fm/result/main}
export ENCODER=csfm
export WEIGHTS=${WEIGHTS:?WEIGHTS 필요 (CSFM 가중치 — Academic Access Agreement)}
export THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-${FM_CSFM_ROOT:-}}
export FM_CSFM_VARIANT=${FM_CSFM_VARIANT:-base}
export EPOCHS=${EPOCHS:-100} LR=${LR:-1e-3} BATCH=${BATCH:-256} DEVICE=${DEVICE:-cuda}
export FEAT_BATCH=${FEAT_BATCH:-64} MAX_SEGMENTS=${MAX_SEGMENTS:-8} FORCE=${FORCE:-0}

run_task () {  # name data_subdir signals window horizons prefix_stem id_fields
  TASK_NAME="$1" DATA_DIR="$DATA_ROOT/$2" SIGNALS="$3" WINDOW="$4" \
  HORIZONS="$5" PREFIX_STEM="$6" ID_FIELDS="$7" \
  OUT_DIR="$OUT_ROOT/${1}_${ENCODER}" \
  bash "$SCRIPT_DIR/run_fm.sh"
}

# CSFM = ECG+PPG 멀티모달 → ecg 또는 ppg 를 포함하는 task 전부.
run_task hypotension    hypotension    ecg_ppg_abp 300  "5 10 15" "hypotension_ecg_ppg_abp_w300s" "case_ids"
run_task cardiac_arrest cardiac_arrest ecg_ppg     600  "5 15 30" "scope_arrest_ecg_ppg_w600s"    "subject_ids case_ids"
run_task ich            ich            abp_icp_ecg 1200 "5 15 30" "ich_abp_icp_ecg_w1200s"        "subject_ids case_ids"
