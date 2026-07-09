#!/bin/bash
# HeartBEiT frozen linear-probe baseline — ECG 전용 (파형→이미지→BEiT).
# ⚠ 가중치는 저자(Mount Sinai) 접근 신청 필요. 획득한 HF BEiT 디렉토리를 WEIGHTS 로.
# 사용법:
#   WEIGHTS=/weights/heartbeit_hf_dir bash downstream/baselines/fm/bash/run_heartbeit.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT=${DATA_ROOT:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream}
OUT_ROOT=${OUT_ROOT:-/home/coder/workspace/k-mimic-/bio_fm/result/main}
export ENCODER=heartbeit
export WEIGHTS=${WEIGHTS:?WEIGHTS 필요 (HeartBEiT HF BEiT 디렉토리 — 접근 신청 필요)}
export THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-${FM_HEARTBEIT_ROOT:-}}
export EPOCHS=${EPOCHS:-100} LR=${LR:-1e-3} BATCH=${BATCH:-256} DEVICE=${DEVICE:-cuda}
# 이미지 ViT 는 무거워 feature 추출 배치를 작게.
export FEAT_BATCH=${FEAT_BATCH:-16} MAX_SEGMENTS=${MAX_SEGMENTS:-8} FORCE=${FORCE:-0}

run_task () {  # name data_subdir signals window horizons prefix_stem id_fields
  TASK_NAME="$1" DATA_DIR="$DATA_ROOT/$2" SIGNALS="$3" WINDOW="$4" \
  HORIZONS="$5" PREFIX_STEM="$6" ID_FIELDS="$7" \
  OUT_DIR="$OUT_ROOT/${1}_${ENCODER}" \
  bash "$SCRIPT_DIR/run_fm.sh"
}

run_task hypotension    hypotension    ecg_ppg_abp 300 "5 10 15" "hypotension_ecg_ppg_abp_w300s" "case_ids"
run_task cardiac_arrest cardiac_arrest ecg_ppg     600 "5 15 30" "scope_arrest_ecg_ppg_w600s"    "subject_ids case_ids"
