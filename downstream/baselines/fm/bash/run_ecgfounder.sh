#!/bin/bash
# ECGFounder frozen linear-probe baseline — ECG 전용 (ecg 채널만 사용).
# 사전: 각 task prepare_data 로 표준 per-(fold,split) chunk (.pt) 생성 + ECGFounder 가중치·레포.
# 사용법:
#   FM_ECGFOUNDER_ROOT=/repos/ECGFounder WEIGHTS=/weights/1_lead_ECGFounder.pth \
#     bash downstream/baselines/fm/bash/run_ecgfounder.sh
#   env: DATA_ROOT/OUT_ROOT/WEIGHTS/THIRD_PARTY_ROOT/EPOCHS/LR/BATCH/DEVICE/FORCE
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT=${DATA_ROOT:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream}
OUT_ROOT=${OUT_ROOT:-/home/coder/workspace/k-mimic-/bio_fm/result/main}
export ENCODER=ecgfounder
export WEIGHTS=${WEIGHTS:?WEIGHTS 필요 (예: /weights/1_lead_ECGFounder.pth)}
export THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-${FM_ECGFOUNDER_ROOT:-}}
export EPOCHS=${EPOCHS:-100} LR=${LR:-1e-3} BATCH=${BATCH:-256} DEVICE=${DEVICE:-cuda}
export FEAT_BATCH=${FEAT_BATCH:-64} MAX_SEGMENTS=${MAX_SEGMENTS:-0} FORCE=${FORCE:-0}

run_task () {  # name data_subdir signals window horizons prefix_stem id_fields
  TASK_NAME="$1" DATA_DIR="$DATA_ROOT/$2" SIGNALS="$3" WINDOW="$4" \
  HORIZONS="$5" PREFIX_STEM="$6" ID_FIELDS="$7" \
  OUT_DIR="$OUT_ROOT/${1}_${ENCODER}" \
  bash "$SCRIPT_DIR/run_fm.sh"
}

# ecg 를 포함하는 task 만 (ECGFounder 는 ecg 채널만 인코딩).
run_task hypotension    hypotension    ecg_ppg_abp 300 "5 10 15" "hypotension_ecg_ppg_abp_w300s" "case_ids"
run_task cardiac_arrest cardiac_arrest ecg_ppg     600 "5 15 30" "scope_arrest_ecg_ppg_w600s"    "subject_ids case_ids"
