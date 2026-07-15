#!/bin/bash
# BIOT frozen linear-probe baseline — 다채널 (task 제공 신호 전부 사용, 200Hz).
# 사전: BIOT 레포 + pretrained-models/*.ckpt (또는 HF braindecode/BIOT).
# 사용법:
#   FM_BIOT_ROOT=/repos/BIOT WEIGHTS=/repos/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt \
#     bash downstream/baselines/fm/bash/run_biot.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT=${DATA_ROOT:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream}
OUT_ROOT=${OUT_ROOT:-/home/coder/workspace/k-mimic-/bio_fm/result/main}
export ENCODER=biot
export WEIGHTS=${WEIGHTS:?WEIGHTS 필요 (예: /repos/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt)}
export THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-${FM_BIOT_ROOT:-}}
export EPOCHS=${EPOCHS:-100} LR=${LR:-1e-3} BATCH=${BATCH:-256} DEVICE=${DEVICE:-cuda}
export FEAT_BATCH=${FEAT_BATCH:-64} MAX_SEGMENTS=${MAX_SEGMENTS:-0} FORCE=${FORCE:-0}

run_task () {  # name data_subdir signals window horizons prefix_stem id_fields
  TASK_NAME="$1" DATA_DIR="$DATA_ROOT/$2" SIGNALS="$3" WINDOW="$4" \
  HORIZONS="$5" PREFIX_STEM="$6" ID_FIELDS="$7" \
  OUT_DIR="$OUT_ROOT/${1}_${ENCODER}" \
  bash "$SCRIPT_DIR/run_fm.sh"
}

# BIOT 는 다채널 → 모든 신호 조합 task 에 적용 가능(데이터 없으면 자동 SKIP).
run_task hypotension    hypotension    ecg_ppg_abp     300  "5 10 15" "hypotension_ecg_ppg_abp_w300s"  "case_ids"
run_task cardiac_arrest cardiac_arrest ecg_ppg         600  "5 15 30" "scope_arrest_ecg_ppg_w600s"     "subject_ids case_ids"
run_task ich            ich            abp_icp_ecg     1200 "5 15 30" "ich_abp_icp_ecg_w1200s"         "subject_ids case_ids"
