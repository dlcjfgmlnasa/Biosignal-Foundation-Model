#!/bin/bash
# Imminent Cardiac Arrest (SCOPE) — 데이터 준비 (window-level).
#
# 소스: SCOPE (SNUH ICU, KHDP icu-cardiac-arrest) — .vital + metadata_v1.0.xlsx.
#   (구 MIMIC manifest 경로는 prepare_data_scope 로 대체됨.)
# 입력: ECG + PPG (2채널). horizon 5/15/30 min. fold split = case_id 단위.
#
# 사용법:
#   bash downstream/acute_event/cardiac_arrest/bash/prepare_data.sh
#   (env override: METADATA=... VITAL_DIR=... OUT_DIR=... TASK=arrest|death bash ...)

set -e

METADATA=${METADATA:-/home/coder/workspace/datasets/icu-cardiac-arrest/1.0/metadata_v1.0.xlsx}
VITAL_DIR=${VITAL_DIR:-/home/coder/workspace/datasets/icu-cardiac-arrest/1.0}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/cardiac_arrest}
TASK=${TASK:-arrest}                 # arrest(pos=Cardiac Arrest) | death(pos=Death), neg=Discharge
INPUT_SIGNALS=${INPUT_SIGNALS:-ecg ppg}
WINDOW_SECS=${WINDOW_SECS:-600}
HORIZON_MINS=${HORIZON_MINS:-5 15 30}
N_FOLDS=${N_FOLDS:-5}
WORKERS=${WORKERS:-8}
TRAIN_NEG_CAP=${TRAIN_NEG_CAP:-0}   # train 음성 case 당 window 캡(0=없음). val/test 불변. 불균형 완화용

echo "============================================================"
echo "  Imminent Cardiac Arrest (SCOPE) — Data Preparation"
echo "  Metadata: $METADATA"
echo "  Vital:    $VITAL_DIR"
echo "  Task=$TASK | Signals: $INPUT_SIGNALS | Window: ${WINDOW_SECS}s | Horizons: $HORIZON_MINS min"
echo "  Output:   $OUT_DIR"
echo "============================================================"

# INPUT_SIGNALS / HORIZON_MINS 는 다중값이라 일부러 unquoted (word-split).
python -m downstream.acute_event.cardiac_arrest.prepare_data_scope \
    --metadata "$METADATA" \
    --vital-dir "$VITAL_DIR" \
    --task $TASK \
    --input-signals $INPUT_SIGNALS \
    --window-secs $WINDOW_SECS \
    --horizon-mins $HORIZON_MINS \
    --n-folds $N_FOLDS \
    --out-dir "$OUT_DIR" \
    --workers $WORKERS \
    --train-neg-cap-per-case $TRAIN_NEG_CAP

echo -e "\n============================================================"
echo "  Done! Output prefix: $OUT_DIR/scope_${TASK}_${INPUT_SIGNALS// /_}_w${WINDOW_SECS}s_h{H}min"
echo "============================================================"
