#!/bin/bash
# Outcome Prediction — Cardiac Arrest (patient-level, SCOPE) — 데이터 준비.
#
# 소스: SCOPE (SNUH ICU, KHDP icu-cardiac-arrest) — .vital + metadata_v1.0.xlsx.
# 코호트: arrest(1) vs discharge(0), death 제외. 입력: ECG + PPG (2채널).
# 예측 시점: outcome 12h 전 (lead). 입력 = 고정 12h history band [anchor−24h, anchor−12h].
#   → 환자당 최대 144 window (12h/5min), aggregator 로 환자 1 예측.
#
# 사용법:
#   bash downstream/outcome/cardiac_arrest/bash/prepare_data.sh
#   (env override: METADATA=... VITAL_DIR=... OUT_DIR=... TASK=arrest bash ...)

set -e

METADATA=${METADATA:-/home/coder/workspace/datasets/icu-cardiac-arrest/1.0/metadata_v1.0.xlsx}
VITAL_DIR=${VITAL_DIR:-/home/coder/workspace/datasets/icu-cardiac-arrest/1.0}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/cardiac_arrest_outcome}
TASK=${TASK:-arrest}                 # arrest(pos=Cardiac Arrest, neg=Discharge, Death 제외)
INPUT_SIGNALS=${INPUT_SIGNALS:-ecg ppg}
WINDOW_SEC=${WINDOW_SEC:-300}
STRIDE_SEC=${STRIDE_SEC:-300}
LEAD_SEC=${LEAD_SEC:-43200}          # 12h lead
HISTORY_SEC=${HISTORY_SEC:-43200}    # 12h 고정 history band
MAX_WINDOWS=${MAX_WINDOWS:-144}      # 12h / 5min = 144
BAND_HOURS=${BAND_HOURS:-48}
N_FOLDS=${N_FOLDS:-5}
WORKERS=${WORKERS:-8}

echo "============================================================"
echo "  Cardiac Arrest Outcome (SCOPE, patient-level) — Data Preparation"
echo "  Metadata: $METADATA"
echo "  Vital:    $VITAL_DIR"
echo "  Task=$TASK | Signals: $INPUT_SIGNALS | Window: ${WINDOW_SEC}s stride ${STRIDE_SEC}s"
echo "  Lead: ${LEAD_SEC}s | History band: ${HISTORY_SEC}s | max_windows: $MAX_WINDOWS"
echo "  Output:   $OUT_DIR"
echo "============================================================"

# INPUT_SIGNALS 는 다중값이라 일부러 unquoted (word-split).
python -m downstream.outcome.cardiac_arrest.prepare_data_scope \
    --metadata "$METADATA" \
    --vital-dir "$VITAL_DIR" \
    --task $TASK \
    --input-signals $INPUT_SIGNALS \
    --window-sec $WINDOW_SEC \
    --stride-sec $STRIDE_SEC \
    --lead-sec $LEAD_SEC \
    --history-sec $HISTORY_SEC \
    --max-windows $MAX_WINDOWS \
    --band-hours $BAND_HOURS \
    --n-folds $N_FOLDS \
    --out-dir "$OUT_DIR" \
    --workers $WORKERS

echo -e "\n============================================================"
echo "  Done! Output prefix: $OUT_DIR/scope_${TASK}_w${WINDOW_SEC}s"
echo "============================================================"
