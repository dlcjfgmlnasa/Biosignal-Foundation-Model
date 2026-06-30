#!/bin/bash
# Imminent Cardiac Arrest — 데이터 준비 (window-level, manifest-driven).
#
# 의존 순서 (중요):
#   build_records_to_download.py (--cohort-csv)
#     → download_waveforms.py (→ download_manifest.json)
#       → 이 스크립트 (prepare_data --manifest)
#         → bash/run.sh
#
# 입력: ECG + PPG (비침습; 전 채널 동시 valid 구간만 span). horizon 5/15/30 min.
#
# 사용법:
#   bash downstream/acute_event/cardiac_arrest/bash/prepare_data.sh
#   (경로/인자 env override 가능: MANIFEST=... OUT_DIR=... bash ...)

set -e

MANIFEST=${MANIFEST:-/home/coder/workspace/k-mimic-/bio_fm/data/raw/mimic3-waveform-cardiac-arrest/download_manifest.json}
WAVEFORM_DIR=${WAVEFORM_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/raw/mimic3-waveform-cardiac-arrest}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/cardiac_arrest}
INPUT_SIGNALS=${INPUT_SIGNALS:-ecg ppg}
WINDOW_SECS=${WINDOW_SECS:-600}
HORIZON_MINS=${HORIZON_MINS:-5 15 30}
N_FOLDS=${N_FOLDS:-5}

echo "============================================================"
echo "  Imminent Cardiac Arrest — Data Preparation (window-level)"
echo "  Manifest: $MANIFEST"
echo "  Waveform: $WAVEFORM_DIR"
echo "  Signals:  $INPUT_SIGNALS | Window: ${WINDOW_SECS}s | Horizons: $HORIZON_MINS min"
echo "  Output:   $OUT_DIR"
echo "============================================================"

# INPUT_SIGNALS / HORIZON_MINS 는 다중값이라 일부러 unquoted (word-split).
python -m downstream.acute_event.cardiac_arrest.prepare_data \
    --manifest "$MANIFEST" \
    --waveform-dir "$WAVEFORM_DIR" \
    --input-signals $INPUT_SIGNALS \
    --window-secs $WINDOW_SECS \
    --horizon-mins $HORIZON_MINS \
    --n-folds $N_FOLDS \
    --out-dir "$OUT_DIR"

echo -e "\n============================================================"
echo "  Done! Output prefix: $OUT_DIR/cardiac_arrest_${INPUT_SIGNALS// /_}_w${WINDOW_SECS}s_h{H}min"
echo "============================================================"
