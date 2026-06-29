#!/bin/bash
# Arrhythmia (VitalDB) — 데이터 준비
# VitalDB-Arrhythmia annotation + VitalDB Open .vital → ECG+PPG 30s 5-class .pt (5-fold)
#
# 사용법 (서버):
#   bash downstream/acute_event/arrhythmia/bash/prepare_vitaldb.sh
#   WORKERS=32 MAX_PER_CLASS=8000 bash .../prepare_vitaldb.sh   # 옵션 override
set -e

# ── 경로 (모두 ${VAR:-default} 라 환경변수로 override 가능) ──
ARR_DIR="${ARR_DIR:-/home/coder/workspace/datasets/vitaldb-arrhythmia/1.0.0}"
VITAL_DIR="${VITAL_DIR:-/home/coder/workspace/datasets/vitaldb_open/1.0.0/vital_files}"
# canonical downstream data 경로 (memory project_kmimic_bio_fm_paths)
OUT_DIR="${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/arrhythmia}"

WORKERS="${WORKERS:-16}"
N_FOLDS="${N_FOLDS:-5}"
WIN_SEC="${WIN_SEC:-30}"
STRIDE_SEC="${STRIDE_SEC:-15}"
MIN_BEATS="${MIN_BEATS:-10}"
MAX_PER_CASE_PER_CLASS="${MAX_PER_CASE_PER_CLASS:-40}"
MAX_PER_CLASS="${MAX_PER_CLASS:-}"          # 빈값=무제한 (먼저 분포 보고 정하기 권장)

# 기본은 canonical 조합(ECG+PPG)만. ecg-only/ppg-only ablation 필요 시 아래 주석 해제.
SIGNAL_COMBOS=("ecg ppg")
# SIGNAL_COMBOS=("ecg ppg" "ecg" "ppg")

echo "============================================================"
echo "  Arrhythmia (VitalDB) data prep"
echo "  Annotation: $ARR_DIR"
echo "  Vital:      $VITAL_DIR"
echo "  Out:        $OUT_DIR"
echo "  Workers:    $WORKERS   Folds: $N_FOLDS"
echo "  Window:     ${WIN_SEC}s  stride ${STRIDE_SEC}s  min_beats $MIN_BEATS"
echo "  Cap:        per-case-per-class=$MAX_PER_CASE_PER_CLASS  per-class=${MAX_PER_CLASS:-none}"
echo "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    echo -e "\n>>> input-signals: $SIGNALS"
    python -m downstream.acute_event.arrhythmia.prepare_data_vitaldb \
        --arrhythmia-dir "$ARR_DIR" \
        --vital-dir "$VITAL_DIR" \
        --input-signals $SIGNALS \
        --out-dir "$OUT_DIR" \
        --win-sec "$WIN_SEC" \
        --stride-sec "$STRIDE_SEC" \
        --min-beats "$MIN_BEATS" \
        --max-per-case-per-class "$MAX_PER_CASE_PER_CLASS" \
        --n-folds "$N_FOLDS" \
        --workers "$WORKERS" \
        ${MAX_PER_CLASS:+--max-per-class $MAX_PER_CLASS}
done

echo -e "\nDone. Saved fold files to $OUT_DIR"
