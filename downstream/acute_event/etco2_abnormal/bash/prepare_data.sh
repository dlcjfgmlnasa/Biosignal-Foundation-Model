#!/bin/bash
# Intraoperative EtCO2 Abnormal — paper canonical + 5-fold CV (Task #5)
# Paper Table 1 FINAL (2026-05-14): ECG + PPG + AWP, window=3min, horizon=10min ahead
# Label: EtCO2 < 35 또는 > 45 mmHg sustained ≥ 1min (CO2 input 제거 — label leakage 방지)
#
# 사용법:
#   bash downstream/acute_event/etco2_abnormal/bash/prepare_data.sh
#
# Sweep ablation:
#   WINDOWS="60 180 300 600" HORIZONS="5 10 15" bash ...
# Single split (legacy):
#   N_FOLDS=1 bash ...

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
RAW_DIR="${RAW_DIR:-/home/coder/workspace/datasets/vitaldb_open/1.0.0}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/etco2_abnormal}"
WINDOWS="${WINDOWS:-180}"           # canonical: 3 min input window
HORIZONS="${HORIZONS:-10}"           # canonical: 10 min ahead
N_FOLDS="${N_FOLDS:-5}"              # clinical AI 표준: 5-fold patient-level CV
REQUIRED="${REQUIRED:-ecg ppg awp}"

echo "============================================================"
echo "  Intraop EtCO2 Abnormal (Task #5) — canonical + ${N_FOLDS}-fold CV"
echo "  Parsed:    $DATA_DIR"
echo "  Raw vital: $RAW_DIR"
echo "  Output:    $OUT_DIR"
echo "  Windows:   $WINDOWS"
echo "  Horizons:  $HORIZONS"
echo "  N folds:   $N_FOLDS"
echo "  Modality:  ECG + PPG + AWP"
echo "============================================================"

python -m downstream.acute_event.etco2_abnormal.prepare_data \
    --data-dir "$DATA_DIR" \
    --raw-dir "$RAW_DIR" \
    --input-signals ecg ppg awp \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS \
    --horizon-mins $HORIZONS \
    --n-folds $N_FOLDS \
    --out-dir "$OUT_DIR"

echo -e "\nDone! Saved to: $OUT_DIR"
