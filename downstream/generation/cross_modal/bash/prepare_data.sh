#!/usr/bin/env bash
# Task 8: Any-to-Any Cross-modal — Data Preparation
# Prepares aligned multi-channel windows for cross-modal evaluation/training.
set -euo pipefail

# ── Config ──────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-outputs/downstream/any_to_any}"
N_CASES="${N_CASES:-50}"
WINDOW_SEC="${WINDOW_SEC:-30}"
STRIDE_SEC="${STRIDE_SEC:-15}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"

# Local .pt directory (for --source local)
PARSED_MIMIC3_PPG="${PARSED_MIMIC3_PPG:-}"

# Task #12 Cross-Modal Recon (2026-05-13 확정): 8 modality
#   ECG, ABP, PPG, CVP, CO2, RESP, ICP, PAP
# Pairs: ECG→ABP, ECG→PPG, ABP→PPG, CO2→RESP, ABP→ICP, ABP→PAP, CVP→PAP

# ── Cardiovascular core (MIMIC-III-Ext-PPG, local .pt) ──────
if [ -n "$PARSED_MIMIC3_PPG" ] && [ -d "$PARSED_MIMIC3_PPG" ]; then
    echo "=== Cardiovascular core (MIMIC-III-Ext-PPG, local) ==="
    python -m downstream.generation.cross_modal.prepare_data \
        --source local \
        --data-dir "$PARSED_MIMIC3_PPG" \
        --signal-types ecg abp ppg \
        --n-cases "$N_CASES" \
        --window-sec "$WINDOW_SEC" \
        --stride-sec "$STRIDE_SEC" \
        --train-ratio "$TRAIN_RATIO" \
        --out-dir "$OUT_DIR"
else
    echo "=== Cardiovascular (VitalDB fallback, ECG+ABP+PPG+CVP) ==="
    python -m downstream.generation.cross_modal.prepare_data \
        --source vitaldb \
        --signal-types ecg abp ppg cvp \
        --n-cases "$N_CASES" \
        --window-sec "$WINDOW_SEC" \
        --stride-sec "$STRIDE_SEC" \
        --train-ratio "$TRAIN_RATIO" \
        --out-dir "$OUT_DIR"
fi

# ── Rare-modality virtual probes (MIMIC-III primary; VitalDB 보조) ──
# ABP→ICP, ABP→PAP, CVP→PAP scenarios 위한 ABP+CVP+ICP+PAP windows
echo ""
echo "=== Rare-modality (VitalDB) ==="
python -m downstream.generation.cross_modal.prepare_data \
    --source vitaldb \
    --signal-types abp cvp icp pap \
    --n-cases "$N_CASES" \
    --window-sec "$WINDOW_SEC" \
    --stride-sec "$STRIDE_SEC" \
    --train-ratio "$TRAIN_RATIO" \
    --out-dir "$OUT_DIR"

# ── Respiratory: VitalDB ────────────────────────────────────
# CO2→RESP scenario 위한 CO2+RESP windows
echo ""
echo "=== Respiratory (VitalDB) ==="
python -m downstream.generation.cross_modal.prepare_data \
    --source vitaldb \
    --signal-types co2 resp \
    --n-cases "$N_CASES" \
    --window-sec "$WINDOW_SEC" \
    --stride-sec "$STRIDE_SEC" \
    --train-ratio "$TRAIN_RATIO" \
    --out-dir "$OUT_DIR"

echo ""
echo "Done. Output: $OUT_DIR"
