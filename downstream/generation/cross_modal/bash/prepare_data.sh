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
#   ECG, ABP, PPG, CVP, CO2, RESP_Impedance, ICP, PAP
# Pairs: ECG→ABP, ECG→PPG, ABP→PPG, CO2→RESP_Impedance, ABP→ICP, ABP→PAP, CVP→PAP
# v2: RESP → resp_impedance(8). RESP_Flow(9)는 downstream 데이터 소스 부재로 제외.

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

# ── Respiratory ─────────────────────────────────────────────
# CO2→RESP_Impedance scenario 위한 CO2+RESP_Impedance windows.
# ⚠️ source 주의: VitalDB Open(SNUADC)에는 RESP 매핑이 0건이라 vitaldb 소스로는
#    resp_impedance 가 비어 시나리오가 skip 된다. impedance 는 MIMIC-III 만 보유 —
#    이 pair 는 MIMIC-III 소스로 돌려야 데이터가 잡힌다(generation-task 소스 점검 필요).
echo ""
echo "=== Respiratory (CO2 + RESP_Impedance) ==="
python -m downstream.generation.cross_modal.prepare_data \
    --source vitaldb \
    --signal-types co2 resp_impedance \
    --n-cases "$N_CASES" \
    --window-sec "$WINDOW_SEC" \
    --stride-sec "$STRIDE_SEC" \
    --train-ratio "$TRAIN_RATIO" \
    --out-dir "$OUT_DIR"

echo ""
echo "Done. Output: $OUT_DIR"
