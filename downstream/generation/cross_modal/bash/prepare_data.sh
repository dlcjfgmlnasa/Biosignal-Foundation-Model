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

# Task #12 Cross-Modal Recon (2026-06-17 갱신): 강결합 4쌍 (약→강 방향)
#   Pairs: ECG→ABP, PPG→ABP, ECG+PPG→ABP (γ cardio), AWP→RESP_Flow (γ respi)
#   ※ cardio 데이터는 ecg/abp/ppg 동일 추출 — 시나리오만 subset 으로 분기.
#   제외: ABP→PPG(강→약), ABP→ICP(ICP VitalDB 부재),
#         CO2→RESP_Impedance(약결합+소스 부재)

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
    echo "=== Cardiovascular (VitalDB fallback, ECG+ABP+PPG) ==="
    python -m downstream.generation.cross_modal.prepare_data \
        --source vitaldb \
        --signal-types ecg abp ppg \
        --n-cases "$N_CASES" \
        --window-sec "$WINDOW_SEC" \
        --stride-sec "$STRIDE_SEC" \
        --train-ratio "$TRAIN_RATIO" \
        --out-dir "$OUT_DIR"
fi

# ── Respiratory (AWP→RESP_Flow, 강결합 (5,9)) ────────────────
# AWP→RESP_Flow scenario 위한 AWP+RESP_Flow windows.
# ⚠️ source 점검: ventilator(Primus) 파형 — VitalDB Primus 에 airway pressure/flow 가
#    있는지 확인. 없으면 MIMIC-III ventilator 소스로 전환 필요.
echo ""
echo "=== Respiratory (AWP + RESP_Flow) ==="
python -m downstream.generation.cross_modal.prepare_data \
    --source vitaldb \
    --signal-types awp resp_flow \
    --n-cases "$N_CASES" \
    --window-sec "$WINDOW_SEC" \
    --stride-sec "$STRIDE_SEC" \
    --train-ratio "$TRAIN_RATIO" \
    --out-dir "$OUT_DIR"

echo ""
echo "Done. Output: $OUT_DIR"
