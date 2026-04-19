#!/usr/bin/env bash
# Task 8: Any-to-Any Cross-modal — Run all scenarios
# Evaluates in both zero_shot and lora modes.
set -euo pipefail

# ── Config ──────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to pretrained model path}"
MODEL_VERSION="${MODEL_VERSION:-v1}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-outputs/downstream/any_to_any}"

# Data files (output of prepare_data.sh)
CARDIO_DATA="${CARDIO_DATA:-$OUT_DIR/task8_any_to_any_local_ecg_abp_ppg_w30s.pt}"
RESP_DATA="${RESP_DATA:-$OUT_DIR/task8_any_to_any_vitaldb_co2_awp_w30s.pt}"

# Fallback: if local cardio data doesn't exist, try vitaldb version
if [ ! -f "$CARDIO_DATA" ]; then
    CARDIO_DATA="$OUT_DIR/task8_any_to_any_vitaldb_ecg_abp_ppg_cvp_w30s.pt"
fi

# LoRA config
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# ── Common args ─────────────────────────────────────────────
COMMON="--checkpoint $CHECKPOINT --model-version $MODEL_VERSION --device $DEVICE --out-dir $OUT_DIR"
LORA_ARGS="--epochs $EPOCHS --lr $LR --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --batch-size $BATCH_SIZE"

# ── Cardiovascular Scenarios ────────────────────────────────
CARDIO_SCENARIOS=(
    "ECG->ABP"
    "ABP->ECG"
    "PPG->ABP"
    "ECG+PPG->ABP"
)

if [ -f "$CARDIO_DATA" ]; then
    echo "=============================================="
    echo " Cardiovascular scenarios: $CARDIO_DATA"
    echo "=============================================="

    # Zero-shot: all scenarios at once
    echo ""
    echo "--- Zero-shot (all cardiovascular) ---"
    for sc in "${CARDIO_SCENARIOS[@]}"; do
        echo ""
        echo ">>> Zero-shot: $sc"
        python -m downstream.generation.cross_modal.run \
            $COMMON --mode zero_shot \
            --data-path "$CARDIO_DATA" \
            --scenario "$sc"
    done

    # LoRA: each scenario separately
    for sc in "${CARDIO_SCENARIOS[@]}"; do
        echo ""
        echo ">>> LoRA: $sc"
        python -m downstream.generation.cross_modal.run \
            $COMMON --mode lora \
            --data-path "$CARDIO_DATA" \
            --scenario "$sc" \
            $LORA_ARGS
    done
else
    echo "WARNING: Cardiovascular data not found: $CARDIO_DATA"
    echo "  Run prepare_data.sh first."
fi

# ── Respiratory Scenarios ───────────────────────────────────
RESP_SCENARIOS=(
    "CO2->AWP"
    "AWP->CO2"
)

if [ -f "$RESP_DATA" ]; then
    echo ""
    echo "=============================================="
    echo " Respiratory scenarios: $RESP_DATA"
    echo "=============================================="

    for sc in "${RESP_SCENARIOS[@]}"; do
        echo ""
        echo ">>> Zero-shot: $sc"
        python -m downstream.generation.cross_modal.run \
            $COMMON --mode zero_shot \
            --data-path "$RESP_DATA" \
            --scenario "$sc"
    done

    for sc in "${RESP_SCENARIOS[@]}"; do
        echo ""
        echo ">>> LoRA: $sc"
        python -m downstream.generation.cross_modal.run \
            $COMMON --mode lora \
            --data-path "$RESP_DATA" \
            --scenario "$sc" \
            $LORA_ARGS
    done
else
    echo "WARNING: Respiratory data not found: $RESP_DATA"
    echo "  Run prepare_data.sh first."
fi

echo ""
echo "=============================================="
echo " All scenarios complete. Results in: $OUT_DIR"
echo "=============================================="
