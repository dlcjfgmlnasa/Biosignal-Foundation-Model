#!/bin/bash
# Arrhythmia Classification — 전체 실험 실행 스크립트
# 3가지 신호 조합 × 2 modes = 6 실험
#
# 사전 조건: prepare_data.sh로 .pt 데이터셋이 생성되어 있어야 함
#
# 사용법:
#   bash downstream/classification/arrhythmia/bash/run.sh

set -e

# ── 설정 ──
CHECKPOINT=/home/coder/workspace/updown/bio_fm/output/phase2/base/checkpoints/best.pt
DATA_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/arrhythmia
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/arrhythmia
DEVICE=cuda

# Linear Probe 설정
LP_EPOCHS=20
LP_LR=1e-3

# LoRA 설정
LORA_EPOCHS=30
LORA_LR=1e-4
LORA_RANK=8
LORA_ALPHA=16

# ── 실험 조합 ──
SIGNAL_COMBOS=("ppg" "ecg" "ppg_ecg")
MODES=("linear_probe" "lora")

TOTAL=$(( ${#SIGNAL_COMBOS[@]} * ${#MODES[@]} ))
COUNT=0

echo "============================================================"
echo "  Arrhythmia Classification — Experiment Sweep"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  Device:     $DEVICE"
echo "  Signals:    ${SIGNAL_COMBOS[*]}"
echo "  Modes:      ${MODES[*]}"
echo "  Classes:    SR / AF / STACH / SBRAD / AFLT"
echo "  Total:      $TOTAL experiments"
echo "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    PT_FILE="${DATA_DIR}/arrhythmia_${SIGNALS}.pt"

    if [ ! -f "$PT_FILE" ]; then
        echo "  SKIP: $PT_FILE not found"
        continue
    fi

    for MODE in "${MODES[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_DIR="${OUT_DIR}/${MODE}/${SIGNALS}"
        mkdir -p "$EXP_DIR"

        echo -e "\n[${COUNT}/${TOTAL}] ${MODE} | ${SIGNALS}"

        if [ "$MODE" = "linear_probe" ]; then
            EPOCHS=$LP_EPOCHS
            LR=$LP_LR
            EXTRA_ARGS=""
        else
            EPOCHS=$LORA_EPOCHS
            LR=$LORA_LR
            EXTRA_ARGS="--lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA"
        fi

        python -m downstream.diagnosis.arrhythmia.run \
            --checkpoint "$CHECKPOINT" \
            --mode "$MODE" \
            --data-path "$PT_FILE" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --device "$DEVICE" \
            --out-dir "$EXP_DIR" \
            $EXTRA_ARGS

    done
done

echo -e "\n============================================================"
echo "  Done! ${COUNT}/${TOTAL} experiments completed"
echo "  Results: $OUT_DIR"
echo "============================================================"
