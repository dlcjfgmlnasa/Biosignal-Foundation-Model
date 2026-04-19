#!/bin/bash
# Sepsis Prediction — 실험 스크립트
#
# 사용법:
#   bash downstream/classification/sepsis/bash/run.sh

set -e

CHECKPOINT=/home/coder/workspace/updown/bio_fm/output/phase2/base/checkpoints/best.pt
DATA_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/sepsis
OUT_DIR=/home/coder/workspace/updown/bio_fm/result/downstream/sepsis
DEVICE=cuda

DATA_PATH="${DATA_DIR}/sepsis_w600s.pt"

echo "============================================================"
echo "  Sepsis Prediction (Transformer Aggregator)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data:       $DATA_PATH"
echo "============================================================"

# Linear Probe
EXP_DIR="${OUT_DIR}/linear_probe"
mkdir -p "$EXP_DIR"

echo -e "\n[1/2] Linear Probe"
python -m downstream.organ_dysfunction.sepsis.run \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --mode linear_probe \
    --epochs 30 --lr 1e-3 \
    --batch-size 8 --max-windows 24 \
    --device "$DEVICE" \
    --out-dir "$EXP_DIR"

# LoRA
EXP_DIR="${OUT_DIR}/lora"
mkdir -p "$EXP_DIR"

echo -e "\n[2/2] LoRA Fine-tuning"
python -m downstream.organ_dysfunction.sepsis.run \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --mode lora \
    --epochs 30 --lr 1e-4 --lora-rank 8 \
    --batch-size 8 --max-windows 24 \
    --device "$DEVICE" \
    --out-dir "$EXP_DIR"

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
