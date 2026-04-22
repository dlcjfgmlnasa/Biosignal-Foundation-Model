#!/bin/bash
# ICU Mortality Prediction — 실험 스크립트
# Linear Probe + LoRA 두 모드 실행
#
# 사전 조건: prepare_data.sh로 데이터 준비 완료
#
# 사용법:
#   bash downstream/classification/mortality/bash/run.sh

set -e

CHECKPOINT=/home/coder/workspace/updown/bio_fm/output/phase2/base/checkpoints/best.pt
DATA_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/mortality
OUT_DIR=/home/coder/workspace/updown/bio_fm/result/downstream/mortality
DEVICE=cuda

WINDOW_SEC=600
EPOCHS_LP=30
EPOCHS_LORA=30
LR_LP=1e-3
LR_LORA=1e-4
LORA_RANK=8
BATCH_SIZE=32

DATA_PATH="${DATA_DIR}/mortality_w${WINDOW_SEC}s.pt"

echo "============================================================"
echo "  ICU Mortality Prediction"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data:       $DATA_PATH"
echo "  Output:     $OUT_DIR"
echo "============================================================"

# ── Linear Probe ──
EXP_DIR="${OUT_DIR}/linear_probe"
mkdir -p "$EXP_DIR"

echo -e "\n[1/2] Linear Probe"
python -m downstream.outcome.mortality.run \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --mode linear_probe \
    --epochs "$EPOCHS_LP" \
    --lr "$LR_LP" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --out-dir "$EXP_DIR"

# ── LoRA ──
EXP_DIR="${OUT_DIR}/lora"
mkdir -p "$EXP_DIR"

echo -e "\n[2/2] LoRA Fine-tuning"
python -m downstream.outcome.mortality.run \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --mode lora \
    --epochs "$EPOCHS_LORA" \
    --lr "$LR_LORA" \
    --lora-rank "$LORA_RANK" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --out-dir "$EXP_DIR"

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
