#!/bin/bash
# Intracranial Hypertension Detection — 실험 스크립트
#
# 사전 조건:
#   1. download_waveforms.py scan → ICP-RECORDS 생성
#   2. download_waveforms.py download → waveform 다운로드
#   3. prepare_data.py → .pt 생성
#
# 사용법:
#   bash downstream/classification/intracranial_hypertension/bash/run.sh

set -e

CHECKPOINT=/home/coder/workspace/updown/bio_fm/output/phase2/base/checkpoints/best.pt
DATA_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/intracranial_hypertension
OUT_DIR=/home/coder/workspace/updown/bio_fm/result/downstream/intracranial_hypertension
DEVICE=cuda

WINDOW_SECS=(30 60 300 600)
HORIZON_MINS=(5 10 15)
EPOCHS_LP=30
EPOCHS_LORA=30
LR_LP=1e-3
LR_LORA=1e-4
LORA_RANK=8

echo "============================================================"
echo "  Intracranial Hypertension Detection (ICP > 20mmHg)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "============================================================"

for WIN in "${WINDOW_SECS[@]}"; do
    for HORIZON in "${HORIZON_MINS[@]}"; do
        DATA_PATH="${DATA_DIR}/ich_icp_w${WIN}s_h${HORIZON}min.pt"

        if [ ! -f "$DATA_PATH" ]; then
            echo "[SKIP] Not found: $DATA_PATH"
            continue
        fi

        EXP_NAME="w${WIN}s_h${HORIZON}min"
        echo -e "\n[${EXP_NAME}]"

        # Linear Probe
        EXP_DIR="${OUT_DIR}/${EXP_NAME}/linear_probe"
        mkdir -p "$EXP_DIR"

        python -m downstream.acute_event.intracranial_hypertension.run \
            --checkpoint "$CHECKPOINT" \
            --data-path "$DATA_PATH" \
            --mode linear_probe \
            --epochs "$EPOCHS_LP" \
            --lr "$LR_LP" \
            --device "$DEVICE" \
            --out-dir "$EXP_DIR"

        # LoRA
        EXP_DIR="${OUT_DIR}/${EXP_NAME}/lora"
        mkdir -p "$EXP_DIR"

        python -m downstream.acute_event.intracranial_hypertension.run \
            --checkpoint "$CHECKPOINT" \
            --data-path "$DATA_PATH" \
            --mode lora \
            --epochs "$EPOCHS_LORA" \
            --lr "$LR_LORA" \
            --lora-rank "$LORA_RANK" \
            --device "$DEVICE" \
            --out-dir "$EXP_DIR"
    done
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
