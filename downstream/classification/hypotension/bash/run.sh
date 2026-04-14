#!/bin/bash
# Hypotension Prediction — 전체 실험 실행 스크립트
# 6가지 신호 조합 × 4 windows × 3 horizons × 2 modes = 144 실험
#
# 사전 조건: prepare_data.sh로 .pt 데이터셋이 생성되어 있어야 함
#
# 사용법:
#   bash downstream/classification/hypotension/run.sh

set -e

# ── 설정 ──
CHECKPOINT=/home/coder/workspace/updown/bio_fm/output/phase2/base/checkpoints/best.pt
DATA_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/hypotension
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/hypotension
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
SIGNAL_COMBOS=("abp" "ecg" "ppg" "ecg_ppg" "ecg_abp" "ecg_ppg_abp")
WINDOWS=(30 60 300 600)
HORIZONS=(5 10 15)
MODES=("linear_probe" "lora")

TOTAL=$(( ${#SIGNAL_COMBOS[@]} * ${#WINDOWS[@]} * ${#HORIZONS[@]} * ${#MODES[@]} ))
COUNT=0

echo "============================================================"
echo "  Hypotension Prediction — Experiment Sweep"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  Device:     $DEVICE"
echo "  Signals:    ${SIGNAL_COMBOS[*]}"
echo "  Windows:    ${WINDOWS[*]}"
echo "  Horizons:   ${HORIZONS[*]}"
echo "  Modes:      ${MODES[*]}"
echo "  Total:      $TOTAL experiments"
echo "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    for W in "${WINDOWS[@]}"; do
        for H in "${HORIZONS[@]}"; do
            # .pt 파일 경로
            PT_FILE="${DATA_DIR}/task1_hypotension_${SIGNALS}_w${W}s_h${H}min.pt"

            if [ ! -f "$PT_FILE" ]; then
                echo "  SKIP: $PT_FILE not found"
                continue
            fi

            for MODE in "${MODES[@]}"; do
                COUNT=$((COUNT + 1))
                EXP_DIR="${OUT_DIR}/${MODE}/${SIGNALS}_w${W}s_h${H}min"
                mkdir -p "$EXP_DIR"

                echo -e "\n[${COUNT}/${TOTAL}] ${MODE} | ${SIGNALS} | w=${W}s h=${H}min"

                # 모드별 인자 설정
                if [ "$MODE" = "linear_probe" ]; then
                    EPOCHS=$LP_EPOCHS
                    LR=$LP_LR
                    EXTRA_ARGS=""
                else
                    EPOCHS=$LORA_EPOCHS
                    LR=$LORA_LR
                    EXTRA_ARGS="--lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA"
                fi

                # --input-signals: underscore → space 변환
                INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')

                python -m downstream.classification.hypotension.run \
                    --checkpoint "$CHECKPOINT" \
                    --mode "$MODE" \
                    --data-path "$PT_FILE" \
                    --input-signals $INPUT_SIGNALS \
                    --window-sec "$W" \
                    --epochs "$EPOCHS" \
                    --lr "$LR" \
                    --device "$DEVICE" \
                    --out-dir "$EXP_DIR" \
                    $EXTRA_ARGS

            done
        done
    done
done

echo -e "\n============================================================"
echo "  Done! ${COUNT}/${TOTAL} experiments completed"
echo "  Results: $OUT_DIR"
echo "============================================================"
