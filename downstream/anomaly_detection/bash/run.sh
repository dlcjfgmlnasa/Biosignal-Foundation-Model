#!/bin/bash
# ICU False Alarm Reduction — 전체 실험 실행 스크립트
# 3가지 신호 조합 × 4 윈도우 × 2 모드 = 24 실험
#
# 사전 조건: prepare_data.sh로 .pt 데이터셋 생성 완료
#
# 사용법:
#   bash downstream/anomaly_detection/bash/run.sh

set -e

# ── 설정 ──
CHECKPOINT=/home/coder/workspace/updown/bio_fm/output/phase2/base/checkpoints/best.pt
DATA_DIR=/home/coder/workspace/updown/bio_fm/datasets/processed/anomaly_detection
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/anomaly_detection
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
SIGNAL_COMBOS=("ecg" "ecg_ppg" "ecg_ppg_abp")
WINDOWS=(30 60 120 300)
MODES=("linear_probe" "lora")

TOTAL=$(( ${#SIGNAL_COMBOS[@]} * ${#WINDOWS[@]} * ${#MODES[@]} ))
COUNT=0

echo "============================================================"
echo "  ICU False Alarm Reduction — Experiment Sweep"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  Device:     $DEVICE"
echo "  Signals:    ${SIGNAL_COMBOS[*]}"
echo "  Windows:    ${WINDOWS[*]}"
echo "  Modes:      ${MODES[*]}"
echo "  Total:      $TOTAL experiments"
echo "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    for W in "${WINDOWS[@]}"; do
        # .pt 파일 경로: underscore → signal 조합
        PT_FILE="${DATA_DIR}/false_alarm_${SIGNALS}_w${W}s.pt"

        if [ ! -f "$PT_FILE" ]; then
            echo "  SKIP: $PT_FILE not found"
            continue
        fi

        for MODE in "${MODES[@]}"; do
            COUNT=$((COUNT + 1))
            EXP_DIR="${OUT_DIR}/${MODE}/${SIGNALS}_w${W}s"
            mkdir -p "$EXP_DIR"

            echo -e "\n[${COUNT}/${TOTAL}] ${MODE} | ${SIGNALS} | w=${W}s"

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

            python -m downstream.anomaly_detection.run \
                --checkpoint "$CHECKPOINT" \
                --mode "$MODE" \
                --data-path "$PT_FILE" \
                --input-signals $INPUT_SIGNALS \
                --epochs "$EPOCHS" \
                --lr "$LR" \
                --device "$DEVICE" \
                --out-dir "$EXP_DIR" \
                $EXTRA_ARGS

        done
    done
done

echo -e "\n============================================================"
echo "  Done! ${COUNT}/${TOTAL} experiments completed"
echo "  Results: $OUT_DIR"
echo "============================================================"
