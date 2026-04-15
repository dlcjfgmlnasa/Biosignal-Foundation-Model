#!/bin/bash
# Anomaly Detection — Reconstruction Error 기반 실험 스크립트
# 3가지 신호 조합 × 3 윈도우 = 9 실험 (학습 없음, zero-shot)
#
# 사전 조건: prepare_data.sh로 .pt 데이터셋 생성 완료
#
# 사용법:
#   bash downstream/anomaly_detection/bash/run.sh

set -e

# ── 설정 ──
CHECKPOINT=/home/coder/workspace/updown/bio_fm/output/phase2/base/checkpoints/best.pt
DATA_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/anomaly_detection
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/anomaly_detection/results
DEVICE=cuda

MASK_RATIO=0.5
N_TRIALS=5

# ── 실험 조합 ──
SIGNAL_COMBOS=("ecg" "ecg_ppg" "ecg_ppg_abp")
WINDOWS=(10 20 30)

TOTAL=$(( ${#SIGNAL_COMBOS[@]} * ${#WINDOWS[@]} ))
COUNT=0

echo "============================================================"
echo "  Anomaly Detection — Reconstruction Error (Zero-Shot)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  Device:     $DEVICE"
echo "  Signals:    ${SIGNAL_COMBOS[*]}"
echo "  Windows:    ${WINDOWS[*]}"
echo "  Mask ratio: $MASK_RATIO, Trials: $N_TRIALS"
echo "  Total:      $TOTAL experiments"
echo "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    for W in "${WINDOWS[@]}"; do
        PT_FILE="${DATA_DIR}/false_alarm_${SIGNALS}_w${W}s.pt"

        if [ ! -f "$PT_FILE" ]; then
            echo "  SKIP: $PT_FILE not found"
            continue
        fi

        COUNT=$((COUNT + 1))
        EXP_DIR="${OUT_DIR}/${SIGNALS}_w${W}s"
        mkdir -p "$EXP_DIR"

        echo -e "\n[${COUNT}/${TOTAL}] ${SIGNALS} | w=${W}s"

        # --input-signals: underscore → space 변환
        INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')

        python -m downstream.anomaly_detection.run \
            --checkpoint "$CHECKPOINT" \
            --data-path "$PT_FILE" \
            --input-signals $INPUT_SIGNALS \
            --window-sec "$W" \
            --mask-ratio "$MASK_RATIO" \
            --n-trials "$N_TRIALS" \
            --device "$DEVICE" \
            --out-dir "$EXP_DIR"

    done
done

echo -e "\n============================================================"
echo "  Done! ${COUNT}/${TOTAL} experiments completed"
echo "  Results: $OUT_DIR"
echo "============================================================"
