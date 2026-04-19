#!/bin/bash
# Anomaly Detection — ECG Signal Quality 실험 스크립트
# Reconstruction error 기반 zero-shot (학습 없음)
#
# 사전 조건: prepare_data.sh로 데이터 준비 완료
#
# 사용법:
#   bash downstream/anomaly_detection/bash/run.sh

set -e

CHECKPOINT=/home/coder/workspace/updown/bio_fm/output/phase2/base/checkpoints/best.pt
DATA_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/signal_quality
OUT_DIR=/home/coder/workspace/updown/bio_fm/result/downstream/anomaly_detection
DEVICE=cuda

MASK_RATIO=0.5
N_TRIALS=5

# Pretraining에서 학습한 ECG lead만 사용
LEADS=("II" "V5")

echo "============================================================"
echo "  Anomaly Detection — ECG Signal Quality (Zero-Shot)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  Leads:      ${LEADS[*]}"
echo "  Mask ratio: $MASK_RATIO, Trials: $N_TRIALS"
echo "============================================================"

for LEAD in "${LEADS[@]}"; do
    EXP_DIR="${OUT_DIR}/lead_${LEAD}"
    mkdir -p "$EXP_DIR"

    echo -e "\n[Lead ${LEAD}]"

    python -m downstream.safety.false_alarm.run \
        --checkpoint "$CHECKPOINT" \
        --data-dir "$DATA_DIR" \
        --lead "$LEAD" \
        --mask-ratio "$MASK_RATIO" \
        --n-trials "$N_TRIALS" \
        --device "$DEVICE" \
        --out-dir "$EXP_DIR"
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
