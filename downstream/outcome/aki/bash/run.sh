#!/bin/bash
# Postoperative AKI (Task #9) — Frozen Linear-Probe protocol, 5-fold CV
# Frozen encoder + TransformerAggregator + LinearProbe (encoder grad 없음).
# prepare_data.sh 로 5-fold chunked .pt 가 준비돼 있어야 함:
#   aki_binary_abp_ecg_ppg_w600s_fold{0..4}_{train,val,test}_chunk*.pt
#
# 사용법:
#   bash downstream/outcome/aki/bash/run.sh
#
# env override:
#   CHECKPOINT=... DATA_DIR=... OUT_DIR=... DEVICE=cuda \
#     [LABEL_MODE=binary WINDOW_SEC=600 N_FOLDS=5 EPOCHS=30 LR=1e-3 BATCH_SIZE=16] \
#     bash downstream/outcome/aki/bash/run.sh

set -e

CHECKPOINT="${CHECKPOINT:-/home/coder/workspace/updown/bio_fm/output/phase2/base/checkpoints/best.pt}"
DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/aki}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/result/downstream/aki/linear_probe}"
DEVICE="${DEVICE:-cuda}"

LABEL_MODE="${LABEL_MODE:-binary}"
WINDOW_SEC="${WINDOW_SEC:-600}"
N_FOLDS="${N_FOLDS:-5}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SIG_STR="${SIG_STR:-abp_ecg_ppg}"

# load_prepared_split_chunked 는 prefix(파일명·확장자 없음)로 fold/split chunk 를 glob.
DATA_PREFIX="${DATA_DIR}/aki_${LABEL_MODE}_${SIG_STR}_w${WINDOW_SEC}s"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Postop AKI (Task #9) — Frozen Linear Probe, ${N_FOLDS}-fold CV"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Data prefix: $DATA_PREFIX"
echo "  Output:      $OUT_DIR"
echo "  Label mode:  $LABEL_MODE"
echo "  Device:      $DEVICE"
echo "============================================================"

for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
    echo -e "\n[fold ${FOLD}/${N_FOLDS}] Linear Probe (frozen encoder)"
    python -m downstream.outcome.aki.run \
        --checkpoint "$CHECKPOINT" \
        --data-path "$DATA_PREFIX" \
        --mode linear_probe \
        --fold "$FOLD" \
        --n-folds "$N_FOLDS" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --out-dir "$OUT_DIR"
done

# ── OOF 집계: 5-fold preds_fold*.npz → bootstrap 95% CI + mean±SD ──
echo -e "\n[eval] OOF aggregation over ${N_FOLDS} folds"
python -m downstream.run_eval \
    --pred-glob "${OUT_DIR}/preds_fold*.npz" \
    --task-name "aki_${LABEL_MODE}" \
    --out "${OUT_DIR}/aki_${LABEL_MODE}_eval_report.json"

echo -e "\n============================================================"
echo "  Done! Frozen ${N_FOLDS}-fold results: $OUT_DIR"
echo "  Report: ${OUT_DIR}/aki_${LABEL_MODE}_eval_report.json"
echo "============================================================"
