#!/bin/bash
# Arrhythmia multi-class 1D-CNN baseline — CARMEN 과 같은 데이터·5-fold·평가.
# 사전: arrhythmia prepare_data_vitaldb 로 arrhythmia_ecg_ppg_fold*.pt 생성.
# 사용법:
#   bash downstream/baselines/bash/run_arrhythmia_cnn.sh
#   CUDA_VISIBLE_DEVICES=0 bash ...
#   env override: SIGNALS/EPOCHS/PATIENCE/BATCH/LR/DATA_DIR/OUT_DIR

set -e

DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/arrhythmia}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/arrhythmia_cnn}
DEVICE=${DEVICE:-cuda}

SIGNALS=${SIGNALS:-ecg_ppg}            # prefix 채널 토큰 (= ecg ppg)
N_FOLDS=${N_FOLDS:-5}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}               # val macro-AUROC 무개선 시 early-stop
LR=${LR:-1e-3}
BATCH=${BATCH:-128}
WORKERS=${WORKERS:-0}                  # arrhythmia 데이터는 RAM 로드(작음) → DataLoader worker 불필요
FORCE=${FORCE:-0}

INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')
PREFIX="${DATA_DIR}/arrhythmia_${SIGNALS}"
EXP_DIR="${OUT_DIR}/${SIGNALS}"
mkdir -p "$EXP_DIR"

echo "============================================================"
echo "  Arrhythmia multi-class 1D-CNN baseline"
echo "  Data:    $PREFIX  | Signals: $INPUT_SIGNALS"
echo "  Output:  $EXP_DIR | Epochs: $EPOCHS (patience $PATIENCE) LR $LR Batch $BATCH"
echo "============================================================"

if ! ls "${PREFIX}"_fold0.pt >/dev/null 2>&1; then
    echo "[SKIP] ${PREFIX}_fold0.pt not found (prepare_data_vitaldb 필요)"
    exit 0
fi

for f in $(seq 0 $((N_FOLDS - 1))); do
    if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/fold${f}.json" ]; then
        echo "  [skip] done: ${EXP_DIR} (fold $f)"
        continue
    fi
    echo -e "\n[arrhythmia-cnn | fold ${f}]"
    python -m downstream.baselines.arrhythmia_cnn \
        --data-path "$PREFIX" \
        --input-signals $INPUT_SIGNALS \
        --n-folds "$N_FOLDS" --fold "$f" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --lr "$LR" --batch-size "$BATCH" \
        --device "$DEVICE" --out-dir "$EXP_DIR"
done

echo -e "\n============================================================"
echo "  Done! Results: $EXP_DIR"
echo "============================================================"
