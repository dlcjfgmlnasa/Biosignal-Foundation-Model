#!/bin/bash
# Massive Transfusion end-to-end 1D-CNN baseline (SNUH OR) — CARMEN 과 같은 데이터·5-fold·평가.
# 사전: massive_transfusion prepare_data 로 massive_transfusion_{signals}_w{win}s_h{H}min_fold*.pt 생성.
# 사용법:
#   bash downstream/baselines/bash/run_masstf_cnn.sh
#   CUDA_VISIBLE_DEVICES=0 bash ...            # GPU 1장
#   HORIZONS=5 bash ...                         # 특정 horizon 만
#   env override: SIGNALS/WINDOW/HORIZONS/EPOCHS/PATIENCE/BATCH/LR/DATA_DIR/OUT_DIR

set -e

DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/massive_transfusion}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/massive_transfusion_cnn}
DEVICE=${DEVICE:-cuda}

SIGNALS=${SIGNALS:-abp_ppg}              # prefix 채널 토큰 (= abp ppg)
WINDOW=${WINDOW:-600}
HORIZONS=${HORIZONS:-5 10 15}
N_FOLDS=${N_FOLDS:-5}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-20}                 # early-stop (end-to-end CNN 과적합 방지)
LR=${LR:-1e-3}
BATCH=${BATCH:-128}
WORKERS=${WORKERS:-4}
FORCE=${FORCE:-0}

INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')

echo "============================================================"
echo "  Massive Transfusion end-to-end 1D-CNN baseline (SNUH OR)"
echo "  Data:    $DATA_DIR"
echo "  Output:  $OUT_DIR"
echo "  Signals=$INPUT_SIGNALS Window=${WINDOW}s Horizons=$HORIZONS"
echo "  Epochs:  $EPOCHS (patience $PATIENCE) | LR $LR | Batch $BATCH"
echo "============================================================"

for H in $HORIZONS; do
    PREFIX="${DATA_DIR}/massive_transfusion_${SIGNALS}_w${WINDOW}s_h${H}min"
    if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1; then
        echo "[SKIP] ${PREFIX}_fold0_*.pt not found (prepare_data 필요)"
        continue
    fi
    EXP_DIR="${OUT_DIR}/${SIGNALS}_w${WINDOW}s_h${H}min"
    mkdir -p "$EXP_DIR"
    for f in $(seq 0 $((N_FOLDS - 1))); do
        if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
            echo "  [skip] done: ${EXP_DIR} (fold $f)"
            continue
        fi
        echo -e "\n[masstf | w${WINDOW}s_h${H}min | fold ${f}]"
        python -m downstream.baselines.masstf_cnn \
            --data-path "$PREFIX" \
            --input-signals $INPUT_SIGNALS \
            --n-folds "$N_FOLDS" --fold "$f" \
            --epochs "$EPOCHS" --patience "$PATIENCE" \
            --lr "$LR" --batch-size "$BATCH" --num-workers "$WORKERS" \
            --device "$DEVICE" --out-dir "$EXP_DIR"
    done
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
