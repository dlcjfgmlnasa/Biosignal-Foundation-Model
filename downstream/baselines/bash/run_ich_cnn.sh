#!/bin/bash
# ICH end-to-end 1D-CNN baseline — CARMEN(frozen/LoRA)과 같은 데이터·5-fold·평가.
# 사전: intracranial_hypertension prepare_data 로 _fold*.pt 생성돼 있어야 함.
# 사용법:
#   bash downstream/baselines/bash/run_ich_cnn.sh
#   CUDA_VISIBLE_DEVICES=0 bash ...            # GPU 1장
#   HORIZONS=30 bash ...                        # 특정 horizon 만
#   env override: SIGNALS/WINDOW/HORIZONS/EPOCHS/PATIENCE/BATCH/LR/DATA_DIR/OUT_DIR

set -e

DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/intracranial_hypertension}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/intracranial_hypertension_cnn}
DEVICE=${DEVICE:-cuda}

# ⚠ prefix 채널 토큰 = prepare_data --input-signals "abp icp ecg" → mode_str "abp_icp_ecg".
SIGNALS=${SIGNALS:-abp_icp_ecg}
WINDOW=${WINDOW:-1200}                  # 20min 고정
HORIZONS=${HORIZONS:-5 15 30}
N_FOLDS=${N_FOLDS:-5}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-20}                 # early-stop (end-to-end CNN 과적합 방지)
LR=${LR:-1e-3}
BATCH=${BATCH:-64}                       # 1200s×ch → IOH(128)보다 작게
WORKERS=${WORKERS:-4}
FORCE=${FORCE:-0}

INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')

echo "============================================================"
echo "  ICH end-to-end 1D-CNN baseline"
echo "  Data:    $DATA_DIR"
echo "  Output:  $OUT_DIR"
echo "  Signals: $INPUT_SIGNALS | Window: ${WINDOW}s | Horizons: $HORIZONS"
echo "  Epochs:  $EPOCHS (patience $PATIENCE) | LR $LR | Batch $BATCH"
echo "============================================================"

for H in $HORIZONS; do
    PREFIX="${DATA_DIR}/intracranial_hypertension_${SIGNALS}_w${WINDOW}s_h${H}min"
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
        echo -e "\n[w${WINDOW}s_h${H}min | fold ${f}]"
        python -m downstream.baselines.ich_cnn \
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
