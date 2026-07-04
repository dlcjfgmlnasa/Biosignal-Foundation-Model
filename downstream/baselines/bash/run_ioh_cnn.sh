#!/bin/bash
# IOH end-to-end 1D-CNN baseline — VitalDB IOH 문헌(end-to-end supervised) 비교군.
# CARMEN(frozen/LoRA)과 같은 prepared 데이터·5-fold·평가로 직접 비교.
#
# 사전: hypotension prepare_data 로 _fold*.pt 생성돼 있어야 함(CARMEN 과 동일 데이터).
# 사용법:
#   bash downstream/baselines/bash/run_ioh_cnn.sh
#   # env override: WINDOWS/HORIZONS/EPOCHS/PATIENCE/BATCH/LR/DATA_DIR/OUT_DIR

set -e

DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/hypotension}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/hypotension_cnn}
DEVICE=${DEVICE:-cuda}

SIGNALS=${SIGNALS:-ecg_ppg_abp}        # prefix 채널 토큰 (= ecg ppg abp)
WINDOWS=${WINDOWS:-300}
HORIZONS=${HORIZONS:-5 10 15}
N_FOLDS=${N_FOLDS:-5}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-5}                # early-stop (end-to-end CNN 과적합 방지)
LR=${LR:-1e-3}
BATCH=${BATCH:-128}
FORCE=${FORCE:-0}
# GPU: fold 하나가 보이는 모든 GPU 를 DataParallel 로 사용(단일 프로세스, 데이터 RAM 1벌).
#   GPU 를 제한하려면 실행 앞에 CUDA_VISIBLE_DEVICES=0,1 지정. fold 는 순차(동시 아님).
WORKERS=${WORKERS:-4}                   # ioh_cnn DataLoader worker 수 (Linux fork 시 데이터 COW 공유)

INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')

echo "============================================================"
echo "  IOH end-to-end 1D-CNN baseline"
echo "  Data:    $DATA_DIR"
echo "  Output:  $OUT_DIR"
echo "  Signals: $INPUT_SIGNALS | Window: ${WINDOWS}s | Horizons: $HORIZONS"
echo "  Epochs:  $EPOCHS (patience $PATIENCE) | LR $LR | Batch $BATCH"
echo "============================================================"

for H in $HORIZONS; do
    PREFIX="${DATA_DIR}/hypotension_${SIGNALS}_w${WINDOWS}s_h${H}min"
    if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1; then
        echo "[SKIP] ${PREFIX}_fold0_*.pt not found (prepare_data 필요)"
        continue
    fi
    EXP_DIR="${OUT_DIR}/${SIGNALS}_w${WINDOWS}s_h${H}min"
    mkdir -p "$EXP_DIR"

    # fold 순차 실행 — 각 fold 가 보이는 모든 GPU 를 ioh_cnn 의 DataParallel 로 사용한다
    # (단일 프로세스 → 데이터는 RAM 에 1벌만, GPU 4개 활용). GPU 를 제한하려면 외부에서
    # CUDA_VISIBLE_DEVICES=0,1 처럼 지정. RAM-resident 라 fold 는 동시 실행하지 않는다.
    for f in $(seq 0 $((N_FOLDS - 1))); do
        if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
            echo "  [skip] done: ${EXP_DIR} (fold $f)"
            continue
        fi
        echo -e "\n[w${WINDOWS}s_h${H}min | fold ${f} | all visible GPUs (DataParallel)]"
        python -m downstream.baselines.ioh_cnn \
            --data-path "$PREFIX" \
            --input-signals $INPUT_SIGNALS \
            --n-folds "$N_FOLDS" --fold "$f" \
            --epochs "$EPOCHS" --patience "$PATIENCE" \
            --lr "$LR" --batch-size "$BATCH" \
            --num-workers "$WORKERS" \
            --device "$DEVICE" --out-dir "$EXP_DIR"
    done
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
