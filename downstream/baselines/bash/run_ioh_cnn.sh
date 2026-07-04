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
PATIENCE=${PATIENCE:-20}               # early-stop (end-to-end CNN 과적합 방지)
LR=${LR:-1e-3}
BATCH=${BATCH:-128}
FORCE=${FORCE:-0}
NUM_GPUS=${NUM_GPUS:-4}                 # fold 를 GPU 에 하나씩 분배해 동시 실행 (0/1=순차)
WORKERS=${WORKERS:-4}                   # ioh_cnn DataLoader worker 수
CACHE_DIR=${CACHE_DIR:-}                # 비우면 EXP_DIR/_cache_fold{f}. 로컬 디스크 권장

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

    # fold 를 GPU 에 라운드로빈 분배해 최대 NUM_GPUS 개를 동시에 실행.
    # 각 프로세스는 CUDA_VISIBLE_DEVICES 로 GPU 1개만 보므로 --device cuda 그대로.
    # 5-fold 는 서로 독립 → embarrassingly parallel (DDP 불필요).
    pids=(); slot=0
    for f in $(seq 0 $((N_FOLDS - 1))); do
        if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
            echo "  [skip] done: ${EXP_DIR} (fold $f)"
            continue
        fi
        gpu=$(( slot % NUM_GPUS ))
        CACHE_ARG=""
        [ -n "$CACHE_DIR" ] && CACHE_ARG="--cache-dir ${CACHE_DIR}/h${H}_fold${f}"
        echo -e "\n[w${WINDOWS}s_h${H}min | fold ${f} → GPU ${gpu}]"
        CUDA_VISIBLE_DEVICES=$gpu python -m downstream.baselines.ioh_cnn \
            --data-path "$PREFIX" \
            --input-signals $INPUT_SIGNALS \
            --n-folds "$N_FOLDS" --fold "$f" \
            --epochs "$EPOCHS" --patience "$PATIENCE" \
            --lr "$LR" --batch-size "$BATCH" \
            --num-workers "$WORKERS" $CACHE_ARG \
            --device "$DEVICE" --out-dir "$EXP_DIR" &
        pids+=($!); slot=$((slot + 1))
        # NUM_GPUS 개 채우면 배치 완료까지 대기 후 다음 배치.
        if [ "$NUM_GPUS" -ge 1 ] && [ ${#pids[@]} -ge "$NUM_GPUS" ]; then
            wait "${pids[@]}" || echo "  [warn] 이 배치에서 non-zero 종료 fold 있음 (로그 확인)"
            pids=(); slot=0
        fi
    done
    # 배치 미충족 잔여 fold 대기.
    [ ${#pids[@]} -gt 0 ] && { wait "${pids[@]}" || echo "  [warn] 잔여 배치 non-zero 종료 fold 있음"; }
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
