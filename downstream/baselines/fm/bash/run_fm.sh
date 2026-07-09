#!/bin/bash
# 범용 외부 FM frozen linear-probe baseline 런처 (encoder × 단일 task).
# CARMEN 과 같은 prepared 데이터·5-fold·평가 위에서 외부 FM 을 붙여 비교한다.
# 보통 encoder 별 wrapper(run_<encoder>.sh)가 task 별로 이 스크립트를 호출한다.
#
# 필수 env: ENCODER WEIGHTS
# 주요 env override:
#   ENCODER          biot|ecgfounder|papagei|pulseppg|heartbeit
#   WEIGHTS          사전학습 가중치 파일(또는 HF 디렉토리)
#   THIRD_PARTY_ROOT upstream 레포 루트 (미지정 시 모델별 FM_*_ROOT env 사용)
#   TASK_NAME        npz task 태그 접두 (예: cardiac_arrest)  [default fm]
#   DATA_DIR         prepared .pt 디렉토리
#   PREFIX_STEM      horizon 앞까지의 파일 stem (예: scope_arrest_ecg_ppg_w600s).
#                    기본 "${TASK_NAME}_${SIGNALS}_w${WINDOW}s". _h${H}min 이 뒤에 붙는다.
#   SIGNALS          채널 토큰 (예: ecg_ppg → --input-signals ecg ppg)
#   WINDOW HORIZONS N_FOLDS
#   ID_FIELDS        grouping id 우선순위 (예: "subject_ids case_ids")  [default case_ids]
#   OUT_DIR EPOCHS LR BATCH DROPOUT MAX_SEGMENTS FEAT_BATCH DEVICE FORCE

set -e

ENCODER=${ENCODER:?ENCODER 필요 (biot|ecgfounder|papagei|pulseppg|heartbeit)}
WEIGHTS=${WEIGHTS:?WEIGHTS 필요 (사전학습 가중치 경로)}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-}

TASK_NAME=${TASK_NAME:-fm}
DATA_DIR=${DATA_DIR:?DATA_DIR 필요}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/${TASK_NAME}_${ENCODER}}
DEVICE=${DEVICE:-cuda}

SIGNALS=${SIGNALS:-ecg}
WINDOW=${WINDOW:-600}
HORIZONS=${HORIZONS:-5 15 30}
N_FOLDS=${N_FOLDS:-5}
ID_FIELDS=${ID_FIELDS:-case_ids}
PREFIX_STEM=${PREFIX_STEM:-${TASK_NAME}_${SIGNALS}_w${WINDOW}s}

EPOCHS=${EPOCHS:-100}
LR=${LR:-1e-3}
BATCH=${BATCH:-256}
DROPOUT=${DROPOUT:-0.1}
MAX_SEGMENTS=${MAX_SEGMENTS:-8}
FEAT_BATCH=${FEAT_BATCH:-64}
PATIENCE=${PATIENCE:-0}
FORCE=${FORCE:-0}

INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')
ROOT_ARG=""
[ -n "$THIRD_PARTY_ROOT" ] && ROOT_ARG="--third-party-root $THIRD_PARTY_ROOT"

echo "============================================================"
echo "  FM frozen linear-probe baseline"
echo "  Encoder: $ENCODER   Weights: $WEIGHTS"
echo "  Task=$TASK_NAME  Signals=$INPUT_SIGNALS  Window=${WINDOW}s  Horizons=$HORIZONS"
echo "  Data:    $DATA_DIR   (stem: ${PREFIX_STEM}_h<H>min)"
echo "  Output:  $OUT_DIR"
echo "============================================================"

for H in $HORIZONS; do
    PREFIX="${DATA_DIR}/${PREFIX_STEM}_h${H}min"
    if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1 && [ ! -f "${PREFIX}_fold0.pt" ]; then
        echo "[SKIP] ${PREFIX}_fold0*.pt not found (prepare_data 필요)"
        continue
    fi
    EXP_DIR="${OUT_DIR}/${SIGNALS}_w${WINDOW}s_h${H}min"
    mkdir -p "$EXP_DIR"
    for f in $(seq 0 $((N_FOLDS - 1))); do
        if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
            echo "  [skip] done: ${EXP_DIR} (fold $f)"
            continue
        fi
        echo -e "\n[$ENCODER | $TASK_NAME | w${WINDOW}s_h${H}min | fold ${f}]"
        python -m downstream.baselines.fm.run_fm_probe \
            --encoder "$ENCODER" --weights "$WEIGHTS" $ROOT_ARG \
            --data-path "$PREFIX" \
            --input-signals $INPUT_SIGNALS \
            --task-name "$TASK_NAME" --id-fields $ID_FIELDS \
            --n-folds "$N_FOLDS" --fold "$f" \
            --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH" \
            --dropout "$DROPOUT" --max-segments "$MAX_SEGMENTS" \
            --feat-batch "$FEAT_BATCH" --patience "$PATIENCE" \
            --device "$DEVICE" --out-dir "$EXP_DIR"
    done
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
