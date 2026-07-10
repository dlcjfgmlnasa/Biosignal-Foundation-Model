#!/bin/bash
# Intraoperative Hypoxemia Prediction — frozen/LoRA sweep (window-level, horizon × 5-fold).
#
#   - IOH(hypotension) 와 동일 protocol: 입력·window 고정, horizon 만 변경. val best-ckpt → test 1회.
#   - Resume: 완료 셀(out-dir/preds_fold{f}.npz)은 건너뛴다. FORCE=1 이면 전부 재실행.
#
# 사전: bash/prepare_data.sh 로 hypoxemia_{signals}_w{W}s_h{H}min_fold*.pt 생성.
#
# 사용법:
#   # (1) 직렬 실행 (NPROC>1 이면 fold 당 torchrun DDP)
#   nohup bash downstream/acute_event/hypoxemia/bash/run.sh > hypox_sweep.log 2>&1 &
#
#   # (2) GPU job-sharding: 명령만 내보내 4 GPU 풀에 분산
#   PRINT_ONLY=1 bash downstream/acute_event/hypoxemia/bash/run.sh > jobs.txt
#   python -m downstream.run_sharded --gpus 0,1,2,3 --jobs-file jobs.txt

set -e

# canonical 경로는 k-mimic-/bio_fm 아래 (updown 기본은 stale).
CHECKPOINT=${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}
DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/hypoxemia}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/hypoxemia}
DEVICE=${DEVICE:-cuda}
MODEL_VERSION=${MODEL_VERSION:-v2}   # v2 필수 (9 modality 단일 embedding). v1 로드 금지.

PRINT_ONLY=${PRINT_ONLY:-0}
log()  { if [ "$PRINT_ONLY" = "1" ]; then echo    "$@" >&2; else echo    "$@"; fi; }
loge() { if [ "$PRINT_ONLY" = "1" ]; then echo -e "$@" >&2; else echo -e "$@"; fi; }

LP_EPOCHS=${LP_EPOCHS:-2000}
LP_LR=${LP_LR:-1e-3}
LP_BATCH=${LP_BATCH:-512}

LORA_EPOCHS=${LORA_EPOCHS:-200}
LORA_PATIENCE=${LORA_PATIENCE:-5}
LORA_LR=${LORA_LR:-1e-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
# ⚠ batch 를 바꾸면 LoRA 최적화 궤적이 달라져 다른 task 의 batch=128 결과와 비교성 깨짐.
LORA_BATCH=${LORA_BATCH:-128}

NPROC=${NPROC:-4}
if [ "$NPROC" -gt 1 ]; then
    LORA_LAUNCH="torchrun --nproc_per_node=$NPROC -m"
else
    LORA_LAUNCH="python -m"
fi
if [ "$NPROC" -gt 1 ] && [ "$PRINT_ONLY" = "1" ]; then
    log "  ⚠ NPROC>1 + PRINT_ONLY=1: run_sharded 는 GPU 1장 핀 → python -m 으로 출력(torchrun 무시)"
fi

# 입력 파형 조합. prepare_data.sh 의 SIGNALS 와 **같은 순서**여야 파일명 prefix 가 맞는다.
# canonical = ecg_ppg_co2_awp (심폐 4-modality). ablation 은 SIGNALS_OVERRIDE=ecg_ppg.
SIGNAL_COMBOS=(${SIGNALS_OVERRIDE:-ecg_ppg_co2_awp})
WINDOWS=(${WINDOWS_OVERRIDE:-300})
HORIZONS=(${HORIZONS_OVERRIDE:-5 10 15})
MODES=(${MODES_OVERRIDE:-linear_probe lora})
N_FOLDS=${N_FOLDS:-5}
FORCE=${FORCE:-0}

# prepare_data.py 는 파일명을 h{int(horizon_min)}min 으로 쓴다. 소수(0.5)나 "5.0" 을
# 넘기면 prefix 가 어긋나 조용히 SKIP 된다 → 정수만 허용.
for H in "${HORIZONS[@]}"; do
    case "$H" in
        ''|*[!0-9]*) echo "ERROR: HORIZONS 는 정수만 (got '$H')" >&2; exit 1 ;;
    esac
done

TOTAL=$(( ${#SIGNAL_COMBOS[@]} * ${#WINDOWS[@]} * ${#HORIZONS[@]} * ${#MODES[@]} * N_FOLDS ))
COUNT=0

log "============================================================"
log "  Intraoperative Hypoxemia — frozen/LoRA sweep"
log "  Checkpoint: $CHECKPOINT"
log "  ModelVer:   $MODEL_VERSION"
log "  Data:       $DATA_DIR"
log "  Output:     $OUT_DIR"
log "  Signals:    ${SIGNAL_COMBOS[*]}"
log "  Windows:    ${WINDOWS[*]}s | Horizons: ${HORIZONS[*]} min"
log "  Modes:      ${MODES[*]}"
log "  Total:      $TOTAL experiments"
log "  LoRA batch: $LORA_BATCH  (NPROC=$NPROC → eff $((LORA_BATCH * NPROC)))"
log "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    for W in "${WINDOWS[@]}"; do
        for H in "${HORIZONS[@]}"; do
            # prepare_data.py 저장 prefix = hypoxemia_{signals}_w{W}s_h{H}min
            PREFIX="${DATA_DIR}/hypoxemia_${SIGNALS}_w${W}s_h${H}min"

            if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1; then
                log "  SKIP: ${PREFIX}_fold0_*.pt not found (prepare_data 필요)"
                continue
            fi

            INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')

            for MODE in "${MODES[@]}"; do
                if [ "$MODE" = "linear_probe" ]; then
                    EPOCHS=$LP_EPOCHS; LR=$LP_LR; BATCH=$LP_BATCH
                    EXTRA_ARGS=""; PAT=0
                else
                    EPOCHS=$LORA_EPOCHS; LR=$LORA_LR; BATCH=$LORA_BATCH
                    EXTRA_ARGS="--lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA"
                    PAT=$LORA_PATIENCE
                fi

                EXP_DIR="${OUT_DIR}/${MODE}/${SIGNALS}_w${W}s_h${H}min"
                if [ "$PRINT_ONLY" != "1" ]; then
                    mkdir -p "$EXP_DIR"
                fi

                if [ "$PRINT_ONLY" = "1" ]; then
                    LAUNCH="python -m"
                else
                    LAUNCH="$LORA_LAUNCH"
                fi

                for f in $(seq 0 $((N_FOLDS - 1))); do
                    COUNT=$((COUNT + 1))
                    if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
                        log "  [skip] done: ${EXP_DIR} (fold $f)"
                        continue
                    fi

                    loge "\n[${COUNT}/${TOTAL}] ${MODE} | ${SIGNALS} | w=${W}s h=${H}min | fold ${f}"

                    CMD="EARLY_STOP_PATIENCE=$PAT $LAUNCH downstream.acute_event.hypoxemia.run \
                        --checkpoint $CHECKPOINT \
                        --model-version $MODEL_VERSION \
                        --mode $MODE \
                        --data-path $PREFIX \
                        --n-folds $N_FOLDS \
                        --fold $f \
                        --input-signals $INPUT_SIGNALS \
                        --window-sec $W \
                        --epochs $EPOCHS \
                        --lr $LR \
                        --batch-size $BATCH \
                        --device $DEVICE \
                        --out-dir $EXP_DIR \
                        $EXTRA_ARGS"

                    if [ "$PRINT_ONLY" = "1" ]; then
                        echo "$CMD" | tr -s ' \t\n' ' '
                        echo
                    else
                        eval "$CMD"
                    fi
                done
            done
        done
    done
done

loge "\n============================================================"
log "  Done! ${COUNT}/${TOTAL} experiments completed"
log "  Results: $OUT_DIR"
log "============================================================"
