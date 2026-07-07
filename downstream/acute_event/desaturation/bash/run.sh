#!/bin/bash
# Desaturation Prediction (SNUH) — 실행 스크립트
# 입력 CO2+AWP+RESP_Flow+PPG(ECG 제외) 고정. horizon 1/3/5min. 15min stress-test 는 HORIZONS override.
# 기본: 1 combo × window 1개 × horizon 1개 × modes(linear_probe,lora) × 5 fold.
# env override: SIGNALS_OVERRIDE / WINDOWS_OVERRIDE / HORIZONS_OVERRIDE / MODES_OVERRIDE
#
# 사전 조건: prepare_data.sh 로 .pt 데이터셋 생성돼 있어야 함
#   (파일명: desaturation_co2_awp_resp_flow_ppg_w300s_h{1,3,5}min_fold{f}_{train,val,test}.pt)
#
# 사용법:
#   # (1) 직렬 실행 (단일 GPU)
#   bash downstream/acute_event/desaturation/bash/run.sh
#
#   # (2) A안 — GPU job-sharding (4 GPU 풀 분산, ~4× throughput)
#   PRINT_ONLY=1 bash downstream/acute_event/desaturation/bash/run.sh > jobs.txt
#   python -m downstream.run_sharded --gpus 0,1,2,3 --jobs-file jobs.txt
#
#   # (3) B안 — torchrun+DDP (단일 fold 를 4 GPU 데이터 병렬). NPROC=4 기본.

set -e

# ── 설정 (canonical 경로 = k-mimic-/bio_fm, memory project_kmimic_bio_fm_paths) ──
# 모두 ${VAR:-default} → 환경변수 override 가능. 서버 실제 경로 확인 후 맞추세요.
CHECKPOINT=${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}
DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/desaturation}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/desaturation}
DEVICE=${DEVICE:-cuda}
MODEL_VERSION=${MODEL_VERSION:-v2}   # v2 필수 (9 modality 단일 embedding). v1 금지.

PRINT_ONLY=${PRINT_ONLY:-0}
log()  { if [ "$PRINT_ONLY" = "1" ]; then echo    "$@" >&2; else echo    "$@"; fi; }
loge() { if [ "$PRINT_ONLY" = "1" ]; then echo -e "$@" >&2; else echo -e "$@"; fi; }

# Linear Probe (frozen feature 캐싱)
LP_EPOCHS=${LP_EPOCHS:-2000}
LP_LR=${LP_LR:-1e-3}
LP_BATCH=${LP_BATCH:-512}

# LoRA
LORA_EPOCHS=${LORA_EPOCHS:-200}
LORA_PATIENCE=${LORA_PATIENCE:-5}
LORA_LR=${LORA_LR:-1e-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_BATCH=${LORA_BATCH:-128}   # ⚠ batch≠32 는 비교성 주의(LR 재튜닝). 보수적이면 32.

# 한 fold 를 여러 GPU DDP (fold 는 순차). NPROC=1 이면 단일 GPU.
NPROC=${NPROC:-4}
if [ "$NPROC" -gt 1 ]; then LORA_LAUNCH="torchrun --nproc_per_node=$NPROC -m"; else LORA_LAUNCH="python -m"; fi

# ── 실험 조합 (canonical: 입력·window·horizon 고정) ──
SIGNAL_COMBOS=(${SIGNALS_OVERRIDE:-co2_awp_resp_flow_ppg})   # CO2·AWP·RESP_Flow·PPG
WINDOWS=(${WINDOWS_OVERRIDE:-300})
HORIZONS=(${HORIZONS_OVERRIDE:-1 3 5})
MODES=(${MODES_OVERRIDE:-linear_probe lora})
N_FOLDS=${N_FOLDS:-5}
FORCE=${FORCE:-0}

TOTAL=$(( ${#SIGNAL_COMBOS[@]} * ${#WINDOWS[@]} * ${#HORIZONS[@]} * ${#MODES[@]} * N_FOLDS ))
COUNT=0

log "============================================================"
log "  Desaturation Prediction (SNUH) — CV Experiment"
log "  Checkpoint: $CHECKPOINT"
log "  ModelVer:   $MODEL_VERSION"
log "  Data:       $DATA_DIR"
log "  Output:     $OUT_DIR"
log "  Signals:    ${SIGNAL_COMBOS[*]}"
log "  Windows:    ${WINDOWS[*]}   Horizons: ${HORIZONS[*]}"
log "  Modes:      ${MODES[*]}     Folds: $N_FOLDS"
log "  Total:      $TOTAL experiments   (NPROC=$NPROC, LoRA batch=$LORA_BATCH)"
log "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    for W in "${WINDOWS[@]}"; do
        for H in "${HORIZONS[@]}"; do
            PREFIX="${DATA_DIR}/desaturation_${SIGNALS}_w${W}s_h${H}min"
            if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1; then
                log "  SKIP: ${PREFIX}_fold0_*.pt not found (prepare_data 필요)"
                continue
            fi

            # ⚠ resp_flow 는 '_' 포함 → tr '_' ' ' 로 쪼개면 안 됨. sed 로 보존 후 분리.
            INPUT_SIGNALS=$(echo "$SIGNALS" | sed 's/resp_flow/respXXflow/g; s/_/ /g; s/respXXflow/resp_flow/g')

            for MODE in "${MODES[@]}"; do
                if [ "$MODE" = "linear_probe" ]; then
                    EPOCHS=$LP_EPOCHS; LR=$LP_LR; BATCH=$LP_BATCH; EXTRA_ARGS=""; PAT=0
                else
                    EPOCHS=$LORA_EPOCHS; LR=$LORA_LR; BATCH=$LORA_BATCH
                    EXTRA_ARGS="--lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA"; PAT=$LORA_PATIENCE
                fi

                EXP_DIR="${OUT_DIR}/${MODE}/${SIGNALS}_w${W}s_h${H}min"
                [ "$PRINT_ONLY" != "1" ] && mkdir -p "$EXP_DIR"

                if [ "$PRINT_ONLY" = "1" ]; then LAUNCH="python -m"; else LAUNCH="$LORA_LAUNCH"; fi

                for f in $(seq 0 $((N_FOLDS - 1))); do
                    COUNT=$((COUNT + 1))
                    if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
                        log "  [skip] done: ${EXP_DIR} (fold $f)"; continue
                    fi
                    loge "\n[${COUNT}/${TOTAL}] ${MODE} | ${SIGNALS} | w=${W}s h=${H}min | fold ${f}"

                    # ⚠ env-var 접두어는 `env` 명령으로 감싼다: 직접 eval(셸)뿐 아니라
                    #   run_sharded(shlex.split+Popen, 셸 미경유)에서도 실행되게 하기 위함.
                    #   (그냥 'VAR=0 python' 이면 run_sharded 가 'VAR=0' 을 실행파일로 오인해 실패)
                    CMD="env EARLY_STOP_PATIENCE=$PAT $LAUNCH downstream.acute_event.desaturation.run \
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
                        echo "$CMD" | tr -s ' \t\n' ' '; echo
                    else
                        eval "$CMD"
                    fi
                done
            done
        done
    done
done

loge "\n============================================================"
log "  Done! ${COUNT}/${TOTAL} experiments"
log "  Results: $OUT_DIR"
log "============================================================"
