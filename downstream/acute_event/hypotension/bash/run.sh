#!/bin/bash
# Hypotension Prediction — 실행 스크립트
# 입력 ECG+PPG+ABP 고정, window 고정, 예측 horizon 만 변경 (canonical 설계).
# 기본: ecg_ppg_abp × window 1개 × horizons × modes(linear_probe,lora).
# env override: SIGNALS_OVERRIDE / WINDOWS_OVERRIDE / HORIZONS_OVERRIDE / MODES_OVERRIDE
#
# 사전 조건: prepare_data.sh로 .pt 데이터셋이 생성되어 있어야 함
#
# 사용법:
#   # (1) 기존 직렬 실행 (단일 GPU)
#   bash downstream/acute_event/hypotension/bash/run.sh
#
#   # (2) A안 — GPU job-sharding: 명령만 내보내 4 GPU 풀에 분산 (~4× throughput)
#   PRINT_ONLY=1 bash downstream/acute_event/hypotension/bash/run.sh > jobs.txt
#   python -m downstream.run_sharded --gpus 0,1,2,3 --jobs-file jobs.txt
#
#   # (3) B안 — torchrun+DDP: 단일 LoRA run 을 4 GPU 로 데이터 병렬 (큰 fold 가속)
#   #     run.sh 가 만드는 단일 명령의 'python -m ...' 를
#   #     'torchrun --nproc_per_node=4 -m ...' 로 바꿔 실행하면 된다.
#   #     A안(job-sharding)과 B안은 보통 A안만으로 충분(fold·config 가 4개 이상이면).

set -e

# ── 설정 ──
# canonical 경로는 k-mimic-/bio_fm 아래 (updown 기본은 stale — memory
# project_kmimic_bio_fm_paths). 모두 ${VAR:-default} 라 환경변수로 override 가능:
#   CHECKPOINT=... DATA_DIR=... OUT_DIR=... bash run.sh
# 서버 실제 파일명·경로는 반드시 확인 후 맞추세요(아래 default 는 예시).
CHECKPOINT=${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}
DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/hypotension}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/hypotension}
DEVICE=${DEVICE:-cuda}
# v2 필수 (9 modality 단일 embedding — memory project_data_spec_v2). v1 로드 금지.
MODEL_VERSION=${MODEL_VERSION:-v2}

# PRINT_ONLY=1 이면 python 명령을 실행하지 않고 stdout 으로 출력만 한다
# (run_sharded 로 파이프하기 위함). 기본 0(직렬 실행).
PRINT_ONLY=${PRINT_ONLY:-0}

# 정보성 출력 헬퍼: PRINT_ONLY 면 stderr 로(stdout 엔 job 명령만 남도록), 아니면 stdout.
log()  { if [ "$PRINT_ONLY" = "1" ]; then echo    "$@" >&2; else echo    "$@"; fi; }
loge() { if [ "$PRINT_ONLY" = "1" ]; then echo -e "$@" >&2; else echo -e "$@"; fi; }

# Linear Probe 설정 (frozen feature 캐싱 — batch 는 probe SGD 미니배치에만 영향)
LP_EPOCHS=${LP_EPOCHS:-2000}
LP_LR=${LP_LR:-1e-3}
LP_BATCH=${LP_BATCH:-512}

# LoRA 설정 (모두 env override 가능)
# epoch 은 상한(ceiling). early-stop(LORA_PATIENCE)이 val 수렴 시 자동 중단하므로
# 넉넉히 두면 "수렴 보장 + 시간 절약". LP 는 캐시 위 학습이라 early-stop 없이 1000 고정.
LORA_EPOCHS=${LORA_EPOCHS:-200}
LORA_PATIENCE=${LORA_PATIENCE:-5}   # val best 5 epoch 미개선 시 중단(run.py EARLY_STOP_PATIENCE). IOH LoRA 는 val 이 ep~11 조기 peak 후 단조 하락 관측 → 5 로 충분·시간 절약
LORA_LR=${LORA_LR:-1e-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
# ── C안: batch-size 상향 (가속) — ⚠ 결과가 바뀌므로 비교성 주의 ──
# bf16+frozen encoder 라 VRAM 여유가 커 batch 를 키우면 step 수가 줄어 빨라진다.
# 단, batch 를 바꾸면 LoRA 최적화 궤적이 달라져 **batch=32 시절 결과와 직접 비교
# 불가**(재실행 필요). LR 은 자동 스케일하지 않으니 새 batch 에서 짧은 LR sweep
# 으로 재튜닝 권장. (AdamW 는 SGD 의 linear-scaling 휴리스틱이 잘 안 맞아 √-scale
# 또는 명시 재튜닝이 적절 — estimator 검증.) 보수적으로 가려면 LORA_BATCH=32 유지.
# torchrun(B안) 에선 effective batch = LORA_BATCH × nproc 이라 더 커짐 — 한 task 의
# 전 fold/config 는 동일 (실행모드 × effective batch × LR) 로 고정해야 표 비교 성립.
LORA_BATCH=${LORA_BATCH:-128}

# ── Scratch 설정 (random init + 전체 파라미터 end-to-end 학습) ──
# 사전학습 대조군(CARMEN-scratch, Table 3). LoRA 는 adapter 만 학습하지만 scratch 는
# encoder 전체를 학습하므로 epoch 상한을 크게 낮춘다(LORA 200 → 30). early-stop
# (SCRATCH_PATIENCE)이 val AUROC 수렴 시 자동 중단한다.
# ⚠ 결과 json 의 best_epoch 이 상한(SCRATCH_EPOCHS)에 붙어 있으면 아직 개선 중이라
#   잘린 것이다 — 그 경우 SCRATCH_EPOCHS 를 올려 재실행해야 대조군이 공정하다.
# LR 은 LoRA(1e-4) 보다 높게 잡는다: 사전학습 가중치에서 출발하는 미세조정이 아니라
# random init 에서 처음 학습하는 것이라 더 큰 step 이 필요하다.
#
# ⚠ batch 는 LoRA(128) 보다 작아야 한다. LoRA 는 base weight 가 frozen 이라 autograd
#   가 그 Linear 들의 입력 activation 을 저장하지 않지만(weight grad 불필요), scratch
#   는 q/k/v/out/fc1/fc2/gate 전부가 학습 대상이라 입력을 backward 까지 들고 있어야
#   한다. 실측: L40S(44 GB)에서 SCRATCH_BATCH=128 은 OOM, 32 로 낮춰야 한다.
#   여전히 OOM 이면 16 으로 내리고 PYTORCH_ALLOC_CONF=expandable_segments:True 를
#   함께 준다 (memory project_expandable_segments_oom).
#   ⚠ 한 task 의 전 fold·horizon 은 같은 batch 로 돌아야 표 비교가 성립한다.
SCRATCH_EPOCHS=${SCRATCH_EPOCHS:-30}
SCRATCH_PATIENCE=${SCRATCH_PATIENCE:-5}
SCRATCH_LR=${SCRATCH_LR:-3e-4}
SCRATCH_BATCH=${SCRATCH_BATCH:-32}

# ── 한 fold 를 여러 GPU 로 DDP 실행 (단일 fold 데이터 병렬, fold 는 순차) ──
# 기본 NPROC=4: 각 fold 를 torchrun --nproc_per_node=4 로 4-GPU DDP 실행한다
# (linear_probe=feature 추출 shard→gather, lora=grad all-reduce; 저장은 rank0 전담).
# 단일 GPU 로 돌리려면 NPROC=1. PRINT_ONLY(A안 run_sharded=GPU 1장 핀)면 torchrun
# 금지 → 강제로 python -m 출력(아래). 두 가속(job-sharding × torchrun) 동시 사용 금지.
NPROC=${NPROC:-4}
if [ "$NPROC" -gt 1 ]; then
    LORA_LAUNCH="torchrun --nproc_per_node=$NPROC -m"
else
    LORA_LAUNCH="python -m"
fi
if [ "$NPROC" -gt 1 ] && [ "$PRINT_ONLY" = "1" ]; then
    log "  ⚠ NPROC>1 + PRINT_ONLY=1: run_sharded 는 GPU 1장 핀 → lora 도 python -m 으로 출력(torchrun 무시)"
fi

# ── 실험 조합 ──
# 입력은 ECG+PPG+ABP 고정, window 도 고정(canonical). 예측 horizon 만 변경한다.
# 모두 env override 가능(공백 구분): 예) WINDOWS_OVERRIDE=900 HORIZONS_OVERRIDE="5 10 15" bash run.sh
SIGNAL_COMBOS=(${SIGNALS_OVERRIDE:-ecg_ppg_abp})
WINDOWS=(${WINDOWS_OVERRIDE:-300})
HORIZONS=(${HORIZONS_OVERRIDE:-5 10 15})
MODES=(${MODES_OVERRIDE:-linear_probe lora})
N_FOLDS=${N_FOLDS:-5}   # stratified k-fold CV — fold 별로 실행(--n-folds/--fold)
FORCE=${FORCE:-0}       # 1 이면 완료된 fold(preds_fold{f}.npz)도 재실행

TOTAL=$(( ${#SIGNAL_COMBOS[@]} * ${#WINDOWS[@]} * ${#HORIZONS[@]} * ${#MODES[@]} * N_FOLDS ))
COUNT=0

log "============================================================"
log "  Hypotension Prediction — Paired Comparison Experiment Sweep"
log "  Checkpoint: $CHECKPOINT"
log "  ModelVer:   $MODEL_VERSION"
log "  Data:       $DATA_DIR"
log "  Output:     $OUT_DIR"
log "  Device:     $DEVICE"
log "  Signals:    ${SIGNAL_COMBOS[*]}"
log "  Windows:    ${WINDOWS[*]}"
log "  Horizons:   ${HORIZONS[*]}"
log "  Modes:      ${MODES[*]}"
log "  Total:      $TOTAL experiments"
log "  LoRA batch: $LORA_BATCH  (NPROC=$NPROC → eff $((LORA_BATCH * NPROC)))"
case " ${MODES[*]} " in
    *" scratch "*)
        log "  Scratch:    epochs≤$SCRATCH_EPOCHS  lr=$SCRATCH_LR  batch=$SCRATCH_BATCH  patience=$SCRATCH_PATIENCE"
        log "              (checkpoint 은 ModelConfig 용으로만 읽고 가중치는 버림)"
        ;;
esac
log "  PRINT_ONLY: $PRINT_ONLY"
if [ "$LORA_BATCH" != "32" ]; then
    log "  ⚠ LoRA batch≠32: 결과가 batch=32 기준선과 비교 불가 — LR 재튜닝 권장"
fi
log "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    for W in "${WINDOWS[@]}"; do
        for H in "${HORIZONS[@]}"; do
            # ⚠ 데이터는 단일 .pt 가 아니라 per-(fold,split)[_chunk] prefix 묶음이다.
            #   예: hypotension_ecg_ppg_abp_w300s_h5min_fold0_train_chunk0.pt
            #   run.py 는 --data-path PREFIX(.pt 없이) + --n-folds/--fold 로 해당 fold
            #   chunk 들을 로드한다(load_prepared_split_chunked).
            PREFIX="${DATA_DIR}/hypotension_${SIGNALS}_w${W}s_h${H}min"

            if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1; then
                log "  SKIP: ${PREFIX}_fold0_*.pt not found (prepare_data 필요)"
                continue
            fi

            # --input-signals: underscore → space 변환
            INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')

            for MODE in "${MODES[@]}"; do
                # 모드별 인자 설정
                if [ "$MODE" = "linear_probe" ]; then
                    EPOCHS=$LP_EPOCHS
                    LR=$LP_LR
                    BATCH=$LP_BATCH
                    EXTRA_ARGS=""
                    PAT=0                       # LP 는 early-stop 없이 1000 고정
                elif [ "$MODE" = "scratch" ]; then
                    EPOCHS=$SCRATCH_EPOCHS
                    LR=$SCRATCH_LR
                    BATCH=$SCRATCH_BATCH
                    EXTRA_ARGS=""               # LoRA 미주입 (전체 파라미터 학습)
                    PAT=$SCRATCH_PATIENCE
                else
                    EPOCHS=$LORA_EPOCHS
                    LR=$LORA_LR
                    BATCH=$LORA_BATCH
                    EXTRA_ARGS="--lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA"
                    PAT=$LORA_PATIENCE          # LoRA 는 early-stop(상한 LORA_EPOCHS)
                fi

                EXP_DIR="${OUT_DIR}/${MODE}/${SIGNALS}_w${W}s_h${H}min"
                if [ "$PRINT_ONLY" != "1" ]; then
                    mkdir -p "$EXP_DIR"
                fi

                # launcher 선택: PRINT_ONLY(run_sharded = GPU 1장 핀)면 python -m,
                # 아니면 $LORA_LAUNCH(NPROC>1 시 torchrun DDP).
                if [ "$PRINT_ONLY" = "1" ]; then
                    LAUNCH="python -m"
                else
                    LAUNCH="$LORA_LAUNCH"
                fi

                for f in $(seq 0 $((N_FOLDS - 1))); do
                    COUNT=$((COUNT + 1))
                    # resume: 완료 fold(preds_fold{f}.npz)는 건너뜀. FORCE=1 이면 재실행.
                    if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
                        log "  [skip] done: ${EXP_DIR} (fold $f)"
                        continue
                    fi

                    loge "\n[${COUNT}/${TOTAL}] ${MODE} | ${SIGNALS} | w=${W}s h=${H}min | fold ${f}"

                    # ⚠ env-var 접두어는 `env` 명령으로 감싼다: 직접 eval(셸)뿐 아니라
                    #   run_sharded(shlex.split+Popen, 셸 미경유)에서도 실행되게 하기 위함.
                    #   (그냥 'VAR=0 python' 이면 run_sharded 가 'VAR=0' 을 실행파일로 오인해 실패)
                    CMD="env EARLY_STOP_PATIENCE=$PAT $LAUNCH downstream.acute_event.hypotension.run \
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
                        # 한 줄 명령으로 정규화해 출력 (run_sharded 가 한 줄=한 작업).
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
