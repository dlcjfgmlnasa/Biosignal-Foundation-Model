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
LP_EPOCHS=${LP_EPOCHS:-20}
LP_LR=${LP_LR:-1e-3}
LP_BATCH=${LP_BATCH:-32}

# LoRA 설정 (모두 env override 가능)
LORA_EPOCHS=${LORA_EPOCHS:-30}
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

# ── B안: NPROC>1 이면 lora 를 torchrun 데이터 병렬(단일 fold DDP)로 실행 ──
# linear_probe 는 frozen feature 캐싱이라 DDP 이득 없음 → 항상 python -m.
# PRINT_ONLY(A안 run_sharded 용 단일 GPU job 출력)와는 배타적: PRINT_ONLY=1 이면
# lora 도 반드시 python -m 으로 출력한다(run_sharded 가 GPU 1장에 핀하므로 torchrun
# 금지). 두 가속(A안 job-sharding × B안 torchrun)을 동시에 쓰면 GPU 이중 배정.
NPROC=${NPROC:-1}
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

TOTAL=$(( ${#SIGNAL_COMBOS[@]} * ${#WINDOWS[@]} * ${#HORIZONS[@]} * ${#MODES[@]} ))
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
log "  PRINT_ONLY: $PRINT_ONLY"
if [ "$LORA_BATCH" != "32" ]; then
    log "  ⚠ LoRA batch≠32: 결과가 batch=32 기준선과 비교 불가 — LR 재튜닝 권장"
fi
log "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    for W in "${WINDOWS[@]}"; do
        for H in "${HORIZONS[@]}"; do
            # .pt 파일 경로
            PT_FILE="${DATA_DIR}/hypotension_${SIGNALS}_w${W}s_h${H}min.pt"

            if [ ! -f "$PT_FILE" ]; then
                log "  SKIP: $PT_FILE not found"
                continue
            fi

            for MODE in "${MODES[@]}"; do
                COUNT=$((COUNT + 1))
                EXP_DIR="${OUT_DIR}/${MODE}/${SIGNALS}_w${W}s_h${H}min"
                # PRINT_ONLY 면 디렉토리 생성 보류(실제 실행 시점에 run.py 가 만든다).
                # stale OUT_DIR 에 디렉토리를 미리 파지 않도록.
                if [ "$PRINT_ONLY" != "1" ]; then
                    mkdir -p "$EXP_DIR"
                fi

                loge "\n[${COUNT}/${TOTAL}] ${MODE} | ${SIGNALS} | w=${W}s h=${H}min"

                # 모드별 인자 설정
                if [ "$MODE" = "linear_probe" ]; then
                    EPOCHS=$LP_EPOCHS
                    LR=$LP_LR
                    BATCH=$LP_BATCH
                    EXTRA_ARGS=""
                else
                    EPOCHS=$LORA_EPOCHS
                    LR=$LORA_LR
                    BATCH=$LORA_BATCH
                    EXTRA_ARGS="--lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA"
                fi
                # launcher 선택 (linear_probe·lora 공통): PRINT_ONLY(A안 run_sharded =
                # GPU 1장 핀)면 반드시 python -m, 아니면 $LORA_LAUNCH(NPROC>1 시 torchrun).
                # 두 모드 모두 torchrun 병렬 지원: lora=grad 데이터병렬,
                # linear_probe=feature 추출 shard→gather(저장은 rank0 전담).
                if [ "$PRINT_ONLY" = "1" ]; then
                    LAUNCH="python -m"
                else
                    LAUNCH="$LORA_LAUNCH"
                fi

                # --input-signals: underscore → space 변환
                INPUT_SIGNALS=$(echo "$SIGNALS" | tr '_' ' ')

                CMD="$LAUNCH downstream.acute_event.hypotension.run \
                    --checkpoint $CHECKPOINT \
                    --model-version $MODEL_VERSION \
                    --mode $MODE \
                    --data-path $PT_FILE \
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

loge "\n============================================================"
log "  Done! ${COUNT}/${TOTAL} experiments completed"
log "  Results: $OUT_DIR"
log "============================================================"
