#!/bin/bash
# Outcome Prediction — Cardiac Arrest (patient-level, SCOPE) — frozen/LoRA, 5-fold.
#
#   - 환자 단위 Transformer Aggregator (mortality outcome 과 동일 구조/포맷).
#   - Linear Probe + LoRA 두 모드 × 5-fold. config·fold 직렬.
#   - Resume: 완료 셀(out-dir/preds_fold{f}.npz)은 건너뛴다. FORCE=1 이면 전부 재실행.
#   - NPROC>1 이면 torchrun DDP (linear_probe=sharded 추출, lora=aggregator DDP).
#     NPROC=1 이면 단일 GPU (python) — 결과 불변.
#
# 사전: bash/prepare_data.sh 로 cardiac_arrest_w300s_fold*_{split}_chunk*.pt 생성.
#
# 사용법:
#   nohup bash downstream/outcome/cardiac_arrest/bash/run.sh > ca_outcome.log 2>&1 &
#   tail -f ca_outcome.log

# ── canonical 경로 (k-mimic-/bio_fm — memory project_kmimic_bio_fm_paths). ──
CHECKPOINT=${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}
# 실제 준비 데이터는 task 별로 분리 디렉토리: scope_outcome_{arrest,death}
#   (prepare_data_scope 저장 위치). TASK 에 따라 자동 선택.
DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/scope_outcome_${TASK:-arrest}}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/scope_outcome_${TASK:-arrest}}
# v2 필수 (9 modality 단일 embedding). v1 로드 금지.
MODEL_VERSION=${MODEL_VERSION:-v2}
DEVICE=${DEVICE:-cuda}
NPROC=${NPROC:-4}

# task = arrest(default) | death(=mortality). prepare 저장 prefix scope_{TASK}_w{win}s 와 일치.
TASK=${TASK:-arrest}
WINDOW_SEC=${WINDOW_SEC:-300}
MAX_WINDOWS=${MAX_WINDOWS:-144}
N_FOLDS=${N_FOLDS:-5}
# 환자-수준 집약(AGG): 양성 극소수(예: 74 arrest)라 파라미터 少/0 을 기본으로.
#   mean(기본): 0-param masked mean. 순수 frozen linear-probe. 급성 신호는 희석 가능.
#   lastk:  event 근접 마지막 LAST_K window 평균 (0-param, 후반 악화 집중).
#   max:    peak-risk window (0-param). 급성 이벤트에 자연스러움 (mean 희석 회피).
#   attention: gated-attention MIL (Ilse 2018, few-param, ATTN_DIM 조절). 학습형 절충.
#   transformer: 학습형(param 多) — 양성 수백+ 코호트에서만.
#   ⇒ aggregator ablation: for AGG in mean lastk max attention; do AGG=$AGG bash run.sh; done
#      (frozen encoder 라 집약기만 바뀌어 저렴. OUT_DIR 에 _${AGG} 접미사로 분리 권장.)
AGG=${AGG:-mean}
LAST_K=${LAST_K:-36}    # [AGG=lastk] 5min window 기준 36=마지막 3h
ATTN_DIM=${ATTN_DIM:-32}  # [AGG=attention] gated-MIL hidden dim. 74 양성엔 16~32 (64=과적합)
# ⚠ SELECT_ON_TEST=1: test 를 val 로 써서 best-epoch 선택(early-stopping). test 로 model 을
#   고르므로 낙관 편향(circular) — 보고 시 명시 필요. 별도 val split 부재로 임시 사용.
#   attention 처럼 학습형 aggregator 가 LP_EPOCHS 를 다 돌면 과적합할 때 유용.
SELECT_ON_TEST=${SELECT_ON_TEST:-0}
SEL_EVAL_EVERY=${SEL_EVAL_EVERY:-5}     # val(test) 평가 주기(epoch)
SEL_PATIENCE=${SEL_PATIENCE:-20}        # early stop: 개선없는 평가 N회 연속 → 중단(0=끔). 실제≈N×eval_every epoch
SEL_MIN_DELTA=${SEL_MIN_DELTA:-0}       # 개선 최소 증가폭
SEL_ARGS=""
[ "$SELECT_ON_TEST" = "1" ] && SEL_ARGS="--select-on-test --select-eval-every $SEL_EVAL_EVERY --select-patience $SEL_PATIENCE --select-min-delta $SEL_MIN_DELTA"

# Linear Probe (frozen feature 캐싱 — batch 는 probe SGD 미니배치에만 영향).
LP_BATCH=${LP_BATCH:-512};    LP_LR=${LP_LR:-1e-3};    LP_EPOCHS=${LP_EPOCHS:-1000}
# LoRA. ⚠ max_windows=144 라 환자당 인코더 forward 가 많다 → batch 보수적으로.
LORA_BATCH=${LORA_BATCH:-8};  LORA_LR=${LORA_LR:-2e-4}; LORA_EPOCHS=${LORA_EPOCHS:-30}
LORA_RANK=${LORA_RANK:-8};    LORA_ALPHA=${LORA_ALPHA:-16}
FORCE=${FORCE:-0}

# DRY_RUN=1: 파이프라인 스모크 테스트. fold 0 만, 소수 환자·소수 윈도우·적은 epoch 로
#   추출→학습→평가→저장을 빠르게 1회 돌려 크래시 여부만 확인한다(산출물 무의미).
#   --n-folds 는 5 를 유지해야 per-fold chunk 가 올바로 로드된다(루프만 fold 0 로 제한).
#   세부값은 env 로 조절: DRY_N(클래스당 환자) DRY_WINDOWS(환자당 윈도우)
#   DRY_EPOCHS(epoch 상한) DRY_CHUNKS(split 당 chunk). 미지정 시 run.py 기본값.
DRY_RUN=${DRY_RUN:-0}
DRY_FLAG=""
FOLD_LIST=$(seq 0 $((N_FOLDS-1)))
if [ "$DRY_RUN" = "1" ]; then
    DRY_FLAG="--dry-run"
    [ -n "$DRY_N" ]       && DRY_FLAG="$DRY_FLAG --dry-run-n $DRY_N"
    [ -n "$DRY_WINDOWS" ] && DRY_FLAG="$DRY_FLAG --dry-run-windows $DRY_WINDOWS"
    [ -n "$DRY_EPOCHS" ]  && DRY_FLAG="$DRY_FLAG --dry-run-epochs $DRY_EPOCHS"
    [ -n "$DRY_CHUNKS" ]  && DRY_FLAG="$DRY_FLAG --dry-run-chunks $DRY_CHUNKS"
    FOLD_LIST=0
    FORCE=1                    # dryrun 산출물은 매번 새로 쓴다
    OUT_DIR="$OUT_DIR/_dryrun" # 실 결과 디렉토리를 덮어쓰지 않도록 격리
fi

if [ "$NPROC" -gt 1 ]; then
    LAUNCH="torchrun --nproc_per_node=$NPROC -m"
else
    LAUNCH="python -m"
fi

# prepare_data_scope.py 저장 prefix = scope_{task}_w{win}s
PREFIX=$DATA_DIR/scope_${TASK}_w${WINDOW_SEC}s

# maybe_run <out-dir> <fold> <command...>
maybe_run() {
  local outdir="$1"; local fold="$2"; shift 2
  if [ "$FORCE" != "1" ] && [ -f "$outdir/preds_fold${fold}.npz" ]; then
    echo "  [skip] done: $outdir (fold $fold)"
    return 0
  fi
  echo "  [run]  $outdir (fold $fold)"
  "$@"
}

echo "============================================================"
[ "$DRY_RUN" = "1" ] && echo "  *** DRY-RUN (fold 0 only, 소수 환자, 산출물→$OUT_DIR) ***"
echo "  Cardiac Arrest Outcome (patient-level) | NPROC=$NPROC | FORCE=$FORCE"
echo "  Checkpoint: $CHECKPOINT | ModelVer: $MODEL_VERSION"
echo "  Data:       $PREFIX (${N_FOLDS}-fold)"
echo "  Window=${WINDOW_SEC}s max_windows=$MAX_WINDOWS aggregator=$AGG"
echo "  LP: batch=$LP_BATCH lr=$LP_LR epochs=$LP_EPOCHS"
echo "  LoRA: batch=$LORA_BATCH (eff $((LORA_BATCH*NPROC))) lr=$LORA_LR epochs=$LORA_EPOCHS rank=$LORA_RANK"
echo "  OUT_DIR: $OUT_DIR"
echo "============================================================"

# 실행할 모드 선택 (기본 둘 다). LP 만: MODES=linear_probe / LoRA 만: MODES=lora
MODES="${MODES:-linear_probe lora}"
has_mode() { case " $MODES " in *" $1 "*) return 0;; *) return 1;; esac; }

for f in $FOLD_LIST; do
  if has_mode linear_probe; then
    LP_OUT=$OUT_DIR/linear_probe
    maybe_run "$LP_OUT" "$f" \
      $LAUNCH downstream.outcome.cardiac_arrest.run \
        --checkpoint "$CHECKPOINT" --model-version "$MODEL_VERSION" \
        --data-path "$PREFIX" --mode linear_probe --n-folds "$N_FOLDS" --fold "$f" \
        --epochs "$LP_EPOCHS" --lr "$LP_LR" --batch-size "$LP_BATCH" \
        --aggregator "$AGG" --last-k "$LAST_K" --attn-dim "$ATTN_DIM" $SEL_ARGS \
        --max-windows "$MAX_WINDOWS" --device "$DEVICE" --out-dir "$LP_OUT" $DRY_FLAG
  fi

  if has_mode lora; then
    LORA_OUT=$OUT_DIR/lora
    maybe_run "$LORA_OUT" "$f" \
      $LAUNCH downstream.outcome.cardiac_arrest.run \
        --checkpoint "$CHECKPOINT" --model-version "$MODEL_VERSION" \
        --data-path "$PREFIX" --mode lora --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
        --n-folds "$N_FOLDS" --fold "$f" \
        --epochs "$LORA_EPOCHS" --lr "$LORA_LR" --batch-size "$LORA_BATCH" \
        --aggregator "$AGG" --last-k "$LAST_K" --attn-dim "$ATTN_DIM" $SEL_ARGS \
        --max-windows "$MAX_WINDOWS" --device "$DEVICE" --out-dir "$LORA_OUT" $DRY_FLAG
  fi
done

echo "============================================================"
echo "  Done! Results under $OUT_DIR/{linear_probe,lora}/"
echo "============================================================"
