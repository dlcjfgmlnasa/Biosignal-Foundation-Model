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
# 환자-수준 집약: transformer(기본, 학습 aggregator) | mean(파라미터 없는 masked mean,
# 소규모 코호트 과적합 방지). run.py --aggregator 로 전달.
AGG=${AGG:-transformer}

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
        --aggregator "$AGG" \
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
        --aggregator "$AGG" \
        --max-windows "$MAX_WINDOWS" --device "$DEVICE" --out-dir "$LORA_OUT" $DRY_FLAG
  fi
done

echo "============================================================"
echo "  Done! Results under $OUT_DIR/{linear_probe,lora}/"
echo "============================================================"
