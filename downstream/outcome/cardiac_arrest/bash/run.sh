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
DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/cardiac_arrest_outcome}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/cardiac_arrest_outcome}
# v2 필수 (9 modality 단일 embedding). v1 로드 금지.
MODEL_VERSION=${MODEL_VERSION:-v2}
DEVICE=${DEVICE:-cuda}
NPROC=${NPROC:-4}

WINDOW_SEC=${WINDOW_SEC:-300}
MAX_WINDOWS=${MAX_WINDOWS:-144}
N_FOLDS=${N_FOLDS:-5}

# Linear Probe (frozen feature 캐싱 — batch 는 probe SGD 미니배치에만 영향).
LP_BATCH=${LP_BATCH:-512};    LP_LR=${LP_LR:-1e-3};    LP_EPOCHS=${LP_EPOCHS:-1000}
# LoRA. ⚠ max_windows=144 라 환자당 인코더 forward 가 많다 → batch 보수적으로.
LORA_BATCH=${LORA_BATCH:-8};  LORA_LR=${LORA_LR:-2e-4}; LORA_EPOCHS=${LORA_EPOCHS:-30}
LORA_RANK=${LORA_RANK:-8};    LORA_ALPHA=${LORA_ALPHA:-16}
FORCE=${FORCE:-0}

if [ "$NPROC" -gt 1 ]; then
    LAUNCH="torchrun --nproc_per_node=$NPROC -m"
else
    LAUNCH="python -m"
fi

# prepare_data_scope.py 저장 prefix = cardiac_arrest_w{win}s
PREFIX=$DATA_DIR/cardiac_arrest_w${WINDOW_SEC}s

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
echo "  Cardiac Arrest Outcome (patient-level) | NPROC=$NPROC | FORCE=$FORCE"
echo "  Checkpoint: $CHECKPOINT | ModelVer: $MODEL_VERSION"
echo "  Data:       $PREFIX (${N_FOLDS}-fold)"
echo "  Window=${WINDOW_SEC}s max_windows=$MAX_WINDOWS"
echo "  LP: batch=$LP_BATCH lr=$LP_LR epochs=$LP_EPOCHS"
echo "  LoRA: batch=$LORA_BATCH (eff $((LORA_BATCH*NPROC))) lr=$LORA_LR epochs=$LORA_EPOCHS rank=$LORA_RANK"
echo "  OUT_DIR: $OUT_DIR"
echo "============================================================"

for f in $(seq 0 $((N_FOLDS-1))); do
  LP_OUT=$OUT_DIR/linear_probe
  maybe_run "$LP_OUT" "$f" \
    $LAUNCH downstream.outcome.cardiac_arrest.run \
      --checkpoint "$CHECKPOINT" --model-version "$MODEL_VERSION" \
      --data-path "$PREFIX" --mode linear_probe --n-folds "$N_FOLDS" --fold "$f" \
      --epochs "$LP_EPOCHS" --lr "$LP_LR" --batch-size "$LP_BATCH" \
      --max-windows "$MAX_WINDOWS" --device "$DEVICE" --out-dir "$LP_OUT"

  LORA_OUT=$OUT_DIR/lora
  maybe_run "$LORA_OUT" "$f" \
    $LAUNCH downstream.outcome.cardiac_arrest.run \
      --checkpoint "$CHECKPOINT" --model-version "$MODEL_VERSION" \
      --data-path "$PREFIX" --mode lora --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
      --n-folds "$N_FOLDS" --fold "$f" \
      --epochs "$LORA_EPOCHS" --lr "$LORA_LR" --batch-size "$LORA_BATCH" \
      --max-windows "$MAX_WINDOWS" --device "$DEVICE" --out-dir "$LORA_OUT"
done

echo "============================================================"
echo "  Done! Results under $OUT_DIR/{linear_probe,lora}/"
echo "============================================================"
