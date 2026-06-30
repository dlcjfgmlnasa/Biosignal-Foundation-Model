#!/bin/bash
# Imminent Cardiac Arrest — frozen/LoRA sweep (window-level, horizon × 5-fold).
#
#   - IOH/ICH 와 동일 protocol (val best-ckpt → test 1회). 옛 patient-aggregation 폐기.
#   - 각 fold = torchrun 4-GPU(DDP). config·fold 직렬. lora eff batch = LORA_BATCH×NPROC.
#   - Resume: 완료 셀(out-dir/preds_fold{f}.npz)은 건너뛴다. FORCE=1 이면 전부 재실행.
#
# 사전: bash/prepare_data.sh 로 cardiac_arrest_ecg_ppg_w600s_h{H}min_fold*.pt 생성.
#
# 사용법:
#   nohup bash downstream/acute_event/cardiac_arrest/bash/run.sh > ca_sweep.log 2>&1 &
#   tail -f ca_sweep.log

CHECKPOINT=${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}
DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/cardiac_arrest}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/cardiac_arrest}
NPROC=${NPROC:-4}
TASK=${TASK:-arrest}          # prepare_data_scope --task (arrest|death) → prefix scope_{task}_...
SIGNALS=${SIGNALS:-ecg_ppg}   # prefix 채널 토큰 (prepare_data_scope --input-signals 순서 = ecg ppg)
WINDOW=${WINDOW:-600}
HORIZONS=${HORIZONS:-5 15 30}

LP_BATCH=${LP_BATCH:-512};    LP_LR=${LP_LR:-1e-3};    LP_EPOCHS=${LP_EPOCHS:-1000}
LORA_BATCH=${LORA_BATCH:-128}; LORA_LR=${LORA_LR:-2e-4}; LORA_EPOCHS=${LORA_EPOCHS:-100}
LORA_RANK=${LORA_RANK:-8};    LORA_ALPHA=${LORA_ALPHA:-16}
FORCE=${FORCE:-0}

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
echo "  Imminent Cardiac Arrest Sweep | NPROC=$NPROC | FORCE=$FORCE"
echo "  Signals=$SIGNALS Window=${WINDOW}s Horizons=$HORIZONS"
echo "  LP: batch=$LP_BATCH lr=$LP_LR epochs=$LP_EPOCHS"
echo "  LoRA: batch=$LORA_BATCH (eff $((LORA_BATCH*NPROC))) lr=$LORA_LR epochs=$LORA_EPOCHS rank=$LORA_RANK"
echo "  CHECKPOINT: $CHECKPOINT | OUT_DIR: $OUT_DIR"
echo "============================================================"

for H in $HORIZONS; do
  PREFIX=$DATA_DIR/cardiac_arrest_${SIGNALS}_w${WINDOW}s_h${H}min
  for f in 0 1 2 3 4; do
    LP_OUT=$OUT_DIR/linear_probe/${SIGNALS}_w${WINDOW}s_h${H}min
    maybe_run "$LP_OUT" "$f" \
      torchrun --nproc_per_node=$NPROC -m downstream.acute_event.cardiac_arrest.run \
        --checkpoint $CHECKPOINT --model-version v2 \
        --data-path $PREFIX --mode linear_probe --n-folds 5 --fold $f \
        --epochs $LP_EPOCHS --lr $LP_LR --batch-size $LP_BATCH --device cuda \
        --out-dir "$LP_OUT"

    LORA_OUT=$OUT_DIR/lora/${SIGNALS}_w${WINDOW}s_h${H}min
    maybe_run "$LORA_OUT" "$f" \
      torchrun --nproc_per_node=$NPROC -m downstream.acute_event.cardiac_arrest.run \
        --checkpoint $CHECKPOINT --model-version v2 \
        --data-path $PREFIX --mode lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA \
        --n-folds 5 --fold $f \
        --epochs $LORA_EPOCHS --lr $LORA_LR --batch-size $LORA_BATCH --device cuda \
        --out-dir "$LORA_OUT"
  done
done

echo "============================================================"
echo "  Done. results under $OUT_DIR/{linear_probe,lora}/"
echo "============================================================"
