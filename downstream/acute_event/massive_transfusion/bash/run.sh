#!/bin/bash
# Massive Transfusion Prediction — CARMEN(LP/LoRA) + 1D-ResNet CNN baseline sweep.
#
#   - IOH/ICH/CardiacArrest 와 동일 protocol (val best-ckpt → test 1회, 5-fold OOF).
#   - CARMEN: linear_probe(헤드라인) + lora. CNN: 사전학습 없는 supervised 1D-ResNet 비교군.
#   - 각 fold = torchrun 4-GPU(DDP, CARMEN). CNN 은 DataParallel. config·fold 직렬.
#   - Resume: 완료 셀(out-dir/preds_fold{f}.npz)은 건너뛴다. FORCE=1 이면 전부 재실행.
#
# 사전: bash/prepare_data.sh 로 massive_transfusion_abp_ppg_w600s_h{H}min_fold*.pt 생성.
#
# 사용법:
#   nohup bash downstream/acute_event/massive_transfusion/bash/run.sh > mt_sweep.log 2>&1 &
#   tail -f mt_sweep.log

CHECKPOINT=${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}
DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/massive_transfusion}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/massive_transfusion}
NPROC=${NPROC:-4}
SIGNALS=${SIGNALS:-abp_ppg}   # prepare_data --input-signals 순서(abp ppg) 와 일치
WINDOW=${WINDOW:-600}
HORIZONS=${HORIZONS:-5 10 15}

LP_BATCH=${LP_BATCH:-512};    LP_LR=${LP_LR:-1e-3};    LP_EPOCHS=${LP_EPOCHS:-1000}
LORA_BATCH=${LORA_BATCH:-128}; LORA_LR=${LORA_LR:-2e-4}; LORA_EPOCHS=${LORA_EPOCHS:-30}
LORA_RANK=${LORA_RANK:-8};    LORA_ALPHA=${LORA_ALPHA:-16}
CNN_BATCH=${CNN_BATCH:-128};  CNN_LR=${CNN_LR:-1e-3};  CNN_EPOCHS=${CNN_EPOCHS:-100}; CNN_PATIENCE=${CNN_PATIENCE:-20}
RUN_CNN=${RUN_CNN:-1}         # 0 이면 CNN baseline 건너뜀
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
echo "  Massive Transfusion Sweep | NPROC=$NPROC | FORCE=$FORCE | CNN=$RUN_CNN"
echo "  Signals=$SIGNALS Window=${WINDOW}s Horizons=$HORIZONS"
echo "  CHECKPOINT: $CHECKPOINT | OUT_DIR: $OUT_DIR"
echo "============================================================"

for H in $HORIZONS; do
  # prepare_data.py 저장 prefix = massive_transfusion_{signals}_w{win}s_h{H}min
  PREFIX=$DATA_DIR/massive_transfusion_${SIGNALS}_w${WINDOW}s_h${H}min
  for f in 0 1 2 3 4; do
    # ── CARMEN: linear_probe (헤드라인) ──
    LP_OUT=$OUT_DIR/linear_probe/${SIGNALS}_w${WINDOW}s_h${H}min
    maybe_run "$LP_OUT" "$f" \
      torchrun --nproc_per_node=$NPROC -m downstream.acute_event.massive_transfusion.run \
        --checkpoint $CHECKPOINT --model-version v2 \
        --data-path $PREFIX --mode linear_probe --n-folds 5 --fold $f \
        --epochs $LP_EPOCHS --lr $LP_LR --batch-size $LP_BATCH --device cuda \
        --out-dir "$LP_OUT"

    # ── CARMEN: LoRA ──
    LORA_OUT=$OUT_DIR/lora/${SIGNALS}_w${WINDOW}s_h${H}min
    maybe_run "$LORA_OUT" "$f" \
      torchrun --nproc_per_node=$NPROC -m downstream.acute_event.massive_transfusion.run \
        --checkpoint $CHECKPOINT --model-version v2 \
        --data-path $PREFIX --mode lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA \
        --n-folds 5 --fold $f \
        --epochs $LORA_EPOCHS --lr $LORA_LR --batch-size $LORA_BATCH --device cuda \
        --out-dir "$LORA_OUT"

    # ── Supervised baseline: 1D-ResNet CNN (사전학습 없음) ──
    if [ "$RUN_CNN" = "1" ]; then
      CNN_OUT=$OUT_DIR/cnn/${SIGNALS}_w${WINDOW}s_h${H}min
      maybe_run "$CNN_OUT" "$f" \
        python -m downstream.baselines.masstf_cnn \
          --data-path $PREFIX --input-signals abp ppg \
          --n-folds 5 --fold $f \
          --epochs $CNN_EPOCHS --patience $CNN_PATIENCE --lr $CNN_LR \
          --batch-size $CNN_BATCH --device cuda \
          --out-dir "$CNN_OUT"
    fi
  done
done

echo "============================================================"
echo "  Done. results under $OUT_DIR/{linear_probe,lora,cnn}/"
echo "============================================================"
