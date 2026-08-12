#!/bin/bash
# Massive Transfusion Prediction — CARMEN(LP/LoRA) + 1D-ResNet CNN baseline sweep.
#
#   - IOH/ICH/CardiacArrest 와 동일 protocol (val best-ckpt → test 1회, 5-fold OOF).
#   - CARMEN: linear_probe(헤드라인) + lora (+ scratch=사전학습 대조군, RUN_SCRATCH=1).
#     CNN: 사전학습 없는 supervised 1D-ResNet 비교군(아키텍처가 CARMEN 과 다름).
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

# ── CARMEN-scratch (random init + 전체 파라미터 end-to-end 학습) ──
# 사전학습 대조군(Table 3). CNN baseline 과 달리 **CARMEN 과 동일 아키텍처**다.
# ⚠ batch 는 LoRA(128) 보다 훨씬 작아야 한다. LoRA 는 base weight 가 frozen 이라
#   autograd 가 그 Linear 들의 입력 activation 을 저장하지 않지만(weight grad 불필요),
#   scratch 는 q/k/v/out/fc1/fc2/gate 전부가 학습 대상이라 입력을 backward 까지
#   보유한다. IOH(w300s)에서 128 은 L40S 44GB OOM 이었다.
#   MT 는 window 가 600s 라 IOH 보다 sample 당 토큰이 ~1.3배(2 sig×300 vs 3 sig×150)
#   → 기본 16 으로 더 보수적으로 잡는다. 여유 있으면 24~32 로 올려도 된다.
#   OOM 시 PYTORCH_ALLOC_CONF=expandable_segments:True 를 함께 준다.
# ⚠ 이 스크립트는 torchrun DDP(NPROC=4) 라 effective batch = SCRATCH_BATCH × NPROC.
# ⚠ 한 task 의 전 fold·horizon 은 같은 batch/LR 로 돌아야 표 비교가 성립한다.
SCRATCH_BATCH=${SCRATCH_BATCH:-16}; SCRATCH_LR=${SCRATCH_LR:-3e-4}; SCRATCH_EPOCHS=${SCRATCH_EPOCHS:-30}

# 모드별 on/off — 기본값은 기존 동작 유지(LP+LoRA+CNN). scratch 만 돌리려면
#   RUN_LP=0 RUN_LORA=0 RUN_CNN=0 RUN_SCRATCH=1 bash run.sh
RUN_LP=${RUN_LP:-1}
RUN_LORA=${RUN_LORA:-1}
RUN_CNN=${RUN_CNN:-1}         # 0 이면 CNN baseline 건너뜀
RUN_SCRATCH=${RUN_SCRATCH:-0} # 1 이면 CARMEN-scratch 대조군 실행
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
echo "  Massive Transfusion Sweep | NPROC=$NPROC | FORCE=$FORCE"
echo "  Modes: LP=$RUN_LP LoRA=$RUN_LORA scratch=$RUN_SCRATCH CNN=$RUN_CNN"
if [ "$RUN_SCRATCH" = "1" ]; then
  echo "  Scratch: epochs=$SCRATCH_EPOCHS lr=$SCRATCH_LR batch=$SCRATCH_BATCH (eff $((SCRATCH_BATCH * NPROC)))"
fi
echo "  Signals=$SIGNALS Window=${WINDOW}s Horizons=$HORIZONS"
echo "  CHECKPOINT: $CHECKPOINT | OUT_DIR: $OUT_DIR"
echo "============================================================"

for H in $HORIZONS; do
  # prepare_data.py 저장 prefix = massive_transfusion_{signals}_w{win}s_h{H}min
  PREFIX=$DATA_DIR/massive_transfusion_${SIGNALS}_w${WINDOW}s_h${H}min
  for f in 0 1 2 3 4; do
    # ── CARMEN: linear_probe (헤드라인) ──
    if [ "$RUN_LP" = "1" ]; then
      LP_OUT=$OUT_DIR/linear_probe/${SIGNALS}_w${WINDOW}s_h${H}min
      maybe_run "$LP_OUT" "$f" \
        torchrun --nproc_per_node=$NPROC -m downstream.acute_event.massive_transfusion.run \
          --checkpoint $CHECKPOINT --model-version v2 \
          --data-path $PREFIX --mode linear_probe --n-folds 5 --fold $f \
          --epochs $LP_EPOCHS --lr $LP_LR --batch-size $LP_BATCH --device cuda \
          --out-dir "$LP_OUT"
    fi

    # ── CARMEN: LoRA ──
    if [ "$RUN_LORA" = "1" ]; then
      LORA_OUT=$OUT_DIR/lora/${SIGNALS}_w${WINDOW}s_h${H}min
      maybe_run "$LORA_OUT" "$f" \
        torchrun --nproc_per_node=$NPROC -m downstream.acute_event.massive_transfusion.run \
          --checkpoint $CHECKPOINT --model-version v2 \
          --data-path $PREFIX --mode lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA \
          --n-folds 5 --fold $f \
          --epochs $LORA_EPOCHS --lr $LORA_LR --batch-size $LORA_BATCH --device cuda \
          --out-dir "$LORA_OUT"
    fi

    # ── CARMEN-scratch: 동일 구조 random init, 전체 파라미터 학습 (사전학습 대조군) ──
    # --checkpoint 은 ModelConfig 공급용 (가중치는 로드하지 않는다).
    if [ "$RUN_SCRATCH" = "1" ]; then
      SCRATCH_OUT=$OUT_DIR/scratch/${SIGNALS}_w${WINDOW}s_h${H}min
      maybe_run "$SCRATCH_OUT" "$f" \
        torchrun --nproc_per_node=$NPROC -m downstream.acute_event.massive_transfusion.run \
          --checkpoint $CHECKPOINT --model-version v2 \
          --data-path $PREFIX --mode scratch --n-folds 5 --fold $f \
          --epochs $SCRATCH_EPOCHS --lr $SCRATCH_LR --batch-size $SCRATCH_BATCH --device cuda \
          --out-dir "$SCRATCH_OUT"
    fi

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
echo "  Done. results under $OUT_DIR/{linear_probe,lora,scratch,cnn}/"
echo "============================================================"
