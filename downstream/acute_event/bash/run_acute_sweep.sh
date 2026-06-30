#!/bin/bash
# Acute Event Figure — IOH + ICP IH frozen/LoRA sweep (horizon × 5-fold).
#
#   - 각 fold = torchrun 4-GPU(DDP). config·fold 직렬 실행.
#   - frozen(linear_probe) 선이 figure 헤드라인, lora 는 main table 용.
#   - 전 cell 하이퍼 통일(비교성 절대조건): LP/LoRA batch·lr·epochs 고정.
#   - lora eff batch = LORA_BATCH × NPROC = 32×4 = 128, LR 2e-4(√-scale 보정).
#   - linear_probe batch 는 feature 추출 batch(결과 byte-identical), lora 만 비교 민감.
#
# 경로/하이퍼는 환경변수로 override 가능:
#   CKPT=... OUT=... LP_EPOCHS=100 bash downstream/acute_event/bash/run_acute_sweep.sh
#
# 백그라운드 권장(60 job 직렬, 김):
#   nohup bash downstream/acute_event/bash/run_acute_sweep.sh > sweep.log 2>&1 &
#   tail -f sweep.log
#
# 사전조건: prepare_data 로 5-fold chunked .pt 가 HYP_DATA/ICH_DATA 에 있어야 함.
#   ls $HYP_DATA  # hypotension_ecg_ppg_abp_w300s_h5min_fold0_test_chunk0.pt 등

CKPT=${CKPT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}
HYP_DATA=${HYP_DATA:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/hypotension}
ICH_DATA=${ICH_DATA:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/intracranial_hypertension}
OUT=${OUT:-/home/coder/workspace/k-mimic-/bio_fm/result/main}
NPROC=${NPROC:-4}

# 고정 하이퍼 (전 cell 통일)
LP_BATCH=${LP_BATCH:-128};    LP_LR=${LP_LR:-1e-3};    LP_EPOCHS=${LP_EPOCHS:-200}
LORA_BATCH=${LORA_BATCH:-32}; LORA_LR=${LORA_LR:-2e-4}; LORA_EPOCHS=${LORA_EPOCHS:-200}
LORA_RANK=${LORA_RANK:-8};    LORA_ALPHA=${LORA_ALPHA:-16}

echo "============================================================"
echo "  Acute Event Sweep | NPROC=$NPROC"
echo "  LP:   batch=$LP_BATCH lr=$LP_LR epochs=$LP_EPOCHS"
echo "  LoRA: batch=$LORA_BATCH (eff $((LORA_BATCH*NPROC))) lr=$LORA_LR epochs=$LORA_EPOCHS rank=$LORA_RANK"
echo "  CKPT: $CKPT"
echo "  OUT:  $OUT"
echo "============================================================"

# ===== IOH (horizon 5/10/15) =====
for H in 5 10 15; do
  PREFIX=$HYP_DATA/hypotension_ecg_ppg_abp_w300s_h${H}min
  for f in 0 1 2 3 4; do
    torchrun --nproc_per_node=$NPROC -m downstream.acute_event.hypotension.run \
      --checkpoint $CKPT --model-version v2 \
      --data-path $PREFIX --mode linear_probe --n-folds 5 --fold $f \
      --input-signals ecg ppg abp \
      --epochs $LP_EPOCHS --lr $LP_LR --batch-size $LP_BATCH --device cuda \
      --out-dir $OUT/hypotension/linear_probe/ecg_ppg_abp_w300s_h${H}min
    torchrun --nproc_per_node=$NPROC -m downstream.acute_event.hypotension.run \
      --checkpoint $CKPT --model-version v2 \
      --data-path $PREFIX --mode lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA \
      --n-folds 5 --fold $f \
      --input-signals ecg ppg abp \
      --epochs $LORA_EPOCHS --lr $LORA_LR --batch-size $LORA_BATCH --device cuda \
      --out-dir $OUT/hypotension/lora/ecg_ppg_abp_w300s_h${H}min
  done
done

# ===== ICP IH (horizon 15/30/60) — ICH run.py 엔 --input-signals 인자 없음(채널은 .pt metadata) =====
for H in 15 30 60; do
  PREFIX=$ICH_DATA/intracranial_hypertension_abp_ecg_icp_w1200s_h${H}min
  for f in 0 1 2 3 4; do
    torchrun --nproc_per_node=$NPROC -m downstream.acute_event.intracranial_hypertension.run \
      --checkpoint $CKPT --model-version v2 \
      --data-path $PREFIX --mode linear_probe --n-folds 5 --fold $f \
      --epochs $LP_EPOCHS --lr $LP_LR --batch-size $LP_BATCH --device cuda \
      --out-dir $OUT/intracranial_hypertension/linear_probe/abp_ecg_icp_w1200s_h${H}min
    torchrun --nproc_per_node=$NPROC -m downstream.acute_event.intracranial_hypertension.run \
      --checkpoint $CKPT --model-version v2 \
      --data-path $PREFIX --mode lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA \
      --n-folds 5 --fold $f \
      --epochs $LORA_EPOCHS --lr $LORA_LR --batch-size $LORA_BATCH --device cuda \
      --out-dir $OUT/intracranial_hypertension/lora/abp_ecg_icp_w1200s_h${H}min
  done
done

echo "============================================================"
echo "  Done. results under $OUT/{hypotension,intracranial_hypertension}/{linear_probe,lora}/"
echo "============================================================"
