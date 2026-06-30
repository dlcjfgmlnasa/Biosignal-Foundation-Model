#!/bin/bash
# Sepsis Prediction — 실험 스크립트 (patient-level Transformer Aggregator)
# Linear Probe + LoRA 두 모드 실행.
#
# 사용법:
#   # (1) 단일 GPU (기존 동작 — 결과 불변)
#   bash downstream/outcome/sepsis/bash/run.sh
#
#   # (2) B안 — torchrun+DDP: LoRA run 을 N GPU 로 데이터 병렬(환자 단위 분할).
#   NPROC=4 bash downstream/outcome/sepsis/bash/run.sh

set -e

# ── canonical 경로 (k-mimic-/bio_fm — updown 기본은 stale, memory
#    project_kmimic_bio_fm_paths). 모두 ${VAR:-default} 라 env override 가능. ──
CHECKPOINT=${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}
DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/sepsis}
OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/sepsis}
DEVICE=${DEVICE:-cuda}
# v2 필수 (9 modality 단일 embedding — memory project_data_spec_v2). v1 로드 금지.
MODEL_VERSION=${MODEL_VERSION:-v2}

WINDOW_SEC=${WINDOW_SEC:-600}
MAX_WINDOWS=${MAX_WINDOWS:-24}

# Linear Probe.
LP_EPOCHS=${LP_EPOCHS:-250}
LP_LR=${LP_LR:-1e-3}
LP_BATCH=${LP_BATCH:-8}

# LoRA 설정.
LORA_EPOCHS=${LORA_EPOCHS:-250}
LORA_LR=${LORA_LR:-1e-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
# ── C안: batch-size 상향(가속) — ⚠ 결과가 바뀌므로 비교성 주의 ──
# batch(=환자 수/step)를 키우면 step 수가 줄어 빨라지나 LoRA 최적화 궤적이 달라져
# **기존 batch(8) 결과와 직접 비교 불가**(재실행 필요). torchrun 에선 effective
# batch = LORA_BATCH × NPROC. 보수적으로 가려면 LORA_BATCH=8.
LORA_BATCH=${LORA_BATCH:-128}

# DDP: NPROC>1 이면 torchrun 데이터 병렬. linear_probe 는 sharded frozen-feature
# 추출(각 rank 가 환자 shard 인코딩 → gather → rank0 단독 aggregator+probe 학습),
# lora 는 aggregator DDP. NPROC=1 이면 단일 GPU(python) — 기존 동작/결과 불변.
NPROC=${NPROC:-1}
if [ "$NPROC" -gt 1 ]; then
    LAUNCH="torchrun --nproc_per_node=$NPROC -m"
else
    LAUNCH="python -m"
fi

DATA_PATH="${DATA_DIR}/sepsis_w${WINDOW_SEC}s.pt"

echo "============================================================"
echo "  Sepsis Prediction (Transformer Aggregator)"
echo "  Checkpoint:  $CHECKPOINT"
echo "  ModelVer:    $MODEL_VERSION"
echo "  Data:        $DATA_PATH"
echo "  Output:      $OUT_DIR"
echo "  Device:      $DEVICE"
echo "  LoRA batch:  $LORA_BATCH  (NPROC=$NPROC → eff $((LORA_BATCH * NPROC)))"
if [ "$LORA_BATCH" != "8" ]; then
    echo "  ⚠ LoRA batch≠8: 결과가 기존 batch=8 기준선과 비교 불가 — LR 재튜닝 권장"
fi
echo "============================================================"

# ── Linear Probe (NPROC>1 이면 sharded 추출 torchrun) ──
EXP_DIR="${OUT_DIR}/linear_probe"
mkdir -p "$EXP_DIR"

echo -e "\n[1/2] Linear Probe (launch: $LAUNCH)"
$LAUNCH downstream.outcome.sepsis.run \
    --checkpoint "$CHECKPOINT" \
    --model-version "$MODEL_VERSION" \
    --data-path "$DATA_PATH" \
    --mode linear_probe \
    --epochs "$LP_EPOCHS" \
    --lr "$LP_LR" \
    --batch-size "$LP_BATCH" \
    --max-windows "$MAX_WINDOWS" \
    --device "$DEVICE" \
    --out-dir "$EXP_DIR"

# ── LoRA (NPROC>1 이면 torchrun DDP) ──
EXP_DIR="${OUT_DIR}/lora"
mkdir -p "$EXP_DIR"

echo -e "\n[2/2] LoRA Fine-tuning (launch: $LAUNCH)"
$LAUNCH downstream.outcome.sepsis.run \
    --checkpoint "$CHECKPOINT" \
    --model-version "$MODEL_VERSION" \
    --data-path "$DATA_PATH" \
    --mode lora \
    --epochs "$LORA_EPOCHS" \
    --lr "$LORA_LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --batch-size "$LORA_BATCH" \
    --max-windows "$MAX_WINDOWS" \
    --device "$DEVICE" \
    --out-dir "$EXP_DIR"

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
