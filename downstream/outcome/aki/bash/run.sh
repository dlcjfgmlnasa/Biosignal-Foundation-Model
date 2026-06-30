#!/bin/bash
# Postoperative AKI (Task #9) — patient-level Transformer Aggregator, 5-fold CV.
# Frozen Linear-Probe + LoRA. prepare_data.sh 로 5-fold chunked .pt 준비 필요:
#   aki_binary_abp_ecg_ppg_w600s_fold{0..4}_{train,val,test}_chunk*.pt
#
# 사용법:
#   # (1) 단일 GPU (linear_probe — 기존 동작, 결과 불변)
#   bash downstream/outcome/aki/bash/run.sh
#
#   # (2) lora 도 함께 (NPROC>1 이면 fold 별 lora 를 torchrun DDP 로 가속)
#   MODES="linear_probe lora" NPROC=4 bash downstream/outcome/aki/bash/run.sh
#
# env override:
#   CHECKPOINT=... DATA_DIR=... OUT_DIR=... DEVICE=cuda MODEL_VERSION=v2 \
#     [LABEL_MODE=binary WINDOW_SEC=600 N_FOLDS=5 MODES="..." NPROC=4 \
#      LP_EPOCHS LP_LR LP_BATCH LORA_EPOCHS LORA_LR LORA_RANK LORA_ALPHA LORA_BATCH]

set -e

# ── canonical 경로 (k-mimic-/bio_fm — updown 기본은 stale, memory
#    project_kmimic_bio_fm_paths). ──
CHECKPOINT="${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}"
DATA_DIR="${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/aki}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/aki}"
DEVICE="${DEVICE:-cuda}"
# v2 필수 (9 modality 단일 embedding — memory project_data_spec_v2). v1 로드 금지.
MODEL_VERSION="${MODEL_VERSION:-v2}"

LABEL_MODE="${LABEL_MODE:-binary}"
WINDOW_SEC="${WINDOW_SEC:-600}"
N_FOLDS="${N_FOLDS:-5}"
SIG_STR="${SIG_STR:-abp_ecg_ppg}"
MAX_WINDOWS="${MAX_WINDOWS:-24}"
# 실행할 모드 (기본 linear_probe 만 — 기존 동작 보존). "linear_probe lora" 로 둘 다.
MODES="${MODES:-linear_probe}"

# Linear Probe.
LP_EPOCHS="${LP_EPOCHS:-250}"
LP_LR="${LP_LR:-1e-3}"
LP_BATCH="${LP_BATCH:-512}"

# LoRA.
LORA_EPOCHS="${LORA_EPOCHS:-250}"
LORA_LR="${LORA_LR:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
# ── C안: batch-size 상향(가속) — ⚠ 결과가 바뀌므로 비교성 주의 ──
# batch(=환자 수/step)를 키우면 step 수가 줄어 빨라지나 LoRA 최적화 궤적이 달라져
# **기존 batch(16) 결과와 직접 비교 불가**(재실행 필요). torchrun 에선 effective
# batch = LORA_BATCH × NPROC. 보수적으로 가려면 LORA_BATCH=16.
LORA_BATCH="${LORA_BATCH:-512}"

# DDP: NPROC>1 이면 fold 별로 torchrun 데이터 병렬. linear_probe 는 sharded
# frozen-feature 추출(각 rank 가 환자 shard 인코딩 → gather → rank0 단독 학습),
# lora 는 aggregator DDP. NPROC=1 이면 단일 GPU(python) — 기존 동작/결과 불변.
NPROC="${NPROC:-1}"
if [ "$NPROC" -gt 1 ]; then
    NPROC_LAUNCH="torchrun --nproc_per_node=$NPROC -m"
else
    NPROC_LAUNCH="python -m"
fi

# load_prepared_split_chunked 는 prefix(파일명·확장자 없음)로 fold/split chunk 를 glob.
DATA_PREFIX="${DATA_DIR}/aki_${LABEL_MODE}_${SIG_STR}_w${WINDOW_SEC}s"

echo "============================================================"
echo "  Postop AKI (Task #9) — ${N_FOLDS}-fold CV"
echo "  Checkpoint:  $CHECKPOINT"
echo "  ModelVer:    $MODEL_VERSION"
echo "  Data prefix: $DATA_PREFIX"
echo "  Output:      $OUT_DIR"
echo "  Label mode:  $LABEL_MODE"
echo "  Modes:       $MODES"
echo "  Device:      $DEVICE"
echo "  LoRA batch:  $LORA_BATCH  (NPROC=$NPROC → eff $((LORA_BATCH * NPROC)))"
if [ "$LORA_BATCH" != "16" ]; then
    echo "  ⚠ LoRA batch≠16: 결과가 기존 batch=16 기준선과 비교 불가 — LR 재튜닝 권장"
fi
echo "============================================================"

for MODE in $MODES; do
    MODE_OUT="${OUT_DIR}/${MODE}"
    mkdir -p "$MODE_OUT"

    # linear_probe·lora 둘 다 NPROC_LAUNCH 사용(NPROC>1 이면 torchrun). linear_probe
    # 는 sharded 추출만 병렬, aggregator+probe 학습은 rank0 단독.
    if [ "$MODE" = "linear_probe" ]; then
        EPOCHS="$LP_EPOCHS"; LR="$LP_LR"; BATCH="$LP_BATCH"
        LAUNCH="$NPROC_LAUNCH"; EXTRA_ARGS=""
    else
        EPOCHS="$LORA_EPOCHS"; LR="$LORA_LR"; BATCH="$LORA_BATCH"
        LAUNCH="$NPROC_LAUNCH"
        EXTRA_ARGS="--lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA"
    fi

    for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
        echo -e "\n[${MODE} fold ${FOLD}/${N_FOLDS}] (launch: $LAUNCH)"
        $LAUNCH downstream.outcome.aki.run \
            --checkpoint "$CHECKPOINT" \
            --model-version "$MODEL_VERSION" \
            --data-path "$DATA_PREFIX" \
            --mode "$MODE" \
            --fold "$FOLD" \
            --n-folds "$N_FOLDS" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --batch-size "$BATCH" \
            --max-windows "$MAX_WINDOWS" \
            --device "$DEVICE" \
            --out-dir "$MODE_OUT" \
            $EXTRA_ARGS
    done

    # ── OOF 집계: 5-fold preds_fold*.npz → bootstrap 95% CI + mean±SD ──
    echo -e "\n[eval] OOF aggregation over ${N_FOLDS} folds (${MODE})"
    python -m downstream.run_eval \
        --pred-glob "${MODE_OUT}/preds_fold*.npz" \
        --task-name "aki_${LABEL_MODE}" \
        --out "${MODE_OUT}/aki_${LABEL_MODE}_eval_report.json"
    echo "  Report: ${MODE_OUT}/aki_${LABEL_MODE}_eval_report.json"
done

echo -e "\n============================================================"
echo "  Done! ${N_FOLDS}-fold results: $OUT_DIR"
echo "============================================================"
