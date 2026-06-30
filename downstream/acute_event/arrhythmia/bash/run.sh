#!/bin/bash
# Arrhythmia Classification — 전체 실험 실행 스크립트
# 3가지 신호 조합 × 2 modes = 6 실험
#
# 사전 조건: prepare_data.sh로 .pt 데이터셋이 생성되어 있어야 함
#
# 사용법:
#   bash downstream/classification/arrhythmia/bash/run.sh

set -e

# ── 설정 ──
# canonical 경로는 k-mimic-/bio_fm 아래 (updown 기본은 stale — memory
# project_kmimic_bio_fm_paths). 모두 ${VAR:-default} 라 환경변수로 override 가능.
CHECKPOINT="${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}"
DATA_DIR="${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/arrhythmia}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/arrhythmia}"
DEVICE="${DEVICE:-cuda}"
# v2 필수 (9 modality 단일 embedding — memory project_data_spec_v2). v1 로드 금지.
MODEL_VERSION="${MODEL_VERSION:-v2}"

# Linear Probe 설정
LP_EPOCHS="${LP_EPOCHS:-250}"
LP_LR="${LP_LR:-1e-3}"

# LoRA 설정
LORA_EPOCHS="${LORA_EPOCHS:-250}"
LORA_LR="${LORA_LR:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
# ── C안: LoRA batch-size 상향 (가속) — ⚠ 결과가 바뀌므로 비교성 주의 ──
# batch≠32 면 LoRA 최적화 궤적이 달라져 batch=32 기준선과 직접 비교 불가(LR 재튜닝
# 필요). torchrun(B안)에선 effective batch = LORA_BATCH × nproc. 보수적: LORA_BATCH=32.
LORA_BATCH="${LORA_BATCH:-512}"

# ── B안: NPROC>1 이면 lora 를 torchrun 데이터 병렬(단일 fold DDP)로 실행 ──
# linear_probe 는 frozen feature 캐싱이라 DDP 이득 없음 → 항상 python -m.
#   NPROC=4 bash run.sh   →   lora 가 torchrun --nproc_per_node=4 로 실행됨.
NPROC=${NPROC:-1}
if [ "$NPROC" -gt 1 ]; then
    LORA_LAUNCH="torchrun --nproc_per_node=$NPROC -m"
else
    LORA_LAUNCH="python -m"
fi

# ── 실험 조합 ──
SIGNAL_COMBOS=("ppg" "ecg" "ppg_ecg")
MODES=("linear_probe" "lora")

TOTAL=$(( ${#SIGNAL_COMBOS[@]} * ${#MODES[@]} ))
COUNT=0

echo "============================================================"
echo "  Arrhythmia Classification — Experiment Sweep"
echo "  Checkpoint: $CHECKPOINT"
echo "  ModelVer:   $MODEL_VERSION"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  Device:     $DEVICE"
echo "  Signals:    ${SIGNAL_COMBOS[*]}"
echo "  Modes:      ${MODES[*]}"
echo "  Classes:    SR / AF / STACH / SBRAD / AFLT"
echo "  LoRA batch: $LORA_BATCH  (NPROC=$NPROC → eff $((LORA_BATCH * NPROC)))"
echo "  Total:      $TOTAL experiments"
echo "============================================================"
if [ "$LORA_BATCH" != "32" ]; then
    echo "  ⚠ LoRA batch≠32: 결과가 batch=32 기준선과 비교 불가 — LR 재튜닝 권장"
fi

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    PT_FILE="${DATA_DIR}/arrhythmia_${SIGNALS}.pt"

    if [ ! -f "$PT_FILE" ]; then
        echo "  SKIP: $PT_FILE not found"
        continue
    fi

    for MODE in "${MODES[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_DIR="${OUT_DIR}/${MODE}/${SIGNALS}"
        mkdir -p "$EXP_DIR"

        echo -e "\n[${COUNT}/${TOTAL}] ${MODE} | ${SIGNALS}"

        # linear_probe·lora 모두 $LORA_LAUNCH(NPROC>1 시 torchrun). linear_probe 는
        # feature 추출만 shard→gather 병렬화하고 probe batch 는 default 유지(결과 불변),
        # lora 만 batch 상향(C안).
        if [ "$MODE" = "linear_probe" ]; then
            EPOCHS=$LP_EPOCHS
            LR=$LP_LR
            EXTRA_ARGS=""
            LAUNCH="$LORA_LAUNCH"
        else
            EPOCHS=$LORA_EPOCHS
            LR=$LORA_LR
            # C안: lora 만 batch 상향(linear_probe 는 frozen feature 캐싱이라 무관).
            EXTRA_ARGS="--lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --batch-size $LORA_BATCH"
            LAUNCH="$LORA_LAUNCH"
        fi

        $LAUNCH downstream.acute_event.arrhythmia.run \
            --checkpoint "$CHECKPOINT" \
            --model-version "$MODEL_VERSION" \
            --mode "$MODE" \
            --data-path "$PT_FILE" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --device "$DEVICE" \
            --out-dir "$EXP_DIR" \
            $EXTRA_ARGS

    done
done

echo -e "\n============================================================"
echo "  Done! ${COUNT}/${TOTAL} experiments completed"
echo "  Results: $OUT_DIR"
echo "============================================================"
