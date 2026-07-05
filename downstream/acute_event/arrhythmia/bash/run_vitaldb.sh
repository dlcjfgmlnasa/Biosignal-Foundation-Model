#!/bin/bash
# Arrhythmia (VitalDB) — 5-fold 실험 실행 (linear_probe + lora)
# 사전조건: prepare_vitaldb.sh 로 arrhythmia_ecg_ppg_fold{0..N-1}.pt 생성
#
# 사용법 (서버):
#   bash downstream/acute_event/arrhythmia/bash/run_vitaldb.sh
#   NPROC=4 bash .../run_vitaldb.sh        # lora 를 torchrun DDP 데이터 병렬
set -e

# ── 경로 (memory project_kmimic_bio_fm_paths — updown 기본은 stale, override 필수) ──
CHECKPOINT="${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}"
DATA_DIR="${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/arrhythmia}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/arrhythmia}"
DEVICE="${DEVICE:-cuda}"
MODEL_VERSION="${MODEL_VERSION:-v2}"   # v2 필수 (9 modality 단일 embedding)

N_FOLDS="${N_FOLDS:-5}"
LP_EPOCHS="${LP_EPOCHS:-1000}"
LP_LR="${LP_LR:-1e-3}"
LP_PATIENCE="${LP_PATIENCE:-0}"    # LP: early-stop 없이 전 epoch, best-val ckpt 선택(관례)
LORA_EPOCHS="${LORA_EPOCHS:-100}"     # ceiling. patience(아래) early-stop 이 수렴 시 자동 중단
LORA_LR="${LORA_LR:-1e-4}"
LORA_PATIENCE="${LORA_PATIENCE:-10}"  # LoRA: val macro-AUROC 무개선 10 epoch → early stop
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_BATCH="${LORA_BATCH:-128}"   # ⚠ batch≠32 면 LR 재튜닝 필요 (memory lora_acceleration)

# NPROC>1 → lora 는 torchrun DDP. linear_probe 는 frozen feature 캐싱이라 항상 python -m.
NPROC=${NPROC:-1}
if [ "$NPROC" -gt 1 ]; then LORA_LAUNCH="torchrun --nproc_per_node=$NPROC -m"; else LORA_LAUNCH="python -m"; fi

# canonical 조합만. ablation 추가 시 "ecg" "ppg" 도 prepare 후 여기에 추가.
SIGNAL_COMBOS=("ecg_ppg")
MODES=("linear_probe" "lora")

echo "============================================================"
echo "  Arrhythmia (VitalDB) — 5-class: NSR/AF/SV/VA/BradyCond"
echo "  Checkpoint: $CHECKPOINT"
echo "  ModelVer:   $MODEL_VERSION   Folds: $N_FOLDS"
echo "  Data:       $DATA_DIR        Out: $OUT_DIR"
echo "  Combos:     ${SIGNAL_COMBOS[*]}   Modes: ${MODES[*]}"
echo "============================================================"

for SIGNALS in "${SIGNAL_COMBOS[@]}"; do
    PREFIX="${DATA_DIR}/arrhythmia_${SIGNALS}"
    if [ ! -f "${PREFIX}_fold0.pt" ]; then
        echo "  SKIP: ${PREFIX}_fold0.pt not found (prepare_vitaldb.sh 먼저 실행)"
        continue
    fi
    for MODE in "${MODES[@]}"; do
        EXP_DIR="${OUT_DIR}/${MODE}/${SIGNALS}"
        mkdir -p "$EXP_DIR"
        if [ "$MODE" = "linear_probe" ]; then
            EPOCHS=$LP_EPOCHS; LR=$LP_LR; PAT=$LP_PATIENCE; EXTRA_ARGS=""; LAUNCH="python -m"
        else
            EPOCHS=$LORA_EPOCHS; LR=$LORA_LR; PAT=$LORA_PATIENCE
            EXTRA_ARGS="--lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --batch-size $LORA_BATCH"
            LAUNCH="$LORA_LAUNCH"
        fi
        for ((F=0; F<N_FOLDS; F++)); do
            echo -e "\n[${MODE} | ${SIGNALS} | fold ${F}/${N_FOLDS}]"
            $LAUNCH downstream.acute_event.arrhythmia.run \
                --checkpoint "$CHECKPOINT" \
                --model-version "$MODEL_VERSION" \
                --mode "$MODE" \
                --data-path "$PREFIX" \
                --fold "$F" \
                --n-folds "$N_FOLDS" \
                --epochs "$EPOCHS" \
                --patience "$PAT" \
                --lr "$LR" \
                --device "$DEVICE" \
                --out-dir "$EXP_DIR" \
                $EXTRA_ARGS
        done
    done
done

echo -e "\n============================================================"
echo "  Done. Results: $OUT_DIR  (per-fold JSON + .npz for OOF aggregation)"
echo "============================================================"
