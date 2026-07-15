#!/bin/bash
# Blood Pressure (SBP/DBP) Estimation — 실험 스크립트 (회귀 task).
#
# 사전 조건: prepare_data.sh 로 chunk 산출물 생성
#   (bp_{signals}_w{win}s_fold{F}_{split}_chunk*.pt).
#
# 사용법:
#   bash downstream/vital_sign/blood_pressure/bash/run.sh
#   MODES=linear_probe HEADS="attn mean" bash ...   # frozen 만, 두 head 비교
#
# canonical 경로는 k-mimic-/bio_fm 아래 (updown 기본은 stale — memory
# project_kmimic_bio_fm_paths). 모두 ${VAR:-default} 라 env 로 override 가능.
# ⚠ DATA_DIR 은 prepare_data.sh 의 OUT_DIR 과 반드시 일치해야 로드된다.

set -e

CHECKPOINT="${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}"
DATA_DIR="${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/bp_estimation}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/bp_estimation}"
MODEL_VERSION="${MODEL_VERSION:-v2}"   # v2 필수 (9 modality 단일 embedding)

# prepare_data 산출물 prefix 결정: bp_{signals}_w{win}s (input-signals 를 "_" 로 join).
SIGNALS="${SIGNALS:-ecg_ppg}"          # 입력 채널 prefix (ppg | ecg_ppg). label=ABP.
WIN="${WIN:-30}"                       # canonical: 30s window
N_FOLDS="${N_FOLDS:-5}"                # patient-level stratified k-fold

# head: attn(권장, attention-pool) / mean(mean-pool linear probe, 비교용 baseline).
HEADS="${HEADS:-attn}"
# 변수명 전 task 공통: LP_*(frozen linear probe) / LORA_*.
LP_EPOCHS="${LP_EPOCHS:-100}"
LORA_EPOCHS="${LORA_EPOCHS:-30}"
LP_LR="${LP_LR:-1e-3}"
LORA_LR="${LORA_LR:-1e-4}"
LP_BATCH="${LP_BATCH:-64}"
LORA_BATCH="${LORA_BATCH:-32}"         # LoRA 는 32 유지(비교성 — memory)
LORA_RANK="${LORA_RANK:-8}"
FORCE="${FORCE:-0}"                    # 1 이면 완료 fold(bp_preds_fold{f}.npz)도 재실행

PREFIX="${DATA_DIR}/bp_${SIGNALS}_w${WIN}s"

echo "============================================================"
echo "  Blood Pressure (SBP/DBP) Estimation — regression"
echo "  Checkpoint: $CHECKPOINT  ($MODEL_VERSION)"
echo "  Data:       $PREFIX  (fold0..$((N_FOLDS - 1)))"
echo "  Output:     $OUT_DIR"
echo "  Heads:      $HEADS   Modes: ${MODES:-linear_probe lora}"
echo "============================================================"

if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1; then
    echo "[ERROR] ${PREFIX}_fold0_*.pt not found — prepare_data.sh 먼저 실행하세요." >&2
    exit 1
fi

for HEAD in $HEADS; do
    for MODE in ${MODES:-linear_probe lora}; do
        if [ "$MODE" = "linear_probe" ]; then
            EPOCHS="$LP_EPOCHS"; LR="$LP_LR"; BATCH="$LP_BATCH"; EXTRA=""
        else
            EPOCHS="$LORA_EPOCHS"; LR="$LORA_LR"; BATCH="$LORA_BATCH"
            EXTRA="--lora-rank $LORA_RANK"
        fi
        EXP_DIR="${OUT_DIR}/head_${HEAD}/${MODE}"
        mkdir -p "$EXP_DIR"

        for f in $(seq 0 $((N_FOLDS - 1))); do
            if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/bp_preds_fold${f}.npz" ]; then
                echo "  [skip] done: ${EXP_DIR} (fold $f)"
                continue
            fi
            echo -e "\n[head=${HEAD} | ${MODE} | fold ${f}]"
            python -m downstream.vital_sign.blood_pressure.run \
                --checkpoint "$CHECKPOINT" \
                --model-version "$MODEL_VERSION" \
                --data-path "$PREFIX" \
                --n-folds "$N_FOLDS" \
                --fold "$f" \
                --mode "$MODE" \
                --head "$HEAD" \
                --epochs "$EPOCHS" \
                --lr "$LR" \
                --batch-size "$BATCH" \
                --out-dir "$EXP_DIR" \
                $EXTRA
        done
    done
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
