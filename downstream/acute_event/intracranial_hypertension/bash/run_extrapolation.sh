#!/bin/bash
# Intracranial Hypertension — LoRA 외삽(context-length) 평가 실행 스크립트
#
# 목적: 입력 window 길이를 60/300/600/1200/1800s 로 바꿔가며(horizon 30min 고정)
#   학습·평가해, 모델이 다양한 컨텍스트 길이에서 보이는 성능 곡선을 얻는다.
#   canonical run.sh(단일 window) 와 달리 여기선 window 를 sweep 한다.
#   ※ run.py 는 window 당 --data-path 하나로 train/val/test 를 모두 그 window 에서
#     쓰므로, 각 window 는 자기 데이터로 학습·평가된다(window 별 곡선).
#
# 사전: prepare_data 로 각 window 의 fold chunk 가 있어야 한다. 없으면:
#   WINDOW_SECS_OVERRIDE="60 300 600 1200 1800" HORIZON_MINS_OVERRIDE=30 \
#     bash downstream/acute_event/intracranial_hypertension/bash/prepare_data.sh
#
# 사용법:
#   bash downstream/acute_event/intracranial_hypertension/bash/run_extrapolation.sh
#   # env override: WINDOW_SECS_OVERRIDE / HORIZON_MINS_OVERRIDE / MODES_OVERRIDE / NPROC ...

set -e

# canonical 경로 (k-mimic-/bio_fm). 모두 env override 가능.
CHECKPOINT="${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}"
# 외삽 sweep 은 prediction(미래 horizon 예측). DATA_DIR 은 task_mode 로 분리된
# prediction 데이터 디렉토리(prepare_data.sh TASK_MODE=prediction 산출물)를 가리킨다.
TASK_MODE="${TASK_MODE:-prediction}"
DATA_DIR="${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/intracranial_hypertension_${TASK_MODE}}"
# 외삽 결과는 canonical 결과와 섞이지 않게 별도 디렉토리에 저장.
OUT_DIR="${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/intracranial_hypertension_extrapolation}"
DEVICE="${DEVICE:-cuda}"
MODEL_VERSION="${MODEL_VERSION:-v2}"

# prefix 채널 토큰 = prepare_data --input-signals "abp icp ecg" → "abp_icp_ecg".
# (run.py 는 --input-signals 없음: prepared 데이터의 모든 채널 사용 → prefix 로만 결합.
#  구 "icp" 토큰은 prepare 산출물과 불일치해 전부 SKIP 되던 버그.)
# prepare_data 가 파일명·디렉토리에 task_mode 를 넣으므로(detection/prediction 분리) 위에서 TASK_MODE 정의.
SIGNALS="${SIGNALS:-abp_icp_ecg}"

# 외삽 sweep: window 여러 개, horizon 고정. (canonical run.sh 는 window 1개)
WINDOW_SECS=(${WINDOW_SECS_OVERRIDE:-60 300 600 1200 1800})
HORIZON_MINS=(${HORIZON_MINS_OVERRIDE:-30})
MODES=(${MODES_OVERRIDE:-linear_probe lora})

# 변수명 전 task 공통 컨벤션. batch: LP=512, LoRA=128(NPROC=4 시 eff 512).
LP_EPOCHS="${LP_EPOCHS:-1000}"
LORA_EPOCHS="${LORA_EPOCHS:-30}"
LP_LR="${LP_LR:-1e-3}"
LORA_LR="${LORA_LR:-1e-4}"
# LP_BATCH 64: 긴 window(최대 1800s→~2700 token) attention O(seq²) OOM 회피.
#   frozen 추출은 window 독립이라 batch 무관하게 결과 byte-identical (run.sh 와 동일 근거).
LP_BATCH="${LP_BATCH:-64}"
LORA_BATCH="${LORA_BATCH:-128}"
LORA_RANK="${LORA_RANK:-8}"
N_FOLDS="${N_FOLDS:-5}"
FORCE="${FORCE:-0}"

# 한 fold 를 여러 GPU 로 DDP 실행 (fold 순차). 단일 GPU 는 NPROC=1.
NPROC=${NPROC:-4}
if [ "$NPROC" -gt 1 ]; then
    LAUNCH="torchrun --nproc_per_node=$NPROC -m"
else
    LAUNCH="python -m"
fi

echo "============================================================"
echo "  ICH — LoRA Extrapolation (context-length) Sweep"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  Windows:    ${WINDOW_SECS[*]}   (외삽 sweep)"
echo "  Horizon:    ${HORIZON_MINS[*]} min"
echo "  Modes:      ${MODES[*]}"
echo "  Batch:      LP=$LP_BATCH  LoRA=$LORA_BATCH (NPROC=$NPROC → eff $((LORA_BATCH * NPROC)))"
echo "============================================================"

for WIN in "${WINDOW_SECS[@]}"; do
    for HORIZON in "${HORIZON_MINS[@]}"; do
        # 데이터는 per-(fold,split)[_chunk] prefix 묶음 (단일 .pt 아님).
        PREFIX="${DATA_DIR}/intracranial_hypertension_${TASK_MODE}_${SIGNALS}_w${WIN}s_h${HORIZON}min"

        if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1; then
            echo "[SKIP] ${PREFIX}_fold0_*.pt not found (prepare_data 필요)"
            continue
        fi

        EXP_NAME="w${WIN}s_h${HORIZON}min"

        for MODE in "${MODES[@]}"; do
            if [ "$MODE" = "linear_probe" ]; then
                EPOCHS="$LP_EPOCHS"; LR="$LP_LR"
                EXTRA="--batch-size $LP_BATCH"
            else
                EPOCHS="$LORA_EPOCHS"; LR="$LORA_LR"
                EXTRA="--lora-rank $LORA_RANK --batch-size $LORA_BATCH"
            fi
            EXP_DIR="${OUT_DIR}/${EXP_NAME}/${MODE}"
            mkdir -p "$EXP_DIR"

            for f in $(seq 0 $((N_FOLDS - 1))); do
                if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
                    echo "  [skip] done: ${EXP_DIR} (fold $f)"
                    continue
                fi
                echo -e "\n[${EXP_NAME} | ${MODE} | fold ${f}]"
                $LAUNCH downstream.acute_event.intracranial_hypertension.run \
                    --checkpoint "$CHECKPOINT" \
                    --model-version "$MODEL_VERSION" \
                    --data-path "$PREFIX" \
                    --n-folds "$N_FOLDS" \
                    --fold "$f" \
                    --mode "$MODE" \
                    --epochs "$EPOCHS" \
                    --lr "$LR" \
                    --device "$DEVICE" \
                    --out-dir "$EXP_DIR" \
                    $EXTRA
            done
        done
    done
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "  window 별 결과: \$OUT_DIR/w{60,300,600,1200,1800}s_h${HORIZON_MINS[*]}min/{linear_probe,lora}/"
echo "============================================================"
