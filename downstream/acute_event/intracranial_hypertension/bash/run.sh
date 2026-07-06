#!/bin/bash
# Intracranial Hypertension Detection — 실험 스크립트
#
# 사전 조건:
#   1. download_waveforms.py scan → ICP-RECORDS 생성
#   2. download_waveforms.py download → waveform 다운로드
#   3. prepare_data.py → .pt 생성
#
# 사용법:
#   bash downstream/classification/intracranial_hypertension/bash/run.sh

set -e

# canonical 경로는 k-mimic-/bio_fm 아래 (updown 기본은 stale — memory
# project_kmimic_bio_fm_paths). 모두 ${VAR:-default} 라 환경변수로 override 가능:
#   CHECKPOINT=... DATA_DIR=... OUT_DIR=... bash run.sh
# 서버 실제 파일명·경로는 반드시 확인 후 맞추세요(아래 default 는 예시).
CHECKPOINT="${CHECKPOINT:-/home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}"
DATA_DIR="${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/intracranial_hypertension}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/intracranial_hypertension}"
DEVICE="${DEVICE:-cuda}"
# v2 필수 (9 modality 단일 embedding — memory project_data_spec_v2). v1 로드 금지.
MODEL_VERSION="${MODEL_VERSION:-v2}"

# 입력은 ABP+ECG 고정(detection: ICP 는 라벨전용, prepare_data 에서 결정). window 고정(canonical).
# env override: WINDOW_SECS_OVERRIDE / HORIZON_MINS_OVERRIDE
# ⚠ prefix 채널 토큰 = prepare_data --input-signals "abp ecg" 의 mode_str "abp_ecg".
#   run.py 는 --input-signals 가 없어 prepared 데이터의 모든 채널을 그대로 사용하므로
#   로드 파일은 이 prefix + horizon 으로만 결정된다. detection 은 horizon=0(h0min).
#   (prediction 모드로 쓰려면 SIGNALS=abp_icp_ecg, HORIZON_MINS_OVERRIDE="5 15 30".)
# detection(aICP식): 입력 ABP+ECG(ICP=라벨전용) → prefix 토큰 "abp_ecg", horizon=0(h0min).
#   prepare_data.py 가 detection 시 horizon 을 0 으로 고정 저장하므로 여기도 0 이어야 매칭.
# ⚠ prepare_data 가 파일명에 task_mode 를 넣으므로(detection/prediction 산출물 분리) 여기도 일치해야.
TASK_MODE="${TASK_MODE:-detection}"
SIGNALS="${SIGNALS:-abp_ecg}"
WINDOW_SECS=(${WINDOW_SECS_OVERRIDE:-10})         # 10s 고정 (aICP npj DM 세그먼트 기준)
HORIZON_MINS=(${HORIZON_MINS_OVERRIDE:-0})        # detection: 동시 라벨(h0min). prediction 시 5/15/30.
# 변수명은 전 task 공통 컨벤션: LP_EPOCHS/LORA_EPOCHS, LP_LR/LORA_LR, LP_BATCH/LORA_BATCH
LP_EPOCHS="${LP_EPOCHS:-1000}"
LORA_EPOCHS="${LORA_EPOCHS:-30}"
LP_LR="${LP_LR:-1e-3}"
LORA_LR="${LORA_LR:-1e-4}"
# LP_BATCH: frozen feature 추출/probe-fit batch. detection 은 window 가 10s(2채널 packed
#   ~10 token)로 짧아 attention O(seq²) VRAM 부담이 없다 → 512 로 복원(OOM 무관, 추출 가속).
LP_BATCH="${LP_BATCH:-512}"
LORA_RANK="${LORA_RANK:-8}"
N_FOLDS="${N_FOLDS:-5}"   # stratified k-fold — fold 별 실행(--n-folds/--fold)
FORCE="${FORCE:-0}"       # 1 이면 완료 fold(preds_fold{f}.npz)도 재실행
# ── C안: LoRA batch-size 상향 (가속) — ⚠ 결과가 바뀌므로 비교성 주의 ──
# bf16+frozen encoder 라 VRAM 여유가 커 batch 를 키우면 step 수가 줄어 빨라진다.
# 단, batch≠32 면 LoRA 최적화 궤적이 달라져 batch=32 시절 결과와 직접 비교 불가
# (LR 재튜닝 필요). torchrun(B안)에선 effective batch = LORA_BATCH × nproc.
# 보수적으로 가려면 LORA_BATCH=32 로 override.
LORA_BATCH="${LORA_BATCH:-128}"

# ── 한 fold 를 여러 GPU 로 DDP 실행 (fold 순차, 각 fold torchrun 4-GPU) ──
# 기본 NPROC=4 (cardiac_arrest/hypotension 과 동일). 단일 GPU 는 NPROC=1.
NPROC=${NPROC:-4}
if [ "$NPROC" -gt 1 ]; then
    LORA_LAUNCH="torchrun --nproc_per_node=$NPROC -m"
else
    LORA_LAUNCH="python -m"
fi

echo "============================================================"
echo "  Intracranial Hypertension Detection (ICP > 20mmHg)"
echo "  Checkpoint: $CHECKPOINT"
echo "  ModelVer:   $MODEL_VERSION"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  LoRA batch: $LORA_BATCH  (NPROC=$NPROC → eff $((LORA_BATCH * NPROC)))"
echo "============================================================"
if [ "$LORA_BATCH" != "32" ]; then
    echo "  ⚠ LoRA batch≠32: 결과가 batch=32 기준선과 비교 불가 — LR 재튜닝 권장"
fi

for WIN in "${WINDOW_SECS[@]}"; do
    for HORIZON in "${HORIZON_MINS[@]}"; do
        # ⚠ 데이터는 단일 .pt 가 아니라 per-(fold,split)[_chunk] prefix 묶음이다.
        #   예: intracranial_hypertension_detection_abp_ecg_w10s_h0min_fold0_train_chunk0.pt
        #   run.py 는 --data-path PREFIX(.pt 없이) + --n-folds/--fold 로 로드.
        PREFIX="${DATA_DIR}/intracranial_hypertension_${TASK_MODE}_${SIGNALS}_w${WIN}s_h${HORIZON}min"

        if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1; then
            echo "[SKIP] ${PREFIX}_fold0_*.pt not found (prepare_data 필요)"
            continue
        fi

        EXP_NAME="w${WIN}s_h${HORIZON}min"

        for MODE in linear_probe lora; do
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
                # resume: 완료 fold(preds_fold{f}.npz)는 건너뜀. FORCE=1 이면 재실행.
                if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
                    echo "  [skip] done: ${EXP_DIR} (fold $f)"
                    continue
                fi
                echo -e "\n[${EXP_NAME} | ${MODE} | fold ${f}]"
                $LORA_LAUNCH downstream.acute_event.intracranial_hypertension.run \
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
echo "============================================================"
