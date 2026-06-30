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

# 입력은 ABP+ECG+ICP 고정(prepare_data 에서 결정). window 도 고정(canonical),
# 예측 horizon 만 변경한다. env override: WINDOW_SECS_OVERRIDE / HORIZON_MINS_OVERRIDE
# 예) WINDOW_SECS_OVERRIDE=600 HORIZON_MINS_OVERRIDE="10 30" bash run.sh
WINDOW_SECS=(${WINDOW_SECS_OVERRIDE:-1200})       # 20min 고정
HORIZON_MINS=(${HORIZON_MINS_OVERRIDE:-5 10 15})  # 이것만 변경
EPOCHS_LP="${EPOCHS_LP:-30}"
EPOCHS_LORA="${EPOCHS_LORA:-30}"
LR_LP="${LR_LP:-1e-3}"
LR_LORA="${LR_LORA:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
# ── C안: LoRA batch-size 상향 (가속) — ⚠ 결과가 바뀌므로 비교성 주의 ──
# bf16+frozen encoder 라 VRAM 여유가 커 batch 를 키우면 step 수가 줄어 빨라진다.
# 단, batch≠32 면 LoRA 최적화 궤적이 달라져 batch=32 시절 결과와 직접 비교 불가
# (LR 재튜닝 필요). torchrun(B안)에선 effective batch = LORA_BATCH × nproc.
# 보수적으로 가려면 LORA_BATCH=32 로 override.
LORA_BATCH="${LORA_BATCH:-128}"

# ── B안: NPROC>1 이면 lora 를 torchrun 데이터 병렬(단일 fold DDP)로 실행 ──
# linear_probe 는 frozen feature 캐싱이라 DDP 이득 없음 → 항상 python -m.
#   NPROC=4 bash run.sh   →   lora 가 torchrun --nproc_per_node=4 로 실행됨.
NPROC=${NPROC:-1}
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
        DATA_PATH="${DATA_DIR}/intracranial_hypertension_icp_w${WIN}s_h${HORIZON}min.pt"

        if [ ! -f "$DATA_PATH" ]; then
            echo "[SKIP] Not found: $DATA_PATH"
            continue
        fi

        EXP_NAME="w${WIN}s_h${HORIZON}min"
        echo -e "\n[${EXP_NAME}]"

        # Linear Probe (NPROC>1 이면 feature 추출을 torchrun shard→gather 병렬화)
        EXP_DIR="${OUT_DIR}/${EXP_NAME}/linear_probe"
        mkdir -p "$EXP_DIR"

        $LORA_LAUNCH downstream.acute_event.intracranial_hypertension.run \
            --checkpoint "$CHECKPOINT" \
            --model-version "$MODEL_VERSION" \
            --data-path "$DATA_PATH" \
            --mode linear_probe \
            --epochs "$EPOCHS_LP" \
            --lr "$LR_LP" \
            --device "$DEVICE" \
            --out-dir "$EXP_DIR"

        # LoRA (NPROC>1 이면 torchrun DDP, 아니면 python -m)
        EXP_DIR="${OUT_DIR}/${EXP_NAME}/lora"
        mkdir -p "$EXP_DIR"

        $LORA_LAUNCH downstream.acute_event.intracranial_hypertension.run \
            --checkpoint "$CHECKPOINT" \
            --model-version "$MODEL_VERSION" \
            --data-path "$DATA_PATH" \
            --mode lora \
            --epochs "$EPOCHS_LORA" \
            --lr "$LR_LORA" \
            --lora-rank "$LORA_RANK" \
            --batch-size "$LORA_BATCH" \
            --device "$DEVICE" \
            --out-dir "$EXP_DIR"
    done
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "============================================================"
