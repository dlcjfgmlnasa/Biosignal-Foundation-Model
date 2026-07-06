#!/usr/bin/env bash
# Task #13 Waveform Forecasting — Zero-shot evaluation (all modalities)
#
# run.py 는 zero-shot 전용(추가 학습 없음):
#   single-input : model.generate()  (block-autoregressive, 임의 horizon)
#   multi-input  : model.forecast()  (K-block 1-shot, horizon 상한 ≈ next_block_size×patch)
# 각 (modality, fold) 마다 preds_fold{f}.npz + results json dump → aggregator 호환.
#
# 사전 조건: bash prepare_data.sh 로 .pt 생성 (per-(fold,split)_chunk 묶음).
#
# 사용법 (서버):
#   CHECKPOINT=... DATA_DIR=... OUT_DIR=... \
#     bash downstream/generation/intra_modal_forecast/bash/run.sh
set -euo pipefail

# ── 서버 canonical 경로 (updown 기본은 stale — memory project_kmimic_bio_fm_paths) ──
BIOFM_ROOT="${BIOFM_ROOT:-/home/coder/workspace/k-mimic-/bio_fm}"
# Phase2 최종 checkpoint (av epoch049_final). 서버 실제 경로 확인 후 override.
CHECKPOINT="${CHECKPOINT:-${BIOFM_ROOT}/outputs/main/phase2/kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt}"
DATA_DIR="${DATA_DIR:-${BIOFM_ROOT}/data/downstream/forecasting}"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/result/main/forecasting}"
# v2 필수 (9 modality 단일 embedding — memory project_data_spec_v2). v1 로드 금지.
MODEL_VERSION="${MODEL_VERSION:-v2}"

# prepare_data.sh 와 반드시 동일해야 파일명 prefix 매칭 (ctx/tgt).
CTX_SEC="${CTX_SEC:-30}"
TGT_SEC="${TGT_SEC:-30}"
# horizon(초) sweep → "MAE(horizon)" curve. TGT_SEC 초과분은 run.py 가 자동 truncate.
# multi-input(icp)은 forecast() 상한까지만 유효(나머지는 상한으로 clamp).
TARGET_SWEEP="${TARGET_SWEEP:-2 6 10 30}"
N_FOLDS="${N_FOLDS:-5}"
FORCE="${FORCE:-0}"   # 1 이면 완료 fold(preds_fold{f}.npz)도 재실행

# 대상 modality 목록 (env 로 축소 가능). icp 는 multi-input 로 자동 분기.
SIGNALS="${SIGNALS:-ecg abp ppg co2 awp cvp icp}"

echo "============================================================"
echo "  Waveform Forecasting — Zero-shot Eval"
echo "  Checkpoint: $CHECKPOINT"
echo "  ModelVer:   $MODEL_VERSION"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUT_DIR"
echo "  Modalities: $SIGNALS"
echo "  Horizons:   ${TARGET_SWEEP}s  (ctx ${CTX_SEC}s, prepared tgt ${TGT_SEC}s)"
echo "============================================================"

for sig in $SIGNALS; do
    # ── modality → 데이터 prefix 라우팅 ──────────────────────
    if [ "$sig" = "icp" ]; then
        # MIMIC multi-input (context=icp → target=icp)
        PREFIX="${DATA_DIR}/forecasting_multi_mimic3_icp_to_icp_ctx${CTX_SEC}s_tgt${TGT_SEC}s"
    else
        # VitalDB single-input
        PREFIX="${DATA_DIR}/forecasting_vitaldb_${sig}_ctx${CTX_SEC}s_tgt${TGT_SEC}s"
    fi

    if ! ls "${PREFIX}"_fold0_*.pt >/dev/null 2>&1; then
        echo "[SKIP] ${sig}: ${PREFIX}_fold0_*.pt 없음 (prepare_data 필요)"
        continue
    fi

    EXP_DIR="${OUT_DIR}/${sig}"
    mkdir -p "$EXP_DIR"

    for f in $(seq 0 $((N_FOLDS - 1))); do
        # resume: 완료 fold 는 건너뜀. FORCE=1 이면 재실행.
        if [ "$FORCE" != "1" ] && [ -f "${EXP_DIR}/preds_fold${f}.npz" ]; then
            echo "  [skip] done: ${EXP_DIR} (fold $f)"
            continue
        fi
        echo -e "\n[${sig} | fold ${f}]"
        python -m downstream.generation.intra_modal_forecast.run \
            --checkpoint "$CHECKPOINT" \
            --model-version "$MODEL_VERSION" \
            --data-path "$PREFIX" \
            --signal-type "$sig" \
            --n-folds "$N_FOLDS" \
            --fold "$f" \
            --target-sec $TARGET_SWEEP \
            --out-dir "$EXP_DIR"
    done
done

echo -e "\n============================================================"
echo "  Done! Results: $OUT_DIR"
echo "  각 modality/ 아래 preds_fold{0..$((N_FOLDS-1))}.npz + forecasting_*_results.json"
echo "============================================================"
