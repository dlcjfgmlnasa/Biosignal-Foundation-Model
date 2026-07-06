#!/usr/bin/env bash
# Task #13 Waveform Forecasting — Data Preparation (all modalities)
# Intra-modal forecast: 과거 context waveform → 미래 target waveform.
#
# 소스 라우팅 (2026-07-06 확정):
#   VitalDB single-input (autoregressive model.generate):
#       ecg, abp, ppg, co2, awp, cvp   (6종; load_pilot_cases PREFERRED_TRACKS)
#   MIMIC multi-input (non-AR model.forecast, ICH waveform-dir):
#       icp   (context=icp → target=icp; ICH 로더가 icp 채널 추출)
#             ※ forecast() 는 K-block(≈ next_block_size × patch) 1-shot → horizon 상한 존재.
#   ⛔ resp_impedance / resp_flow: 즉시 추출 경로 없음 (본 launcher 제외)
#       - ICH 로더(load_patient_signals) 채널맵에 RESP 없음
#       - mimic3_waveform.scan 은 ABP-anchored (RESP 전용 스캔 함수 부재)
#       - resp_flow 는 다운스트림 소스 자체 부재 (memory project_respflow_no_downstream_source)
#       활성화하려면 RESP 포함 MIMIC waveform dir + 채널맵 확장 후 아래 NOTE 블록 참조.
#
# 사용법 (서버):
#   OUT_DIR=... VITAL_DIR=... WAVEFORM_DIR=... \
#     bash downstream/generation/intra_modal_forecast/bash/prepare_data.sh
set -euo pipefail

# ── 서버 canonical 경로 (updown 기본은 stale — memory project_kmimic_bio_fm_paths) ──
REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
UPDOWN_ROOT="${UPDOWN_ROOT:-/home/coder/workspace/updown}"
BIOFM_ROOT="${BIOFM_ROOT:-/home/coder/workspace/k-mimic-/bio_fm}"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/forecasting}"
# icp multi-input 용 ICH waveform 디렉토리 (ICH prepare_data 다운로드 산출물 재사용)
WAVEFORM_DIR="${WAVEFORM_DIR:-${UPDOWN_ROOT}/raw/mimic3-waveform-ich}"
# 로컬 VitalDB Open .vital 디렉토리 (없으면 VitalDB API 경유)
VITAL_DIR="${VITAL_DIR:-}"

N_CASES="${N_CASES:-100}"
CTX_SEC="${CTX_SEC:-30}"        # context window (초)
TGT_SEC="${TGT_SEC:-30}"        # target(미래) 길이 — run.sh 의 TARGET_SWEEP 상한
STRIDE_SEC="${STRIDE_SEC:-15}"
N_FOLDS="${N_FOLDS:-5}"         # patient(case)-level stratified K-fold
SEED="${SEED:-42}"

# VitalDB single-input 대상 6종 (env 로 축소/변경 가능)
VITALDB_SIGNALS="${VITALDB_SIGNALS:-ecg abp ppg co2 awp cvp}"

VITAL_ARGS=""
[ -n "$VITAL_DIR" ] && VITAL_ARGS="--vital-dir $VITAL_DIR"

echo "============================================================"
echo "  Waveform Forecasting — Data Preparation"
echo "  Out:        $OUT_DIR"
echo "  VitalDB:    $VITALDB_SIGNALS"
echo "  Context:    ${CTX_SEC}s → Target: ${TGT_SEC}s (stride ${STRIDE_SEC}s)"
echo "  K-fold:     $N_FOLDS  |  n_cases: $N_CASES"
echo "============================================================"

# ── VitalDB single-input (autoregressive) ─────────────────────
for sig in $VITALDB_SIGNALS; do
    echo -e "\n=== [VitalDB single-input] ${sig} ==="
    python -m downstream.generation.intra_modal_forecast.prepare_data \
        --source vitaldb $VITAL_ARGS \
        --signal-type "$sig" \
        --n-cases "$N_CASES" \
        --context-sec "$CTX_SEC" \
        --target-sec "$TGT_SEC" \
        --stride-sec "$STRIDE_SEC" \
        --n-folds "$N_FOLDS" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR"
done

# ── MIMIC multi-input: icp (context=icp → target=icp) ─────────
if [ -d "$WAVEFORM_DIR" ]; then
    echo -e "\n=== [MIMIC multi-input] icp → icp (ICH waveform-dir) ==="
    python -m downstream.generation.intra_modal_forecast.prepare_data \
        --source mimic3 \
        --multi-input \
        --context-channels icp \
        --target-channel icp \
        --waveform-dir "$WAVEFORM_DIR" \
        --context-sec "$CTX_SEC" \
        --target-sec "$TGT_SEC" \
        --stride-sec "$STRIDE_SEC" \
        --n-folds "$N_FOLDS" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR"
else
    echo -e "\n[SKIP] ICH waveform-dir 없음: $WAVEFORM_DIR — icp forecast 건너뜀"
    echo "        (ICH prepare_data.sh 로 ICP waveform 선다운로드 필요)"
fi

# ── ⛔ resp_impedance (즉시 경로 없음) ────────────────────────
# 활성화 조건: RESP(impedance) 채널을 가진 MIMIC waveform dir + 로더 채널맵 확장.
# 예 (구현 후):
#   python -m downstream.generation.intra_modal_forecast.prepare_data \
#       --source mimic3 --multi-input \
#       --context-channels resp_impedance --target-channel resp_impedance \
#       --waveform-dir <RESP 포함 dir> ...
echo -e "\n[NOTE] resp_impedance / resp_flow: 소스 부재로 본 launcher 에서 제외."

echo -e "\n============================================================"
echo "  Done. Output: $OUT_DIR"
echo "  → run.sh 로 zero-shot 평가 진행."
echo "============================================================"
