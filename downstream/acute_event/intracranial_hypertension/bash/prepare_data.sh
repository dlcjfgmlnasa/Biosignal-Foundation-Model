#!/bin/bash
# Intracranial Hypertension Detection — 데이터 준비
# Paper Table 3 #3 (detection, aICP식):
#   Input:    ABP + ECG (두개외 신호만; ICP 는 라벨 전용 → circularity 차단)
#   Window:   10 s — aICP(npj DM)=1s 세그먼트 기준. CARMEN 차별점 = 사전학습
#             multimodal FM 표현 전이(frozen/LoRA) + 다-beat/호흡결합 문맥(1s→10s).
#             (per-window 분류·mean-pool probe. aICP 의 stay-level 빈도 집계와 다름)
#   Label:    동시(same-window) ICP > 20 mmHg 지속(SUSTAINED_SEC) = IH high/low 이진
#             (Czosnyka & Pickard 2004 IH 정의; aICP 는 임계 15 사용)
#
# Step 1: ICP 레코드 스캔 (헤더만 읽어 ICP 채널 존재 확인)
# Step 2: ICP 레코드 다운로드
# Step 3: 윈도우 추출 + 라벨링 (.pt 생성, paired comparison)
#
# 사용법:
#   bash downstream/acute_event/intracranial_hypertension/bash/prepare_data.sh
#
# env override:
#   WAVEFORM_DIR=... OUT_DIR=... WINDOWS=... HORIZONS=... STRIDE=... \
#     bash downstream/acute_event/intracranial_hypertension/bash/prepare_data.sh

set -e

REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
UPDOWN_ROOT="${UPDOWN_ROOT:-/home/coder/workspace/updown}"
BIOFM_ROOT="${BIOFM_ROOT:-/home/coder/workspace/k-mimic-/bio_fm}"
RECORDS_FILE="${RECORDS_FILE:-${REPO_ROOT}/downstream/outcome/sepsis/RECORDS-waveforms}"
ICP_RECORDS="${ICP_RECORDS:-downstream/acute_event/intracranial_hypertension/ICP-RECORDS}"
WAVEFORM_DIR="${WAVEFORM_DIR:-${UPDOWN_ROOT}/raw/mimic3-waveform-ich}"
# ICH 는 prediction 전용 (미래 horizon IH 예측). anchored 산출물도 prediction 이름.
TASK_MODE="prediction"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/intracranial_hypertension_${TASK_MODE}}"
# ── 확정 스펙 (Güiza anchored, 2026-07-08) ──
WINDOWS="${WINDOWS:-1200}"        # 20min(1200s) 입력 window
HORIZONS="${HORIZONS:-5 10 15}"   # 5/10/15분 전 예측
MODE="${MODE:-anchored}"          # anchored(onset-anchor 양성 + 균일 stride 음성)
NEG_STRIDE="${NEG_STRIDE:-1800}"  # 음성 균일 stride 30min → prevalence ~3.5%
EVENT_CONSEC="${EVENT_CONSEC:-5}" # 5×1min median>20mmHg = crisis (Güiza)
# 입력 신호: ABP + ICP (prediction 이라 현재 ICP 를 입력에 포함). label 은 항상 ICP.
INPUT_SIGNALS="${INPUT_SIGNALS:-abp icp}"
# 저장 dtype: float16 이 기본(디스크·RAM 절반).
SIGNAL_DTYPE="${SIGNAL_DTYPE:-float16}"
# 데이터는 이미 flat-numeric(3008477.hea/_0001.dat)로 받아져 있음 → 기본 스킵.
#   (구 matched ICP-RECORDS 다운로드 경로는 이 데이터와 레이아웃 불일치라 미사용.)
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"
LABEL_ARGS=""
[ -n "$VALID_RATIO" ] && LABEL_ARGS="$LABEL_ARGS --valid-ratio $VALID_RATIO"

echo "============================================================"
echo "  Intracranial Hypertension — Data Preparation (Güiza anchored)"
echo "  Waveform: $WAVEFORM_DIR"
echo "  Output:   $OUT_DIR"
echo "  Input:    $INPUT_SIGNALS  (dtype=$SIGNAL_DTYPE)"
echo "  Window:   ${WINDOWS}s (20min)   Horizons: $HORIZONS min"
echo "  Sampling: $MODE  neg-stride=${NEG_STRIDE}s  event-consec=${EVENT_CONSEC}"
echo "============================================================"

if [ "$SKIP_DOWNLOAD" != "1" ]; then
    # (선택) matched subset ICP-RECORDS 스캔+다운로드. 현 데이터는 flat-numeric 이라
    # 기본 스킵. matched 데이터를 새로 받을 때만 SKIP_DOWNLOAD=0.
    echo -e "\n[Step 1] Scanning for ICP records..."
    python -m downstream.acute_event.intracranial_hypertension.download_waveforms \
        scan --records-file "$RECORDS_FILE" --out-file "$ICP_RECORDS"
    echo -e "\n[Step 2] Downloading ICP waveforms..."
    python -m downstream.acute_event.intracranial_hypertension.download_waveforms \
        download --icp-records-file "$ICP_RECORDS" --out-dir "$WAVEFORM_DIR"
else
    echo -e "\n[Step 1-2] SKIP_DOWNLOAD=1 — using existing waveforms at $WAVEFORM_DIR"
fi

# Step 3: anchored canonical (ABP+ICP, 20min window, --scan-dir flat-numeric)
run_combo() {
    local label="$1"
    local sigs="$2"
    echo -e "\n[Step 3 — $label] $sigs"
    python -m downstream.acute_event.intracranial_hypertension.prepare_data \
        --waveform-dir "$WAVEFORM_DIR" \
        --input-signals $sigs \
        --window-secs $WINDOWS \
        --horizon-mins $HORIZONS \
        --sampling-mode $MODE \
        --neg-stride-sec $NEG_STRIDE \
        --event-consec $EVENT_CONSEC \
        --scan-dir \
        --signal-dtype $SIGNAL_DTYPE \
        --task-mode $TASK_MODE \
        $LABEL_ARGS \
        --out-dir "$OUT_DIR"
}

# canonical: anchored 입력 ABP+ICP (INPUT_SIGNALS env 로 override 가능).
run_combo "canonical" "$INPUT_SIGNALS"

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "  canonical: input ${INPUT_SIGNALS}, ${WINDOWS}s window, dtype=${SIGNAL_DTYPE}, task=${TASK_MODE}"
echo "============================================================"
