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
# ICH 는 prediction 전용 (detection 폐기 — 미래 horizon IH 예측).
TASK_MODE="prediction"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/intracranial_hypertension_${TASK_MODE}}"
WINDOWS="${WINDOWS:-900}"      # 15min(900s) 입력 window (prediction 표준)
HORIZONS="${HORIZONS:-5 15 30}"  # 5/15/30분 전 예측
STRIDE="${STRIDE:-30}"      # prediction: 30s stride (15min window 겹침 완화 — 데이터/메모리 폭증 방지)
MODE="${MODE:-unbiased}"    # unbiased(현실적·기본) | biased(과대평가·선행비교)
# 저장 dtype: float16 이 기본(디스크·RAM 절반). float32 로 바꾸려면 SIGNAL_DTYPE=float32.
SIGNAL_DTYPE="${SIGNAL_DTYPE:-float16}"
# 입력 신호: prediction=ABP+ICP+ECG (미래 horizon 예측이라 현재 ICP 를 입력에 포함).
INPUT_SIGNALS="${INPUT_SIGNALS:-abp icp ecg}"
MIN_GAP="${MIN_GAP:-1200}"  # biased 모드 sparse 간격(초)
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
# 라벨 지속시간: SUSTAINED_SEC = "horizon 구간 내 ICP>임계 지속(초)". ICP 임계(20)는 유지.
#   VALID_RATIO 미설정 시 prepare_data.py 기본값 사용.
SUSTAINED_SEC="${SUSTAINED_SEC:-60}"
LABEL_ARGS=""
[ -n "$SUSTAINED_SEC" ] && LABEL_ARGS="$LABEL_ARGS --sustained-sec $SUSTAINED_SEC"
[ -n "$VALID_RATIO" ]   && LABEL_ARGS="$LABEL_ARGS --valid-ratio $VALID_RATIO"

echo "============================================================"
echo "  Intracranial Hypertension — Data Preparation"
echo "  Waveform: $WAVEFORM_DIR"
echo "  Output:   $OUT_DIR"
echo "  Task:     $TASK_MODE"
echo "  Input:    $INPUT_SIGNALS  (dtype=$SIGNAL_DTYPE)"
echo "  Windows:  ${WINDOWS}s (15min prediction)"
echo "  Horizons: $HORIZONS min"
echo "  Stride:   ${STRIDE}s"
echo "============================================================"

if [ "$SKIP_DOWNLOAD" != "1" ]; then
    # Step 1: ICP 레코드 스캔
    echo -e "\n[Step 1] Scanning for ICP records..."
    python -m downstream.acute_event.intracranial_hypertension.download_waveforms \
        scan \
        --records-file "$RECORDS_FILE" \
        --out-file "$ICP_RECORDS"

    # Step 2: 다운로드
    echo -e "\n[Step 2] Downloading ICP waveforms..."
    python -m downstream.acute_event.intracranial_hypertension.download_waveforms \
        download \
        --icp-records-file "$ICP_RECORDS" \
        --out-dir "$WAVEFORM_DIR"
else
    echo -e "\n[Step 1-2] SKIP_DOWNLOAD=1 — using existing waveforms at $WAVEFORM_DIR"
fi

# Step 3: prediction canonical 조합 (ABP+ICP+ECG, 15min window)
run_combo() {
    local label="$1"
    local sigs="$2"
    echo -e "\n[Step 3 — $label] $sigs"
    python -m downstream.acute_event.intracranial_hypertension.prepare_data \
        --waveform-dir "$WAVEFORM_DIR" \
        --input-signals $sigs \
        --window-secs $WINDOWS \
        --horizon-mins $HORIZONS \
        --stride-sec $STRIDE \
        --signal-dtype $SIGNAL_DTYPE \
        --min-sample-gap-sec $MIN_GAP --sampling-mode $MODE \
        --task-mode $TASK_MODE \
        $LABEL_ARGS \
        --out-dir "$OUT_DIR"
}

# canonical: prediction 입력 ABP+ICP+ECG (INPUT_SIGNALS env 로 override 가능).
run_combo "canonical" "$INPUT_SIGNALS"
# S7 modality-subset ablation 필요 시 아래 주석 해제 (+ WINDOWS/HORIZONS sweep env override):
# run_combo "subset-2ch" "abp icp"
# run_combo "subset-4ch" "abp icp ecg cvp"

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "  canonical: input ${INPUT_SIGNALS}, ${WINDOWS}s window, dtype=${SIGNAL_DTYPE}, task=${TASK_MODE}"
echo "============================================================"
