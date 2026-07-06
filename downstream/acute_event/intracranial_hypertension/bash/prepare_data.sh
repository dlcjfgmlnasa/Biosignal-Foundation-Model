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
# task-mode: detection(aICP식 동시 탐지, 입력=ABP+ECG·ICP=라벨전용, circularity 차단) 기본.
#   prediction(미래 horizon IH 예측, 현재-ICH 제외)로 바꾸려면 TASK_MODE=prediction.
TASK_MODE="${TASK_MODE:-detection}"
# ⚠ 출력 디렉토리를 task_mode 로 완전히 분리한다 → detection/prediction 산출물이
#   서로 다른 폴더에 저장(파일명 분리에 더해 디렉토리까지 분리). run.sh 도 동일 규칙.
#   detection  → .../data/downstream/intracranial_hypertension_detection
#   prediction → .../data/downstream/intracranial_hypertension_prediction
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/intracranial_hypertension_${TASK_MODE}}"
WINDOWS="${WINDOWS:-10}"       # 10s 입력 window (aICP npj DM 세그먼트 기준; detection=동시 IH 탐지)
HORIZONS="${HORIZONS:-5 15 30}"  # (detection 이면 무시·h0min) prediction 모드 시 5/15/30분 전
STRIDE="${STRIDE:-10}"      # detection: window(10s)와 동일 → 연속 타일링(겹침·건너뜀 없이 전구간 커버)
MODE="${MODE:-unbiased}"    # unbiased(현실적·기본) | biased(과대평가·선행비교)
MIN_GAP="${MIN_GAP:-1200}"  # biased 모드 sparse 간격(초)
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
# detection 10s window 라벨: SUSTAINED_SEC 기본 10 = "이 10s 구간 평균 ICP>임계"(aICP식 동시 탐지).
#   코드가 sustained 를 label 구간(bucket 수)으로 clamp 하므로 10 이상은 10s window 에선 10 과 동일.
#   VALID_RATIO 미설정 시 prepare_data.py 기본값 사용. ICP 임계(20)는 유지.
SUSTAINED_SEC="${SUSTAINED_SEC:-10}"
LABEL_ARGS=""
[ -n "$SUSTAINED_SEC" ] && LABEL_ARGS="$LABEL_ARGS --sustained-sec $SUSTAINED_SEC"
[ -n "$VALID_RATIO" ]   && LABEL_ARGS="$LABEL_ARGS --valid-ratio $VALID_RATIO"

echo "============================================================"
echo "  Intracranial Hypertension — Data Preparation"
echo "  Waveform: $WAVEFORM_DIR"
echo "  Output:   $OUT_DIR"
echo "  Task:     $TASK_MODE"
echo "  Windows:  ${WINDOWS}s"
if [ "$TASK_MODE" = "detection" ]; then
    echo "  Horizons: (detection — 동시 라벨, horizon 무시 → h0min)"
else
    echo "  Horizons: $HORIZONS"
fi
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

# Step 3: Table 3 #3 canonical 단일 조합 (ABP+ICP+ECG)
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
        --min-sample-gap-sec $MIN_GAP --sampling-mode $MODE \
        --task-mode $TASK_MODE \
        $LABEL_ARGS \
        --out-dir "$OUT_DIR"
}

# Table 3 canonical (detection): 입력 ABP+ECG (ICP 는 라벨 전용 — circularity 차단, aICP식).
#   TASK_MODE=detection 이면 horizon 무시(파일명 h0min). prediction 모드로 쓰려면
#   TASK_MODE=prediction 로 두고 입력에 icp 를 포함(구 canonical "abp icp ecg").
run_combo "canonical" "abp ecg"
# S7 modality-subset ablation 필요 시 아래 주석 해제 (+ WINDOWS/HORIZONS sweep env override):
# run_combo "subset-2ch" "abp icp"
# run_combo "subset-4ch" "abp icp ecg cvp"

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "  canonical: input ABP+ECG (ICP=label), ${WINDOWS}s window, task=${TASK_MODE}"
echo "============================================================"
