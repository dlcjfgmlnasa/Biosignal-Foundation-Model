#!/bin/bash
# Intracranial Hypertension Detection — 데이터 준비
# Paper Table S7 (Task 3) 정렬: 3 combos × 4 windows × 3 horizons
#   Combos:   ABP+ICP / ABP+ICP+ECG / ABP+ICP+ECG+CVP
#   Horizons: 10 / 15 / 30 min ahead
#   Label:    ICP > 22 mmHg ≥ 1 min (BTF 4th ed., Carney 2017)
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
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/intracranial_hypertension}"
WINDOWS="${WINDOWS:-600}"   # 600s=10min: Table 3 #3 canonical 단일 (10분=모델 최대 컨텍스트). S7 sweep 은 env override
HORIZONS="${HORIZONS:-30}"  # 30min: Table 3 #3 canonical 단일 horizon
STRIDE="${STRIDE:-30}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "============================================================"
echo "  Intracranial Hypertension — Data Preparation"
echo "  Waveform: $WAVEFORM_DIR"
echo "  Output:   $OUT_DIR"
echo "  Windows:  $WINDOWS"
echo "  Horizons: $HORIZONS"
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
        --out-dir "$OUT_DIR"
}

# Table 3 canonical: ABP+ICP+ECG 만 생성.
run_combo "canonical" "abp icp ecg"
# S7 modality-subset ablation 필요 시 아래 주석 해제 (+ WINDOWS/HORIZONS sweep env override):
# run_combo "subset-2ch" "abp icp"
# run_combo "subset-4ch" "abp icp ecg cvp"

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "  canonical: ABP+ICP+ECG, ${WINDOWS}s window, ${HORIZONS}min horizon"
echo "============================================================"
