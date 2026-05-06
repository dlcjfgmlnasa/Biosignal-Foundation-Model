#!/bin/bash
# Intracranial Hypertension Detection — 데이터 준비
# Paper Table S7 (Task 3) 정렬: 3 combos × 4 windows × 3 horizons
#   Combos:   ABP+ICP / ABP+ICP+ECG / ABP+ICP+ECG+CVP
#   Horizons: 10 / 15 / 30 min ahead
#   Label:    ICP > 20 mmHg ≥ 1 min
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

RECORDS_FILE="${RECORDS_FILE:-/home/coder/workspace/updown/bio_fm/downstream/classification/sepsis/RECORDS-waveforms}"
ICP_RECORDS="${ICP_RECORDS:-downstream/acute_event/intracranial_hypertension/ICP-RECORDS}"
WAVEFORM_DIR="${WAVEFORM_DIR:-/home/coder/workspace/updown/bio_fm/data/raw/mimic3-waveform-ich}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/intracranial_hypertension}"
WINDOWS="${WINDOWS:-60 180 300 600}"
HORIZONS="${HORIZONS:-10 15 30}"
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

# Step 3: 3 combos paired comparison
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

# Paper Table S7: ABP+ICP / ABP+ICP+ECG / ABP+ICP+ECG+CVP
run_combo "1/3" "abp icp"
run_combo "2/3" "abp icp ecg"
run_combo "3/3" "abp icp ecg cvp"

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "  3 combos × $(echo $WINDOWS | wc -w) windows × $(echo $HORIZONS | wc -w) horizons"
echo "============================================================"
