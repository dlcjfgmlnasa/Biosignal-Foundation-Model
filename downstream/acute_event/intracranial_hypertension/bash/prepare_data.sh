#!/bin/bash
# Intracranial Hypertension Detection — 데이터 준비
#
# Step 1: ICP 레코드 스캔 (헤더만 읽어 ICP 채널 존재 확인)
# Step 2: ICP 레코드 다운로드
# Step 3: 윈도우 추출 + 라벨링 (.pt 생성)
#
# 사용법:
#   bash downstream/classification/intracranial_hypertension/bash/prepare_data.sh

set -e

RECORDS_FILE=/home/coder/workspace/updown/bio_fm/downstream/classification/sepsis/RECORDS-waveforms
ICP_RECORDS=downstream/classification/intracranial_hypertension/ICP-RECORDS
WAVEFORM_DIR=/home/coder/workspace/updown/bio_fm/data/raw/mimic3-waveform-ich
OUT_DIR=/home/coder/workspace/updown/bio_fm/data/downstream/intracranial_hypertension

echo "============================================================"
echo "  Intracranial Hypertension — Data Preparation"
echo "============================================================"

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

# Step 3: 윈도우 추출 (다양한 window × horizon 조합)
echo -e "\n[Step 3] Extracting windows..."
python -m downstream.acute_event.intracranial_hypertension.prepare_data \
    --waveform-dir "$WAVEFORM_DIR" \
    --input-signals icp \
    --window-secs 30 60 300 600 \
    --horizon-mins 5 10 15 \
    --out-dir "$OUT_DIR"

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "============================================================"
