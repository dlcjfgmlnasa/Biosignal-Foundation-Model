#!/bin/bash
# Blood Pressure (SBP/DBP) Estimation — VitalDB, 회귀 task, 5-fold patient-level CV.
# 입력 PPG(+ECG), 라벨은 같은 window 의 ABP beat 단위 SBP/DBP. horizon 없음(즉시 추정).
# PPG-FM(CSFM·PaPaGei·AnyPPG 등)의 표준 BP 벤치마크와 직접 비교용.
#
# 사용법:
#   bash downstream/vital_sign/blood_pressure/bash/prepare_data.sh
#   INPUTS="ecg ppg" bash ...          # 멀티모달 입력
#   WINDOWS="10 30" bash ...           # window sweep
#   N_FOLDS=1 bash ...                 # single split
#   env override: DATA_DIR / OUT_DIR / INPUTS / WINDOWS / N_FOLDS / REQUIRED

set -e

DATA_DIR="${DATA_DIR:-/home/coder/workspace/updown/parser/vitaldb}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/updown/bio_fm/data/downstream/bp_estimation}"
INPUTS="${INPUTS:-ppg}"                # 입력 modality (ppg | "ecg ppg")
WINDOWS="${WINDOWS:-30}"               # canonical: 30s window
N_FOLDS="${N_FOLDS:-5}"
REQUIRED="${REQUIRED:-ecg ppg abp}"    # paired: 모든 조합 동일 환자
STRIDE="${STRIDE:-30}"

echo "============================================================"
echo "  Blood Pressure (SBP/DBP) Estimation — VitalDB, ${N_FOLDS}-fold CV"
echo "  Data:    $DATA_DIR"
echo "  Output:  $OUT_DIR"
echo "  Input:   $INPUTS   (label: ABP, 입력 제외)"
echo "  Windows: $WINDOWS   Stride: ${STRIDE}s"
echo "============================================================"

python -m downstream.vital_sign.blood_pressure.prepare_data \
    --data-dir "$DATA_DIR" --input-signals $INPUTS \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS --stride-sec $STRIDE \
    --n-folds $N_FOLDS --out-dir "$OUT_DIR"

echo -e "\n============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
