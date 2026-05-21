#!/bin/bash
# Mechanical Ventilation Need Prediction (#11) — MIMIC-III Waveform Matched
# Label: ICU 입원 24h 내 invasive ventilation 시작 여부
#
# v2(2026-05-21): 입력 RESP 는 prepare_data 내부 MIMIC_SIGNAL_MAP 에서 자동으로
#   resp_impedance(8) 로 매핑된다(CLI signal 인자 없음). #11 임상 정의가 삽관 전
#   비침습 흉부 임피던스이므로 RESP=impedance 가 정합. SSOT: data/spatial_map.py.
#   AWP/CO2 는 명시적 제외(insertion 전 ventilator/capnography 없음 — 임상적 불가능).
#
# 주의: 본 스크립트의 경로는 서버 기본값(env 로 override 가능). 실행 전 확인할 것.

set -e

REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
UPDOWN_ROOT="${UPDOWN_ROOT:-/home/coder/workspace/updown}"
BIOFM_ROOT="${BIOFM_ROOT:-${UPDOWN_ROOT}/bio_fm}"
COHORT_CSV="${COHORT_CSV:-${REPO_ROOT}/downstream/outcome/vent_need/bquxjob_6b9a6adb_19e203a5e97.csv}"
WAVEFORM_DIR="${WAVEFORM_DIR:-${UPDOWN_ROOT}/raw/mimic3-waveform-vent-need}"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/vent_need}"
WINDOW_SEC="${WINDOW_SEC:-600}"
STRIDE_SEC="${STRIDE_SEC:-300}"

echo "============================================================"
echo "  Mech Vent Need (#11) — MIMIC-III Waveform, 24h horizon"
echo "  Cohort:      $COHORT_CSV"
echo "  Waveform:    $WAVEFORM_DIR"
echo "  Output:      $OUT_DIR"
echo "  Window/Strd: ${WINDOW_SEC}s / ${STRIDE_SEC}s  (RESP→resp_impedance 자동)"
echo "============================================================"

python -m downstream.outcome.vent_need.prepare_data \
    --cohort-csv "$COHORT_CSV" \
    --waveform-dir "$WAVEFORM_DIR" \
    --out-dir "$OUT_DIR" \
    --window-sec "$WINDOW_SEC" \
    --stride-sec "$STRIDE_SEC"

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "============================================================"
