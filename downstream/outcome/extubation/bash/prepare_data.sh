#!/bin/bash
# Extubation Failure Prediction (#10) — MIMIC-III Waveform Matched
# Cohort: BigQuery sql/01_extubation_cohort.sql 결과 CSV
# Label: 같은 ICU stay 내 발관(vent_end) 후 48h 내 재삽관(reintubation)
#
# v2(2026-05-21): 입력 RESP 는 prepare_data 내부 MIMIC_SIGNAL_MAP 에서 자동으로
#   resp_impedance(8) 로 매핑된다(CLI signal 인자 없음). MIMIC-III 에 flow 채널이
#   없어 RESP/Flow → RESP/Impedance proxy 사용(한계는 paper limitation). SSOT:
#   data/spatial_map.py SIGNAL_KEY_TO_TYPE.
#
# 주의: 본 스크립트의 경로는 서버 기본값(env 로 override 가능). 실행 전 확인할 것.

set -e

REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
UPDOWN_ROOT="${UPDOWN_ROOT:-/home/coder/workspace/updown}"
BIOFM_ROOT="${BIOFM_ROOT:-${UPDOWN_ROOT}/bio_fm}"
COHORT_CSV="${COHORT_CSV:-${REPO_ROOT}/downstream/outcome/extubation/bquxjob_4f1dd355_19e203e8e82.csv}"
WAVEFORM_DIR="${WAVEFORM_DIR:-${UPDOWN_ROOT}/raw/mimic3-waveform-extubation}"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/extubation}"
WINDOW_SEC="${WINDOW_SEC:-600}"
STRIDE_SEC="${STRIDE_SEC:-300}"

echo "============================================================"
echo "  Extubation Failure (#10) — MIMIC-III Waveform"
echo "  Cohort:      $COHORT_CSV"
echo "  Waveform:    $WAVEFORM_DIR"
echo "  Output:      $OUT_DIR"
echo "  Window/Strd: ${WINDOW_SEC}s / ${STRIDE_SEC}s  (RESP→resp_impedance 자동)"
echo "============================================================"

python -m downstream.outcome.extubation.prepare_data \
    --cohort-csv "$COHORT_CSV" \
    --waveform-dir "$WAVEFORM_DIR" \
    --out-dir "$OUT_DIR" \
    --window-sec "$WINDOW_SEC" \
    --stride-sec "$STRIDE_SEC"

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "============================================================"
