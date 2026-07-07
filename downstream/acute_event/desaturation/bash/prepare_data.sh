#!/bin/bash
# Intraoperative Desaturation Prediction — SNUH ventilation_eventcases (self-contained)
# Task: Acute Event — respiratory desaturation *prediction* (detection 아님)
# Label: future horizon 내 SpO2 < 92% "at any point" + 아티팩트 게이트 (category 텍스트 미사용)
#        = 소아논문(PLOS One 2023) 프로토콜: any-point 는 반드시 품질게이트와 세트.
#        게이트 = SpO2∈[50,100] · |pulseHR-ecgHR|≤20% · PI≥0.3. threshold 92=Prescience/WHO.
# Input: CO2 + AWP + RESP_Flow + PPG (ECG 제외). PPG 포함=competitive(문헌 관행). RESP_Flow SNUH 고유.
# Source: 188 raw .vital (Intellivue 파형 125~250Hz → 100Hz 다운샘플)
#
# N: 게이트 후 재측정 필요. dwell variant 는 SUSTAINED=10/30. 게이트 끄기: NO_GATE=1.
#
# 사용법:
#   bash downstream/acute_event/desaturation/bash/prepare_data.sh
#   SUSTAINED=30 bash .../prepare_data.sh          # 완화판
#   EXTERNAL_TEST=1 bash .../prepare_data.sh        # VitalDB 학습모델 외부검증용 test-only

set -e

REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
UPDOWN_ROOT="${UPDOWN_ROOT:-/home/coder/workspace/updown}"
BIOFM_ROOT="${BIOFM_ROOT:-${UPDOWN_ROOT}/bio_fm}"

# SNUH ventilation_eventcases 위치 (서버에 업로드한 경로로 override)
VITAL_DIR="${VITAL_DIR:-${UPDOWN_ROOT}/datasets/ventilation_eventcases/data/vital_files}"
META_XLSX="${META_XLSX:-${UPDOWN_ROOT}/datasets/ventilation_eventcases/ventilation_cases.xlsx}"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/desaturation}"

WINDOWS="${WINDOWS:-300}"            # 5 min input window
HORIZONS="${HORIZONS:-1 3 5}"        # 1/3/5min 임상정렬. 15min stress-test 는 HORIZONS="1 3 5 15"
STRIDE="${STRIDE:-30}"
THRESHOLD="${THRESHOLD:-95}"         # SpO2 < 95% (SNUH 소아 PLOS One 2023). 92=Prescience, 90=conservative
SUSTAINED="${SUSTAINED:-1}"          # 1='any point'(게이트와 세트, 기본). dwell variant: 10/30

GATE_ARGS=""
if [ "${NO_GATE:-0}" = "1" ]; then GATE_ARGS="--no-artifact-gate"; fi
N_FOLDS="${N_FOLDS:-5}"
INPUT="${INPUT:-co2 awp resp_flow ppg}"   # CO2·AWP·RESP_Flow·PPG (ECG 제외). override 가능
REQUIRED="${REQUIRED:-co2 awp}"
# prediction=미래 desat 예측(기본) / detection=입력 window 내 concurrent desat 분류(h0)
# detection 권장: TASK_MODE=detection INPUT="co2 awp resp_flow" (PPG 빼야 non-trivial)
TASK_MODE="${TASK_MODE:-prediction}"

EXTRA=""
if [ "${EXTERNAL_TEST:-0}" = "1" ]; then EXTRA="--external-test"; fi

echo "============================================================"
echo "  Intraop Desaturation Prediction — SNUH (${N_FOLDS}-fold CV)"
echo "  Vital:     $VITAL_DIR"
echo "  Meta:      $META_XLSX"
echo "  Output:    $OUT_DIR"
echo "  Window/Horizon: ${WINDOWS}s / ${HORIZONS}min   Stride: ${STRIDE}s"
echo "  Label:     SpO2 < ${THRESHOLD}% sustained >= ${SUSTAINED}s"
echo "  Input:     $INPUT   (required: $REQUIRED)"
echo "  Mode:      ${EXTRA:-kfold}"
echo "============================================================"

cd "$REPO_ROOT"
python -m downstream.acute_event.desaturation.prepare_data \
    --vital-dir "$VITAL_DIR" \
    --meta-xlsx "$META_XLSX" \
    --input-signals $INPUT \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS \
    --horizon-mins $HORIZONS \
    --stride-sec $STRIDE \
    --spo2-threshold $THRESHOLD \
    --sustained-sec $SUSTAINED \
    --task-mode $TASK_MODE \
    --n-folds $N_FOLDS \
    --out-dir "$OUT_DIR" \
    $GATE_ARGS \
    $EXTRA

echo -e "\n============================================================"
echo "  Done! Saved to: $OUT_DIR"
echo "============================================================"
