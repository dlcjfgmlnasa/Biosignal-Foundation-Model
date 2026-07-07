#!/bin/bash
# ============================================================
#  Desaturation DETECTION — 데이터 준비 (원커맨드 wrapper)
# ============================================================
# concurrent desat 분류: 입력 = 환기 파형 CO2·AWP·RESP_Flow (SpO2/PPG는 라벨용만).
# 라벨 = 입력 window 내 SpO2<90% (아티팩트 게이트 후). 파형에서 "지금 저산소 상태"를 판별.
# CNN sanity check: 간단 1D-CNN 도 AUROC ~0.8 (신호 확인됨) → CARMEN 은 더 안정적 기대.
# 산출: desaturation_co2_awp_resp_flow_w300s_h0min_fold{0-4}_{train,val,test}.pt (15개)
#
# 사용법:
#   VITAL_DIR=... META_XLSX=... OUT_DIR=... \
#   bash downstream/acute_event/desaturation/bash/prepare_detection.sh
#
# override 가능: THRESHOLD(기본 90) / STRIDE(기본 30) / WINDOWS(기본 300) / N_FOLDS(5)
set -e

export TASK_MODE=detection
export INPUT="${INPUT:-co2 awp resp_flow}"     # 환기 파형만 (SpO2/PPG 제외)
export THRESHOLD="${THRESHOLD:-90}"            # concurrent desat 임계 (CNN 검증값)
# window=30s: CNN window sweep(10/30/60/300s) 최적점. AUROC 0.87/lift 12× (뒤집힌 U,
#   300s=희석·과적합 0.76, 10s=morphology 부족 0.81, 30~60s sweet spot). stride=window.
export WINDOWS="${WINDOWS:-30}"
export STRIDE="${STRIDE:-30}"

DIR="$(cd "$(dirname "$0")" && pwd)"
echo ">>> DETECTION 모드로 prepare_data 실행 (TASK_MODE=$TASK_MODE, INPUT='$INPUT', THRESHOLD=$THRESHOLD)" >&2
bash "$DIR/prepare_data.sh"
