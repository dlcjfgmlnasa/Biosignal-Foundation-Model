#!/bin/bash
# Intracranial Hypertension — LoRA 외삽(context-length extrapolation) 평가용 데이터 준비
#
# 목적: 동일 horizon(30min)에서 여러 input window(60/300/600/1200/1800s)의 데이터를
#   만들어, 한 window 로 학습한 LoRA 가 학습 때 보지 못한 다른 window 길이로
#   외삽되는 능력을 평가한다. prepare_data 는 --window-secs 를 다중값으로 받아
#   window sweep 을 한 번에 생성한다(label=ICP>20mmHg ≥1min, abp+ecg+icp 고정).
#
# 사전: waveform 이 이미 $WAVEFORM_DIR 에 받아져 있어야 한다(없으면 bash/prepare_data.sh
#   의 scan/download 단계 먼저 수행). 이 스크립트는 prepare(.pt 생성) 단계만 한다.
#
# 사용법:
#   bash downstream/acute_event/intracranial_hypertension/bash/prepare_data_extrapolation.sh
#   # env override:
#   WINDOW_SECS="60 300 600 1200 1800" HORIZON_MINS=30 OUT_DIR=... bash ...

set -e

# canonical 경로는 k-mimic-/bio_fm 아래 (updown 기본은 stale — memory
# project_kmimic_bio_fm_paths). 모두 ${VAR:-default} 라 환경변수로 override 가능.
WAVEFORM_DIR="${WAVEFORM_DIR:-/home/coder/workspace/updown/raw/mimic3-waveform-ich}"
OUT_DIR="${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/intracranial_hypertension}"
INPUT_SIGNALS="${INPUT_SIGNALS:-abp ecg icp}"
WINDOW_SECS="${WINDOW_SECS:-60 300 600 1200 1800}"   # 외삽 평가용 multi-window sweep
HORIZON_MINS="${HORIZON_MINS:-30}"                   # horizon 고정
N_FOLDS="${N_FOLDS:-5}"

echo "============================================================"
echo "  Intracranial Hypertension — Extrapolation Data Preparation"
echo "  Waveform: $WAVEFORM_DIR"
echo "  Output:   $OUT_DIR"
echo "  Signals:  $INPUT_SIGNALS"
echo "  Windows:  $WINDOW_SECS   (외삽 평가 sweep)"
echo "  Horizon:  $HORIZON_MINS min"
echo "  Folds:    $N_FOLDS"
echo "============================================================"

# WINDOW_SECS / HORIZON_MINS / INPUT_SIGNALS 는 다중값이라 일부러 unquoted(word-split).
python -m downstream.acute_event.intracranial_hypertension.prepare_data \
    --waveform-dir "$WAVEFORM_DIR" \
    --input-signals $INPUT_SIGNALS \
    --window-secs $WINDOW_SECS \
    --horizon-mins $HORIZON_MINS \
    --n-folds $N_FOLDS \
    --out-dir "$OUT_DIR"

echo -e "\n============================================================"
echo "  Done! Output: $OUT_DIR"
echo "  생성된 window 별 prefix: intracranial_hypertension_icp_w{60,300,600,1200,1800}s_h${HORIZON_MINS}min_fold*"
echo "============================================================"
