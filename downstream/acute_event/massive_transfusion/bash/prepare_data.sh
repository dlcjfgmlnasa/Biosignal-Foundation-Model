#!/bin/bash
# Massive Transfusion Prediction — 데이터 준비 (SNUH OR, VitalDB .vital).
#
#   study_population.csv(masstf 라벨 + correct_start_time onset) → raw .vital 직접
#   파싱 → anchor-relative band window (5/10/15분 horizon) × stratified 5-fold CV.
#
#   음성(17,570)은 전량 파싱 비현실적(≈374GB) → --neg-per-pos 배만 seeded 서브샘플.
#   prevalence 인위적 → AUROC 주지표(true≈2.3% 병기).
#
# 사용법:
#   nohup bash downstream/acute_event/massive_transfusion/bash/prepare_data.sh \
#       > mt_prepare.log 2>&1 &
#   tail -f mt_prepare.log

set -euo pipefail

# Windows 콘솔(cp949) 유니코드 출력 크래시 방지.
export PYTHONIOENCODING=utf-8

STUDY_CSV=${STUDY_CSV:-F:/Massive_Transfusion_Raw/study_population.csv}
RAW_DIR=${RAW_DIR:-F:/Massive_Transfusion_Raw}
OUT_DIR=${OUT_DIR:-F:/Massive_Transfusion_Downstream}
SIGNALS=${SIGNALS:-abp ppg}
WINDOW=${WINDOW:-600}
HORIZONS=${HORIZONS:-5 10 15}
MAX_LEAD=${MAX_LEAD:-300}
NEG_PER_POS=${NEG_PER_POS:-3}
NFOLDS=${NFOLDS:-5}
WORKERS=${WORKERS:-8}
SEED=${SEED:-42}

echo "============================================================"
echo "  Massive Transfusion — prepare_data"
echo "  Signals=$SIGNALS Window=${WINDOW}s Horizons=$HORIZONS Neg/Pos=$NEG_PER_POS"
echo "  RAW=$RAW_DIR  OUT=$OUT_DIR  Workers=$WORKERS"
echo "============================================================"

python -m downstream.acute_event.massive_transfusion.prepare_data \
    --study-csv "$STUDY_CSV" \
    --raw-dir "$RAW_DIR" \
    --input-signals $SIGNALS \
    --window-secs "$WINDOW" \
    --horizon-mins $HORIZONS \
    --max-lead-sec "$MAX_LEAD" \
    --neg-per-pos "$NEG_PER_POS" \
    --n-folds "$NFOLDS" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR"

echo "  Done → $OUT_DIR"
