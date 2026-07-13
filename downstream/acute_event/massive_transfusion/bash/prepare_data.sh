#!/bin/bash
# Massive Transfusion Prediction — 데이터 준비 (SNUH OR, VitalDB .vital).
#
#   study_population.csv(masstf 라벨 + correct_start_time onset) → raw .vital 직접
#   파싱 → Kwon 2024 프로토콜(case당 window 1개) × stratified 5-fold CV.
#
#   sampling=per_case (기본, Kwon 2024): 관측 10분 window 1개/case.
#     - MT+ : onset h분 전에 끝나는 10분 window (h=5/10/15)
#     - 음성: 수술 중 랜덤 valid endpoint 의 10분 window, **전체 음성 사용**(자연 prevalence ~2.3%)
#   최적화: 필요 구간만 crop + 필요 트랙만 로드 → 전체 17,984 파일도 파싱 빠름.
#
# 사용법:
#   nohup bash downstream/acute_event/massive_transfusion/bash/prepare_data.sh \
#       > mt_prepare.log 2>&1 &
#   tail -f mt_prepare.log
#   # band(legacy) 모드로 서브샘플: SAMPLING=band NEG_PER_POS=3 bash ...

set -euo pipefail

# Windows 콘솔(cp949) 유니코드 출력 크래시 방지.
export PYTHONIOENCODING=utf-8

STUDY_CSV=${STUDY_CSV:-F:/Massive_Transfusion_Raw/study_population.csv}
RAW_DIR=${RAW_DIR:-F:/Massive_Transfusion_Raw}
OUT_DIR=${OUT_DIR:-F:/Massive_Transfusion_Downstream}
SIGNALS=${SIGNALS:-abp ppg}
SAMPLING=${SAMPLING:-per_case}   # per_case(Kwon2024) | band(legacy)
WINDOW=${WINDOW:-600}            # per_case 에선 관측 window(10분)
HORIZONS=${HORIZONS:-5 10 15}
MAX_LEAD=${MAX_LEAD:-300}        # band 모드 전용
NEG_PER_POS=${NEG_PER_POS:-3}   # band/서브샘플 전용 (per_case full 은 전체 음성 자동)
NFOLDS=${NFOLDS:-5}
WORKERS=${WORKERS:-16}           # 서버 64 vCPU 면 32 권장
SEED=${SEED:-42}

echo "============================================================"
echo "  Massive Transfusion — prepare_data (sampling=$SAMPLING)"
echo "  Signals=$SIGNALS Window=${WINDOW}s Horizons=$HORIZONS"
echo "  RAW=$RAW_DIR  OUT=$OUT_DIR  Workers=$WORKERS"
echo "============================================================"

python -m downstream.acute_event.massive_transfusion.prepare_data \
    --study-csv "$STUDY_CSV" \
    --raw-dir "$RAW_DIR" \
    --input-signals $SIGNALS \
    --sampling "$SAMPLING" \
    --window-secs "$WINDOW" \
    --horizon-mins $HORIZONS \
    --max-lead-sec "$MAX_LEAD" \
    --neg-per-pos "$NEG_PER_POS" \
    --n-folds "$NFOLDS" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR"

echo "  Done → $OUT_DIR"
