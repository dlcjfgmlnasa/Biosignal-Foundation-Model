#!/bin/bash
# Intraoperative Hypoxemia Prediction — 데이터 준비 (VitalDB Open, window-level).
#
# 라벨: 미래 horizon 구간에 SpO2 < 92% 가 ≥60초 지속되면 positive.
#   임계 92% = Lundberg et al., Nat Biomed Eng 2018 (Prescience) 및 WHO 중재 수준.
#   Prescience 는 numeric AIMS 시계열로 5분 전 예측 — 우리는 raw waveform 으로 같은 문제를 푼다.
# 입력: parsed .pt 파형 (ECG/PPG; CO2/AWP 는 파싱 부재 — 아래 주석). SpO2 numeric 은
#   라벨 전용이며 입력에 넣지 않는다 (넣으면 persistence baseline 이 압도 — SWIFT 계열).
#
# 시간 정렬: parser manifest 의 start_sample(dtstart 원점 100Hz) 로 SpO2 1Hz trend 를 인덱싱.
# Baseline 가드: 예측 시점에 이미 SpO2<92% 인 윈도우는 제외(기본) — 지속성 누수 차단.
#
# ⚠ 먼저 코호트 크기부터 재세요 (co2/awp 를 required 로 두면 전신마취+capnography 케이스로 좁혀짐):
#   MAX_SUBJECTS=200 bash downstream/acute_event/hypoxemia/bash/prepare_data.sh
#   → 로그의 "intersection intervals / subject(s)" 와 "Patient-level positive" 확인 후 전체 실행.
#
# 사용법:
#   bash downstream/acute_event/hypoxemia/bash/prepare_data.sh
#   (env override: DATA_DIR=... RAW_DIR=... OUT_DIR=... REQUIRED="ecg ppg" bash ...)

set -e

REPO_ROOT="${REPO_ROOT:-/home/coder/workspace/Biosignal-Foundation-Model}"
UPDOWN_ROOT="${UPDOWN_ROOT:-/home/coder/workspace/updown}"
BIOFM_ROOT="${BIOFM_ROOT:-/home/coder/workspace/k-mimic-/bio_fm}"
DATA_DIR="${DATA_DIR:-${UPDOWN_ROOT}/parser/vitaldb}"
RAW_DIR="${RAW_DIR:-/home/coder/workspace/datasets/vitaldb_open/1.0.0}"
OUT_DIR="${OUT_DIR:-${BIOFM_ROOT}/data/downstream/hypoxemia}"

WINDOWS="${WINDOWS:-300}"              # canonical: 5 min input window
HORIZONS="${HORIZONS:-5 10 15}"        # Table 3(a) 공통 lead-time 격자
# ⚠ CO2/AWP 는 현재 parsed VitalDB 에 사실상 없다 (300 subject 중 1명, 2026-07-10 실측).
#   원인 후보: 파서 품질 게이트(co2 flatline≤0.3·clip≤0.1)가 capnogram 의 정상적인
#   plateau/baseline 을 탈락시킴. probe_raw_tracks.py 로 확인할 것.
#   → canonical 은 ECG+PPG. 재파싱으로 CO2 를 살리면 그때 조합을 넓힌다.
SIGNALS="${SIGNALS:-ecg ppg}"          # 입력 파형 (라벨은 SpO2 numeric)
REQUIRED="${REQUIRED:-ecg ppg}"        # SIGNALS ⊆ REQUIRED 필수 (아니면 채널 소실)
N_FOLDS="${N_FOLDS:-5}"
WORKERS="${WORKERS:-16}"               # 네트워크 마운트 → SpO2 로딩 ThreadPool
SPO2_THRESHOLD="${SPO2_THRESHOLD:-92}"
SUSTAINED_SEC="${SUSTAINED_SEC:-60}"
MAX_SUBJECTS="${MAX_SUBJECTS:-}"       # 설정 시 dry-run (코호트 크기 확인용)

EXTRA_ARGS=""
if [ -n "$MAX_SUBJECTS" ]; then
    EXTRA_ARGS="--max-subjects $MAX_SUBJECTS"
    OUT_DIR="${OUT_DIR}_dryrun"
fi

echo "============================================================"
echo "  Intraoperative Hypoxemia — Data Preparation (VitalDB Open)"
echo "  Parsed:    $DATA_DIR"
echo "  Raw vital: $RAW_DIR"
echo "  Output:    $OUT_DIR"
echo "  Input:     $SIGNALS   (required: $REQUIRED)"
echo "  Window:    ${WINDOWS}s | Horizons: ${HORIZONS} min | Folds: $N_FOLDS"
echo "  Label:     SpO2 < ${SPO2_THRESHOLD}% sustained >= ${SUSTAINED_SEC}s"
if [ -n "$MAX_SUBJECTS" ]; then echo "  ⚠ DRY-RUN: max-subjects=$MAX_SUBJECTS"; fi
echo "============================================================"

# SIGNALS / REQUIRED / HORIZONS / EXTRA_ARGS 는 다중값이라 일부러 unquoted (word-split).
python -m downstream.acute_event.hypoxemia.prepare_data \
    --data-dir "$DATA_DIR" \
    --raw-dir "$RAW_DIR" \
    --input-signals $SIGNALS \
    --required-signals $REQUIRED \
    --window-secs $WINDOWS \
    --horizon-mins $HORIZONS \
    --n-folds $N_FOLDS \
    --workers $WORKERS \
    --spo2-threshold $SPO2_THRESHOLD \
    --sustained-sec $SUSTAINED_SEC \
    --out-dir "$OUT_DIR" \
    $EXTRA_ARGS

SIG_TOKEN=$(echo "$SIGNALS" | tr ' ' '_')
echo -e "\n============================================================"
echo "  Done! Output prefix: $OUT_DIR/hypoxemia_${SIG_TOKEN}_w${WINDOWS}s_h{H}min"
echo "============================================================"
