#!/bin/bash
# Outcome Prediction — Mortality (patient-level, SCOPE) — 데이터 준비.
#
# cardiac_arrest 과 동일 엔진(prepare_data_scope.py --task death). 코호트만 다름:
#   death(1) vs discharge(0), Cardiac Arrest 제외. 나머지 설계 동일:
#   ECG+PPG, lead 12h, 고정 12h history band [anchor−24h, anchor−12h], max 144 window.
# 저장 prefix = scope_death_w300s (arrest 데이터와 파일명 충돌 없음).
#
# 사용법:
#   bash downstream/outcome/cardiac_arrest/bash/prepare_data_mortality.sh
#   (env override: METADATA=... VITAL_DIR=... OUT_DIR=... bash ...)

# mortality 전용 기본 out-dir (arrest 와 분리). 그 외는 prepare_data.sh 기본을 승계.
export TASK=death
export OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/mortality_outcome}

exec bash "$(dirname "$0")/prepare_data.sh"
