#!/bin/bash
# Outcome Prediction — Mortality (patient-level, SCOPE) — 5-fold linear_probe + LoRA.
#
# cardiac_arrest run.sh 와 동일 엔진(TASK=death). prefix scope_death_w300s 로드,
# 결과는 mortality_outcome 로 분리 저장.
#
# 사용법:
#   NPROC=4 nohup bash downstream/outcome/cardiac_arrest/bash/run_mortality.sh \
#     > mortality_outcome.log 2>&1 &

export TASK=death
export DATA_DIR=${DATA_DIR:-/home/coder/workspace/k-mimic-/bio_fm/data/downstream/mortality_outcome}
export OUT_DIR=${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/mortality_outcome}

exec bash "$(dirname "$0")/run.sh"
