#!/bin/bash
# ============================================================
#  Desaturation DETECTION — 학습 (원커맨드 wrapper)
# ============================================================
# concurrent desat 검출: 입력 CO2·AWP·RESP_Flow 파형 → "지금 저산소 상태냐" 이진 분류.
# h0 데이터(prepare_detection.sh 산출)를 linear_probe / lora 로 5-fold 학습.
#
# 사전조건: prepare_detection.sh 로 desaturation_co2_awp_resp_flow_w300s_h0min_*.pt 생성.
#
# 사용법 (직렬):
#   CHECKPOINT=... DATA_DIR=... bash downstream/acute_event/desaturation/bash/run_detection.sh
#
# 4-GPU 분산:
#   PRINT_ONLY=1 CHECKPOINT=... DATA_DIR=... \
#     bash downstream/acute_event/desaturation/bash/run_detection.sh > jobs_det.txt
#   python -m downstream.run_sharded --gpus 0,1,2,3 --jobs-file jobs_det.txt
set -e

export HORIZONS_OVERRIDE=0                      # detection = h0 (concurrent, 파일명 _h0min)
export SIGNALS_OVERRIDE=co2_awp_resp_flow      # 환기 파형 3종 (파일명·--input-signals 일치)
# detection 결과는 prediction 과 분리된 폴더로 (미지정 시 기본)
export OUT_DIR="${OUT_DIR:-/home/coder/workspace/k-mimic-/bio_fm/result/main/desaturation_detection}"

DIR="$(cd "$(dirname "$0")" && pwd)"
echo ">>> DETECTION 학습 실행 (HORIZONS=0, SIGNALS=$SIGNALS_OVERRIDE, OUT_DIR=$OUT_DIR)" >&2
bash "$DIR/run.sh"
