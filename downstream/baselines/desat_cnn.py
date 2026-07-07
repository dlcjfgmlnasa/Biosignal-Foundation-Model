# -*- coding: utf-8 -*-
"""Intraoperative Desaturation (detection) end-to-end 1D-CNN baseline (SNUH cohort).

CARMEN(frozen / LoRA) 과 **같은 prepared 데이터·5-fold·평가** 위에서 사전학습 없이
task 에 처음부터 지도학습하는 1D-ResNet. 공용 코어(:mod:`ioh_cnn`)의 RAM-resident
fp16 스트리밍 로더·모델·학습/평가/저장을 재사용하고, desaturation 고유의 입력 채널
(환기 파형 CO2+AWP+RESP_Flow — SpO2/PPG/ECG 제외로 label leakage 차단)과
**환자단위 patient_ids grouping** 만 지정한다.

라벨은 SpO2 파형에서 재생성한 concurrent desat(<90%, 아티팩트 게이트 후 any-point);
category 텍스트 라벨은 사용하지 않는다. 소규모(≈167명)·rare(≈1.2%) 이벤트라 코어의
``pos_weight = n_neg/n_pos`` BCE 가 그대로 적용된다
([[project_supervised_baseline_resnet1d]]).

사용법:
    python -m downstream.baselines.desat_cnn \
        --data-path .../desaturation_co2_awp_resp_flow_w30s_h0min \
        --input-signals co2 awp resp_flow --n-folds 5 --fold 0 \
        --epochs 100 --patience 20 --lr 1e-3 --batch-size 128 --device cuda \
        --out-dir .../result/main/desaturation_detection_cnn/co2_awp_resp_flow_w30s_h0min
"""
from __future__ import annotations

import argparse

from downstream.baselines.ioh_cnn import add_common_args, run_cnn_baseline


def main():
    p = add_common_args(
        argparse.ArgumentParser(description="Desaturation detection end-to-end 1D-CNN baseline"),
        default_signals=["co2", "awp", "resp_flow"],
    )
    args = p.parse_args()
    # desaturation: 한 환자(patient)가 여러 window → 환자단위 patient_ids grouping
    # (prepared split 이 patient_ids 미저장이면 case_ids fallback). CARMEN desaturation
    # run.py 의 patient-cluster OOF bootstrap grouping 과 정합.
    run_cnn_baseline(args, task_name="desaturation_cnn", tag="DESAT-CNN",
                     id_fields=("patient_ids", "case_ids"))


if __name__ == "__main__":
    main()
