# -*- coding: utf-8 -*-
"""Cardiac Arrest end-to-end 1D-CNN baseline (SCOPE cohort).

CARMEN(frozen / LoRA) 과 **같은 prepared 데이터·5-fold·평가** 위에서 사전학습 없이
task 에 처음부터 지도학습하는 1D-ResNet. 공용 코어(:mod:`ioh_cnn`)의 RAM-resident
fp16 스트리밍 로더·모델·학습/평가/저장을 재사용하고, arrest 고유의 입력 채널
(default ECG+PPG)과 **환자단위 subject_ids grouping** 만 지정한다.

SCOPE 논문 baseline(PPG-only 1D-ResNet, AUROC 0.888)과 문헌표준 아키텍처를 맞춘
end-to-end 지도학습 비교군 ([[project_supervised_baseline_resnet1d]]).

사용법:
    python -m downstream.baselines.arrest_cnn \
        --data-path .../scope_arrest_ecg_ppg_w600s_h30min \
        --input-signals ecg ppg --n-folds 5 --fold 0 \
        --epochs 100 --patience 20 --lr 1e-3 --batch-size 128 --device cuda \
        --out-dir .../result/main/cardiac_arrest_cnn/ecg_ppg_w600s_h30min
"""
from __future__ import annotations

import argparse

from downstream.baselines.ioh_cnn import add_common_args, run_cnn_baseline


def main():
    p = add_common_args(
        argparse.ArgumentParser(description="Cardiac Arrest end-to-end 1D-CNN baseline"),
        default_signals=["ecg", "ppg"],
    )
    args = p.parse_args()
    # arrest: 한 환자(subject)가 여러 window → 환자단위 subject_ids grouping
    # (없으면 case_ids fallback). CARMEN arrest run.py 와 동일.
    run_cnn_baseline(args, task_name="cardiac_arrest_cnn", tag="ARREST-CNN",
                     id_fields=("subject_ids", "case_ids"))


if __name__ == "__main__":
    main()
