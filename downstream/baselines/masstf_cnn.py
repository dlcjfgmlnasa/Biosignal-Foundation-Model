# -*- coding: utf-8 -*-
"""Massive Transfusion end-to-end 1D-CNN baseline (SNUH OR cohort).

CARMEN(frozen / LoRA) 과 **같은 prepared 데이터·5-fold·평가** 위에서 사전학습 없이
task 에 처음부터 지도학습하는 1D-ResNet. 공용 코어(:mod:`ioh_cnn`)의 RAM-resident
fp16 스트리밍 로더·모델·학습/평가/저장을 재사용하고, MT 고유의 입력 채널
(default ABP+PPG)과 **case 단위 subject_ids grouping** 만 지정한다.

문헌표준 1D-ResNet end-to-end 지도학습 비교군 ([[project_supervised_baseline_resnet1d]]).

사용법:
    python -m downstream.baselines.masstf_cnn \
        --data-path .../massive_transfusion_abp_ppg_w600s_h5min \
        --input-signals abp ppg --n-folds 5 --fold 0 \
        --epochs 100 --patience 20 --lr 1e-3 --batch-size 128 --device cuda \
        --out-dir .../result/main/massive_transfusion_cnn/abp_ppg_w600s_h5min
"""
from __future__ import annotations

import argparse

from downstream.baselines.ioh_cnn import add_common_args, run_cnn_baseline


def main():
    p = add_common_args(
        argparse.ArgumentParser(
            description="Massive Transfusion end-to-end 1D-CNN baseline"
        ),
        default_signals=["abp", "ppg"],
    )
    args = p.parse_args()
    # MT: 한 surgical case(subject)가 여러 window → case 단위 subject_ids grouping
    # (없으면 case_ids fallback). CARMEN massive_transfusion run.py 와 동일.
    run_cnn_baseline(args, task_name="massive_transfusion_cnn", tag="MASSTF-CNN",
                     id_fields=("subject_ids", "case_ids"))


if __name__ == "__main__":
    main()
