# -*- coding: utf-8 -*-
"""ICH(Intracranial Hypertension) end-to-end 1D-CNN baseline.

CARMEN(frozen / LoRA) 과 **같은 prepared 데이터·5-fold·평가** 위에서 사전학습 없이
task 에 처음부터 지도학습하는 1D-ResNet 을 돌린다. 공용 코어(:mod:`ioh_cnn`)의
RAM-resident fp16 스트리밍 로더·모델·학습/평가/저장을 그대로 재사용하고, ICH 고유의
입력 채널(default ABP+ICP+ECG)과 **환자단위 subject_ids grouping** 만 지정한다
(CARMEN ICH run.py 의 OOF bootstrap grouping 정합).

⚠ ICH 는 window=1200s(20min) 로 커서 RAM peak 가 크다 — ioh_cnn 의 스트리밍 fp16
   방식이라 split 1개분 fp16 만 상주하지만, fold 동시 실행 수는 컨테이너 RAM 에 맞춰라.

사용법:
    python -m downstream.baselines.ich_cnn \
        --data-path .../intracranial_hypertension_abp_icp_ecg_w1200s_h30min \
        --input-signals abp icp ecg --n-folds 5 --fold 0 \
        --epochs 100 --patience 20 --lr 1e-3 --batch-size 64 --device cuda \
        --out-dir .../result/main/intracranial_hypertension_cnn/w1200s_h30min
"""
from __future__ import annotations

import argparse

from downstream.baselines.ioh_cnn import add_common_args, run_cnn_baseline


def main():
    p = add_common_args(
        argparse.ArgumentParser(description="ICH end-to-end 1D-CNN baseline"),
        default_signals=["abp", "icp", "ecg"],
    )
    args = p.parse_args()
    # ICH: 한 환자(subject)가 여러 window → 환자단위 subject_ids 로 grouping
    # (없으면 case_ids fallback). CARMEN ICH run.py 와 동일.
    run_cnn_baseline(args, task_name="ich_cnn", tag="ICH-CNN",
                     id_fields=("subject_ids", "case_ids"))


if __name__ == "__main__":
    main()
