# -*- coding: utf-8 -*-
"""VTaC False VT Alarm (binary detection) end-to-end 1D-CNN baseline.

CARMEN(frozen / LoRA) 과 **같은 prepared 데이터·5-fold·평가·grouping** 위에서
사전학습 없이 task 에 처음부터 지도학습하는 1D-ResNet. 공용 코어(:mod:`ioh_cnn`)의
RAM-resident fp16 스트리밍 로더·모델·학습/평가/저장을 재사용하고, VTaC 고유의 입력
채널(ECG+PLETH) 과 **record 단위 grouping**(``patients`` = VTaC record, CARMEN
vtac/run.py 의 patient-cluster OOF bootstrap grouping 과 정합) 만 지정한다.

⚠ VTaC prepare(``downstream.acute_event.vtac.prepare_data``)는 fold 당 **단일 통합
파일**(``vtac_<sig>_fold{i}.pt`` = {train,val,test,metadata})을 낸다. 코어 로더는
per-split chunk glob 을 기대하므로, ``--data-path`` 가 prefix 면 ``_fold{fold}.pt`` 로
해석해 단일 파일을 직접 로드한다(iter_prepared_split_chunks 의 is_file 분기).

라벨은 True(진짜 VT)=1 / False(오알람)=0 이진. rare(~28% 양성)라기보다 moderate
imbalance 라 코어의 ``pos_weight = n_neg/n_pos`` BCE 가 그대로 적용된다
([[project_supervised_baseline_resnet1d]]).

사용법:
    # prefix 형 (권장) — 내부에서 _fold{f}.pt 로 해석
    python -m downstream.baselines.vtac_cnn \
        --data-path .../vtac/vtac_ecg_ppg \
        --input-signals ecg ppg --n-folds 5 --fold 0 \
        --epochs 100 --patience 20 --lr 1e-3 --batch-size 128 --device cuda \
        --out-dir .../result/main/vtac/supervised
"""
from __future__ import annotations

import argparse
from pathlib import Path

from downstream.baselines.ioh_cnn import add_common_args, run_cnn_baseline


def main():
    p = add_common_args(
        argparse.ArgumentParser(description="VTaC False VT Alarm end-to-end 1D-CNN baseline"),
        default_signals=["ecg", "ppg"],
    )
    args = p.parse_args()

    # VTaC = single-file-per-fold. prefix 면 fold 파일로 해석(코어는 exact file 도 처리).
    if not Path(args.data_path).is_file():
        args.data_path = f"{args.data_path}_fold{args.fold}.pt"

    # record(=patients) 단위 grouping → CARMEN vtac/run.py 의 patient_ids(=record) 정합.
    run_cnn_baseline(args, task_name="vtac_cnn", tag="VTAC-CNN",
                     id_fields=("patients", "segment_ids"))


if __name__ == "__main__":
    main()
