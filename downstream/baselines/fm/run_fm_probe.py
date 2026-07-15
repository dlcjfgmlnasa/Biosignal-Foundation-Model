# -*- coding: utf-8 -*-
"""외부 FM baseline 통합 실행 진입점 (frozen encoder + linear probe).

하나의 CLI 로 외부 FM(BIOT / ECGFounder / PaPaGei / Pulse-PPG / ST-MEM / AnyPPG)을
``--encoder`` 로 선택해, CARMEN 과 **같은 prepared 데이터·같은 5-fold·같은 평가**
위에서 frozen linear-probe 로 돌린다. 산출물(preds_fold{f}.npz + fold{f}.json)은
CARMEN·CNN baseline 과 동일 스키마 → ``python -m downstream.run_eval`` 로 함께 집계.

사용 예:
    python -m downstream.baselines.fm.run_fm_probe \
        --encoder ecgfounder --weights /weights/1_lead_ECGFounder.pth \
        --third-party-root /repos/ECGFounder \
        --data-path .../scope_arrest_ecg_ppg_w600s_h15min \
        --input-signals ecg ppg --task-name cardiac_arrest \
        --id-fields subject_ids case_ids \
        --n-folds 5 --fold 0 --epochs 100 --lr 1e-3 --batch-size 256 \
        --out-dir .../result/main/cardiac_arrest_ecgfounder/ecg_ppg_w600s_h15min

modality 비호환(예: PPG 전용 모델에 ECG-only task)이면 "SKIP" 후 exit 0.
"""
from __future__ import annotations

import argparse
import sys

from downstream.baselines.ioh_cnn import add_common_args
from downstream.baselines.fm.encoders import ENCODER_NAMES, build_encoder
from downstream.baselines.fm._fm_common import run_fm_baseline


def build_argparser() -> argparse.ArgumentParser:
    p = add_common_args(
        argparse.ArgumentParser(description="External FM frozen linear-probe baseline"),
        default_signals=["ecg"],
    )
    p.add_argument("--encoder", type=str, required=True, choices=ENCODER_NAMES)
    p.add_argument("--weights", type=str, required=True, help="사전학습 가중치 파일(또는 HF 디렉토리)")
    p.add_argument("--third-party-root", type=str, default=None,
                   help="upstream 레포 루트 (미지정 시 모델별 FM_*_ROOT env 사용)")
    p.add_argument("--feat-batch", type=int, default=64, help="frozen feature 추출 배치")
    p.add_argument("--dropout", type=float, default=0.1, help="LinearProbe dropout")
    p.add_argument("--max-segments", type=int, default=0,
                   help="긴 window 를 세그먼트로 분할해 인코딩할 최대 개수(균등 샘플·평균). "
                        "0=window 전체 커버(겹침 없이 연속, 권장). CARMEN 은 window 전체를 pool 하므로 "
                        "부분 커버는 외부 FM 을 불리하게 만든다.")
    p.add_argument("--task-name", type=str, default=None,
                   help="npz task 태그 접두(미지정 시 'fm'). 실제 태그=<task-name>_<encoder>")
    p.add_argument("--id-fields", type=str, nargs="+", default=["case_ids"],
                   help="환자 grouping id payload 키 우선순위 (task 별: arrest/ich=subject_ids case_ids 등)")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    encoder = build_encoder(
        args.encoder,
        weights_path=args.weights,
        device=args.device,
        max_segments=args.max_segments,
        third_party_root=args.third_party_root,
    )

    # modality 호환성 사전 체크 (비호환이면 sweep 을 막지 않도록 exit 0).
    sm = encoder.supported_modalities
    if sm is not None and not (set(args.input_signals) & set(sm)):
        print(f"[SKIP] encoder={args.encoder} 지원 modality {set(sm)} 가 "
              f"입력 {args.input_signals} 에 없음 → 이 task 는 평가 불가.", flush=True)
        sys.exit(0)

    task_tag = f"{args.task_name or 'fm'}_{args.encoder}"
    run_fm_baseline(
        args, encoder,
        task_name=task_tag,
        tag=f"{args.encoder.upper()}",
        id_fields=tuple(args.id_fields),
    )


if __name__ == "__main__":
    main()
