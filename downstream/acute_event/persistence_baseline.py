# -*- coding:utf-8 -*-
"""Persistence (trivial-extrapolation) baseline — acute event figure의 dashed 선.

"미래값 = 현재값" 가정. 입력 윈도우의 **마지막 임계신호값**을 risk score로 써서
FM 과 **동일한 test set·동일한 fold·동일한 CI 방법**(OOF pooled patient-level
bootstrap, `_eval_utils`)으로 AUROC/AUPRC 를 낸다. 미래 구간은 전혀 쓰지 않으므로
label leakage 가 없다(현재 신호만으로 미래 임계 초과를 "예측").

Task 별 score (higher = positive risk 가 되도록 sign 적용):
  - hypotension (IOH)              : signal=abp, sign=-1  (현재 MAP 낮을수록 hypo 위험)
  - intracranial_hypertension (ICH): signal=icp, sign=+1  (현재 ICP 높을수록 ICH 위험)
  - cardiac_arrest (CA, *provisional*): signal=abp, sign=-1 (pre-arrest hypotension proxy)
        ⚠️ CA 는 carry-forward 할 연속 임계신호가 없어 "현재 MAP" 를 잠정 baseline
        으로 둔다. HR/RR trend 등 더 적절한 baseline 으로 --signal/--sign 으로 교체 가능.

임계신호는 prepare_data 가 **입력 채널로 저장한 경우에만** 계산 가능하다
(예: ICH 는 --input-signals 에 icp 가 포함돼야 함).

산출: FM run.py 와 동일한 `preds_fold{i}.npz` (task="{task}_persistence") + OOF 집계
`persistence_report.json` (AUROC/AUPRC point + 95% CI + fold SD). figure 는 이 점을
FM 점 옆 dashed 선으로 얹는다.

사용법 (horizon 1개 = data-path 1개 = 호출 1개; figure 의 각 점):
    python -m downstream.acute_event.persistence_baseline \
        --task hypotension \
        --data-path <DATA>/hypotension_ecg_ppg_abp_w300s_h5min \
        --n-folds 5 \
        --out-dir <OUT>/hypotension/persistence/ecg_ppg_abp_w300s_h5min

    python -m downstream.acute_event.persistence_baseline \
        --task intracranial_hypertension \
        --data-path <DATA>/intracranial_hypertension_abp_ecg_icp_w1200s_h30min \
        --n-folds 5 --out-dir <OUT>/ich/persistence/abp_ecg_icp_w1200s_h30min
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from downstream._save_utils import load_prepared_split_chunked
from downstream._eval_utils import (
    auprc_binary,
    auroc_binary,
    dump_fold_predictions,
    evaluate_oof,
)


TARGET_SR: float = 100.0

# task → persistence score 정의. sign 은 "higher score = positive risk" 가 되게 한다.
# group_by: FM run.py 와 **동일한** patient-bootstrap grouping 키를 명시(silent divergence
# 방지). hypotension 은 case_ids, ICH/CA 는 subject_ids 로 그룹화한다.
PERSIST_CONFIG: dict[str, dict] = {
    "hypotension": {
        "signal": "abp", "sign": -1.0, "tail_sec": 30.0, "group_by": "case_ids",
    },
    "intracranial_hypertension": {
        "signal": "icp", "sign": +1.0, "tail_sec": 30.0, "group_by": "subject_ids",
    },
    # provisional — 현재 MAP proxy (pre-arrest hypotension). 교체 가능.
    "cardiac_arrest": {
        "signal": "abp", "sign": -1.0, "tail_sec": 30.0, "group_by": "subject_ids",
    },
}


def _row_tail_means(
    sig: torch.Tensor,            # (N, T) fp16/fp32
    gap: torch.Tensor | None,     # (N, T) bool, True=gap(원본 NaN)
    tail_sec: float,
) -> np.ndarray:                  # (N,) float
    """각 윈도우의 **마지막 tail_sec** 구간에서 gap 제외 평균을 구한다.

    ABP 파형은 ≥1박동 시간평균 ≈ MAP, ICP 도 동일하게 평균≈평균 ICP. tail 이 전부
    gap 이면 전체 윈도우 valid 평균으로, 그래도 없으면 NaN 으로 두고 호출측이 중앙값
    대체한다.
    """
    X = sig.float().numpy()
    N, T = X.shape
    tail = max(1, min(T, int(tail_sec * TARGET_SR)))
    if gap is not None:
        G = gap.numpy().astype(bool)
        Xv = np.where(G, np.nan, X)
    else:
        Xv = X.astype(np.float64)

    with np.errstate(invalid="ignore"):
        m = np.nanmean(Xv[:, T - tail:], axis=1)
        # tail 전부 gap → 전체 윈도우 valid 평균으로 폴백
        nan_rows = np.isnan(m)
        if nan_rows.any():
            m_full = np.nanmean(Xv, axis=1)
            m[nan_rows] = m_full[nan_rows]
    return m


def _persistence_scores(
    payload: dict,
    signal_key: str,
    sign: float,
    tail_sec: float,
) -> np.ndarray:
    """test payload → (N,) persistence risk score (higher = positive)."""
    signals = payload.get("signals", {})
    if signal_key not in signals:
        print(
            f"ERROR: persistence 임계신호 '{signal_key}' 가 prepared data 에 없습니다. "
            f"가용 채널={list(signals.keys())}. prepare_data 의 --input-signals 에 "
            f"'{signal_key}' 를 포함해 다시 만들거나, --signal 로 다른 채널을 지정하세요.",
            file=sys.stderr,
        )
        sys.exit(1)
    gap = payload.get("gap_masks", {}).get(signal_key)
    m = _row_tail_means(signals[signal_key], gap, tail_sec)
    # 전부 gap 이라 여전히 NaN 인 row → 중앙값(중립) 대체
    still = np.isnan(m)
    if still.any():
        med = np.nanmedian(m)
        m[still] = med if np.isfinite(med) else 0.0
    return sign * m


def _patient_ids(payload: dict, group_by: str) -> list[str]:
    """FM run.py 와 동일한 patient grouping. group_by 는 task별 명시(PERSIST_CONFIG):
    hypotension=case_ids, ICH/CA=subject_ids. 지정 키가 없으면 다른 id 키로 폴백."""
    ids = payload.get(group_by)
    if ids:
        return [str(x) for x in ids]
    for alt in ("subject_ids", "case_ids"):
        if alt != group_by and payload.get(alt):
            print(f"  WARN: grouping 키 '{group_by}' 부재 → '{alt}'로 폴백 "
                  "(FM 과 grouping 불일치 가능)", file=sys.stderr)
            return [str(x) for x in payload[alt]]
    n = int(payload.get("labels", torch.tensor([])).numel())
    return [str(i) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistence (trivial-extrapolation) baseline for acute event figure",
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=sorted(PERSIST_CONFIG.keys()),
        help="acute event task (persistence score 정의 선택)",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="prepared data prefix (run.py 와 동일 — fold/split/chunk 뗀 prefix)",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="preds_fold*.npz + persistence_report.json 저장 경로 "
             "(⚠️ FM run.py out-dir 와 분리 — preds 덮어쓰기 방지)",
    )
    # override (CA baseline 교체 등)
    parser.add_argument("--signal", type=str, default=None,
                        help="임계신호 채널 override (기본: task 설정)")
    parser.add_argument("--sign", type=float, default=None,
                        help="score sign override (+1/-1; higher=positive risk)")
    parser.add_argument("--tail-sec", type=float, default=None,
                        help="윈도우 끝에서 평균 낼 길이(초) override (기본 30)")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = dict(PERSIST_CONFIG[args.task])
    if args.signal is not None:
        cfg["signal"] = args.signal
    if args.sign is not None:
        cfg["sign"] = args.sign
    if args.tail_sec is not None:
        cfg["tail_sec"] = args.tail_sec

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  Persistence baseline — {args.task}")
    print(f"  Data:   {args.data_path}")
    print(f"  Signal: {cfg['signal']} (sign={cfg['sign']:+.0f}, "
          f"tail={cfg['tail_sec']:.0f}s)")
    print(f"  Folds:  {args.n_folds}")
    if args.task == "cardiac_arrest":
        print("  ⚠️ CA persistence 는 provisional(현재 MAP proxy) — "
              "필요 시 --signal/--sign 으로 교체")
    print(f"{'=' * 60}")

    fold_paths: list[Path] = []
    horizon_sec = None
    for f in range(args.n_folds):
        load_fold = f if args.n_folds > 1 else None
        # test split 만 로드(persistence 는 train/val 불필요 — 네트워크 IO/RAM 절약).
        data = load_prepared_split_chunked(
            args.data_path, fold=load_fold, splits=("test",),
        )
        if "test" not in data:
            print(f"ERROR: fold {f} test split 이 없습니다: {args.data_path}",
                  file=sys.stderr)
            sys.exit(1)
        test = data["test"]
        meta = data.get("metadata", {})
        if horizon_sec is None:
            horizon_sec = meta.get("horizon_sec")

        y_true = test["labels"].numpy().astype(int)
        scores = _persistence_scores(test, cfg["signal"], cfg["sign"], cfg["tail_sec"])
        pids = _patient_ids(test, cfg["group_by"])

        n_pos = int(y_true.sum())
        print(f"  fold {f}: n={len(y_true)}, pos={n_pos} "
              f"({100.0 * n_pos / max(len(y_true), 1):.1f}%)")

        p = dump_fold_predictions(
            out_dir, task=f"{args.task}_persistence",
            fold_idx=f, n_folds=args.n_folds,
            y_true=y_true, y_score=scores, patient_ids=pids,
        )
        fold_paths.append(p)

    # ── OOF pooled + patient-level bootstrap CI (FM 과 동일 머신) ──
    res = evaluate_oof(
        fold_paths,
        {"AUROC": auroc_binary, "AUPRC": auprc_binary},
        n_bootstrap=args.bootstrap_iters,
        seed=args.seed,
    )

    report = {
        "task": args.task,
        "method": "persistence",
        "data_path": args.data_path,
        "horizon_sec": horizon_sec,
        "horizon_min": (horizon_sec / 60.0) if horizon_sec else None,
        "signal": cfg["signal"],
        "sign": cfg["sign"],
        "tail_sec": cfg["tail_sec"],
        "group_by": cfg["group_by"],
        "n_folds": args.n_folds,
        "provisional": args.task == "cardiac_arrest" and args.signal is None,
        "metrics": {
            name: {
                "point": d["point"],
                "ci_low": d["ci_low"],
                "ci_high": d["ci_high"],
                "fold_mean": d["fold_mean"],
                "fold_sd": d["fold_sd"],
                "format_plain": d["format_plain"],
            }
            for name, d in res.items()
        },
    }
    report_path = out_dir / "persistence_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\n{'=' * 60}")
    for name, d in res.items():
        print(f"  {name}: {d['format_plain']}")
    print(f"  Report: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
