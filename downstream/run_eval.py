# -*- coding:utf-8 -*-
"""Downstream classification task 의 fold prediction(.npz) → 통합 평가 리포트.

각 task run.py 가 fold 별로 ``save_predictions`` 로 저장한 ``*.npz`` 를 모아
OOF concat → patient-level bootstrap CI → 5-fold mean ± SD 를 계산한다
(memory: project_classification_eval_protocol).

binary / multi-class 는 첫 fold 의 y_score shape 로 자동 판별한다.

사용법:
    # 단일 task: fold .npz 가 있는 디렉토리 (또는 glob)
    python -m downstream.run_eval \\
        --pred-glob "outputs/downstream/hypotension/preds_fold*.npz" \\
        --out outputs/downstream/hypotension/eval_report.json

    # 여러 task 한 번에 (각 task 디렉토리 하위 preds_fold*.npz)
    python -m downstream.run_eval \\
        --tasks-root outputs/downstream \\
        --out outputs/downstream/all_tasks_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from downstream._eval_utils import (
    auprc_binary,
    auprc_macro,
    auroc_binary,
    auroc_macro,
    evaluate_oof,
    load_predictions,
)


def _is_multiclass(fold_path: str | Path) -> bool:
    """첫 fold .npz 의 y_score shape 로 multi-class 판별."""
    fp = load_predictions(fold_path)
    if fp.y_score.ndim == 2 and fp.y_score.shape[1] > 2:
        return True
    # classes 메타가 3개 이상이면 multi-class
    if fp.classes is not None and len(fp.classes) > 2:
        return True
    return False


def _metric_fns(multiclass: bool) -> dict:
    if multiclass:
        return {"AUROC": auroc_macro, "AUPRC": auprc_macro}
    return {"AUROC": auroc_binary, "AUPRC": auprc_binary}


def evaluate_one_task(
    fold_paths: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """단일 task 의 fold .npz 리스트 → 평가 결과 dict."""
    if not fold_paths:
        raise ValueError("no fold prediction files")
    multiclass = _is_multiclass(fold_paths[0])
    metric_fns = _metric_fns(multiclass)
    result = evaluate_oof(fold_paths, metric_fns, n_bootstrap=n_bootstrap, seed=seed)
    result["_meta"] = {
        "n_folds_found": len(fold_paths),
        "multiclass": multiclass,
        "task": load_predictions(fold_paths[0]).task,
    }
    return result


def _discover_tasks(tasks_root: Path) -> dict[str, list[str]]:
    """tasks_root 하위에서 task별 fold .npz 그룹 수집.

    구조 가정: <tasks_root>/<task>/preds_fold*.npz
    """
    out: dict[str, list[str]] = {}
    for sub in sorted(tasks_root.iterdir()):
        if not sub.is_dir():
            continue
        npzs = sorted(glob.glob(str(sub / "preds_fold*.npz")))
        if npzs:
            out[sub.name] = npzs
    return out


def _print_table(task_results: dict[str, dict]) -> str:
    """task별 결과 → markdown 표 (plain ASCII, console 안전)."""
    lines = [
        "| Task | AUROC [95% CI] (5-fold SD) | AUPRC [95% CI] (5-fold SD) | folds |",
        "|---|---|---|---|",
    ]
    for task, res in task_results.items():
        meta = res.get("_meta", {})
        au = res["AUROC"]
        ap = res["AUPRC"]
        au_str = (
            f"{au['point']:.3f} [{au['ci_low']:.3f}-{au['ci_high']:.3f}] "
            f"(+/-{au['fold_sd']:.3f})"
        )
        ap_str = (
            f"{ap['point']:.3f} [{ap['ci_low']:.3f}-{ap['ci_high']:.3f}] "
            f"(+/-{ap['fold_sd']:.3f})"
        )
        lines.append(
            f"| {task} | {au_str} | {ap_str} | {meta.get('n_folds_found', '?')} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred-glob", type=str, help="단일 task fold .npz glob 패턴")
    src.add_argument("--tasks-root", type=Path, help="여러 task 의 상위 디렉토리")
    p.add_argument("--task-name", type=str, default=None,
                   help="--pred-glob 모드에서 리포트에 쓸 task 이름 (기본: 자동)")
    p.add_argument("--out", type=Path, default=None, help="결과 JSON 출력 경로")
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    task_results: dict[str, dict] = {}

    if args.pred_glob:
        fold_paths = sorted(glob.glob(args.pred_glob))
        if not fold_paths:
            print(f"[ERROR] no files match {args.pred_glob}", file=sys.stderr)
            sys.exit(1)
        name = args.task_name or load_predictions(fold_paths[0]).task
        print(f"Evaluating task '{name}' from {len(fold_paths)} fold files...")
        task_results[name] = evaluate_one_task(
            fold_paths, n_bootstrap=args.bootstrap_iters, seed=args.seed
        )
    else:
        if not args.tasks_root.is_dir():
            print(f"[ERROR] {args.tasks_root} not a directory", file=sys.stderr)
            sys.exit(1)
        discovered = _discover_tasks(args.tasks_root)
        if not discovered:
            print(f"[ERROR] no preds_fold*.npz under {args.tasks_root}", file=sys.stderr)
            sys.exit(1)
        for task, fold_paths in discovered.items():
            print(f"Evaluating '{task}' ({len(fold_paths)} folds)...")
            try:
                task_results[task] = evaluate_one_task(
                    fold_paths, n_bootstrap=args.bootstrap_iters, seed=args.seed
                )
            except Exception as e:
                print(f"  [WARN] {task} 평가 실패: {e}", file=sys.stderr)

    print("\n" + _print_table(task_results) + "\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # numpy 타입 정리 후 저장
        serializable = {}
        for task, res in task_results.items():
            serializable[task] = {
                k: (
                    {kk: (list(vv) if isinstance(vv, (list, np.ndarray)) else vv)
                     for kk, vv in v.items()}
                    if isinstance(v, dict) else v
                )
                for k, v in res.items()
            }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"Report saved: {args.out}")


if __name__ == "__main__":
    main()
