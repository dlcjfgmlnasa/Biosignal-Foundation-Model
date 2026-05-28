# -*- coding: utf-8 -*-
"""Ablation downstream 결과 집계 + bootstrap CI + significance.

run_ablation_downstream.py 가 생성한
``<result_root>/<variant>/<task>/<mode>/fold_<k>/preds_fold{k}.npz``
구조를 스캔하여:

1. variant × task × mode 단위로 OOF concat → patient-level bootstrap 95% CI
   (downstream._eval_utils.evaluate_oof 재사용 — DRY).
2. 5-fold per-fold mean ± SD.
3. baseline (00_full) 대비 paired patient-cluster bootstrap significance.
4. 출력:
     * <out_dir>/ablation_main_table.md      — variant × task × mode 표
     * <out_dir>/ablation_per_task.csv       — 상세 행
     * <out_dir>/significance.json           — paired Δ + bootstrap p / CI

Classification (binary / multi-class) 와 Generation (forecasting) 모두 처리한다.
generation task 의 .npz 는 ``y_true`` / ``y_score`` 컬럼에 raw waveform
(예: (N, L) reshape) 을 저장한다는 가정. 저장 형식이 없으면 N/A 로 표시한다.

본 스크립트는 변경 금지:
  - downstream/_eval_utils.py 등 task 코드는 import 만.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from downstream._eval_utils import (  # noqa: E402
    auprc_binary, auprc_macro, auroc_binary, auroc_macro,
    bootstrap_ci, concat_oof, evaluate_oof, fold_stability, load_predictions,
)


# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_variants(doc: dict) -> list[dict]:
    if isinstance(doc, dict) and "variants" in doc:
        return list(doc["variants"])
    if isinstance(doc, list):
        return list(doc)
    return []


def normalize_tasks(doc: dict) -> list[dict]:
    if isinstance(doc, dict) and "tasks" in doc:
        return list(doc["tasks"])
    if isinstance(doc, list):
        return list(doc)
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Cell discovery
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CellResult:
    variant: str
    task: str
    mode: str
    fold_paths: list[Path] = field(default_factory=list)
    metrics: dict[str, dict] = field(default_factory=dict)
    task_kind: str = "classification"   # or "generation"
    missing_folds: list[int] = field(default_factory=list)
    err: str | None = None


def discover_cells(result_root: Path) -> dict[tuple[str, str, str], CellResult]:
    """``<root>/<variant>/<task>/<mode>/fold_*/preds_fold*.npz`` 스캔."""
    cells: dict[tuple[str, str, str], CellResult] = {}
    pattern = str(result_root / "*" / "*" / "*" / "fold_*" / "preds_fold*.npz")
    for path_str in sorted(glob.glob(pattern)):
        p = Path(path_str)
        # path: <root>/<variant>/<task>/<mode>/fold_<k>/preds_foldK.npz
        try:
            fold_dir = p.parent
            mode_dir = fold_dir.parent
            task_dir = mode_dir.parent
            variant_dir = task_dir.parent
        except Exception:
            continue
        key = (variant_dir.name, task_dir.name, mode_dir.name)
        cells.setdefault(
            key,
            CellResult(variant=key[0], task=key[1], mode=key[2]),
        ).fold_paths.append(p)
    return cells


# ──────────────────────────────────────────────────────────────────────────────
# Per-cell evaluation
# ──────────────────────────────────────────────────────────────────────────────


def _classification_metric_fns(fold_path: Path) -> dict[str, Callable]:
    fp = load_predictions(fold_path)
    multiclass = (
        (fp.y_score.ndim == 2 and fp.y_score.shape[1] > 2)
        or (fp.classes is not None and len(fp.classes) > 2)
    )
    if multiclass:
        return {"AUROC": auroc_macro, "AUPRC": auprc_macro}
    return {"AUROC": auroc_binary, "AUPRC": auprc_binary}


def _generation_metric_fns() -> dict[str, Callable]:
    def mse(y_true, y_pred):
        return float(np.mean((y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2))

    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64))))

    return {"MSE": mse, "MAE": mae}


def _task_kind(fold_path: Path, task_kind_hint: str | None) -> str:
    if task_kind_hint:
        return task_kind_hint
    fp = load_predictions(fold_path)
    # heuristic: y_true 가 모두 정수형 + 작은 카디널리티 → classification.
    if fp.y_true.dtype.kind in ("i", "u") and len(np.unique(fp.y_true)) < 64:
        return "classification"
    return "generation"


def evaluate_cell(
    cell: CellResult,
    n_bootstrap: int,
    seed: int,
    task_kind_hint: str | None = None,
) -> None:
    """cell 단위 평가 — cell.metrics, cell.task_kind 채움."""
    if not cell.fold_paths:
        cell.err = "no fold .npz"
        return
    fp0 = cell.fold_paths[0]
    try:
        cell.task_kind = _task_kind(fp0, task_kind_hint)
        if cell.task_kind == "classification":
            metric_fns = _classification_metric_fns(fp0)
        else:
            metric_fns = _generation_metric_fns()
        cell.metrics = evaluate_oof(
            cell.fold_paths, metric_fns, n_bootstrap=n_bootstrap, seed=seed,
        )
    except Exception as e:
        cell.err = f"{type(e).__name__}: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Significance — paired patient-cluster bootstrap (variant vs baseline)
# ──────────────────────────────────────────────────────────────────────────────


def paired_bootstrap_delta(
    baseline_oof,
    variant_oof,
    metric_fn: Callable,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Patient-cluster paired bootstrap (variant - baseline).

    동일한 patient_id 가 양쪽에 있다고 가정 (같은 CV split). 환자 단위 resample
    → 두 OOF 에서 같은 환자 행을 뽑아 metric 차이.
    """
    if baseline_oof is None or variant_oof is None:
        return {"delta": None, "ci_low": None, "ci_high": None, "p_value": None}

    b_pid = baseline_oof.patient_id
    v_pid = variant_oof.patient_id
    common = np.intersect1d(np.unique(b_pid), np.unique(v_pid))
    if len(common) == 0:
        return {"delta": None, "ci_low": None, "ci_high": None,
                "p_value": None, "note": "no overlapping patients"}

    b_idx_by_pid = {pid: np.where(b_pid == pid)[0] for pid in common}
    v_idx_by_pid = {pid: np.where(v_pid == pid)[0] for pid in common}

    def _metric_on(oof, idx):
        return float(metric_fn(oof.y_true[idx], oof.y_score[idx]))

    b_all = np.concatenate([b_idx_by_pid[pid] for pid in common])
    v_all = np.concatenate([v_idx_by_pid[pid] for pid in common])
    point_b = _metric_on(baseline_oof, b_all)
    point_v = _metric_on(variant_oof, v_all)
    delta = point_v - point_b

    rng = np.random.default_rng(seed)
    boot_deltas: list[float] = []
    for _ in range(n_iter):
        sampled = rng.choice(common, size=len(common), replace=True)
        bi = np.concatenate([b_idx_by_pid[pid] for pid in sampled])
        vi = np.concatenate([v_idx_by_pid[pid] for pid in sampled])
        try:
            m = _metric_on(variant_oof, vi) - _metric_on(baseline_oof, bi)
            if np.isfinite(m):
                boot_deltas.append(m)
        except Exception:
            continue

    if not boot_deltas:
        return {"delta": delta, "ci_low": None, "ci_high": None, "p_value": None}

    arr = np.asarray(boot_deltas)
    # two-sided p-value: 2 * min(P(Δ≤0), P(Δ≥0))
    p_pos = float(np.mean(arr >= 0))
    p_neg = float(np.mean(arr <= 0))
    p_val = float(min(1.0, 2.0 * min(p_pos, p_neg)))
    return {
        "delta": float(delta),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "p_value": p_val,
        "baseline_point": point_b,
        "variant_point": point_v,
    }


def compute_significance(
    cells_by_key: dict[tuple[str, str, str], CellResult],
    baseline_variant: str,
    n_iter: int,
    seed: int,
) -> dict:
    """각 (task, mode) 에서 baseline vs 다른 variant 비교."""
    # baseline OOF cache
    base_oof_cache: dict[tuple[str, str], Any] = {}
    base_metric_cache: dict[tuple[str, str], dict[str, Callable]] = {}

    out: dict[str, Any] = {}
    for (variant, task, mode), cell in cells_by_key.items():
        if variant == baseline_variant:
            continue
        base_key = (task, mode)
        base_cell = cells_by_key.get((baseline_variant, task, mode))
        if base_cell is None or not base_cell.fold_paths:
            continue
        if not cell.fold_paths or cell.err:
            continue

        if base_key not in base_oof_cache:
            try:
                base_oof_cache[base_key] = concat_oof(base_cell.fold_paths)
                if base_cell.task_kind == "generation":
                    base_metric_cache[base_key] = _generation_metric_fns()
                else:
                    base_metric_cache[base_key] = _classification_metric_fns(
                        base_cell.fold_paths[0])
            except Exception as e:
                base_oof_cache[base_key] = None
                base_metric_cache[base_key] = {}
                out.setdefault("_errors", []).append(
                    f"baseline OOF concat failed {base_key}: {e}")
                continue

        try:
            var_oof = concat_oof(cell.fold_paths)
        except Exception as e:
            out.setdefault("_errors", []).append(
                f"variant OOF concat failed ({variant},{task},{mode}): {e}")
            continue

        base_oof = base_oof_cache[base_key]
        metric_fns = base_metric_cache[base_key]
        per_metric: dict[str, dict] = {}
        for mname, fn in metric_fns.items():
            per_metric[mname] = paired_bootstrap_delta(
                base_oof, var_oof, fn, n_iter=n_iter, seed=seed,
            )
        out.setdefault(task, {}).setdefault(mode, {})[variant] = per_metric
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Output writers
# ──────────────────────────────────────────────────────────────────────────────


def _fmt_metric(d: dict, key: str = "point", precision: int = 3) -> str:
    if not d or d.get(key) is None:
        return "-"
    v = d[key]
    lo = d.get("ci_low")
    hi = d.get("ci_high")
    sd = d.get("fold_sd")
    if lo is None or hi is None:
        return f"{v:.{precision}f}"
    sd_str = f" (±{sd:.{precision}f})" if sd is not None else ""
    return f"{v:.{precision}f} [{lo:.{precision}f}-{hi:.{precision}f}]{sd_str}"


def _sig_star(p_value: float | None) -> str:
    if p_value is None:
        return ""
    if p_value < 0.001:
        return " ★★★"
    if p_value < 0.01:
        return " ★★"
    if p_value < 0.05:
        return " ★"
    return ""


def write_main_table(
    cells_by_key: dict[tuple[str, str, str], CellResult],
    significance: dict,
    out_path: Path,
    baseline_variant: str,
    variants_order: list[str],
) -> None:
    """variant × task,mode 표 (markdown). baseline 대비 Δ 포함."""
    # 컬럼: (task, mode) 정렬
    tm_set = sorted({(t, m) for _, t, m in cells_by_key.keys()})
    headers = ["variant"] + [f"{t}/{m}" for (t, m) in tm_set]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]

    ordered = variants_order or sorted({v for v, _, _ in cells_by_key.keys()})
    for v in ordered:
        row = [v]
        for (t, m) in tm_set:
            cell = cells_by_key.get((v, t, m))
            if cell is None or cell.err or not cell.metrics:
                row.append("-")
                continue
            primary = "AUROC" if "AUROC" in cell.metrics else (
                "MSE" if "MSE" in cell.metrics else next(iter(cell.metrics)))
            txt = _fmt_metric(cell.metrics[primary])
            if v != baseline_variant:
                sig = (significance.get(t, {}).get(m, {}).get(v, {}).get(primary, {}))
                if sig.get("delta") is not None:
                    delta = sig["delta"]
                    star = _sig_star(sig.get("p_value"))
                    sign = "+" if delta >= 0 else ""
                    txt += f"<br>Δ={sign}{delta:.3f}{star}"
            row.append(txt)
        lines.append("| " + " | ".join(row) + " |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Ablation main table\n\n")
        f.write(f"Baseline variant: `{baseline_variant}`. "
                f"Cell = primary metric [95% CI] (5-fold SD), "
                f"Δ = variant − baseline, "
                f"★ p<.05  ★★ p<.01  ★★★ p<.001 (paired patient-cluster bootstrap).\n\n")
        f.write("\n".join(lines))
        f.write("\n")


def write_per_task_csv(
    cells_by_key: dict[tuple[str, str, str], CellResult],
    out_path: Path,
) -> None:
    rows: list[dict] = []
    for (variant, task, mode), cell in cells_by_key.items():
        if cell.err or not cell.metrics:
            rows.append({
                "variant": variant, "task": task, "mode": mode,
                "metric": "", "fold": "", "value": "",
                "ci_low": "", "ci_high": "", "fold_sd": "",
                "n_folds_found": len(cell.fold_paths),
                "err": cell.err or "",
            })
            continue
        for metric, d in cell.metrics.items():
            if metric.startswith("_"):
                continue
            # OOF row
            rows.append({
                "variant": variant, "task": task, "mode": mode,
                "metric": metric, "fold": "oof",
                "value": d.get("point", ""),
                "ci_low": d.get("ci_low", ""),
                "ci_high": d.get("ci_high", ""),
                "fold_sd": d.get("fold_sd", ""),
                "n_folds_found": len(cell.fold_paths),
                "err": "",
            })
            # per-fold rows
            for i, v in enumerate(d.get("fold_values", []) or []):
                rows.append({
                    "variant": variant, "task": task, "mode": mode,
                    "metric": metric, "fold": i,
                    "value": v,
                    "ci_low": "", "ci_high": "", "fold_sd": "",
                    "n_folds_found": len(cell.fold_paths),
                    "err": "",
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "task", "mode", "metric", "fold",
            "value", "ci_low", "ci_high", "fold_sd",
            "n_folds_found", "err"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_significance(significance: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(significance, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result-root", type=Path,
                   default=REPO_ROOT / "result" / "ablation")
    p.add_argument("--tasks-yaml", type=Path,
                   default=REPO_ROOT / "configs" / "ablation" / "downstream_tasks.yaml")
    p.add_argument("--variants-yaml", type=Path,
                   default=REPO_ROOT / "configs" / "ablation" / "variants.yaml")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "result" / "ablation" / "_aggregated")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline", type=str, default="00_full",
                   help="baseline variant name (default 00_full)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.result_root.exists():
        print(f"[ERROR] result-root not found: {args.result_root}", file=sys.stderr)
        return 2

    # task_kind hint (yaml 의 task entry 에 ``kind: generation`` 명시 가능)
    task_kind_hint: dict[str, str] = {}
    variants_order: list[str] = []
    if args.tasks_yaml.exists():
        tdoc = load_yaml(args.tasks_yaml)
        for t in normalize_tasks(tdoc):
            if "kind" in t and t.get("name"):
                task_kind_hint[t["name"]] = t["kind"]
    if args.variants_yaml.exists():
        vdoc = load_yaml(args.variants_yaml)
        variants_order = [v.get("name", "") for v in normalize_variants(vdoc)]

    cells_by_key = discover_cells(args.result_root)
    print(f"[aggregate] discovered {len(cells_by_key)} cells under {args.result_root}")
    if not cells_by_key:
        return 1

    # Evaluate each cell
    missing_summary: list[str] = []
    for key, cell in cells_by_key.items():
        evaluate_cell(
            cell,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            task_kind_hint=task_kind_hint.get(cell.task),
        )
        if cell.err:
            missing_summary.append(f"  [err]    {'/'.join(key)}: {cell.err}")
        elif len(cell.fold_paths) < 5:
            missing_summary.append(
                f"  [partial] {'/'.join(key)}: {len(cell.fold_paths)} fold(s)")

    if missing_summary:
        print(f"\n[aggregate] {len(missing_summary)} cell(s) incomplete:")
        for line in missing_summary[:30]:
            print(line)
        if len(missing_summary) > 30:
            print(f"  ... (+{len(missing_summary) - 30} more)")

    # Significance (vs baseline)
    significance = compute_significance(
        cells_by_key, baseline_variant=args.baseline,
        n_iter=args.bootstrap, seed=args.seed,
    )

    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_main_table(
        cells_by_key, significance,
        args.out_dir / "ablation_main_table.md",
        baseline_variant=args.baseline,
        variants_order=variants_order,
    )
    write_per_task_csv(cells_by_key, args.out_dir / "ablation_per_task.csv")
    write_significance(significance, args.out_dir / "significance.json")

    print(f"\n[aggregate] outputs:")
    print(f"  - {args.out_dir / 'ablation_main_table.md'}")
    print(f"  - {args.out_dir / 'ablation_per_task.csv'}")
    print(f"  - {args.out_dir / 'significance.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
