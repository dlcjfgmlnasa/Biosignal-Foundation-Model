# -*- coding: utf-8 -*-
"""Ablation downstream sweep orchestrator.

variants.yaml × downstream_tasks.yaml × folds × modes 의 cartesian product 를
큐로 만들고, 각 셀에 대해 task `run.py` 를 subprocess 로 invoke 한다.

핵심 책임:
  1. variants.yaml + downstream_tasks.yaml 로드.
  2. 각 variant 의 Phase 2 best.pt (또는 Phase 1 only / reuse) ckpt resolve.
  3. task 의 ``applies_to_groups`` 와 variant 의 ``ablation_group`` 매칭 필터.
  4. CLI 필터 (--variants / --groups / --tasks / --folds / --modes) 적용.
  5. (variant, task, mode, fold) 셀 마다 task `run.py` subprocess 실행.
  6. ``--nproc N`` + ``--gpu-ids`` 로 동시 실행 (round-robin CUDA_VISIBLE_DEVICES).
  7. ``--skip-existing`` 으로 ``preds_fold{k}.npz`` 존재 시 skip.
  8. 실패 셀은 마지막에 요약. 한 셀이 실패해도 나머지는 계속 실행.

본 스크립트는 변경 금지:
  - variants.yaml
  - scripts/run_ablation.py
  - downstream/**/run.py

사전학습된 ckpt 가 모두 존재한다고 가정한다.

Usage 예시
==========

    # dry-run (큐 구성만 확인)
    python -m scripts.run_ablation_downstream \\
        --variants-yaml configs/ablation/variants.yaml \\
        --tasks-yaml    configs/ablation/downstream_tasks.yaml \\
        --dry-run

    # 단일 variant × 단일 task, 4-GPU 동시 실행
    python -m scripts.run_ablation_downstream \\
        --variants 00_full \\
        --tasks    hypotension \\
        --nproc 4 --gpu-ids 0 1 2 3 \\
        --skip-existing

downstream_tasks.yaml 스키마 (가정)
=================================

::

    defaults:
      epochs: 20
      lr: 1.0e-3
      batch_size: 32
      n_folds: 5

    tasks:
      - name: hypotension                     # 셀 식별자
        module: downstream.acute_event.hypotension.run
        data_path: outputs/downstream/hypotension/cohort.pt
        modes: [linear_probe, lora]
        applies_to_groups: [baseline, loss_component, lscnorm, cross_modal,
                            phase_stage, sampling, attn_bias, model_scale]
        extra_args:                            # optional. CLI flag (kebab) → value.
          input-signals: [abp]
          window-sec: 600
        omit_flags: [mode, fold, n-folds, epochs, lr, batch-size]  # optional.
                                               # 해당 flag 는 자동 주입 skip
                                               # (예: generation task 가 mode/fold 미지원)

스키마는 robust 하게 처리한다:
  * ``defaults`` 없으면 사용 안 함.
  * ``extra_args`` 의 value 는 scalar | list[scalar].
  * ``applies_to_groups`` 빈/없음 → 모든 variant.
  * ``omit_flags`` 는 자동 주입 flag(mode/fold/...) blacklist (extra_args 는 통과).
  * 알 수 없는 키는 warning 후 통과.
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_variants(doc: dict) -> list[dict]:
    """variants.yaml → list[variant_dict]. doc 가 dict 면 ``variants`` 키, list 면 그대로."""
    if isinstance(doc, dict) and "variants" in doc:
        return list(doc["variants"])
    if isinstance(doc, list):
        return list(doc)
    raise ValueError("variants.yaml 형식 오류: 최상위 'variants' 키 또는 list 필요")


def normalize_tasks(doc: dict) -> tuple[list[dict], dict]:
    """downstream_tasks.yaml → (tasks list, defaults dict)."""
    if not isinstance(doc, dict):
        raise ValueError("downstream_tasks.yaml 최상위는 mapping 이어야 함")
    tasks = doc.get("tasks", [])
    defaults = doc.get("defaults", {}) or {}
    if not isinstance(tasks, list):
        raise ValueError("downstream_tasks.yaml 'tasks' 는 list 필요")
    return list(tasks), dict(defaults)


# ──────────────────────────────────────────────────────────────────────────────
# Ckpt resolution
# ──────────────────────────────────────────────────────────────────────────────


def _exp_name_of(variant: dict, phase: str, fallback: str) -> str:
    """variant 의 phase1/phase2.overrides.exp_name 추출 (없으면 fallback)."""
    block = variant.get(phase, {}) or {}
    return (block.get("overrides", {}) or {}).get("exp_name", fallback)


def _resolve_phase1_exp_name(variant: dict, all_variants: list[dict]) -> str:
    """phase1.reuse 체인을 따라가 최종 phase1 exp_name 을 반환."""
    p1 = variant.get("phase1", {}) or {}
    reuse = p1.get("reuse")
    if not reuse:
        return _exp_name_of(variant, "phase1", f"ablation_{variant['name']}_p1")
    referent = _find_variant(all_variants, reuse)
    return _resolve_phase1_exp_name(referent, all_variants)


def _find_variant(all_variants: list[dict], name: str) -> dict:
    for v in all_variants:
        if v.get("name") == name:
            return v
    raise KeyError(f"variant not found: {name}")


def resolve_ckpt_path(
    variant: dict,
    all_variants: list[dict],
    ckpt_root: Path,
) -> Path:
    """variant 의 평가용 ckpt 경로 결정.

    규약:
      * run_phase2 == False  → Phase 1 best.pt (예: 04_phase1_only)
      * phase2.reuse 있음    → 해당 variant 의 Phase 2 best.pt
      * 그 외                 → 본 variant 의 Phase 2 best.pt
    """
    name = variant["name"]
    run_phase2 = variant.get("run_phase2", True)

    if not run_phase2:
        exp = _resolve_phase1_exp_name(variant, all_variants)
        return ckpt_root / exp / "checkpoints" / "best.pt"

    p2 = variant.get("phase2", {}) or {}
    reuse = p2.get("reuse")
    if reuse:
        referent = _find_variant(all_variants, reuse)
        return resolve_ckpt_path(referent, all_variants, ckpt_root)

    exp = _exp_name_of(variant, "phase2", f"ablation_{name}_p2")
    return ckpt_root / exp / "checkpoints" / "best.pt"


# ──────────────────────────────────────────────────────────────────────────────
# Cell construction
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Cell:
    variant: str
    variant_group: str
    task: str
    module: str
    mode: str
    fold: int
    n_folds: int
    ckpt: Path
    data_path: str | None
    out_dir: Path
    extra_args: dict = field(default_factory=dict)
    epochs: int | None = None
    lr: float | None = None
    batch_size: int | None = None
    omit_flags: set[str] = field(default_factory=set)

    def key(self) -> str:
        return f"{self.variant}/{self.task}/{self.mode}/fold{self.fold}"


def _filter_tasks(
    tasks: list[dict],
    variant_group: str,
    task_filter: set[str] | None,
) -> list[dict]:
    out = []
    for t in tasks:
        name = t.get("name")
        if task_filter and name not in task_filter:
            continue
        groups = t.get("applies_to_groups") or []
        if groups and variant_group not in groups:
            continue
        out.append(t)
    return out


def _filter_variants(
    variants: list[dict],
    name_filter: set[str] | None,
    group_filter: set[str] | None,
) -> list[dict]:
    if not name_filter and not group_filter:
        return list(variants)
    out = []
    for v in variants:
        if name_filter and v.get("name") in name_filter:
            out.append(v)
            continue
        if group_filter and v.get("ablation_group") in group_filter:
            out.append(v)
    return out


def build_cells(
    variants: list[dict],
    tasks: list[dict],
    defaults: dict,
    ckpt_root: Path,
    result_root: Path,
    fold_filter: list[int] | None,
    mode_filter: set[str] | None,
    variant_filter: set[str] | None,
    group_filter: set[str] | None,
    task_filter: set[str] | None,
) -> list[Cell]:
    sel_variants = _filter_variants(variants, variant_filter, group_filter)
    cells: list[Cell] = []

    default_n_folds = int(defaults.get("n_folds", 5))

    for v in sel_variants:
        vname = v.get("name")
        vgroup = v.get("ablation_group", "unknown")
        ckpt = resolve_ckpt_path(v, variants, ckpt_root)

        for t in _filter_tasks(tasks, vgroup, task_filter):
            tname = t.get("name")
            modes = t.get("modes") or ["linear_probe"]
            if mode_filter:
                modes = [m for m in modes if m in mode_filter]
            if not modes:
                continue

            n_folds = int(t.get("n_folds", default_n_folds))
            folds = list(range(n_folds)) if fold_filter is None else list(fold_filter)

            for mode in modes:
                for fold in folds:
                    out_dir = (
                        result_root / vname / tname / mode / f"fold_{fold}"
                    )
                    cells.append(
                        Cell(
                            variant=vname,
                            variant_group=vgroup,
                            task=tname,
                            module=t.get("module", ""),
                            mode=mode,
                            fold=fold,
                            n_folds=n_folds,
                            ckpt=ckpt,
                            data_path=t.get("data_path"),
                            out_dir=out_dir,
                            extra_args=t.get("extra_args", {}) or {},
                            epochs=t.get("epochs", defaults.get("epochs")),
                            lr=t.get("lr", defaults.get("lr")),
                            batch_size=t.get("batch_size", defaults.get("batch_size")),
                            omit_flags=set(t.get("omit_flags", []) or []),
                        )
                    )
    return cells


# ──────────────────────────────────────────────────────────────────────────────
# Command building
# ──────────────────────────────────────────────────────────────────────────────


def _flag(name: str) -> str:
    """yaml key (snake or kebab) → CLI ``--kebab-case`` flag."""
    return f"--{name.replace('_', '-')}"


def _append_kv(cmd: list[str], key: str, value: Any) -> None:
    """Append ``--key v1 [v2 ...]`` to cmd. None → skip."""
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(_flag(key))
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        cmd.append(_flag(key))
        cmd.extend(str(x) for x in value)
        return
    cmd.append(_flag(key))
    cmd.append(str(value))


def build_cell_cmd(cell: Cell) -> list[str]:
    """Cell → ``python -m <module> ...`` argv."""
    if not cell.module:
        raise ValueError(f"task {cell.task!r} missing 'module' field")

    cmd: list[str] = [sys.executable, "-m", cell.module]
    omit = cell.omit_flags

    def _add(flag: str, value: Any) -> None:
        if flag in omit:
            return
        _append_kv(cmd, flag, value)

    _add("checkpoint", str(cell.ckpt))
    _add("mode", cell.mode)
    _add("fold", cell.fold)
    _add("n-folds", cell.n_folds)
    _add("out-dir", str(cell.out_dir))
    if cell.data_path is not None:
        _add("data-path", cell.data_path)
    if cell.epochs is not None:
        _add("epochs", cell.epochs)
    if cell.lr is not None:
        _add("lr", cell.lr)
    if cell.batch_size is not None:
        _add("batch-size", cell.batch_size)
    for k, v in cell.extra_args.items():
        _append_kv(cmd, k, v)   # extra_args bypass omit (의도된 명시 인자)
    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────────────────────


def preflight(cells: list[Cell], strict: bool) -> list[str]:
    """ckpt / data_path 누락 검출. strict=True 면 즉시 예외."""
    issues: list[str] = []
    seen_ckpt: dict[Path, bool] = {}
    seen_data: dict[str, bool] = {}
    for c in cells:
        if c.ckpt not in seen_ckpt:
            seen_ckpt[c.ckpt] = c.ckpt.exists()
        if not seen_ckpt[c.ckpt]:
            issues.append(f"[ckpt-missing] variant={c.variant} → {c.ckpt}")
        if c.data_path is not None and c.data_path not in seen_data:
            seen_data[c.data_path] = Path(c.data_path).exists()
        if c.data_path is not None and not seen_data[c.data_path]:
            issues.append(f"[data-missing]  task={c.task}   → {c.data_path}")
    # dedup
    issues = sorted(set(issues))
    if strict and issues:
        msg = "\n".join(issues)
        raise FileNotFoundError(
            f"preflight 실패 ({len(issues)} issue):\n{msg}\n"
            f"--skip-preflight 로 무시 가능."
        )
    return issues


def cell_done(cell: Cell) -> bool:
    """``preds_fold{fold}.npz`` 가 cell.out_dir 에 존재하면 True."""
    return (cell.out_dir / f"preds_fold{cell.fold}.npz").exists()


# ──────────────────────────────────────────────────────────────────────────────
# Worker (subprocess execution)
# ──────────────────────────────────────────────────────────────────────────────


def _run_cell(
    cell: Cell,
    gpu_id: str | None,
    dry_run: bool,
) -> tuple[str, int, float, str]:
    """단일 cell subprocess 실행. (key, returncode, elapsed_s, stderr_tail) 반환."""
    cell.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cell_cmd(cell)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = cell.out_dir / "run.log"
    if dry_run:
        return cell.key(), 0, 0.0, " ".join(cmd)

    start = time.time()
    with open(log_path, "wb") as log_f:
        log_f.write(f"# CMD: {' '.join(cmd)}\n".encode("utf-8"))
        log_f.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n".encode("utf-8"))
        log_f.write(b"#" + b"-" * 78 + b"\n")
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start

    # stderr/stdout tail (마지막 20 line) 으로 실패 사유 요약
    tail = ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        tail = "".join(lines[-20:])
    except Exception:
        pass
    return cell.key(), proc.returncode, elapsed, tail


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants-yaml", type=Path,
                   default=REPO_ROOT / "configs" / "ablation" / "variants.yaml")
    p.add_argument("--tasks-yaml", type=Path,
                   default=REPO_ROOT / "configs" / "ablation" / "downstream_tasks.yaml")
    p.add_argument("--ckpt-root", type=Path,
                   default=REPO_ROOT / "outputs" / "ablation")
    p.add_argument("--result-root", type=Path,
                   default=REPO_ROOT / "result" / "ablation")

    p.add_argument("--variants", nargs="+", default=None,
                   help="실행할 variant name 화이트리스트 (생략 시 전체)")
    p.add_argument("--groups", nargs="+", default=None,
                   help="실행할 ablation_group 화이트리스트")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="실행할 task name 화이트리스트")
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="실행할 fold index 화이트리스트 (생략 시 task.n_folds 전체)")
    p.add_argument("--modes", nargs="+", default=None,
                   help="실행할 mode 화이트리스트 (linear_probe / lora / zero_shot)")

    p.add_argument("--nproc", type=int, default=1, help="동시 실행 셀 수 (GPU 단위)")
    p.add_argument("--gpu-ids", nargs="+", default=None,
                   help="CUDA_VISIBLE_DEVICES round-robin 대상 (예: 0 1 2 3)")

    p.add_argument("--skip-existing", action="store_true",
                   help="preds_fold{k}.npz 존재 시 cell skip")
    p.add_argument("--skip-preflight", action="store_true",
                   help="ckpt/data 누락 검사 무시")
    p.add_argument("--dry-run", action="store_true",
                   help="실제 실행 없이 큐만 출력")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.variants_yaml.exists():
        print(f"[ERROR] variants.yaml not found: {args.variants_yaml}", file=sys.stderr)
        return 2
    if not args.tasks_yaml.exists():
        print(f"[ERROR] tasks.yaml not found: {args.tasks_yaml}\n"
              f"  (downstream_tasks.yaml 가 아직 작성 중이면 작성 후 재실행)",
              file=sys.stderr)
        return 2

    variants = normalize_variants(load_yaml(args.variants_yaml))
    tasks, defaults = normalize_tasks(load_yaml(args.tasks_yaml))

    cells = build_cells(
        variants=variants,
        tasks=tasks,
        defaults=defaults,
        ckpt_root=args.ckpt_root,
        result_root=args.result_root,
        fold_filter=args.folds,
        mode_filter=set(args.modes) if args.modes else None,
        variant_filter=set(args.variants) if args.variants else None,
        group_filter=set(args.groups) if args.groups else None,
        task_filter=set(args.tasks) if args.tasks else None,
    )

    if not cells:
        print("[WARN] no cells after filtering — check --variants/--groups/--tasks/--modes/--folds",
              file=sys.stderr)
        return 0

    issues = preflight(cells, strict=not (args.skip_preflight or args.dry_run))
    if issues:
        print(f"[preflight] {len(issues)} issue(s) detected:")
        for i in issues[:20]:
            print(f"  {i}")
        if len(issues) > 20:
            print(f"  ... (+{len(issues) - 20} more)")

    skipped = [c for c in cells if args.skip_existing and cell_done(c)]
    todo = [c for c in cells if not (args.skip_existing and cell_done(c))]

    print(f"\n[run_ablation_downstream] cells: total={len(cells)}, "
          f"todo={len(todo)}, skipped(existing)={len(skipped)}")
    if args.dry_run:
        print(f"[dry-run] showing first/last 5 of {len(todo)} cells:")
        preview = todo[:5] + ([] if len(todo) <= 10 else [None]) + todo[-5:]  # type: ignore[list-item]
        for c in preview:
            if c is None:
                print("    ...")
                continue
            cmd = build_cell_cmd(c)
            print(f"    [{c.key()}] ckpt={c.ckpt.name} → {' '.join(cmd[:6])} ...")
        if args.gpu_ids:
            print(f"[dry-run] GPU round-robin: {args.gpu_ids}")
        return 0

    # GPU round-robin
    gpu_pool = list(args.gpu_ids) if args.gpu_ids else [None]

    # 실행
    n_workers = max(1, args.nproc)
    print(f"[run_ablation_downstream] launching {n_workers} worker(s), "
          f"GPUs={gpu_pool}")

    failed: list[tuple[str, int, str]] = []
    completed: int = 0

    if n_workers == 1:
        for i, c in enumerate(todo):
            gpu = gpu_pool[i % len(gpu_pool)]
            key, rc, elapsed, tail = _run_cell(c, gpu, dry_run=False)
            completed += 1
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{completed}/{len(todo)}] {key}  ({status}, {elapsed:.1f}s, gpu={gpu})")
            if rc != 0:
                failed.append((key, rc, tail))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {}
            for i, c in enumerate(todo):
                gpu = gpu_pool[i % len(gpu_pool)]
                futs[ex.submit(_run_cell, c, gpu, False)] = (c, gpu)
            for fut in as_completed(futs):
                c, gpu = futs[fut]
                try:
                    key, rc, elapsed, tail = fut.result()
                except Exception as e:
                    key, rc, elapsed, tail = c.key(), -1, 0.0, str(e)
                completed += 1
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  [{completed}/{len(todo)}] {key}  ({status}, {elapsed:.1f}s, gpu={gpu})")
                if rc != 0:
                    failed.append((key, rc, tail))

    print()
    print(f"[run_ablation_downstream] DONE — {completed - len(failed)}/{completed} OK, "
          f"{len(failed)} FAILED, {len(skipped)} skipped(existing).")
    if failed:
        print("Failed cells:")
        for key, rc, tail in failed:
            print(f"  - {key}  rc={rc}")
            for line in (tail or "").splitlines()[-5:]:
                print(f"      | {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
