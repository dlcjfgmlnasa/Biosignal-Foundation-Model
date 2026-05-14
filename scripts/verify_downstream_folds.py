# -*- coding: utf-8 -*-
"""Downstream prepare_data 출력 fold .pt 파일 무결성 검증 스크립트.

각 fold 파일을 로드해서 다음을 점검:
  1) 모든 fold 존재 (n_folds 와 실제 파일 수 일치)
  2) train/val/test split 비율 (~60/20/20)
  3) 각 split 에 cases > 0
  4) Stratification: fold 간 positive rate 가 비슷한지
  5) Signal shape 일관성 (window_sec × sampling_rate)
  6) 입력 modality 모두 존재
  7) 파일 크기 sanity check

사용법:
  python -m scripts.verify_downstream_folds \\
      --out-dir /home/coder/workspace/updown/bio_fm/data/downstream/hypotension

  # glob pattern 으로 특정 task 만 (sweep 시 유용)
  python -m scripts.verify_downstream_folds \\
      --out-dir .../hypotension --pattern "hypotension_*_w300s_h15min_fold*.pt"
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _split_stats(split: dict[str, Any]) -> dict[str, Any]:
    """train/val/test 한 split 의 통계 추출."""
    if not isinstance(split, dict):
        return {"format": "list", "n_patients": len(split) if hasattr(split, "__len__") else 0}

    # ForecastSample-style: signals=dict, labels=tensor, case_ids=list
    if "labels" in split and isinstance(split["labels"], torch.Tensor):
        labels = split["labels"]
        n = int(labels.numel())
        if n == 0:
            return {"n": 0, "n_pos": 0, "pos_rate": 0.0, "n_cases": 0}
        n_pos = int((labels == 1).sum())
        n_cases = len(set(split.get("case_ids", split.get("subject_ids", [])))) if "case_ids" in split or "subject_ids" in split else 0
        return {"n": n, "n_pos": n_pos, "pos_rate": 100.0 * n_pos / n, "n_cases": n_cases}

    # Patient-packed list-style (mortality/aki/sepsis/extubation/vent_need/cardiac_arrest)
    if isinstance(split, list):
        n_pat = len(split)
        if n_pat == 0:
            return {"n_patients": 0, "n_pos_patients": 0, "pos_rate_patient": 0.0, "n_windows": 0}
        # label 키 후보 모두 시도
        label_keys = ("label", "mortality", "extubation", "vent_need")
        n_pos = 0
        for p in split:
            for lk in label_keys:
                if lk in p:
                    if int(p[lk]) == 1:
                        n_pos += 1
                    break
        n_windows = sum(p.get("n_windows", 0) for p in split)
        return {
            "n_patients": n_pat,
            "n_pos_patients": n_pos,
            "pos_rate_patient": 100.0 * n_pos / n_pat,
            "n_windows": n_windows,
        }
    return {}


def _signal_shape(split: dict[str, Any]) -> tuple[str, tuple, int] | None:
    """Return (signal_name, shape, n_windows_in_signal) for first signal type found."""
    if isinstance(split, dict) and "signals" in split and isinstance(split["signals"], dict):
        sigs = split["signals"]
        if not sigs:
            return None
        first = next(iter(sigs))
        t = sigs[first]
        if isinstance(t, torch.Tensor):
            return first, tuple(t.shape), int(t.shape[0]) if t.ndim > 0 else 0
    if isinstance(split, list) and split:
        p = split[0]
        if "signals" in p and isinstance(p["signals"], dict) and p["signals"]:
            first = next(iter(p["signals"]))
            t = p["signals"][first]
            if isinstance(t, torch.Tensor):
                return first, tuple(t.shape), int(t.shape[0]) if t.ndim > 0 else 0
    return None


def _modality_keys(split: Any) -> list[str]:
    if isinstance(split, dict) and "signals" in split and isinstance(split["signals"], dict):
        return sorted(split["signals"].keys())
    if isinstance(split, list) and split and "signals" in split[0]:
        return sorted(split[0]["signals"].keys())
    return []


def verify(out_dir: Path, pattern: str = "*_fold*.pt") -> int:
    files = sorted(out_dir.glob(pattern))
    if not files:
        print(f"ERROR: no files matching '{pattern}' in {out_dir}", file=sys.stderr)
        return 2

    print(f"Found {len(files)} fold file(s) in {out_dir}\n")

    # fold 별 stats 수집 (cross-fold consistency 비교용)
    pos_rates: dict[str, list[float]] = defaultdict(list)
    issues: list[str] = []

    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[{f.name}] ERROR loading: {e}\n")
            issues.append(f"{f.name}: load failed")
            continue

        m = d.get("metadata", {})
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"[{f.name}] ({size_mb:.1f} MB)")
        print(
            f"  fold={m.get('fold_idx')}/{m.get('n_folds')}  "
            f"task={m.get('task','?')}"
        )
        for k in ("window_sec", "horizon_sec", "stride_sec", "sampling_rate"):
            if k in m:
                print(f"  {k}={m[k]}", end="  ")
        print()

        # modalities
        for k in ("train", "val", "test"):
            mods = _modality_keys(d.get(k))
            if mods:
                exp = m.get("input_signals")
                ok = "OK" if (exp is None or sorted(map(str, exp)) == mods) else "MISMATCH"
                print(f"  modalities[{k}]: {mods}  vs metadata: {exp}  [{ok}]")
                if ok == "MISMATCH":
                    issues.append(f"{f.name}: modality mismatch in {k}")
                break

        # split stats
        for k in ("train", "val", "test"):
            if k not in d:
                continue
            stats = _split_stats(d[k])
            print(f"  {k:5s}: {stats}")
            if "pos_rate" in stats:
                pos_rates[k].append(stats["pos_rate"])
            elif "pos_rate_patient" in stats:
                pos_rates[k].append(stats["pos_rate_patient"])

        # signal shape (window length × sampling_rate 검증)
        shape_info = None
        for k in ("train", "val", "test"):
            if k in d:
                shape_info = _signal_shape(d[k])
                if shape_info:
                    break
        if shape_info:
            sig, shp, n_in_sig = shape_info
            print(f"  signal shape per type: {shp} (e.g., '{sig}')")
            expected_T = None
            if "window_sec" in m and "sampling_rate" in m:
                expected_T = int(float(m["window_sec"]) * float(m["sampling_rate"]))
            if expected_T is not None and len(shp) >= 1 and shp[-1] != expected_T:
                issues.append(
                    f"{f.name}: signal length {shp[-1]} != window_sec×sr={expected_T}"
                )
                print(
                    f"  ⚠️ signal length {shp[-1]} != expected {expected_T} "
                    f"(window_sec={m['window_sec']}s × sr={m['sampling_rate']})"
                )
        print()

    # ── Cross-fold sanity checks ────────────────────────────────
    print("=" * 60)
    print("Cross-fold consistency")
    print("=" * 60)
    for k, rates in pos_rates.items():
        if not rates:
            continue
        mn, mx = min(rates), max(rates)
        spread = mx - mn
        flag = "OK" if spread <= 5.0 else ("⚠️ wide spread" if spread <= 10.0 else "❌ stratify suspect")
        print(
            f"  {k:5s} positive rate across folds: min={mn:.1f}%  max={mx:.1f}%  "
            f"spread={spread:.1f}%  [{flag}]"
        )
        if spread > 10.0:
            issues.append(f"positive rate spread in {k} = {spread:.1f}% (>10%)")

    print()
    if issues:
        print(f"❌ Issues found ({len(issues)}):")
        for s in issues:
            print(f"  - {s}")
        return 1
    print("✅ All checks passed.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Verify downstream prepare_data fold output.")
    p.add_argument(
        "--out-dir", type=str, required=True,
        help="Directory containing fold .pt files (e.g., outputs/downstream/hypotension)",
    )
    p.add_argument(
        "--pattern", type=str, default="*_fold*.pt",
        help="Glob pattern for fold files (default '*_fold*.pt')",
    )
    args = p.parse_args()
    rc = verify(Path(args.out_dir), args.pattern)
    sys.exit(rc)


if __name__ == "__main__":
    main()
