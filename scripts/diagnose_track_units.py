"""특정 track 의 raw 값 분포(min/p1/p50/p99/max) 를 N개 파일에서 집계.

단위 미스매치 의심 시 (e.g. SNUADCM/CVP, Intellivue/ART) raw sample 값이
parser 의 valid_range 와 어긋나는지 빠르게 확정하기 위한 도구.

사용법:
    python -m scripts.diagnose_track_units \
        --list /path/to/kmimic_vital_list.txt \
        --track SNUADCM/CVP \
        --n-samples 30
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

try:
    import vitaldb
except ImportError:
    print("ERROR: vitaldb 패키지 필요")
    raise SystemExit(1)

from data.parser.vitaldb import SIGNAL_CONFIGS, TRACK_MAP


def percentile_stats(values: np.ndarray) -> dict | None:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return None
    return {
        "n": int(values.size),
        "n_finite": int(finite.size),
        "nan_pct": 100.0 * (1 - finite.size / values.size),
        "min": float(finite.min()),
        "p01": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list", type=Path, required=True,
                   help=".vital file paths 리스트 (한 줄 하나)")
    p.add_argument("--track", type=str, required=True,
                   help="조사할 track 이름 (e.g. SNUADCM/CVP, Intellivue/ART)")
    p.add_argument("--n-samples", type=int, default=30,
                   help="track 이 존재하는 파일 N 개까지 조사")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-scan", type=int, default=2000,
                   help="track 찾기 위해 최대 스캔할 파일 수")
    args = p.parse_args()

    rng = random.Random(args.seed)
    with open(args.list, encoding="utf-8") as f:
        all_paths = [ln.strip() for ln in f if ln.strip()]
    rng.shuffle(all_paths)
    candidates = all_paths[: args.max_scan]
    print(f"List size={len(all_paths)} scan_cap={len(candidates)} target_track={args.track}")

    mapping = TRACK_MAP.get(args.track)
    if mapping is None:
        print(f"WARNING: {args.track} not in TRACK_MAP — proceeding anyway")
        cfg = None
        valid_range = None
    else:
        stype_key, _ = mapping
        cfg = SIGNAL_CONFIGS.get(stype_key)
        valid_range = cfg.valid_range if cfg else None
        print(f"Mapping: {args.track} → {stype_key}, valid_range={valid_range}")

    found = 0
    n_load_err = 0
    file_summaries: list[dict] = []

    for path in candidates:
        if found >= args.n_samples:
            break
        try:
            vf = vitaldb.VitalFile(path)
            tracks = vf.get_track_names()
        except Exception as e:
            n_load_err += 1
            continue

        if args.track not in tracks:
            continue

        try:
            trk = vf.find_track(args.track)
            sr = trk.srate if trk is not None and trk.srate > 0 else 500.0
            samples = vf.to_numpy(args.track, interval=1.0 / sr).flatten()
        except Exception as e:
            print(f"  to_numpy ERROR ({path}): {e}")
            continue

        st = percentile_stats(samples)
        if st is None:
            print(f"  EMPTY  {path}")
            continue

        if valid_range is not None:
            lo, hi = valid_range
            mask = (samples < lo) | (samples > hi)
            in_range_pct = 100.0 * (1 - mask.sum() / max(1, samples.size))
        else:
            in_range_pct = float("nan")

        st["in_range_pct"] = in_range_pct
        st["sr"] = sr
        st["path"] = path
        file_summaries.append(st)
        found += 1

        print(
            f"  [{found:2d}] sr={sr:.0f} n={st['n']:>7d} nan%={st['nan_pct']:5.1f} "
            f"min={st['min']:>9.2f} p01={st['p01']:>8.2f} p50={st['p50']:>8.2f} "
            f"p99={st['p99']:>8.2f} max={st['max']:>9.2f} in_range%={in_range_pct:5.1f}"
        )

    print(f"\n{'='*80}")
    print(f"Summary: found {found} files with track {args.track}")
    print(f"  Load errors: {n_load_err}")
    if not file_summaries:
        print("  (no aggregate)")
        return

    keys = ["min", "p01", "p50", "p99", "max", "in_range_pct"]
    arr = {k: np.array([s[k] for s in file_summaries]) for k in keys}
    print(f"  {'metric':>13s}  {'min':>9s}  {'median':>9s}  {'max':>9s}")
    for k in keys:
        v = arr[k]
        print(f"  {k:>13s}  {v.min():>9.2f}  {np.median(v):>9.2f}  {v.max():>9.2f}")
    if valid_range is not None:
        print(f"\n  valid_range = {valid_range}")
        worst = min(s["in_range_pct"] for s in file_summaries)
        median_in = float(np.median(arr["in_range_pct"]))
        print(f"  in_range% — median={median_in:.1f}, worst-file={worst:.1f}")


if __name__ == "__main__":
    main()
