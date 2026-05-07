"""Intellivue .vital 파일 진단 — 왜 모든 track 이 SKIP 되는지 빠른 검사.

K-MIMIC 재파싱이 99.96% 파일에서 0개 recording 으로 종료되는 문제를 진단하기 위해
샘플 파일 N 개에 대해:
  1. 가용 track 이름과 길이
  2. 각 track 의 sample 값 통계 (min/p1/p50/p99/max, NaN%)
  3. 파싱 valid_range 와 비교
  4. 연속 NaN-free 최장 구간 길이

사용법:
    python -m scripts.diagnose_intellivue \
        --list /home/coder/workspace/updown/kmimic_vital_remaining.txt \
        --n-samples 10 \
        --filter-substring Intellivue
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

from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    TRACK_MAP,
    _apply_range_check,
    _detect_electrocautery,
    _detect_motion_artifact,
    _detect_step_change,
    _extract_nan_free_segments,
)

# data/parser/vitaldb.py 의 valid_range 와 동기화 (참고용)
VALID_RANGE = {
    "ECG": (-5.0, 5.0),
    "ABP": (20.0, 300.0),
    "PLETH": (0.0, 2000.0),
    "CVP": (-5.0, 40.0),
    "CO2": (0.0, 100.0),
    "AWP": (-20.0, 80.0),
    "RESP": (-200.0, 5000.0),
}


def stats(values: np.ndarray) -> dict:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return {"n": int(values.size), "nan_pct": 100.0, "values_finite": 0}
    return {
        "n": int(values.size),
        "nan_pct": 100.0 * (1 - finite.size / values.size),
        "min": float(finite.min()),
        "p1": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
    }


def longest_nanfree_run(values: np.ndarray) -> int:
    """NaN 이 아닌 가장 긴 연속 구간의 sample 수."""
    isnan = np.isnan(values)
    if not isnan.any():
        return int(values.size)
    valid = (~isnan).astype(np.int8)
    diff = np.diff(valid, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        return 0
    return int((ends - starts).max())


def short_track_label(track_name: str) -> str:
    """ECG_II → ECG, FLOW_WAV → AWP, CVP → CVP 같이 단순화."""
    upper = track_name.upper()
    for key in VALID_RANGE:
        if key in upper:
            return key
    return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--filter-substring", type=str, default="Intellivue",
                   help="이 substring 가진 path 만 샘플링 (default Intellivue)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    with open(args.list, encoding="utf-8") as f:
        all_paths = [
            line.strip() for line in f
            if line.strip() and (
                args.filter_substring in line
                if args.filter_substring else True
            )
        ]
    print(f"Filtered paths matching '{args.filter_substring}': {len(all_paths)}")
    if not all_paths:
        print("(no match — try --filter-substring '' to disable)")
        return

    sampled = rng.sample(all_paths, k=min(args.n_samples, len(all_paths)))

    for idx, vital_path in enumerate(sampled, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(sampled)}] {vital_path}")
        try:
            vf = vitaldb.VitalFile(vital_path)
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            continue

        try:
            track_names = vf.get_track_names()
        except Exception as e:
            print(f"  TRACK LIST ERROR: {e}")
            continue
        print(f"  Tracks ({len(track_names)}): {track_names[:20]}"
              + (" ..." if len(track_names) > 20 else ""))

        for tn in track_names:
            mapping = TRACK_MAP.get(tn)
            if mapping is None:
                continue  # numeric/unmapped track 은 파서가 무시
            stype_key, _ = mapping
            cfg = SIGNAL_CONFIGS.get(stype_key)
            if cfg is None:
                continue

            try:
                trk = vf.find_track(tn)
                native_sr = trk.srate if trk is not None and trk.srate > 0 else 500.0
                samples = vf.to_numpy(tn, interval=1.0 / native_sr)
            except Exception as e:
                print(f"    {tn}: to_numpy ERROR: {e}")
                continue
            if samples is None or samples.size == 0:
                print(f"    {tn}: EMPTY")
                continue
            samples = samples.flatten()

            n_total = samples.size
            run_raw = longest_nanfree_run(samples) / native_sr

            # 파서와 동일 pipeline 단계별 적용
            data = samples.copy()
            stages: list[tuple[str, float]] = [("raw", run_raw)]

            if cfg.valid_range is not None:
                data, _ = _apply_range_check(data, cfg.valid_range)
                stages.append(("range", longest_nanfree_run(data) / native_sr))

            if cfg.spike_detection:
                data, _ = _detect_electrocautery(
                    data, native_sr, threshold_std=cfg.spike_threshold_std
                )
                stages.append(("spike", longest_nanfree_run(data) / native_sr))

            if stype_key in ("abp", "ppg"):
                data, _ = _detect_step_change(data, native_sr)
                stages.append(("step", longest_nanfree_run(data) / native_sr))

            if stype_key == "ppg":
                data, _ = _detect_motion_artifact(data, native_sr)
                stages.append(("motion", longest_nanfree_run(data) / native_sr))

            min_samples = int(60.0 * native_sr)
            final_segs = _extract_nan_free_segments(data, min_samples)
            n_final_seg = len(final_segs)

            stage_str = " → ".join(f"{name}={v:.0f}s" for name, v in stages)
            verdict = (
                f"OK ({n_final_seg} seg)"
                if n_final_seg > 0
                else "SKIP (<60s 연속 없음)"
            )
            print(f"    {tn:30s} sr={native_sr:.0f} n={n_total:>8d} "
                  f"longest_nanfree[{stage_str}] → {verdict}")


if __name__ == "__main__":
    main()
