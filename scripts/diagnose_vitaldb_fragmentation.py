# -*- coding:utf-8 -*-
"""VitalDB parsed manifest 의 segment fragmentation 진단.

각 subject 의 manifest.json (parser 출력) 을 읽어:
  - (signal_type) 별 timeline 재구성 (start_sample + n_timesteps)
  - segment 간 gap 분포, 평균 segment 길이, longest contiguous span
  - 다양한 max_gap_sec 임계에서 "merge 후" 가능한 longest span 통계

목표: Option A (downstream merge utility) 의 효과 사전 추정.
      "어느 정도 합치면 paper window (win+horizon ≤ 25분) 가 얼마나 확보되는가."

사용법:
    python -m scripts.diagnose_vitaldb_fragmentation \
        --data-dir /home/coder/workspace/updown/parser/vitaldb \
        --signal-types abp ecg ppg \
        --max-gaps 0 5 30 60 300 \
        --target-spans 600 900 1500 \
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median, mean

from tqdm import tqdm

TARGET_SR: float = 100.0

# signal_type int → str (parser 와 동일 매핑)
SIGNAL_TYPE_STR: dict[int, str] = {
    0: "ecg", 1: "abp", 2: "ppg", 3: "cvp",
    4: "co2", 5: "awp", 6: "pap", 7: "icp", 8: "resp",
}
STR_TO_SIGNAL_TYPE: dict[str, int] = {v: k for k, v in SIGNAL_TYPE_STR.items()}


# ── Manifest 로딩 ──────────────────────────────────────────────

def _load_manifest(subj_dir: Path) -> dict | None:
    """subject 디렉토리에서 manifest.json 로드."""
    mp = subj_dir / "manifest.json"
    if not mp.is_file():
        return None
    try:
        with open(mp, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def scan_manifests(
    root: Path,
    max_subjects: int | None,
    workers: int,
) -> dict[str, dict]:
    """전체 subject manifest 병렬 로딩."""
    print(f"  Listing {root} ...", flush=True)
    subject_dirs: list[Path] = []
    for entry in tqdm(os.scandir(root), desc="  Listing entries", unit="entry",
                      dynamic_ncols=True):
        if entry.is_dir():
            subject_dirs.append(Path(entry.path))
    subject_dirs.sort()
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]
    print(f"  → {len(subject_dirs)} subject directory(ies)", flush=True)

    manifests: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_load_manifest, sd): sd for sd in subject_dirs}
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="  Loading manifests", unit="subj", dynamic_ncols=True)
        for fut in pbar:
            sd = futures[fut]
            m = fut.result()
            if m is not None:
                manifests[sd.name] = m
    print(f"  → {len(manifests):,} manifests loaded "
          f"({100 * len(manifests) / max(len(subject_dirs), 1):.1f}%)",
          flush=True)
    return manifests


# ── 분석 ───────────────────────────────────────────────────────

def build_timeline(
    manifests: dict[str, dict],
    signal_type_int: int,
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """{(subject_id, session_id): [(start_sample, n_timesteps), ...]} — start_sample 정렬됨.

    하나의 (subject, session) 안의 같은 signal_type 의 segment 들을 timeline 순으로 정렬.
    """
    out: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for sid, m in manifests.items():
        for sess in m.get("sessions", []):
            sess_id = sess.get("session_id", "")
            for rec in sess.get("recordings", []):
                if rec.get("signal_type") != signal_type_int:
                    continue
                ss = rec.get("start_sample")
                nt = rec.get("n_timesteps")
                if ss is None or nt is None:
                    continue
                out[(sid, sess_id)].append((int(ss), int(nt)))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def compute_per_case_stats(
    timeline: dict[tuple[str, str], list[tuple[int, int]]],
    sr: float = TARGET_SR,
) -> list[dict]:
    """case 별 fragmentation 통계."""
    stats = []
    for (sid, sess_id), segs in timeline.items():
        if not segs:
            continue
        n_seg = len(segs)
        valid_samples = sum(nt for _, nt in segs)
        first_start = segs[0][0]
        last_end = segs[-1][0] + segs[-1][1]
        span_samples = last_end - first_start
        gaps_samples = []
        for (s1, n1), (s2, _n2) in zip(segs[:-1], segs[1:]):
            gap = s2 - (s1 + n1)
            gaps_samples.append(max(gap, 0))
        longest_seg_samples = max((nt for _, nt in segs), default=0)
        stats.append({
            "subject_id": sid,
            "session_id": sess_id,
            "n_segments": n_seg,
            "valid_sec": valid_samples / sr,
            "span_sec": span_samples / sr,
            "valid_ratio": valid_samples / max(span_samples, 1),
            "longest_segment_sec": longest_seg_samples / sr,
            "gaps_sec": [g / sr for g in gaps_samples],
        })
    return stats


def simulate_merge(
    segs: list[tuple[int, int]],
    max_gap_sec: float,
    sr: float = TARGET_SR,
) -> tuple[int, int, int]:
    """max_gap_sec 임계로 segment 들을 group 묶기.

    Returns (n_groups, longest_group_samples, longest_group_filled_samples)
        longest_group_filled_samples: 그 그룹 내 gap 합 (= 채워야 할 분량)
    """
    if not segs:
        return 0, 0, 0
    max_gap_samples = int(max_gap_sec * sr)
    groups: list[list[tuple[int, int]]] = [[segs[0]]]
    for prev, curr in zip(segs[:-1], segs[1:]):
        prev_end = prev[0] + prev[1]
        gap = curr[0] - prev_end
        if gap <= max_gap_samples:
            groups[-1].append(curr)
        else:
            groups.append([curr])

    longest_span = 0
    longest_fill = 0
    for grp in groups:
        span = (grp[-1][0] + grp[-1][1]) - grp[0][0]
        valid = sum(nt for _, nt in grp)
        fill = span - valid
        if span > longest_span:
            longest_span = span
            longest_fill = fill
    return len(groups), longest_span, longest_fill


# ── 리포트 ─────────────────────────────────────────────────────

def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = int(q * (len(vs) - 1))
    return vs[k]


def print_signal_report(
    signal_str: str,
    stats: list[dict],
    timelines: dict[tuple[str, str], list[tuple[int, int]]],
    max_gaps_sec: list[float],
    target_spans_sec: list[float],
) -> None:
    bar = "─" * 100
    print(f"\n{bar}")
    print(f"  Signal: {signal_str.upper()}")
    print(bar)

    if not stats:
        print("  (no cases with this signal)")
        return

    n_case = len(stats)
    n_segs_list = [s["n_segments"] for s in stats]
    valid_list = [s["valid_sec"] for s in stats]
    span_list = [s["span_sec"] for s in stats]
    ratio_list = [s["valid_ratio"] for s in stats]
    longest_seg_list = [s["longest_segment_sec"] for s in stats]
    all_gaps = [g for s in stats for g in s["gaps_sec"]]

    print(f"  Cases : {n_case:,}")
    print()
    print("  Per-case 통계 (median [p10..p90])")
    print(f"    n_segments     : {median(n_segs_list):>8.0f}    "
          f"[{pct(n_segs_list,0.1):.0f} .. {pct(n_segs_list,0.9):.0f}]")
    print(f"    valid 시간 (s)  : {median(valid_list):>8.0f}    "
          f"[{pct(valid_list,0.1):.0f} .. {pct(valid_list,0.9):.0f}]")
    print(f"    span (s)       : {median(span_list):>8.0f}    "
          f"[{pct(span_list,0.1):.0f} .. {pct(span_list,0.9):.0f}]")
    print(f"    valid/span     : {median(ratio_list):>8.2f}    "
          f"[{pct(ratio_list,0.1):.2f} .. {pct(ratio_list,0.9):.2f}]")
    print(f"    longest seg (s): {median(longest_seg_list):>8.0f}    "
          f"[{pct(longest_seg_list,0.1):.0f} .. {pct(longest_seg_list,0.9):.0f}]")

    if all_gaps:
        print()
        print("  Gap 분포 (전체 segment 간격, sec)")
        print(f"    n_gaps       : {len(all_gaps):,}")
        print(f"    p10/p50/p90  : "
              f"{pct(all_gaps,0.1):.1f} / {pct(all_gaps,0.5):.1f} / {pct(all_gaps,0.9):.1f}")
        print(f"    min / max    : {min(all_gaps):.1f} / {max(all_gaps):.1f}")

    # Merge simulation
    print()
    print(f"  Merge simulation @ 다양한 max_gap_sec — "
          f"longest merged span 분포")
    print(f"    max_gap   |  median  |  p10    |  p90    | filled_pct")
    print(f"    ----------|----------|---------|---------|----------")
    for mg in max_gaps_sec:
        spans = []
        fill_pcts = []
        for case_key, segs in timelines.items():
            _, longest_span, longest_fill = simulate_merge(segs, mg)
            spans.append(longest_span / TARGET_SR)
            if longest_span > 0:
                fill_pcts.append(100 * longest_fill / longest_span)
        if not spans:
            continue
        print(f"    {mg:>6.0f} s  |  {median(spans):>5.0f} s |"
              f" {pct(spans,0.1):>5.0f} s |"
              f" {pct(spans,0.9):>5.0f} s |"
              f"  median {median(fill_pcts):.1f}%")

    # Coverage at target spans
    print()
    print(f"  Target span 달성률 — '한 case 의 가장 긴 merged group ≥ T 초' 인 case 비율")
    print(f"    max_gap  |" + "".join(f"  ≥ {int(t):>4}s   " for t in target_spans_sec))
    print(f"    ---------|" + "-" * (15 * len(target_spans_sec)))
    for mg in max_gaps_sec:
        cells = []
        for tgt in target_spans_sec:
            ok = 0
            for case_key, segs in timelines.items():
                _, longest_span, _ = simulate_merge(segs, mg)
                if longest_span / TARGET_SR >= tgt:
                    ok += 1
            pct_ok = 100 * ok / max(len(timelines), 1)
            cells.append(f"  {ok:>5,d} ({pct_ok:>4.1f}%)")
        print(f"    {mg:>5.0f} s  |" + "".join(cells))


# ── 메인 ────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--signal-types", nargs="+", default=["abp", "ecg", "ppg"],
                    help="진단할 signal type 문자열 (default: abp ecg ppg)")
    ap.add_argument("--max-gaps", nargs="+", type=float,
                    default=[0.0, 5.0, 30.0, 60.0, 300.0],
                    help="merge 시뮬레이션 max_gap_sec 후보 (default: 0 5 30 60 300)")
    ap.add_argument("--target-spans", nargs="+", type=float,
                    default=[600.0, 900.0, 1500.0],
                    help="달성률 평가용 target span (초). default: 600 900 1500 "
                         "(= win=600/h=15min, win=300/h=10min 등)")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        print(f"ERROR: data-dir not found: {root}", file=sys.stderr)
        return 1

    print(f"  Scanning manifests from {root} (max_subjects={args.max_subjects})",
          flush=True)
    manifests = scan_manifests(root, args.max_subjects, args.workers)
    if not manifests:
        print("ERROR: no manifests loaded", file=sys.stderr)
        return 1

    unknown = [s for s in args.signal_types if s not in STR_TO_SIGNAL_TYPE]
    if unknown:
        print(f"ERROR: unknown signal_type(s): {unknown}", file=sys.stderr)
        return 1

    for sig_str in args.signal_types:
        st_int = STR_TO_SIGNAL_TYPE[sig_str]
        timelines = build_timeline(manifests, st_int)
        stats = compute_per_case_stats(timelines)
        print_signal_report(
            sig_str, stats, timelines, args.max_gaps, args.target_spans,
        )

    print("\n  해석 가이드:")
    print("  • valid/span 비율이 낮으면 (예: 0.2 = 20%) raw 시간선의 80%가 gap.")
    print("  • max_gap=0 vs max_gap=300 의 longest merged span 차이가 작으면 → 합쳐도 효과 미미.")
    print("  • Target span 달성률이 max_gap 증가에도 안 늘면 → 짧은 segment 수 자체가 문제.")
    print("  • filled_pct 가 20% 넘으면 합치기 어려움 (model 학습 distortion 위험).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
