# -*- coding:utf-8 -*-
"""VitalDB parsed manifest 만 읽어 signal 조합별 코호트 크기를 진단한다.

`prepare_data.py` 가 "0 intersection intervals" 로 죽을 때, 원인이
  (a) 특정 signal 자체가 파싱돼 있지 않다  vs
  (b) 조합은 있으나 요구 **연속 길이**(min_dur)를 만족하는 구간이 없다
중 무엇인지 가른다. `.pt` 는 열지 않으므로 네트워크 마운트에서도 빠르다.

min_dur 는 prepare 의 계산과 같다:
    기본            = window + max_horizon + stride   (예: 300+900+30 = 1230s)
    --allow-tail-windows = window + stride            (예: 300+30    =  330s)

사용법:
    python -m downstream.acute_event.hypoxemia.inspect_cohort \
        --data-dir /home/coder/workspace/updown/parser/vitaldb --max-subjects 300
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from downstream._seg_intersect import (
    SIGNAL_TYPE_STR,
    _SegmentRef,
    _find_containing_seg,
    _intersect_intervals,
    _merge_overlapping_intervals,
)

TARGET_SR = 100.0

DEFAULT_COMBOS = [
    ("ecg",),
    ("ppg",),
    ("co2",),
    ("awp",),
    ("ecg", "ppg"),
    ("ecg", "ppg", "abp"),
    ("ecg", "ppg", "co2"),
    ("ecg", "ppg", "awp"),
    ("ecg", "ppg", "co2", "awp"),
]
DEFAULT_THRESHOLDS = [60.0, 330.0, 630.0, 930.0, 1230.0]


def _subject_segments(subj_dir: Path) -> dict[str, list[_SegmentRef]] | None:
    """manifest.json → {signal_key: [_SegmentRef]} (session 무시, 단일 세션 가정)."""
    mp = subj_dir / "manifest.json"
    if not mp.is_file():
        return None
    try:
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    sig_segs: dict[str, list[_SegmentRef]] = defaultdict(list)
    for sess in m.get("sessions", []):
        for rec in sess.get("recordings", []):
            st, ss = rec.get("signal_type"), rec.get("start_sample")
            nt, fn = rec.get("n_timesteps"), rec.get("file")
            if st is None or ss is None or nt is None or not fn:
                continue
            key = SIGNAL_TYPE_STR.get(int(st))
            if key is None:
                continue
            sig_segs[key].append(_SegmentRef(int(ss), int(nt), str(fn)))
    for k in sig_segs:
        sig_segs[k].sort(key=lambda x: x.start_sample)
    return dict(sig_segs)


def _best_interval(sig_segs: dict[str, list[_SegmentRef]], combo: tuple[str, ...]) -> float:
    """combo 의 교집합 중, 각 신호가 **단일 segment 로 완전히 덮는** 최장 구간(초).

    `_seg_intersect._load_one_subject` 의 채택 규칙을 그대로 재현한다.
    """
    intervals_by_sig: dict[str, list[tuple[int, int]]] = {}
    for sig in combo:
        recs = sig_segs.get(sig, [])
        if not recs:
            return 0.0
        intervals_by_sig[sig] = _merge_overlapping_intervals(
            [(s.start_sample, s.start_sample + s.n_timesteps) for s in recs]
        )
    best = 0
    for i_start, i_end in _intersect_intervals(intervals_by_sig):
        if all(_find_containing_seg(sig_segs[s], i_start, i_end) for s in combo):
            best = max(best, i_end - i_start)
        else:
            # 단일 seg 로 못 덮으면 prepare 도 drop → 0 으로 취급
            continue
    return best / TARGET_SR


def main() -> None:
    ap = argparse.ArgumentParser(description="VitalDB parsed cohort inspector (manifest only)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()

    root = Path(args.data_dir)
    subj_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if args.max_subjects:
        subj_dirs = subj_dirs[: args.max_subjects]
    print(f"subjects scanned: {len(subj_dirs)}  ({root})\n")

    per_subject: list[dict[str, list[_SegmentRef]]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_subject_segments, d) for d in subj_dirs]
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                per_subject.append(r)
    print(f"manifest 로드 성공: {len(per_subject)}\n")

    # (1) signal 존재 여부
    print("── signal 별 보유 subject 수 (구간 길이 무관) ──")
    present = defaultdict(int)
    for segs in per_subject:
        for k in segs:
            present[k] += 1
    for k in sorted(present, key=lambda x: -present[k]):
        print(f"  {k:16s} {present[k]:5d}  ({100.0*present[k]/max(1,len(per_subject)):.1f}%)")
    missing = [k for k in ("ecg", "ppg", "co2", "awp", "abp") if k not in present]
    if missing:
        print(f"  ⚠ 아예 없는 signal: {missing}  → 이 조합은 어떤 min_dur 로도 0 건")

    # (2) 조합 × 임계 길이
    print("\n── 조합별 subject 수 (최장 연속 교집합 ≥ 임계) ──")
    header = "  " + "combo".ljust(26) + "".join(f"{int(t)}s".rjust(9) for t in args.thresholds)
    print(header)
    print("  " + "-" * (26 + 9 * len(args.thresholds)))
    for combo in DEFAULT_COMBOS:
        bests = [_best_interval(s, combo) for s in per_subject]
        row = "  " + "+".join(combo).ljust(26)
        for t in args.thresholds:
            row += str(sum(1 for b in bests if b >= t)).rjust(9)
        print(row)

    print("\n  임계 해설: 1230s = window300 + horizon900 + stride30 (canonical 기본)")
    print("             330s = window300 + stride30 (--allow-tail-windows)")
    print("             630s/930s = horizon 5분/10분만 쓸 때")


if __name__ == "__main__":
    main()
