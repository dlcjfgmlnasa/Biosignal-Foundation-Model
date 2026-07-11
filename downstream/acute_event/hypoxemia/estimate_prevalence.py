# -*- coding:utf-8 -*-
"""전체 코호트 hypoxemia prevalence + 양성 환자 수를 **파형 저장 없이** 추정한다.

prepare_data.py 는 파형 .pt 를 로드·저장(수 시간 + 100GB+)해야 prevalence 가 나온다.
이 도구는 그 전에 "이 라벨 정의가 평가 가능한 코호트를 만드는가"를 싸게 확인한다:
  - manifest 만으로 ecg+ppg 교집합 구간(start,end) 계산 (파형 .pt 미로드)
  - 그 구간에 대해 raw .vital 의 SpO2 1Hz trend 만 로드(작음)
  - window(=stride) 별로 prepare 와 동일한 라벨 규칙 적용 → 라벨 변이별 집계
    · window prevalence (양성 window / 전체 window)
    · **양성 환자 수** (평가 가능성의 실제 병목 — stratified CV 는 환자 단위)

파형 quality(valid_ratio) drop 은 반영하지 않으므로 window prevalence 는 상한 근사다.
양성 환자 수는 정확에 가깝다(라벨은 SpO2 에서만 옴).

사용법:
    python -m downstream.acute_event.hypoxemia.estimate_prevalence \
        --data-dir /home/coder/workspace/updown/parser/vitaldb \
        --raw-dir  /home/coder/workspace/datasets/vitaldb_open/1.0.0 \
        --window-sec 300 --horizons 5 10 15 --workers 16
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from downstream._seg_intersect import (
    _find_containing_seg,
    _intersect_intervals,
    _merge_overlapping_intervals,
)
from downstream.acute_event.hypoxemia.inspect_cohort import _subject_segments
from downstream.acute_event.hypoxemia.prepare_data import (
    _load_spo2_trend,
    _resolve_raw_vital_path,
    build_raw_vital_index,
)

TARGET_SR = 100.0
BASELINE = 30

# (name, threshold, sustain_sec, gate)  sustain<=1 → any-point
VARIANTS = [
    ("anypoint92", 92.0, 1, True),
    ("sus95_60s", 95.0, 60, False),
    ("sus92_60s", 92.0, 60, False),
]


def _spans_ecg_ppg(sig_segs: dict, min_samples: int) -> list[tuple[int, int]]:
    """manifest segment → ecg+ppg 동시 valid 연속 구간 (단일 seg full-cover) 목록."""
    req = ("ecg", "ppg")
    iv = {}
    for s in req:
        recs = sig_segs.get(s, [])
        if not recs:
            return []
        iv[s] = _merge_overlapping_intervals(
            [(r.start_sample, r.start_sample + r.n_timesteps) for r in recs])
    out = []
    for a, b in _intersect_intervals(iv):
        if b - a < min_samples:
            continue
        if all(_find_containing_seg(sig_segs[s], a, b) for s in req):
            out.append((a, b))
    return out


def _label_windows(spo2, spans, window_sec, horizon_sec, stride_sec,
                   thr, sustain, allow_tail):
    """한 환자의 spans 에서 window 별 라벨 집계. Returns (n_win, n_pos)."""
    win = int(window_sec * TARGET_SR)
    stride = int(stride_sec * TARGET_SR)
    hz = int(horizon_sec * TARGET_SR)
    need = win if allow_tail else win + hz
    n_win = n_pos = 0
    for a, b in spans:
        seg_len = b - a
        if seg_len < need:
            continue
        for st in range(0, seg_len - need + 1, stride):
            we_abs = (a + st + win) / TARGET_SR            # window end 절대 초 (dtstart 원점)
            f0, f1 = int(round(we_abs)), int(round(we_abs + horizon_sec))
            if f0 < 0 or f1 > len(spo2):
                continue
            base = spo2[max(0, f0 - BASELINE):f0]
            base = base[(~np.isnan(base)) & (base >= 50) & (base <= 100)]
            if base.size == 0 or np.median(base) < thr:     # baseline 가드
                continue
            fut = spo2[f0:f1]
            fut = fut[(~np.isnan(fut)) & (fut >= 50) & (fut <= 100)]
            if fut.size < max(1, sustain // 2):
                continue
            if sustain <= 1:
                lab = bool(np.any(fut < thr))
            else:
                c = 0; lab = False
                for v in fut:
                    if v < thr:
                        c += 1
                        if c >= sustain:
                            lab = True; break
                    else:
                        c = 0
            n_win += 1
            n_pos += int(lab)
    return n_win, n_pos


def _process(subj_dir: Path, raw_index, window_sec, horizons, stride_sec,
             min_samples, allow_tail):
    """한 환자 → 변이×horizon 별 (n_win, n_pos) + 양성여부."""
    sig_segs = _subject_segments(subj_dir)
    if not sig_segs:
        return None
    spans = _spans_ecg_ppg(sig_segs, min_samples)
    if not spans:
        return None
    rv = _resolve_raw_vital_path(raw_index, subj_dir.name)
    if rv is None:
        return None
    # 게이트는 변이마다 다르지만, 양성 환자 수 상한 확인엔 무게 게이트 없이 로드(빠름).
    spo2 = _load_spo2_trend(rv, artifact_gate=False)
    if spo2 is None:
        return None
    out = {}
    for name, thr, sustain, _gate in VARIANTS:
        for h in horizons:
            nw, npz = _label_windows(spo2, spans, window_sec, h * 60.0,
                                     stride_sec, thr, sustain, allow_tail)
            out[(name, h)] = (nw, npz)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="hypoxemia 전체 코호트 prevalence 추정(저장 없음)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--window-sec", type=float, default=300.0)
    ap.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 15])
    ap.add_argument("--stride-sec", type=float, default=30.0)
    ap.add_argument("--allow-tail-windows", action="store_true")
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="양성 환자 '비율' 추정엔 무작위 ~800 이면 충분(전체 4.5h 회피). "
                         "--sample-seed 와 함께 대표 표본.")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="max-subjects 제한 시 정렬 앞 N 대신 이 seed 로 무작위 N (대표성).")
    ap.add_argument("--workers", type=int, default=16,
                    help="raw .vital SpO2 로딩 I/O bound → 서버(64 vCPU)에선 32~48 권장.")
    args = ap.parse_args()

    max_h = max(args.horizons) * 60.0
    min_dur = args.window_sec + (args.stride_sec if args.allow_tail_windows
                                 else max_h + args.stride_sec)
    min_samples = int(min_dur * TARGET_SR)

    root = Path(args.data_dir)
    subj_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    total_dirs = len(subj_dirs)
    if args.max_subjects and total_dirs > args.max_subjects:
        if args.sample_seed is not None:
            import random
            subj_dirs = sorted(random.Random(args.sample_seed).sample(subj_dirs, args.max_subjects))
        else:
            subj_dirs = subj_dirs[:args.max_subjects]
    print(f"subjects: {len(subj_dirs)}/{total_dirs}  | window={args.window_sec:.0f}s "
          f"horizons={args.horizons} allow_tail={args.allow_tail_windows} "
          f"(min_dur={min_dur:.0f}s)", file=sys.stderr)
    print("indexing raw .vital ...", file=sys.stderr)
    raw_index = build_raw_vital_index(Path(args.raw_dir))

    # agg[(variant,h)] = [n_win, n_pos, pos_patients]
    agg = {(n, h): [0, 0, 0] for n, _, _, _ in VARIANTS for h in args.horizons}
    n_pat = 0
    try:
        from tqdm import tqdm
        prog = lambda it, **k: tqdm(it, **k)  # noqa: E731
    except ImportError:
        prog = lambda it, **k: it  # noqa: E731

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process, d, raw_index, args.window_sec, args.horizons,
                          args.stride_sec, min_samples, args.allow_tail_windows)
                for d in subj_dirs]
        for fut in prog(as_completed(futs), total=len(futs), desc="scan", unit="subj"):
            r = fut.result()
            if r is None:
                continue
            n_pat += 1
            for key, (nw, npz) in r.items():
                agg[key][0] += nw
                agg[key][1] += npz
                if npz > 0:
                    agg[key][2] += 1

    # 표본 → 전체 코호트 외삽 배수 (대표 표본 가정).
    scale = total_dirs / max(1, len(subj_dirs))
    est_full_ecgppg = int(round(n_pat * scale))
    print(f"\n=== hypoxemia prevalence 추정 "
          f"(표본 ecg+ppg {n_pat} 환자 → 전체 ~{est_full_ecgppg} 추정, ×{scale:.1f}) ===")
    print(f"{'variant':>12} {'horizon':>8} {'win prev':>9} {'양성환자':>8} "
          f"{'환자율':>7} {'전체추정':>9}")
    for name, _, _, _ in VARIANTS:
        for h in args.horizons:
            nw, npz, pp = agg[(name, h)]
            wp = 100 * npz / max(1, nw)
            pr = 100 * pp / max(1, n_pat)
            est_pos = int(round(pp * scale))   # 전체 코호트 양성 환자 외삽
            print(f"{name:>12} {h:>6}분 {wp:>8.2f}% {pp:>8} {pr:>6.1f}% {est_pos:>9}")
    print("\n판정 기준: '전체추정' 양성환자 ≥ ~50 이면 5-fold 평가 가능(각 test ~10명).")
    print("  window prevalence 는 gap-policy drop 미반영 상한(실제는 약간 낮음).")
    print("  게이트는 양성환자 수에 거의 영향 없어 미적용(로딩 가속).")


if __name__ == "__main__":
    main()
