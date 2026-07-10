# -*- coding:utf-8 -*-
"""raw VitalDB `.vital` 에서 트랙 존재 여부 + 파서 품질 게이트 통과율을 측정한다.

`parser/vitaldb` 산출물에 CO2/AWP 가 거의 없을 때(예: 300 subject 중 1명),
원인이 셋 중 무엇인지 가른다.
  (A) 원본 .vital 에 트랙이 없다              → 트랙 존재율이 낮게 나옴
  (B) 파싱 시 --signal-types 로 제외했다      → 트랙은 있는데 통과율은 높음
  (C) 품질 게이트가 떨어뜨린다                → 트랙 있고 통과율이 0 근처

(C) 라면 어떤 체크(flatline/clip/high_freq/amplitude/domain)가 죽이는지까지 출력한다.
파서 코드는 읽기만 하고 수정하지 않는다.

사용법:
    python -m downstream.acute_event.hypoxemia.probe_raw_tracks \
        --raw-dir /home/coder/workspace/datasets/vitaldb_open/1.0.0 --n-files 12
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.parser._common import (
    resample_to_target,
    segment_quality_score,
)
from data.parser._quality_checks import domain_quality_check
from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    TRACK_MAP,
    _apply_filter,
    _apply_median_filter,
    _apply_notch_filter,
    _apply_range_check,
    _extract_nan_free_segments,
)

TARGET_SR = 100.0
WATCH = ("ecg", "ppg", "abp", "co2", "awp")


def _preprocess(seg: np.ndarray, cfg, native_sr: float) -> np.ndarray:
    """파서와 동일한 순서: median → notch → filter → resample(100Hz)."""
    if cfg.median_kernel > 0:
        seg = _apply_median_filter(seg, kernel_size=cfg.median_kernel)
    if cfg.notch_freq is not None:
        seg = _apply_notch_filter(seg, freq=cfg.notch_freq, sr=native_sr)
    seg = _apply_filter(seg, cfg, native_sr)
    if native_sr != TARGET_SR:
        seg = resample_to_target(seg, orig_sr=native_sr, target_sr=TARGET_SR)
    return seg


def probe_track(vf, track_name: str, stype_key: str) -> dict | None:
    """한 트랙에 대해 파서 게이트를 재현하고 실패 사유를 집계."""
    cfg = SIGNAL_CONFIGS[stype_key]
    trk = vf.find_track(track_name)
    native_sr = trk.srate if trk is not None and trk.srate > 0 else 0.0
    if native_sr <= 0:
        return None
    try:
        data = vf.to_numpy(track_name, interval=1.0 / native_sr).flatten()
    except Exception:
        return None
    if data.size == 0:
        return None

    if cfg.valid_range is not None:
        data, _ = _apply_range_check(data, cfg.valid_range)

    segs = _extract_nan_free_segments(data, min_samples=int(60 * native_sr))
    if not segs:
        return {"native_sr": native_sr, "n_windows": 0, "pass": 0,
                "fail": {}, "max_good_s": 0.0, "no_segment": True}

    win_samples = int(cfg.quality_window_s * TARGET_SR)
    fail = defaultdict(int)
    n_win = n_pass = 0
    best_run = run = 0

    for _, seg in segs:
        proc = _preprocess(np.asarray(seg, dtype=np.float64), cfg, native_sr)
        for i in range(0, len(proc) - win_samples + 1, win_samples):
            win = proc[i:i + win_samples]
            n_win += 1
            q = segment_quality_score(
                win,
                max_flatline_ratio=cfg.max_flatline_ratio,
                max_clip_ratio=cfg.max_clip_ratio,
                max_high_freq_ratio=cfg.max_high_freq_ratio,
                min_amplitude=cfg.min_amplitude,
                max_amplitude=cfg.max_amplitude,
                min_high_freq_ratio=cfg.min_high_freq_ratio,
            )
            if not q["pass"]:
                # 어떤 항목이 임계를 넘었는지 개별 판정
                if q["flatline_ratio"] > cfg.max_flatline_ratio:
                    fail["flatline"] += 1
                if q["clip_ratio"] > cfg.max_clip_ratio:
                    fail["clip"] += 1
                if q["high_freq_ratio"] > cfg.max_high_freq_ratio:
                    fail["high_freq"] += 1
                if cfg.min_amplitude > 0 and q["amplitude"] < cfg.min_amplitude:
                    fail["amplitude_low"] += 1
                run = 0
                continue
            d = domain_quality_check(stype_key, win, sr=TARGET_SR)
            if not d.get("pass", True):
                fail["domain"] += 1
                run = 0
                continue
            n_pass += 1
            run += 1
            best_run = max(best_run, run)

    return {
        "native_sr": native_sr,
        "n_windows": n_win,
        "pass": n_pass,
        "fail": dict(fail),
        "max_good_s": best_run * cfg.quality_window_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="raw .vital 트랙 존재 + 품질 게이트 probe")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--n-files", type=int, default=12)
    args = ap.parse_args()

    try:
        import vitaldb
    except ImportError:
        print("ERROR: pip install vitaldb", file=sys.stderr)
        sys.exit(1)

    files = sorted(Path(args.raw_dir).glob("**/*.vital"))[: args.n_files]
    if not files:
        print(f"ERROR: .vital 없음: {args.raw_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"probe {len(files)} files from {args.raw_dir}\n")

    key_tracks: dict[str, list[str]] = defaultdict(list)
    for trk, (key, _sid) in TRACK_MAP.items():
        if key in WATCH:
            key_tracks[key].append(trk)

    present = defaultdict(int)
    agg: dict[str, list[dict]] = defaultdict(list)

    for fp in files:
        try:
            vf = vitaldb.VitalFile(str(fp))
            avail = set(vf.get_track_names())
        except Exception as exc:
            print(f"  [WARN] {fp.name}: {exc}", file=sys.stderr)
            continue
        for key in WATCH:
            hit = next((t for t in key_tracks[key] if t in avail), None)
            if hit is None:
                continue
            present[key] += 1
            r = probe_track(vf, hit, key)
            if r:
                r["track"] = hit
                agg[key].append(r)
        print(f"  {fp.name}: tracks " + ", ".join(
            f"{k}={'Y' if any(t in avail for t in key_tracks[k]) else '-'}" for k in WATCH))

    n = len(files)
    print("\n── (A) 원본 트랙 존재율 ──")
    for k in WATCH:
        print(f"  {k:4s} {present[k]:3d}/{n}  ({100.0*present[k]/max(1,n):.0f}%)")

    print("\n── (C) 파서 품질 게이트 통과율 (트랙 보유 파일만) ──")
    print(f"  {'sig':4s} {'srate':>7s} {'windows':>9s} {'pass':>8s} {'pass%':>7s} {'max_good':>10s}   fail 사유")
    for k in WATCH:
        rs = agg[k]
        if not rs:
            print(f"  {k:4s} {'-':>7s} {'-':>9s} {'-':>8s} {'-':>7s} {'-':>10s}")
            continue
        nw = sum(r["n_windows"] for r in rs)
        npass = sum(r["pass"] for r in rs)
        sr = rs[0]["native_sr"]
        mg = max(r["max_good_s"] for r in rs)
        fails = defaultdict(int)
        for r in rs:
            for kk, vv in r["fail"].items():
                fails[kk] += vv
        fs = ", ".join(f"{kk}={vv}" for kk, vv in sorted(fails.items(), key=lambda x: -x[1]))
        pct = 100.0 * npass / max(1, nw)
        print(f"  {k:4s} {sr:7.1f} {nw:9d} {npass:8d} {pct:6.1f}% {mg:9.0f}s   {fs}")

    print("\n해석:")
    print("  (A) 낮음        → 원본에 트랙 없음. 조합에서 제외할 것.")
    print("  (A) 높고 pass%≈0 → 품질 게이트가 죽임. fail 사유 열이 범인.")
    print("  (A)·pass% 둘 다 높음 → 파싱 때 --signal-types 로 제외됐을 가능성.")


if __name__ == "__main__":
    main()
