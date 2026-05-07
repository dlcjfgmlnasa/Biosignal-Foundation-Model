"""--skip-quality-window 로 새로 저장된 .pt 시각화 (sanity check).

K-MIMIC ICU 재파싱 시 quality_window check 를 skip 했기 때문에
저장된 신호가 garbage (flatline/saturated) 인지 정상 신호인지 직접 확인.

mtime 기반 필터로 최근에 저장된 파일만 샘플링한다.

Usage:
    # 최근 1시간 내 saved 된 ECG/PPG/ABP 각 5개 샘플
    python -m scripts.visualize_saved_pt \
        --processed /home/coder/workspace/updown/bio_fm/data/train/k_mimic_full \
        --since-minutes 60 \
        --n-per-signal 5 \
        --signal-types ecg ppg abp \
        --out-dir /tmp/saved_pt_check

    # 특정 cutoff time 이후 saved 만
    python -m scripts.visualize_saved_pt \
        --processed ... \
        --since-timestamp 1715173800 \
        --n-per-signal 8
"""
from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# .pt 파일명 패턴: <session_id>_<stype>_<spatial_id>_seg<i>_<j>.pt
PT_RE = re.compile(r"^(.+)_(ecg|abp|ppg|cvp|co2|awp|pap|icp|resp)_(\d+)_seg(\d+)_(\d+)\.pt$")


def collect_recent_pts(
    processed: Path, since_ts: float, signal_types: list[str]
) -> dict[str, list[Path]]:
    """processed/<subject>/<file>.pt 중 mtime >= since_ts 만 signal_type 별로 모음."""
    by_type: dict[str, list[Path]] = {st: [] for st in signal_types}

    for subj_dir in processed.iterdir():
        if not subj_dir.is_dir():
            continue
        try:
            for entry in subj_dir.iterdir():
                if not entry.is_file() or entry.suffix != ".pt":
                    continue
                m = PT_RE.match(entry.name)
                if not m:
                    continue
                stype = m.group(2)
                if stype not in by_type:
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime < since_ts:
                    continue
                by_type[stype].append(entry)
        except OSError:
            continue

    return by_type


def plot_signal(ax, signal: np.ndarray, sr: float, title: str) -> None:
    """단일 채널 신호 플롯 + 통계 annotation."""
    t = np.arange(signal.size) / sr
    ax.plot(t, signal, linewidth=0.5)
    ax.set_xlabel("time (s)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)

    # 통계
    finite = signal[~np.isnan(signal)]
    if finite.size > 0:
        stats_str = (
            f"n={signal.size}  "
            f"min={finite.min():.2f}  "
            f"p1={np.percentile(finite,1):.2f}  "
            f"p50={np.percentile(finite,50):.2f}  "
            f"p99={np.percentile(finite,99):.2f}  "
            f"max={finite.max():.2f}"
        )
        # flatline 비율 (연속 동일값 비율)
        diffs = np.abs(np.diff(finite))
        flat_ratio = (diffs < 1e-6).sum() / max(1, diffs.size)
        stats_str += f"  flat={flat_ratio*100:.1f}%"
        ax.text(0.01, 0.97, stats_str, transform=ax.transAxes,
                fontsize=7, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--since-minutes", type=float, default=None,
                   help="최근 N 분 내에 saved 된 .pt 만")
    p.add_argument("--since-timestamp", type=float, default=None,
                   help="이 unix timestamp 이후 saved 된 .pt 만 (--since-minutes 와 택일)")
    p.add_argument("--n-per-signal", type=int, default=5)
    p.add_argument("--signal-types", nargs="+",
                   default=["ecg", "abp", "ppg"],
                   choices=["ecg", "abp", "ppg", "cvp", "co2", "awp", "pap", "icp", "resp"])
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/saved_pt_check"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.since_minutes is None and args.since_timestamp is None:
        args.since_minutes = 60.0
    if args.since_timestamp is not None:
        since_ts = args.since_timestamp
    else:
        since_ts = time.time() - args.since_minutes * 60

    print(f"Scanning {args.processed} for .pt with mtime >= "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since_ts))}")

    by_type = collect_recent_pts(args.processed, since_ts, args.signal_types)
    for st in args.signal_types:
        print(f"  {st}: {len(by_type[st])} new .pt files")

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for st in args.signal_types:
        files = by_type[st]
        if not files:
            print(f"\n[{st}] no new files — skip")
            continue

        sampled = rng.sample(files, k=min(args.n_per_signal, len(files)))
        print(f"\n[{st}] sampled {len(sampled)} files for visualization")

        n = len(sampled)
        ncols = 1
        nrows = n
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(14, 2.0 * nrows),
            squeeze=False,
        )
        for i, pt_path in enumerate(sampled):
            try:
                tensor = torch.load(pt_path, weights_only=True)
            except Exception as e:
                print(f"  load fail: {pt_path}: {e}")
                continue

            arr = tensor.squeeze().numpy().astype(np.float32)
            if arr.ndim == 0 or arr.size == 0:
                continue

            plot_signal(
                axes[i][0], arr, sr=100.0,
                title=f"{pt_path.parent.name}/{pt_path.name}",
            )

        out_path = args.out_dir / f"saved_{st}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path}")

    print(f"\nDone. Inspect figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
