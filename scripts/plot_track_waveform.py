"""특정 track 의 raw 파형 + 값 분포 히스토그램 시각화.

단위 미스매치/sentinel 의심 track 을 눈으로 확인 (서버에서 PNG 저장).

사용법:
    # 리스트에서 track 가진 파일 N 개 무작위 샘플
    python -m scripts.plot_track_waveform \
        --list /home/coder/workspace/updown/kmimic_vital_remaining.txt \
        --track SNUADCM/CVP \
        --n-files 4 \
        --out /tmp/cvp_check.png

    # 특정 파일 직접 지정
    python -m scripts.plot_track_waveform \
        --vital-file /path/to/foo.vital \
        --track SNUADCM/CVP \
        --out /tmp/cvp_one.png
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt

try:
    import vitaldb
except ImportError:
    print("ERROR: vitaldb 패키지 필요")
    raise SystemExit(1)

from data.parser.vitaldb import SIGNAL_CONFIGS, TRACK_MAP


def load_track(path: str, track: str) -> tuple[np.ndarray, float] | None:
    try:
        vf = vitaldb.VitalFile(path)
    except Exception as e:
        print(f"  LOAD ERR {path}: {e}")
        return None
    if track not in vf.get_track_names():
        return None
    try:
        trk = vf.find_track(track)
        sr = trk.srate if trk is not None and trk.srate > 0 else 500.0
        samples = vf.to_numpy(track, interval=1.0 / sr).flatten()
    except Exception as e:
        print(f"  to_numpy ERR {path}: {e}")
        return None
    return samples, sr


def select_paths(args) -> list[str]:
    if args.vital_file:
        return [args.vital_file]
    rng = random.Random(args.seed)
    with open(args.list, encoding="utf-8") as f:
        all_paths = [ln.strip() for ln in f if ln.strip()]
    rng.shuffle(all_paths)
    selected: list[str] = []
    for p in all_paths[: args.max_scan]:
        if len(selected) >= args.n_files:
            break
        try:
            vf = vitaldb.VitalFile(p)
            if args.track in vf.get_track_names():
                selected.append(p)
        except Exception:
            continue
    return selected


def plot_grid(paths: list[str], track: str, out_path: Path,
              valid_range: tuple[float, float] | None,
              window_s: float) -> None:
    n = len(paths)
    if n == 0:
        print("ERROR: no file with target track")
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 3.0 * n), squeeze=False)
    fig.suptitle(f"Track: {track}   valid_range={valid_range}", fontsize=12)

    for i, path in enumerate(paths):
        loaded = load_track(path, track)
        ax_wave = axes[i][0]
        ax_hist = axes[i][1]
        if loaded is None:
            ax_wave.text(0.5, 0.5, "load failed", ha="center", va="center")
            ax_hist.text(0.5, 0.5, "load failed", ha="center", va="center")
            continue
        samples, sr = loaded
        finite = samples[~np.isnan(samples)]
        n_total = samples.size
        n_finite = finite.size

        # ── 좌: 시간축 파형 (앞 window_s 초) ──
        n_show = min(int(window_s * sr), n_total)
        t = np.arange(n_show) / sr
        ax_wave.plot(t, samples[:n_show], lw=0.4)
        if valid_range is not None:
            lo, hi = valid_range
            ax_wave.axhline(lo, color="r", ls="--", lw=0.6)
            ax_wave.axhline(hi, color="r", ls="--", lw=0.6)
        ax_wave.set_xlabel("seconds")
        ax_wave.set_ylabel("value")
        title = Path(path).name
        ax_wave.set_title(f"{title}  sr={sr:.0f}Hz  ({window_s:.0f}s window)",
                          fontsize=9)

        # ── 우: 값 분포 히스토그램 ──
        if n_finite > 0:
            p1, p99 = np.percentile(finite, [0.5, 99.5])
            pad = max(1.0, (p99 - p1) * 0.05)
            ax_hist.hist(finite, bins=200, range=(p1 - pad, p99 + pad))
            stat = (
                f"min={finite.min():.2f}  p50={np.median(finite):.2f}  "
                f"max={finite.max():.2f}\nfinite={n_finite}/{n_total} "
                f"({100*n_finite/n_total:.1f}%)"
            )
        else:
            stat = "all NaN"
        if valid_range is not None:
            ax_hist.axvline(valid_range[0], color="r", ls="--", lw=0.6)
            ax_hist.axvline(valid_range[1], color="r", ls="--", lw=0.6)
        ax_hist.set_title(stat, fontsize=8)
        ax_hist.set_xlabel("value")
        ax_hist.set_ylabel("count")
        ax_hist.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--list", type=Path, help=".vital path 리스트")
    grp.add_argument("--vital-file", type=str, help="단일 파일 경로")
    p.add_argument("--track", type=str, required=True)
    p.add_argument("--n-files", type=int, default=4)
    p.add_argument("--max-scan", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-s", type=float, default=60.0,
                   help="파형 시각화 구간 길이(초)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    mapping = TRACK_MAP.get(args.track)
    valid_range = None
    if mapping is not None:
        cfg = SIGNAL_CONFIGS.get(mapping[0])
        if cfg is not None:
            valid_range = cfg.valid_range
    print(f"track={args.track}  valid_range={valid_range}")

    paths = select_paths(args)
    print(f"selected {len(paths)} file(s):")
    for p_ in paths:
        print(f"  {p_}")
    plot_grid(paths, args.track, args.out, valid_range, args.window_s)


if __name__ == "__main__":
    main()
