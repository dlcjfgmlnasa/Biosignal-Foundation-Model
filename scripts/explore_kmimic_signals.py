"""K-MIMIC-MORTAL 신호 탐색 및 시각화 스크립트.

.vital 파일에서 PAP/ICP 등 신호를 찾아 raw waveform을 시각화한다.
서버에서 실행하여 신호 특성을 육안으로 확인한다.

사용법:
    # PAP 신호 탐색 (랜덤 50개 파일에서 검색)
    python scripts/explore_kmimic_signals.py \
        --raw ../datasets/K-MIMIC-MORTAL/1.0.0/VITALDB \
        --signal PAP --n-search 50

    # ICP 신호 탐색
    python scripts/explore_kmimic_signals.py \
        --raw ../datasets/K-MIMIC-MORTAL/1.0.0/VITALDB \
        --signal ICP --n-search 50

    # 특정 파일에서 모든 SNUADCM 트랙 시각화
    python scripts/explore_kmimic_signals.py \
        --raw ../datasets/K-MIMIC-MORTAL/1.0.0/VITALDB \
        --all-tracks --n-search 10
"""

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_vital_files(raw_dir: str, n_search: int) -> list[Path]:
    """재귀적으로 .vital 파일을 찾아 랜덤 샘플링한다."""
    root = Path(raw_dir)
    all_files = list(root.glob("**/*.vital"))
    print(f"전체 .vital 파일: {len(all_files)}개")
    if len(all_files) > n_search:
        random.seed(42)
        all_files = random.sample(all_files, n_search)
    print(f"탐색 대상: {len(all_files)}개\n")
    return all_files


def discover_tracks(vf_path: Path) -> list[str]:
    """vital 파일의 트랙 목록을 반환한다."""
    import vitaldb
    vf = vitaldb.VitalFile(str(vf_path))
    return [t["name"] for t in vf.trks.values() if "name" in t]


def load_track(vf_path: Path, track_name: str, sr: float = 500.0) -> np.ndarray | None:
    """vital 파일에서 특정 트랙을 로드한다."""
    import vitaldb
    data = vitaldb.VitalFile(str(vf_path))
    vals = data.to_numpy([track_name], interval=1.0 / sr)
    if vals is None or len(vals) == 0:
        return None
    arr = vals[:, 0].flatten()
    valid_ratio = (~np.isnan(arr)).sum() / len(arr)
    if valid_ratio < 0.1:
        return None
    return arr


def plot_signal(arr: np.ndarray, sr: float, title: str, out_path: str,
                max_seconds: float = 30.0) -> None:
    """신호를 시각화하여 저장한다."""
    n_samples = min(len(arr), int(max_seconds * sr))
    # NaN이 아닌 구간 찾기
    valid = ~np.isnan(arr)
    if not valid.any():
        return

    # 유효 구간 시작점 찾기
    starts = np.where(valid)[0]
    if len(starts) == 0:
        return

    # 가장 긴 연속 유효 구간 찾기
    diffs = np.diff(starts)
    breaks = np.where(diffs > 1)[0]
    if len(breaks) == 0:
        seg_start = starts[0]
        seg_end = starts[-1] + 1
    else:
        seg_lengths = []
        prev = 0
        for b in breaks:
            seg_lengths.append((prev, b + 1))
            prev = b + 1
        seg_lengths.append((prev, len(starts)))
        longest = max(seg_lengths, key=lambda x: x[1] - x[0])
        seg_start = starts[longest[0]]
        seg_end = starts[longest[1] - 1] + 1

    seg = arr[seg_start:min(seg_start + n_samples, seg_end)]
    seg = np.nan_to_num(seg, nan=0.0)
    t = np.arange(len(seg)) / sr

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t, seg, linewidth=0.5, color="steelblue")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    # 통계 표시
    stats = f"mean={np.nanmean(seg):.1f}, std={np.nanstd(seg):.1f}, min={np.nanmin(seg):.1f}, max={np.nanmax(seg):.1f}"
    ax.text(0.01, 0.95, stats, transform=ax.transAxes, fontsize=7,
            va="top", ha="left", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def search_and_plot_signal(vital_files: list[Path], target_track: str,
                           out_dir: str, max_plots: int = 5, sr: float = 500.0) -> None:
    """target_track이 있는 파일을 찾아 시각화한다."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    found = 0
    for i, vf in enumerate(vital_files):
        try:
            tracks = discover_tracks(vf)
        except Exception:
            continue

        matching = [t for t in tracks if target_track in t]
        if not matching:
            continue

        for track_name in matching:
            arr = load_track(vf, track_name, sr)
            if arr is None:
                continue

            found += 1
            title = f"{vf.name} — {track_name} ({len(arr)/sr:.0f}s)"
            fname = f"{target_track}_{found:02d}_{vf.stem}.png"
            plot_signal(arr, sr, title, str(out_path / fname))

            if found >= max_plots:
                print(f"\n{target_track}: {found}개 시각화 완료")
                return

    print(f"\n{target_track}: {found}개 찾음 (탐색 {len(vital_files)}개 파일)")


def search_and_plot_all_snuadcm(vital_files: list[Path], out_dir: str,
                                 max_files: int = 3, sr: float = 500.0) -> None:
    """SNUADCM 트랙이 있는 파일에서 모든 트랙을 시각화한다."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    plotted = 0
    for vf in vital_files:
        try:
            tracks = discover_tracks(vf)
        except Exception:
            continue

        snuadcm_tracks = [t for t in tracks if "SNUADCM" in t]
        if not snuadcm_tracks:
            continue

        print(f"\n[{vf.name}] — {len(snuadcm_tracks)} SNUADCM tracks")

        fig, axes = plt.subplots(len(snuadcm_tracks), 1,
                                  figsize=(14, 2.5 * len(snuadcm_tracks)), squeeze=False)

        for j, track_name in enumerate(snuadcm_tracks):
            ax = axes[j, 0]
            arr = load_track(vf, track_name, sr)
            if arr is None:
                ax.set_title(f"{track_name} — NO DATA", fontsize=9)
                ax.axis("off")
                continue

            # 30초만 표시
            n_show = min(len(arr), int(30 * sr))
            valid = ~np.isnan(arr)
            starts = np.where(valid)[0]
            if len(starts) == 0:
                ax.set_title(f"{track_name} — ALL NaN", fontsize=9)
                ax.axis("off")
                continue

            seg_start = starts[0]
            seg = arr[seg_start:seg_start + n_show]
            seg = np.nan_to_num(seg, nan=0.0)
            t = np.arange(len(seg)) / sr

            ax.plot(t, seg, linewidth=0.5, color="steelblue")
            stats = f"mean={np.nanmean(seg):.1f}, std={np.nanstd(seg):.1f}, range=[{np.nanmin(seg):.1f}, {np.nanmax(seg):.1f}]"
            ax.set_title(f"{track_name} — {stats}", fontsize=9)
            ax.set_ylabel("Amp", fontsize=8)
            ax.tick_params(labelsize=7)

        axes[-1, 0].set_xlabel("Time (s)", fontsize=8)
        plt.suptitle(vf.name, fontsize=11)
        plt.tight_layout()

        fname = f"all_tracks_{plotted:02d}_{vf.stem}.png"
        fig.savefig(str(out_path / fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

        plotted += 1
        if plotted >= max_files:
            break

    print(f"\n전체 {plotted}개 파일 시각화 완료")


def main() -> None:
    parser = argparse.ArgumentParser(description="K-MIMIC 신호 탐색/시각화")
    parser.add_argument("--raw", required=True, help="K-MIMIC .vital 디렉토리")
    parser.add_argument("--signal", type=str, default=None,
                        help="탐색할 신호 (PAP, ICP, ECG_II, ART, PLETH, CVP)")
    parser.add_argument("--all-tracks", action="store_true",
                        help="SNUADCM 전체 트랙 시각화")
    parser.add_argument("--n-search", type=int, default=50,
                        help="탐색할 파일 수")
    parser.add_argument("--max-plots", type=int, default=5,
                        help="최대 시각화 수")
    parser.add_argument("--out-dir", type=str, default="outputs/explore_kmimic",
                        help="출력 디렉토리")
    args = parser.parse_args()

    vital_files = find_vital_files(args.raw, args.n_search)

    if args.all_tracks:
        search_and_plot_all_snuadcm(vital_files, args.out_dir, max_files=args.max_plots)
    elif args.signal:
        target = f"SNUADCM/{args.signal}"
        search_and_plot_signal(vital_files, target, args.out_dir, args.max_plots)
    else:
        print("--signal 또는 --all-tracks를 지정하세요")


if __name__ == "__main__":
    main()
