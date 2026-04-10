"""K-MIMIC-MORTAL 신호 탐색 및 전처리 파이프라인 시각화.

.vital 파일에서 신호를 찾아 raw → filtered → QC 결과를 시각화한다.

사용법:
    # PAP 신호 탐색 (전처리 파이프라인 포함)
    python scripts/explore_kmimic_signals.py \
        --raw ../datasets/K-MIMIC-MORTAL/1.0.0/VITALDB \
        --signal PAP --n-search 100

    # ICP 신호 탐색
    python scripts/explore_kmimic_signals.py \
        --raw ../datasets/K-MIMIC-MORTAL/1.0.0/VITALDB \
        --signal ICP --n-search 100

    # 전체 SNUADCM 트랙 시각화
    python scripts/explore_kmimic_signals.py \
        --raw ../datasets/K-MIMIC-MORTAL/1.0.0/VITALDB \
        --all-tracks --n-search 50
"""

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    TRACK_MAP,
    TARGET_SR,
    _apply_filter,
    _apply_median_filter,
    _apply_notch_filter,
    _apply_range_check,
    _detect_electrocautery,
    _detect_motion_artifact,
    _extract_nan_free_segments,
)
from data.parser._common import (
    domain_quality_check,
    resample_to_target,
    segment_quality_score,
)

# 트랙 → native SR
NATIVE_SR = {
    "ecg": 500.0, "abp": 500.0, "ppg": 500.0,
    "cvp": 500.0, "pap": 500.0, "icp": 500.0,
    "co2": 62.5, "awp": 62.5,
}


def find_vital_files(raw_dir: str, n_search: int) -> list[Path]:
    root = Path(raw_dir)
    all_files = list(root.glob("**/*.vital"))
    print(f"전체 .vital 파일: {len(all_files)}개")
    if len(all_files) > n_search:
        random.seed(42)
        all_files = random.sample(all_files, n_search)
    print(f"탐색 대상: {len(all_files)}개\n")
    return all_files


def discover_tracks(vf_path: Path) -> list[str]:
    import vitaldb
    vf = vitaldb.VitalFile(str(vf_path))
    return [t["name"] for t in vf.trks.values() if "name" in t]


def load_track_raw(vf_path: Path, track_name: str, sr: float) -> np.ndarray | None:
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


def apply_pipeline_steps(raw: np.ndarray, stype_key: str, native_sr: float) -> dict:
    """전처리 파이프라인을 단계별로 적용하고 각 단계 결과를 반환한다."""
    cfg = SIGNAL_CONFIGS.get(stype_key)
    if cfg is None:
        return {"error": f"No config for {stype_key}"}

    result = {"raw": raw.copy(), "stype": stype_key, "native_sr": native_sr}

    # Step 1: Range check
    data = raw.copy()
    if cfg.valid_range is not None:
        data, n_removed = _apply_range_check(data, cfg.valid_range)
        result["range_check"] = data.copy()
        result["range_removed"] = int(n_removed)
    else:
        result["range_check"] = data.copy()
        result["range_removed"] = 0

    # Step 2: Spike detection
    if cfg.spike_detection:
        data, spike_mask = _detect_electrocautery(data, native_sr, threshold_std=cfg.spike_threshold_std)
        result["spike_detect"] = data.copy()
        result["spike_count"] = int(spike_mask.sum()) if spike_mask is not None else 0
    else:
        result["spike_detect"] = data.copy()
        result["spike_count"] = 0

    # Step 2b: PPG motion artifact
    if stype_key == "ppg":
        data, motion_mask = _detect_motion_artifact(data, native_sr)
        result["motion_detect"] = data.copy()
    else:
        result["motion_detect"] = data.copy()

    # Step 3: NaN-free segments
    min_samples = int(60.0 * native_sr)
    segments = _extract_nan_free_segments(data, min_samples)
    if not segments:
        result["error"] = "No valid segments (all NaN)"
        return result

    segment = max(segments, key=len)
    result["longest_segment"] = segment.copy()
    result["n_segments"] = len(segments)
    result["segment_duration_s"] = len(segment) / native_sr

    # Step 4: Median → Notch → Filter
    if cfg.median_kernel > 0:
        segment = _apply_median_filter(segment, kernel_size=cfg.median_kernel)
    if cfg.notch_freq is not None:
        segment = _apply_notch_filter(segment, freq=cfg.notch_freq, sr=native_sr)
    filtered = _apply_filter(segment, cfg, native_sr)
    result["filtered"] = filtered.copy()

    # Step 5: Resample to 100Hz
    if native_sr != TARGET_SR:
        resampled = resample_to_target(filtered, orig_sr=native_sr, target_sr=TARGET_SR)
    else:
        resampled = filtered
    result["resampled"] = resampled.copy()

    # Step 6: Quality check (5초 윈도우)
    qc_window_s = cfg.quality_window_s
    qc_window = int(qc_window_s * TARGET_SR)
    qc_results = []
    for start in range(0, len(resampled) - qc_window + 1, qc_window):
        win = resampled[start:start + qc_window]
        basic = segment_quality_score(
            win,
            max_flatline_ratio=cfg.max_flatline_ratio,
            max_clip_ratio=cfg.max_clip_ratio,
            max_high_freq_ratio=cfg.max_high_freq_ratio,
            min_amplitude=cfg.min_amplitude,
            max_amplitude=cfg.max_amplitude,
            min_high_freq_ratio=cfg.min_high_freq_ratio,
        )
        if basic["pass"]:
            domain = domain_quality_check(stype_key, win, sr=TARGET_SR)
            passed = domain["pass"]
            qc_results.append({"start": start, "basic": basic, "domain": domain, "pass": passed})
        else:
            qc_results.append({"start": start, "basic": basic, "domain": {}, "pass": False})

    result["qc_results"] = qc_results
    n_pass = sum(1 for q in qc_results if q["pass"])
    n_total = len(qc_results)
    result["qc_pass_ratio"] = n_pass / n_total if n_total > 0 else 0
    result["qc_summary"] = f"{n_pass}/{n_total} windows passed ({result['qc_pass_ratio']:.0%})"

    return result


def plot_pipeline(result: dict, title: str, out_path: str, show_seconds: float = 30.0) -> None:
    """전처리 파이프라인 단계별 시각화."""
    if "error" in result:
        print(f"  SKIP: {result['error']}")
        return

    stype = result["stype"]
    native_sr = result["native_sr"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)

    # Row 0: Raw signal
    raw = result["raw"]
    n_show_raw = min(len(raw), int(show_seconds * native_sr))
    valid = ~np.isnan(raw)
    starts = np.where(valid)[0]
    if len(starts) > 0:
        s0 = starts[0]
        seg = np.nan_to_num(raw[s0:s0 + n_show_raw], nan=0.0)
        t = np.arange(len(seg)) / native_sr
        axes[0].plot(t, seg, linewidth=0.5, color="steelblue")
        stats = f"mean={np.nanmean(seg):.1f} | std={np.nanstd(seg):.1f} | range=[{np.nanmin(seg):.1f}, {np.nanmax(seg):.1f}]"
        axes[0].set_title(f"① Raw ({native_sr:.0f}Hz) — {stats}", fontsize=9)
        axes[0].text(0.01, 0.92, f"range_removed={result['range_removed']} | spikes={result['spike_count']}",
                     transform=axes[0].transAxes, fontsize=7, va="top",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Row 1: Filtered (at native SR)
    filtered = result["filtered"]
    n_show_filt = min(len(filtered), int(show_seconds * native_sr))
    seg_f = filtered[:n_show_filt]
    t_f = np.arange(len(seg_f)) / native_sr
    axes[1].plot(t_f, seg_f, linewidth=0.5, color="darkorange")
    cfg = SIGNAL_CONFIGS[stype]
    filter_info = f"{cfg.filter_type} {cfg.filter_freq}"
    if cfg.notch_freq:
        filter_info += f" | notch={cfg.notch_freq}Hz"
    if cfg.median_kernel > 0:
        filter_info += f" | median_k={cfg.median_kernel}"
    stats_f = f"mean={np.mean(seg_f):.1f} | std={np.std(seg_f):.1f} | range=[{np.min(seg_f):.1f}, {np.max(seg_f):.1f}]"
    axes[1].set_title(f"② Filtered — {filter_info} — {stats_f}", fontsize=9)

    # Row 2: Resampled (100Hz)
    resampled = result["resampled"]
    n_show_resamp = min(len(resampled), int(show_seconds * TARGET_SR))
    seg_r = resampled[:n_show_resamp]
    t_r = np.arange(len(seg_r)) / TARGET_SR
    axes[2].plot(t_r, seg_r, linewidth=0.5, color="seagreen")
    axes[2].set_title(f"③ Resampled ({TARGET_SR:.0f}Hz) — {len(resampled)/TARGET_SR:.0f}s total", fontsize=9)

    # Row 3: QC results overlay
    qc_results = result["qc_results"]
    qc_window = int(cfg.quality_window_s * TARGET_SR)
    axes[3].plot(t_r, seg_r, linewidth=0.5, color="gray", alpha=0.5)

    for qc in qc_results:
        start = qc["start"]
        end = start + qc_window
        if start >= n_show_resamp:
            continue
        t_start = start / TARGET_SR
        t_end = min(end, n_show_resamp) / TARGET_SR
        color = "#2ecc71" if qc["pass"] else "#e74c3c"
        alpha = 0.2 if qc["pass"] else 0.3
        axes[3].axvspan(t_start, t_end, alpha=alpha, color=color)

    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.3, label="QC Pass"),
        Patch(facecolor="#e74c3c", alpha=0.3, label="QC Fail"),
    ]
    axes[3].legend(handles=legend_elements, loc="upper right", fontsize=7)
    axes[3].set_title(f"④ QC Results — {result['qc_summary']}", fontsize=9)
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.tick_params(labelsize=7)

    plt.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def search_and_visualize(vital_files: list[Path], target_signal: str,
                          out_dir: str, max_plots: int = 5) -> None:
    """target_signal을 찾아 전처리 파이프라인 시각화."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_track = f"SNUADCM/{target_signal}"

    # TRACK_MAP에서 stype_key 확인
    track_info = TRACK_MAP.get(target_track)
    if track_info is None:
        print(f"WARNING: {target_track} not in TRACK_MAP. 탐색은 진행하지만 전처리 설정이 없을 수 있습니다.")
        stype_key = target_signal.lower()
    else:
        stype_key = track_info[0]

    native_sr = NATIVE_SR.get(stype_key, 500.0)

    found = 0
    for i, vf in enumerate(vital_files):
        try:
            tracks = discover_tracks(vf)
        except Exception:
            continue

        if target_track not in tracks:
            continue

        print(f"[{i+1}] {vf.name} — {target_track} found")

        arr = load_track_raw(vf, target_track, native_sr)
        if arr is None:
            print(f"  SKIP: insufficient valid data")
            continue

        result = apply_pipeline_steps(arr, stype_key, native_sr)

        found += 1
        title = f"{vf.name} — {target_track} ({stype_key.upper()})"
        fname = f"{target_signal}_{found:02d}_{vf.stem}.png"
        plot_pipeline(result, title, str(out_path / fname))

        if found >= max_plots:
            break

    print(f"\n{target_signal}: {found}개 시각화 완료")


def visualize_all_tracks(vital_files: list[Path], out_dir: str,
                          max_files: int = 3) -> None:
    """SNUADCM 전체 트랙을 파일별로 시각화."""
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

        print(f"\n[{vf.name}] — {len(snuadcm_tracks)} SNUADCM tracks: {snuadcm_tracks}")

        for track_name in snuadcm_tracks:
            track_info = TRACK_MAP.get(track_name)
            if track_info is None:
                print(f"  {track_name}: not in TRACK_MAP, skipping")
                continue

            stype_key = track_info[0]
            native_sr = NATIVE_SR.get(stype_key, 500.0)

            arr = load_track_raw(vf, track_name, native_sr)
            if arr is None:
                print(f"  {track_name}: no valid data")
                continue

            result = apply_pipeline_steps(arr, stype_key, native_sr)

            signal_name = track_name.split("/")[-1]
            title = f"{vf.name} — {track_name} ({stype_key.upper()})"
            fname = f"all_{plotted:02d}_{vf.stem}_{signal_name}.png"
            plot_pipeline(result, title, str(out_path / fname))

        plotted += 1
        if plotted >= max_files:
            break

    print(f"\n전체 {plotted}개 파일 시각화 완료")


def main() -> None:
    parser = argparse.ArgumentParser(description="K-MIMIC 신호 탐색/전처리 파이프라인 시각화")
    parser.add_argument("--raw", required=True, help="K-MIMIC .vital 디렉토리")
    parser.add_argument("--signal", type=str, default=None,
                        help="탐색할 신호 (PAP, ICP, ECG_II, ART, PLETH, CVP)")
    parser.add_argument("--all-tracks", action="store_true",
                        help="SNUADCM 전체 트랙 시각화")
    parser.add_argument("--n-search", type=int, default=100,
                        help="탐색할 파일 수")
    parser.add_argument("--max-plots", type=int, default=5,
                        help="최대 시각화 수")
    parser.add_argument("--out-dir", type=str, default="outputs/explore_kmimic",
                        help="출력 디렉토리")
    args = parser.parse_args()

    vital_files = find_vital_files(args.raw, args.n_search)

    if args.all_tracks:
        visualize_all_tracks(vital_files, args.out_dir, max_files=args.max_plots)
    elif args.signal:
        search_and_visualize(vital_files, args.signal, args.out_dir, args.max_plots)
    else:
        print("--signal 또는 --all-tracks를 지정하세요")


if __name__ == "__main__":
    main()
