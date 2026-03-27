# -*- coding:utf-8 -*-
"""CVP 전용 상세 품질 시각화.

실제 VitalDB 데이터에서 CVP PASS/FAIL 세그먼트를 수집하고,
각 subplot에 어떤 지표에서 위반됐는지 상세히 표시한다.

사용법:
    python -m scripts.viz_cvp_detail
    python -m scripts.viz_cvp_detail --out cvp_detail.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.parser._common import (
    cvp_quality_check,
    resample_to_target,
    segment_quality_score,
)
from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    _apply_filter,
    _apply_median_filter,
    _apply_notch_filter,
    _apply_range_check,
    _detect_electrocautery,
    _extract_nan_free_segments,
)

TARGET_SR = 100.0
WINDOW_S = 10.0
TRACK_NAME = "SNUADC/CVP"

# CVP quality check thresholds (matching cvp_quality_check defaults)
THRESHOLDS = {
    "autocorr_peak": 0.25,
    "hr_min": 30.0,
    "hr_max": 200.0,
    "regularity": 0.7,
    "flatline_ratio": 0.3,
}


@dataclass
class CvpWindow:
    signal: np.ndarray
    case_id: int
    passed: bool
    domain: dict
    basic: dict


def _load_cvp_windows(case_id: int, max_windows: int = 15) -> list[CvpWindow]:
    """단일 케이스에서 CVP 윈도우를 로드하고 품질 검사."""
    import vitaldb

    cfg = SIGNAL_CONFIGS["cvp"]
    native_sr = 500.0

    try:
        data = vitaldb.load_case(case_id, [TRACK_NAME], interval=1.0 / native_sr)
    except Exception as exc:
        print(f"  [WARN] case {case_id} load failed: {exc}", file=sys.stderr)
        return []

    if data is None or len(data) == 0:
        return []

    data = data[:, 0].flatten()

    # Pipeline: range check → spike detection → NaN segments → filter → resample
    if cfg.valid_range is not None:
        data, _ = _apply_range_check(data, cfg.valid_range)
    if cfg.spike_detection:
        data, _ = _detect_electrocautery(data, native_sr, threshold_std=cfg.spike_threshold_std)

    min_samples = int(60.0 * native_sr)
    segments = _extract_nan_free_segments(data, min_samples)
    if not segments:
        return []

    results: list[CvpWindow] = []
    win_samples = int(WINDOW_S * TARGET_SR)

    for segment in segments:
        if cfg.median_kernel > 0:
            segment = _apply_median_filter(segment, kernel_size=cfg.median_kernel)
        if cfg.notch_freq is not None:
            segment = _apply_notch_filter(segment, freq=cfg.notch_freq, sr=native_sr)
        segment = _apply_filter(segment, cfg, native_sr)

        if native_sr != TARGET_SR:
            segment = resample_to_target(segment, orig_sr=native_sr, target_sr=TARGET_SR)

        for win_start in range(0, len(segment) - win_samples + 1, win_samples):
            win = segment[win_start:win_start + win_samples]

            basic = segment_quality_score(
                win,
                max_flatline_ratio=cfg.max_flatline_ratio,
                max_clip_ratio=cfg.max_clip_ratio,
                max_high_freq_ratio=cfg.max_high_freq_ratio,
                min_amplitude=cfg.min_amplitude,
                max_amplitude=cfg.max_amplitude,
                min_high_freq_ratio=cfg.min_high_freq_ratio,
            )

            if not basic["pass"]:
                domain = {"pass": False, "hr": 0.0, "n_peaks": 0, "regularity": 1.0,
                          "flatline_ratio": basic.get("flatline_ratio", 0), "autocorr_peak": 0.0,
                          "resp_power_ratio": 0.0, "_basic_fail": True}
            else:
                domain = cvp_quality_check(win, sr=TARGET_SR)
                domain["_basic_fail"] = False

            results.append(CvpWindow(
                signal=win, case_id=case_id,
                passed=basic["pass"] and domain["pass"],
                domain=domain, basic=basic,
            ))
            if len(results) >= max_windows:
                return results

    return results


def _format_metric(key: str, value: float, domain: dict) -> tuple[str, str]:
    """(표시 텍스트, 색상) 반환. 위반된 지표는 빨간색."""
    if key == "autocorr_peak":
        thr = THRESHOLDS["autocorr_peak"]
        violated = value < thr
        text = f"autocorr = {value:.3f}  (>= {thr})"
    elif key == "hr":
        lo, hi = THRESHOLDS["hr_min"], THRESHOLDS["hr_max"]
        violated = value < lo or value > hi if value > 0 else True
        text = f"HR = {value:.1f} bpm  ({lo:.0f}~{hi:.0f})"
    elif key == "regularity":
        thr = THRESHOLDS["regularity"]
        violated = value >= thr
        text = f"regularity = {value:.4f}  (< {thr})"
    elif key == "flatline_ratio":
        thr = THRESHOLDS["flatline_ratio"]
        violated = value >= thr
        text = f"flatline = {value:.4f}  (< {thr})"
    elif key == "resp_power_ratio":
        text = f"resp_power = {value:.4f}"
        violated = False
    elif key == "n_peaks":
        violated = value < 2
        text = f"peaks = {int(value)}"
    else:
        text = f"{key} = {value}"
        violated = False

    color = "#d32f2f" if violated else "#2e7d32"
    return text, color


def main() -> None:
    parser = argparse.ArgumentParser(description="CVP 상세 품질 시각화")
    parser.add_argument("--out", default="cvp_detail.png", help="출력 PNG")
    parser.add_argument("--cases", type=int, nargs="+",
                        default=[14, 19, 26, 38, 44, 52, 55, 4105, 4112, 4120])
    parser.add_argument("--n-pass", type=int, default=9, help="PASS 예시 수")
    parser.add_argument("--n-fail", type=int, default=9, help="FAIL 예시 수")
    args = parser.parse_args()

    all_pass: list[CvpWindow] = []
    all_fail: list[CvpWindow] = []

    for cid in args.cases:
        print(f"Loading case {cid}...")
        windows = _load_cvp_windows(cid, max_windows=10)
        for w in windows:
            if w.passed:
                all_pass.append(w)
            else:
                all_fail.append(w)
        n_p = sum(1 for w in windows if w.passed)
        n_f = sum(1 for w in windows if not w.passed)
        if windows:
            print(f"  CVP: {len(windows)} windows ({n_p} pass, {n_f} fail)")

    # Select diverse examples (from different cases)
    def _select_diverse(pool: list[CvpWindow], n: int) -> list[CvpWindow]:
        seen_cases: set[int] = set()
        selected: list[CvpWindow] = []
        selected_ids: set[int] = set()
        # First pass: one per case
        for i, w in enumerate(pool):
            if w.case_id not in seen_cases and len(selected) < n:
                selected.append(w)
                selected_ids.add(id(w))
                seen_cases.add(w.case_id)
        # Second pass: fill remaining
        for i, w in enumerate(pool):
            if len(selected) >= n:
                break
            if id(w) not in selected_ids:
                selected.append(w)
                selected_ids.add(id(w))
        return selected[:n]

    passes = _select_diverse(all_pass, args.n_pass)
    fails = _select_diverse(all_fail, args.n_fail)

    print(f"\nSelected: {len(passes)} PASS, {len(fails)} FAIL")

    if not passes and not fails:
        print("No CVP data found.", file=sys.stderr)
        sys.exit(1)

    # Layout: n_cols=3, PASS rows on top, FAIL rows on bottom
    n_cols = 3
    n_pass_rows = (len(passes) + n_cols - 1) // n_cols
    n_fail_rows = (len(fails) + n_cols - 1) // n_cols
    n_rows = n_pass_rows + n_fail_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)
    t = np.arange(int(WINDOW_S * TARGET_SR)) / TARGET_SR

    # Metric keys in display order
    metric_keys = ["hr", "n_peaks", "regularity", "autocorr_peak", "flatline_ratio", "resp_power_ratio"]

    def _plot_window(ax: plt.Axes, w: CvpWindow, is_pass: bool) -> None:
        bg = "#e8f5e9" if is_pass else "#fbe9e7"
        line_color = "steelblue" if is_pass else "indianred"
        title_color = "#2e7d32" if is_pass else "#c62828"
        status = "PASS" if is_pass else "FAIL"

        ax.set_facecolor(bg)
        ax.plot(t, w.signal, color=line_color, linewidth=0.7)
        ax.set_title(f"Case #{w.case_id}  [{status}]", fontsize=11, fontweight="bold", color=title_color)

        # Build metrics text with per-line coloring
        d = w.domain
        lines = []
        colors = []

        if d.get("_basic_fail"):
            lines.append("** basic quality FAIL **")
            colors.append("#d32f2f")
            # Show basic metrics
            for bk in ["flatline_ratio", "high_freq_ratio", "amplitude"]:
                if bk in w.basic:
                    lines.append(f"  {bk} = {w.basic[bk]:.4f}")
                    colors.append("#666666")
        else:
            for mk in metric_keys:
                if mk in d:
                    text, color = _format_metric(mk, d[mk], d)
                    lines.append(text)
                    colors.append(color)

        # Render multi-color text
        y_pos = 0.96
        for line_text, line_color in zip(lines, colors):
            ax.text(
                0.98, y_pos, line_text,
                transform=ax.transAxes, fontsize=8,
                ha="right", va="top", color=line_color,
                fontweight="bold" if line_color == "#d32f2f" else "normal",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
            )
            y_pos -= 0.09

        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("mmHg", fontsize=8)
        ax.tick_params(labelsize=7)

    # Plot PASS examples
    for i, w in enumerate(passes):
        row, col = divmod(i, n_cols)
        _plot_window(axes[row][col], w, is_pass=True)

    # Plot FAIL examples
    for i, w in enumerate(fails):
        row, col = divmod(i, n_cols)
        _plot_window(axes[n_pass_rows + row][col], w, is_pass=False)

    # Hide unused subplots
    for i in range(len(passes), n_pass_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].axis("off")
    for i in range(len(fails), n_fail_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[n_pass_rows + row][col].axis("off")

    # Section labels
    if n_pass_rows > 0:
        axes[0][0].annotate(
            "PASS", xy=(0, 1.15), xycoords="axes fraction",
            fontsize=14, fontweight="bold", color="#2e7d32",
        )
    if n_fail_rows > 0:
        axes[n_pass_rows][0].annotate(
            "FAIL", xy=(0, 1.15), xycoords="axes fraction",
            fontsize=14, fontweight="bold", color="#c62828",
        )

    fig.suptitle(
        "CVP Domain Quality Check — Real VitalDB Data (Detail)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()