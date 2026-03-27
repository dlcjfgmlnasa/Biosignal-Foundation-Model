# -*- coding:utf-8 -*-
"""Autocorrelation threshold sweep 시각화.

각 signal type(ECG/ABP/PPG/CVP)별로 다양한 노이즈 수준의 synthetic 신호를 생성하고,
autocorrelation peak 값과 현재 threshold 기준으로 PASS/FAIL을 시각화한다.

상단: 각 케이스 파형 (grid), 하단: bar chart + threshold line.

사용법:
    python -m scripts.viz_threshold_sweep
    python -m scripts.viz_threshold_sweep --out-dir .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.parser._common import _autocorrelation_peak
from data.parser.vitaldb import _apply_bandpass, _apply_lowpass


SR = 100.0
DURATION_S = 10.0
RNG = np.random.default_rng(42)


def _make_time() -> np.ndarray:
    return np.arange(int(SR * DURATION_S)) / SR


# ── Signal generators ─────────────────────────────────────────


def _gaussian(t: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    """Gaussian pulse."""
    return amp * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))


def _ecg_base(t: np.ndarray) -> np.ndarray:
    """Realistic ECG: sharp R-peak (Gaussian), P-wave, T-wave, flat baseline."""
    rr_s = 60.0 / 72.0  # 0.833s per beat
    ecg = np.zeros_like(t)
    beat_start = 0.0
    while beat_start < t[-1]:
        # P-wave: small bump before QRS, ~0.08s duration
        ecg += _gaussian(t, beat_start + 0.16, sigma=0.04, amp=0.15)
        # Q-wave: small negative deflection
        ecg += _gaussian(t, beat_start + 0.28, sigma=0.008, amp=-0.1)
        # R-peak: sharp, narrow spike (~10ms width)
        ecg += _gaussian(t, beat_start + 0.30, sigma=0.012, amp=1.2)
        # S-wave: negative after R
        ecg += _gaussian(t, beat_start + 0.32, sigma=0.010, amp=-0.25)
        # T-wave: broad, moderate bump
        ecg += _gaussian(t, beat_start + 0.52, sigma=0.06, amp=0.3)
        beat_start += rr_s
    return ecg


def _abp_base(t: np.ndarray) -> np.ndarray:
    """Realistic ABP: fast systolic upstroke, dicrotic notch, diastolic decay."""
    rr_s = 60.0 / 72.0
    abp = np.full_like(t, 60.0)  # diastolic baseline ~60 mmHg
    beat_start = 0.0
    while beat_start < t[-1]:
        # Systolic upstroke: skewed Gaussian (sharp rise, slower fall)
        dt = t - (beat_start + 0.10)
        # Rising phase: narrow Gaussian
        systolic = 60.0 * np.exp(-(dt ** 2) / (2 * 0.03 ** 2))
        # Asymmetric decay: exponential after peak
        decay_mask = dt > 0
        systolic[decay_mask] = 60.0 * np.exp(-dt[decay_mask] / 0.15)
        abp += systolic
        # Dicrotic notch: small dip then bump at ~0.35s after beat start
        abp += _gaussian(t, beat_start + 0.32, sigma=0.012, amp=-8.0)
        abp += _gaussian(t, beat_start + 0.38, sigma=0.03, amp=10.0)
        beat_start += rr_s
    return abp


def _ppg_base(t: np.ndarray) -> np.ndarray:
    """Realistic PPG: rounded systolic pulse, smaller dicrotic bump."""
    rr_s = 60.0 / 72.0
    ppg = np.full_like(t, 500.0)  # baseline
    beat_start = 0.0
    while beat_start < t[-1]:
        # Systolic peak: broader than ABP
        dt = t - (beat_start + 0.12)
        pulse = 40.0 * np.exp(-(dt ** 2) / (2 * 0.04 ** 2))
        decay_mask = dt > 0
        pulse[decay_mask] = 40.0 * np.exp(-dt[decay_mask] / 0.18)
        ppg += pulse
        # Dicrotic wave: rounded, smaller than ABP notch
        ppg += _gaussian(t, beat_start + 0.40, sigma=0.05, amp=8.0)
        beat_start += rr_s
    return ppg


def _cvp_base(t: np.ndarray) -> np.ndarray:
    """Realistic CVP: a-wave, c-wave, x-descent, v-wave, y-descent + respiratory variation."""
    rr_s = 60.0 / 72.0
    cvp = np.full_like(t, 8.0)  # mean CVP ~8 mmHg
    beat_start = 0.0
    while beat_start < t[-1]:
        # a-wave: atrial contraction (largest)
        cvp += _gaussian(t, beat_start + 0.10, sigma=0.04, amp=2.5)
        # c-wave: tricuspid bulging (small)
        cvp += _gaussian(t, beat_start + 0.22, sigma=0.02, amp=0.8)
        # x-descent: atrial relaxation (negative)
        cvp += _gaussian(t, beat_start + 0.35, sigma=0.05, amp=-1.5)
        # v-wave: venous filling
        cvp += _gaussian(t, beat_start + 0.55, sigma=0.05, amp=2.0)
        # y-descent: tricuspid opening (negative)
        cvp += _gaussian(t, beat_start + 0.70, sigma=0.04, amp=-1.2)
        beat_start += rr_s
    # Respiratory variation (0.25Hz, ~15 breaths/min)
    cvp += 1.5 * np.sin(2 * np.pi * 0.25 * t)
    return cvp


# ── Signal type configs ──────────────────────────────────────


SIGNAL_CONFIGS = {
    "ECG": {
        "base_fn": _ecg_base,
        "threshold": 0.25,
        "filter_fn": lambda sig: sig,  # ECG: no additional filter (bandpass already in pipeline)
        "cases": [
            ("Clean", 0.0),
            ("Mild noise", 0.1),
            ("Moderate", 0.2),
            ("Heavy", 0.3),
            ("Very heavy", 0.5),
            ("Pure noise", None),
        ],
        "pure_fn": lambda: _apply_bandpass(RNG.normal(0, 1, int(SR * DURATION_S)), 0.5, 40.0, SR),
    },
    "ABP": {
        "base_fn": _abp_base,
        "threshold": 0.3,
        "filter_fn": lambda sig: _apply_lowpass(sig, hi=15.0, sr=SR),
        "cases": [
            ("Clean", 0.0),
            ("Mild noise", 5.0),
            ("Moderate", 25.0),
            ("Heavy", 35.0),
            ("Very heavy", 50.0),
            ("Pure noise", None),
        ],
        "pure_fn": lambda: _apply_lowpass(RNG.normal(80, 30, int(SR * DURATION_S)), hi=15.0, sr=SR),
    },
    "PPG": {
        "base_fn": _ppg_base,
        "threshold": 0.3,
        "filter_fn": lambda sig: _apply_lowpass(sig, hi=8.0, sr=SR),
        "cases": [
            ("Clean", 0.0),
            ("Mild noise", 5.0),
            ("Moderate", 25.0),
            ("Heavy", 40.0),
            ("Very heavy", 60.0),
            ("Pure noise", None),
        ],
        "pure_fn": lambda: _apply_lowpass(RNG.normal(500, 50, int(SR * DURATION_S)), hi=8.0, sr=SR),
    },
    "CVP": {
        "base_fn": _cvp_base,
        "threshold": 0.25,
        "filter_fn": lambda sig: _apply_lowpass(sig, hi=10.0, sr=SR),
        "cases": [
            ("Clean", 0.0),
            ("Mild noise", 0.5),
            ("Moderate", 1.5),
            ("Heavy", 3.0),
            ("Very heavy", 6.0),
            ("Pure noise", None),
        ],
        "pure_fn": lambda: _apply_lowpass(RNG.normal(8, 3, int(SR * DURATION_S)), hi=10.0, sr=SR),
    },
}


def _generate_cases(cfg: dict) -> list[tuple[str, np.ndarray, float]]:
    """(label, signal, autocorr_peak) 리스트를 생성한다."""
    t = _make_time()
    base = cfg["base_fn"](t)
    filter_fn = cfg["filter_fn"]
    min_lag = 60.0 / 200.0
    max_lag = 60.0 / 30.0

    results = []
    for label, noise_std in cfg["cases"]:
        if noise_std is None:
            # Pure noise
            sig = cfg["pure_fn"]()
        elif noise_std == 0.0:
            sig = base.copy()
        else:
            sig = base + RNG.normal(0, noise_std, len(t))
            sig = filter_fn(sig)

        ac = _autocorrelation_peak(sig, SR, min_lag, max_lag)
        results.append((label, sig, ac))

    return results


def _plot_signal_type(sig_name: str, cfg: dict, out_dir: Path) -> Path:
    """단일 signal type의 threshold sweep figure를 생성한다."""
    cases = _generate_cases(cfg)
    n_cases = len(cases)
    threshold = cfg["threshold"]

    # Layout: top rows = waveforms (3 cols), bottom row = bar chart (full width)
    n_wave_cols = 3
    n_wave_rows = (n_cases + n_wave_cols - 1) // n_wave_cols

    fig = plt.figure(figsize=(5 * n_wave_cols, 3 * n_wave_rows + 3.5))
    gs = fig.add_gridspec(n_wave_rows + 1, n_wave_cols, height_ratios=[1] * n_wave_rows + [1.2])

    t = _make_time()

    labels = []
    acorr_vals = []

    # ── Waveform subplots ──
    for i, (label, sig, ac) in enumerate(cases):
        row, col = divmod(i, n_wave_cols)
        ax = fig.add_subplot(gs[row, col])

        passed = ac >= threshold
        bg_color = "#e8f5e9" if passed else "#fbe9e7"
        line_color = "steelblue" if passed else "indianred"
        title_color = "green" if passed else "red"
        status = "PASS" if passed else "FAIL"

        ax.set_facecolor(bg_color)
        ax.plot(t, sig, color=line_color, linewidth=0.6)
        ax.set_title(f"{label}  [{status}]", fontsize=9, fontweight="bold", color=title_color)
        ax.text(
            0.98, 0.92, f"acorr={ac:.3f}",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
        )
        ax.tick_params(labelsize=6)
        if row == n_wave_rows - 1:
            ax.set_xlabel("Time (s)", fontsize=7)

        labels.append(label)
        acorr_vals.append(ac)

    # Hide unused waveform subplots
    for i in range(n_cases, n_wave_rows * n_wave_cols):
        row, col = divmod(i, n_wave_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

    # ── Bar chart subplot ──
    ax_bar = fig.add_subplot(gs[n_wave_rows, :])
    colors = ["#4caf50" if v >= threshold else "#ef5350" for v in acorr_vals]
    bars = ax_bar.bar(range(n_cases), acorr_vals, color=colors, edgecolor="white", width=0.7)

    # Threshold line
    ax_bar.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold = {threshold}")

    # Labels
    ax_bar.set_xticks(range(n_cases))
    ax_bar.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax_bar.set_ylabel("Autocorrelation Peak", fontsize=9)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(fontsize=9, loc="upper right")
    ax_bar.tick_params(labelsize=8)

    # Value labels on bars
    for bar, val in zip(bars, acorr_vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    fig.suptitle(
        f"{sig_name} — Autocorrelation Threshold Sweep (threshold = {threshold})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / f"{sig_name.lower()}_threshold_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Autocorrelation threshold sweep 시각화")
    parser.add_argument("--out-dir", default=".", help="출력 디렉토리")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sig_name, cfg in SIGNAL_CONFIGS.items():
        path = _plot_signal_type(sig_name, cfg, out_dir)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()