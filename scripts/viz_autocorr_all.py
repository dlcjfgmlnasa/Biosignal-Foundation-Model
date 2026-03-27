# -*- coding:utf-8 -*-
"""4개 signal type (ECG/ABP/PPG/CVP)의 autocorrelation 주기성 체크 비교 시각화.

Usage
-----
python -m scripts.viz_autocorr_all --out autocorr_all_signals.png
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

from data.parser._common import _autocorrelation_peak

# ── Signal type 설정 ──────────────────────────────────────────

SR = 100.0          # sampling rate (Hz)
DURATION = 10.0     # seconds
N = int(SR * DURATION)
MIN_HR = 30.0
MAX_HR = 200.0
MIN_LAG_S = 60.0 / MAX_HR   # 0.3s
MAX_LAG_S = 60.0 / MIN_HR   # 2.0s

SIGNAL_CONFIGS = [
    {
        "name": "ECG",
        "threshold": 0.25,
        "gen_normal": "_gen_ecg",
    },
    {
        "name": "ABP",
        "threshold": 0.3,
        "gen_normal": "_gen_abp",
    },
    {
        "name": "PPG",
        "threshold": 0.3,
        "gen_normal": "_gen_ppg",
    },
    {
        "name": "CVP",
        "threshold": 0.25,
        "gen_normal": "_gen_cvp",
    },
]


# ── Synthetic signal generators ──────────────────────────────

def _gen_ecg() -> np.ndarray:
    """Synthetic ECG: sharp QRS-like spikes at ~75 bpm."""
    t = np.arange(N) / SR
    hr_bpm = 75.0
    period = 60.0 / hr_bpm
    sig = np.zeros(N)
    for beat_time in np.arange(0, DURATION, period):
        idx = int(beat_time * SR)
        if idx < N:
            # Sharp R-peak (Gaussian, sigma ~10ms)
            window = np.arange(max(0, idx - 10), min(N, idx + 10))
            sig[window] += np.exp(-0.5 * ((window - idx) / 1.0) ** 2)
            # Small T-wave
            t_idx = min(idx + 15, N - 1)
            window_t = np.arange(max(0, t_idx - 8), min(N, t_idx + 8))
            sig[window_t] += 0.3 * np.exp(-0.5 * ((window_t - t_idx) / 2.0) ** 2)
    # Add slight baseline wander and noise
    sig += 0.05 * np.sin(2 * np.pi * 0.15 * t)
    sig += 0.02 * np.random.randn(N)
    return sig


def _gen_abp() -> np.ndarray:
    """Synthetic ABP: periodic pulse wave at ~72 bpm."""
    t = np.arange(N) / SR
    hr_bpm = 72.0
    freq = hr_bpm / 60.0
    # Sawtooth-like waveform (fast upstroke, slow decay)
    phase = (t * freq) % 1.0
    sig = np.exp(-3.0 * phase) * np.sin(np.pi * phase) * 2.0
    # Add harmonics for realistic shape
    sig += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    sig += 0.05 * np.random.randn(N)
    return sig


def _gen_ppg() -> np.ndarray:
    """Synthetic PPG: smooth periodic pulse at ~70 bpm."""
    t = np.arange(N) / SR
    hr_bpm = 70.0
    freq = hr_bpm / 60.0
    # PPG is smoother than ABP
    sig = np.sin(2 * np.pi * freq * t)
    sig += 0.4 * np.sin(2 * np.pi * 2 * freq * t - np.pi / 4)
    sig += 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    # Respiratory modulation
    sig *= 1.0 + 0.1 * np.sin(2 * np.pi * 0.2 * t)
    sig += 0.03 * np.random.randn(N)
    return sig


def _gen_cvp() -> np.ndarray:
    """Synthetic CVP: low-amplitude a/c/v waves at ~68 bpm + respiratory variation."""
    t = np.arange(N) / SR
    hr_bpm = 68.0
    freq = hr_bpm / 60.0
    # a-wave + c-wave + v-wave (smaller amplitude than ABP)
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    sig += 0.2 * np.sin(2 * np.pi * 2 * freq * t + np.pi / 3)
    sig += 0.1 * np.sin(2 * np.pi * 3 * freq * t + np.pi / 2)
    # Strong respiratory variation
    sig += 0.3 * np.sin(2 * np.pi * 0.2 * t)
    sig += 0.05 * np.random.randn(N)
    return sig


_GENERATORS = {
    "_gen_ecg": _gen_ecg,
    "_gen_abp": _gen_abp,
    "_gen_ppg": _gen_ppg,
    "_gen_cvp": _gen_cvp,
}


def _gen_filtered_noise() -> np.ndarray:
    """Lowpass-filtered random noise (cutoff ~5Hz)."""
    raw = np.random.randn(N)
    b, a = butter(4, 5.0 / (SR / 2), btype="low")
    return filtfilt(b, a, raw)


# ── Visualization ────────────────────────────────────────────

def main(out_path: str) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(
        "Autocorrelation Periodicity Check — All Signal Types",
        fontsize=14, fontweight="bold", y=0.98,
    )

    t = np.arange(N) / SR

    for row, cfg in enumerate(SIGNAL_CONFIGS):
        name = cfg["name"]
        threshold = cfg["threshold"]
        gen_fn = _GENERATORS[cfg["gen_normal"]]

        # Generate signals
        signals = [
            ("Normal", gen_fn()),
            ("Noise (LP-filtered)", _gen_filtered_noise()),
        ]

        for col, (label, sig) in enumerate(signals):
            ax = axes[row, col]
            peak = _autocorrelation_peak(sig, SR, MIN_LAG_S, MAX_LAG_S)
            passed = peak >= threshold

            # Plot waveform
            ax.plot(t, sig, linewidth=0.6, color="k")

            # Background color
            bg_color = "#d4edda" if passed else "#f8d7da"  # green / red
            ax.set_facecolor(bg_color)

            # Title with result
            status = "PASS" if passed else "FAIL"
            status_color = "#155724" if passed else "#721c24"
            ax.set_title(
                f"{name} — {label}\n"
                f"autocorr_peak={peak:.4f}  threshold={threshold}  [{status}]",
                fontsize=9, color=status_color, fontweight="bold",
            )

            ax.set_xlim(0, DURATION)
            if col == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold")
            if row == 3:
                ax.set_xlabel("Time (s)", fontsize=9)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize autocorrelation periodicity check for 4 signal types",
    )
    parser.add_argument(
        "--out", type=str, default="autocorr_all_signals.png",
        help="Output image path (default: autocorr_all_signals.png)",
    )
    args = parser.parse_args()
    main(args.out)