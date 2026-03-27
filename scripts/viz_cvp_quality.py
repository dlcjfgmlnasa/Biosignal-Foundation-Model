# -*- coding:utf-8 -*-
"""CVP quality check 시각화 스크립트.

synthetic CVP 데이터로 cvp_quality_check()의 pass/fail 결과를 시각화한다.
다양한 케이스(정상, flatline, 노이즈, 불규칙, 저진폭, 빈맥)를 그리드 플롯으로 출력.

사용법:
    python -m scripts.viz_cvp_quality
    python -m scripts.viz_cvp_quality --out cvp_quality.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.parser._common import cvp_quality_check


# ── Synthetic CVP 생성 ────────────────────────────────────────


def _make_normal_cvp(sr: float = 100.0, duration_s: float = 10.0) -> np.ndarray:
    """정상 CVP: a/c/v wave (HR ~72 bpm), respiratory variation 포함."""
    t = np.arange(int(sr * duration_s)) / sr
    hr_hz = 72.0 / 60.0  # 1.2 Hz
    resp_hz = 0.25  # 15 breaths/min

    # a/c/v wave: 기본파 + 고조파
    cvp = (
        2.0 * np.sin(2 * np.pi * hr_hz * t)           # a-wave dominant
        + 0.8 * np.sin(2 * np.pi * hr_hz * 2 * t)     # c-wave (2nd harmonic)
        + 0.5 * np.sin(2 * np.pi * hr_hz * 3 * t)     # v-wave (3rd harmonic)
    )
    # respiratory variation (baseline shift)
    cvp += 1.5 * np.sin(2 * np.pi * resp_hz * t)
    # DC offset (mean CVP ~8 mmHg)
    cvp += 8.0
    return cvp


def _make_flatline_cvp(sr: float = 100.0, duration_s: float = 10.0) -> np.ndarray:
    """센서 분리: 일정 값으로 고정."""
    return np.full(int(sr * duration_s), 5.0)


def _make_noisy_cvp(sr: float = 100.0, duration_s: float = 10.0) -> np.ndarray:
    """고주파 노이즈가 심한 CVP."""
    cvp = _make_normal_cvp(sr, duration_s)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 3.0, len(cvp))  # 강한 노이즈
    return cvp + noise


def _make_irregular_cvp(sr: float = 100.0, duration_s: float = 10.0) -> np.ndarray:
    """불규칙 심박 CVP (심방세동 유사)."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    rng = np.random.default_rng(123)

    cvp = np.zeros(n)
    pos = 0.0
    while pos < duration_s:
        # R-R interval: 0.3~1.5s (매우 불규칙)
        rr = rng.uniform(0.3, 1.5)
        idx_start = int(pos * sr)
        idx_end = min(int((pos + rr) * sr), n)
        if idx_start >= n:
            break
        local_t = np.arange(idx_end - idx_start) / sr
        amp = rng.uniform(0.5, 3.0)
        cvp[idx_start:idx_end] = amp * np.sin(2 * np.pi / rr * local_t)
        pos += rr

    cvp += 8.0
    return cvp


def _make_low_amplitude_cvp(sr: float = 100.0, duration_s: float = 10.0) -> np.ndarray:
    """극저진폭 CVP (IQR < 0.1)."""
    t = np.arange(int(sr * duration_s)) / sr
    hr_hz = 72.0 / 60.0
    cvp = 0.02 * np.sin(2 * np.pi * hr_hz * t) + 5.0
    return cvp


def _make_tachy_cvp(sr: float = 100.0, duration_s: float = 10.0) -> np.ndarray:
    """빈맥 CVP (HR ~160 bpm)."""
    t = np.arange(int(sr * duration_s)) / sr
    hr_hz = 160.0 / 60.0
    resp_hz = 0.3

    cvp = (
        1.5 * np.sin(2 * np.pi * hr_hz * t)
        + 0.5 * np.sin(2 * np.pi * hr_hz * 2 * t)
    )
    cvp += 1.0 * np.sin(2 * np.pi * resp_hz * t)
    cvp += 10.0
    return cvp


def _make_brady_cvp(sr: float = 100.0, duration_s: float = 10.0) -> np.ndarray:
    """서맥 CVP (HR ~45 bpm)."""
    t = np.arange(int(sr * duration_s)) / sr
    hr_hz = 45.0 / 60.0
    resp_hz = 0.2

    cvp = (
        2.5 * np.sin(2 * np.pi * hr_hz * t)
        + 1.0 * np.sin(2 * np.pi * hr_hz * 2 * t)
        + 0.6 * np.sin(2 * np.pi * hr_hz * 3 * t)
    )
    cvp += 1.5 * np.sin(2 * np.pi * resp_hz * t)
    cvp += 7.0
    return cvp


def _make_partial_flatline_cvp(sr: float = 100.0, duration_s: float = 10.0) -> np.ndarray:
    """부분 flatline: 전반 정상 + 후반 flat (센서 접촉 불량)."""
    half = int(sr * duration_s / 2)
    normal = _make_normal_cvp(sr, duration_s / 2)
    flat = np.full(int(sr * duration_s) - half, 5.0)
    return np.concatenate([normal, flat])


# ── 시각화 ────────────────────────────────────────────────────


CASES: list[tuple[str, callable]] = [
    ("Normal (72 bpm)", _make_normal_cvp),
    ("Bradycardia (45 bpm)", _make_brady_cvp),
    ("Tachycardia (160 bpm)", _make_tachy_cvp),
    ("Irregular (AF-like)", _make_irregular_cvp),
    ("Flatline (sensor off)", _make_flatline_cvp),
    ("Partial flatline", _make_partial_flatline_cvp),
    ("High noise", _make_noisy_cvp),
    ("Low amplitude", _make_low_amplitude_cvp),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="CVP quality check 시각화")
    parser.add_argument("--out", default="cvp_quality_check.png", help="출력 PNG 경로")
    parser.add_argument("--sr", type=float, default=100.0, help="sampling rate (Hz)")
    parser.add_argument("--duration", type=float, default=10.0, help="세그먼트 길이 (초)")
    args = parser.parse_args()

    sr = args.sr
    duration_s = args.duration

    n_cases = len(CASES)
    n_cols = 2
    n_rows = (n_cases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3 * n_rows), squeeze=False)

    for i, (label, gen_fn) in enumerate(CASES):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        segment = gen_fn(sr=sr, duration_s=duration_s)
        result = cvp_quality_check(segment, sr=sr)
        passed = result["pass"]

        t = np.arange(len(segment)) / sr
        color = "steelblue" if passed else "indianred"
        bg_color = "#e8f5e9" if passed else "#fbe9e7"

        ax.set_facecolor(bg_color)
        ax.plot(t, segment, color=color, linewidth=0.8)

        # 품질 지표 텍스트
        status = "PASS" if passed else "FAIL"
        title_color = "green" if passed else "red"
        ax.set_title(f"{label}  [{status}]", fontsize=10, fontweight="bold", color=title_color)

        metrics = []
        if "hr" in result and result["hr"] > 0:
            metrics.append(f"HR={result['hr']:.0f}")
        if "n_peaks" in result:
            metrics.append(f"peaks={result['n_peaks']}")
        if "regularity" in result and result["regularity"] < 1.0:
            metrics.append(f"reg={result['regularity']:.3f}")
        if "autocorr_peak" in result:
            metrics.append(f"acorr={result['autocorr_peak']:.3f}")
        if "resp_power_ratio" in result and result["resp_power_ratio"] > 0:
            metrics.append(f"resp={result['resp_power_ratio']:.3f}")
        if "flatline_ratio" in result:
            metrics.append(f"flat={result['flatline_ratio']:.2f}")

        if metrics:
            ax.text(
                0.98, 0.95, "  ".join(metrics),
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("mmHg", fontsize=8)
        ax.tick_params(labelsize=7)

    # 남는 subplot 숨기기
    for i in range(n_cases, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].axis("off")

    fig.suptitle("CVP Quality Check — Synthetic Cases", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()