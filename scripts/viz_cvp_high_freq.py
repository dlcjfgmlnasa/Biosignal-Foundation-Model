# -*- coding:utf-8 -*-
"""CVP high_freq_ratio 필터링 시각화.

다양한 노이즈 수준의 synthetic CVP에 대해 lowpass 10Hz 필터 전후의
high_freq_ratio를 비교하고, threshold=0.5 기준 PASS/FAIL을 시각화한다.

핵심: "현재 threshold 0.5에서 어떤 수준의 노이즈가 통과/차단되는지" 확인.

사용법:
    python -m scripts.viz_cvp_high_freq
    python -m scripts.viz_cvp_high_freq --out cvp_high_freq_ratio.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.parser._common import segment_quality_score
from data.parser.vitaldb import _apply_lowpass


SR = 100.0
DURATION_S = 10.0
HF_THRESHOLD = 0.5  # CVP max_high_freq_ratio


def _make_base_cvp() -> np.ndarray:
    """기본 CVP 파형: a/c/v wave (72bpm) + respiratory variation."""
    t = np.arange(int(SR * DURATION_S)) / SR
    hr_hz = 72.0 / 60.0
    resp_hz = 0.25
    cvp = (
        2.0 * np.sin(2 * np.pi * hr_hz * t)
        + 0.8 * np.sin(2 * np.pi * hr_hz * 2 * t)
        + 0.5 * np.sin(2 * np.pi * hr_hz * 3 * t)
    )
    cvp += 1.5 * np.sin(2 * np.pi * resp_hz * t)
    cvp += 8.0
    return cvp


def _compute_hf_ratio(segment: np.ndarray) -> float:
    """segment_quality_score와 동일한 hf ratio 계산."""
    diffs = np.diff(segment)
    sig_energy = float(np.mean(segment ** 2))
    diff_energy = float(np.mean(diffs ** 2))
    if sig_energy < 1e-10:
        return 1.0
    return diff_energy / sig_energy


# 케이스: (label, noise_std)
CASES: list[tuple[str, float]] = [
    ("Clean", 0.0),
    ("Moderate noise", 3.0),
    ("Borderline (~threshold)", 5.0),
    ("High noise (reject)", 6.0),
    ("Very high noise", 8.0),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="CVP high_freq_ratio 필터링 시각화")
    parser.add_argument("--out", default="cvp_high_freq_ratio.png", help="출력 PNG 경로")
    args = parser.parse_args()

    base_cvp = _make_base_cvp()
    rng = np.random.default_rng(42)

    n_cases = len(CASES)
    fig, axes = plt.subplots(n_cases, 2, figsize=(14, 3 * n_cases), squeeze=False)

    col_titles = ["Before Filter (raw)", "After Lowpass 10Hz"]

    for row, (label, noise_std) in enumerate(CASES):
        # 노이즈 추가
        if noise_std > 0:
            noise = rng.normal(0, noise_std, len(base_cvp))
            raw = base_cvp + noise
        else:
            raw = base_cvp.copy()

        # Lowpass 10Hz 적용 (실제 파이프라인과 동일)
        filtered = _apply_lowpass(raw, hi=10.0, sr=SR)

        t = np.arange(len(raw)) / SR

        for col, (signal, col_label) in enumerate([(raw, col_titles[0]), (filtered, col_titles[1])]):
            ax = axes[row][col]
            hf = _compute_hf_ratio(signal)
            passed = hf < HF_THRESHOLD

            # 배경 색상
            bg_color = "#e8f5e9" if passed else "#fbe9e7"
            ax.set_facecolor(bg_color)

            # 신호 플롯
            line_color = "steelblue" if passed else "indianred"
            ax.plot(t, signal, color=line_color, linewidth=0.7)

            # 타이틀
            status = "PASS" if passed else "FAIL"
            title_color = "green" if passed else "red"

            if col == 0:
                title_text = f"{label} (std={noise_std:.0f})"
            else:
                title_text = f"{col_label}"
            ax.set_title(
                f"{title_text}  [{status}]",
                fontsize=10, fontweight="bold", color=title_color,
            )

            # hf ratio 텍스트 + threshold 표시
            metrics_text = f"hf_ratio = {hf:.4f}  (threshold = {HF_THRESHOLD})"
            ax.text(
                0.98, 0.95, metrics_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )

            ax.set_ylabel("mmHg", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == n_cases - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

            # y축 범위 통일 (raw 기준)
            if col == 0:
                ylim = (raw.min() - 1, raw.max() + 1)
            ax.set_ylim(ylim)

    # threshold 수평선 범례
    fig.suptitle(
        f"CVP High-Freq Ratio Filtering (threshold = {HF_THRESHOLD})\n"
        f"Left: raw signal  |  Right: after lowpass 10Hz (pipeline filter)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()