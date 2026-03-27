# -*- coding:utf-8 -*-
"""실제 VitalDB 데이터로 domain quality check 결과를 시각화.

VitalDB API로 실제 환자 데이터를 다운로드하고, 파이프라인 필터 적용 후
각 signal type별 PASS/FAIL 예시를 grid plot으로 보여준다.

사용법:
    python -m scripts.viz_real_quality
    python -m scripts.viz_real_quality --out real_quality_check.png --cases 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.parser._common import (
    domain_quality_check,
    resample_to_target,
    segment_quality_score,
)
from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    SIGNAL_TYPES,
    TRACK_MAP,
    _apply_filter,
    _apply_median_filter,
    _apply_notch_filter,
    _apply_range_check,
    _detect_electrocautery,
    _detect_motion_artifact,
    _extract_nan_free_segments,
)

TARGET_SR = 100.0
WINDOW_S = 10.0

# 대표 트랙: signal type별 1개만
REPRESENTATIVE_TRACKS: dict[str, str] = {
    "ecg": "SNUADC/ECG_II",
    "abp": "SNUADC/ART",
    "ppg": "SNUADC/PLETH",
    "cvp": "SNUADC/CVP",
}

SIGNAL_LABELS = {"ecg": "ECG", "abp": "ABP", "ppg": "PPG", "cvp": "CVP"}


@dataclass
class WindowResult:
    """윈도우 품질 검사 결과."""
    signal: np.ndarray      # (win_samples,) filtered + resampled
    stype_key: str
    case_id: int
    win_idx: int
    passed: bool
    domain_result: dict
    basic_result: dict


def _load_and_process(case_id: int, stype_key: str, track_name: str,
                      max_windows: int = 20) -> list[WindowResult]:
    """단일 케이스의 단일 트랙을 로드하고 윈도우 단위 품질 검사 결과를 반환."""
    import vitaldb

    cfg = SIGNAL_CONFIGS[stype_key]

    # 트랙별 native SR 추정
    native_sr_map = {
        "ecg": 500.0, "abp": 500.0, "ppg": 500.0,
        "cvp": 500.0, "eeg": 128.0, "co2": 62.5, "awp": 62.5,
    }
    native_sr = native_sr_map.get(stype_key, 500.0)

    try:
        data = vitaldb.load_case(case_id, [track_name], interval=1.0 / native_sr)
    except Exception as exc:
        print(f"  [WARN] case {case_id} {track_name} load failed: {exc}", file=sys.stderr)
        return []

    if data is None or len(data) == 0:
        return []

    data = data[:, 0].flatten()

    # Step 1: Range check
    if cfg.valid_range is not None:
        data, _ = _apply_range_check(data, cfg.valid_range)

    # Step 2: Spike detection
    if cfg.spike_detection:
        data, _ = _detect_electrocautery(data, native_sr, threshold_std=cfg.spike_threshold_std)

    # Step 2b: PPG motion artifact
    if stype_key == "ppg":
        data, _ = _detect_motion_artifact(data, native_sr)

    # Step 3: NaN-free segments
    min_samples = int(60.0 * native_sr)
    segments = _extract_nan_free_segments(data, min_samples)
    if not segments:
        return []

    results: list[WindowResult] = []
    win_samples = int(WINDOW_S * TARGET_SR)

    for segment in segments:
        # Median filter → Notch → Bandpass/Lowpass
        if cfg.median_kernel > 0:
            segment = _apply_median_filter(segment, kernel_size=cfg.median_kernel)
        if cfg.notch_freq is not None:
            segment = _apply_notch_filter(segment, freq=cfg.notch_freq, sr=native_sr)
        segment = _apply_filter(segment, cfg, native_sr)

        # Resample to 100Hz
        if native_sr != TARGET_SR:
            segment = resample_to_target(segment, orig_sr=native_sr, target_sr=TARGET_SR)

        # Window-level quality check
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
                results.append(WindowResult(
                    signal=win, stype_key=stype_key, case_id=case_id,
                    win_idx=len(results), passed=False,
                    domain_result={"pass": False, "reason": "basic_fail"},
                    basic_result=basic,
                ))
                if len(results) >= max_windows:
                    return results
                continue

            domain = domain_quality_check(stype_key, win, sr=TARGET_SR)
            results.append(WindowResult(
                signal=win, stype_key=stype_key, case_id=case_id,
                win_idx=len(results), passed=domain["pass"],
                domain_result=domain, basic_result=basic,
            ))
            if len(results) >= max_windows:
                return results

    return results


def _select_examples(
    all_results: list[WindowResult],
    n_pass: int = 3,
    n_fail: int = 3,
) -> list[WindowResult]:
    """PASS/FAIL 각 n개씩 선택. 부족하면 있는 만큼."""
    passes = [r for r in all_results if r.passed]
    fails = [r for r in all_results if not r.passed]
    selected = passes[:n_pass] + fails[:n_fail]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="실제 VitalDB 데이터 quality check 시각화")
    parser.add_argument("--out", default="real_quality_check.png", help="출력 PNG 경로")
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3, 5, 8],
                        help="VitalDB case IDs (기본: 1 2 3 5 8)")
    parser.add_argument("--n-examples", type=int, default=3,
                        help="signal type별 PASS/FAIL 예시 수 (기본 3)")
    args = parser.parse_args()

    n_ex = args.n_examples
    stypes = list(REPRESENTATIVE_TRACKS.keys())

    # Collect results per signal type
    by_stype: dict[str, list[WindowResult]] = {s: [] for s in stypes}

    for case_id in args.cases:
        print(f"Loading case {case_id}...")
        for stype_key, track_name in REPRESENTATIVE_TRACKS.items():
            results = _load_and_process(case_id, stype_key, track_name, max_windows=10)
            by_stype[stype_key].extend(results)
            n_pass = sum(1 for r in results if r.passed)
            n_fail = sum(1 for r in results if not r.passed)
            if results:
                print(f"  {SIGNAL_LABELS[stype_key]}: {len(results)} windows ({n_pass} pass, {n_fail} fail)")

    # Select examples
    examples: dict[str, list[WindowResult]] = {}
    for stype_key in stypes:
        examples[stype_key] = _select_examples(by_stype[stype_key], n_pass=n_ex, n_fail=n_ex)

    # Plot
    active_stypes = [s for s in stypes if examples[s]]
    if not active_stypes:
        print("No data found. Check network connectivity to VitalDB.", file=sys.stderr)
        sys.exit(1)

    n_rows = len(active_stypes)
    n_cols = max(len(examples[s]) for s in active_stypes)
    if n_cols == 0:
        print("No windows collected.", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)
    t = np.arange(int(WINDOW_S * TARGET_SR)) / TARGET_SR

    for row, stype_key in enumerate(active_stypes):
        exs = examples[stype_key]
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(exs):
                ax.axis("off")
                continue

            wr = exs[col]
            passed = wr.passed
            bg = "#e8f5e9" if passed else "#fbe9e7"
            line_color = "steelblue" if passed else "indianred"
            title_color = "green" if passed else "red"
            status = "PASS" if passed else "FAIL"

            ax.set_facecolor(bg)
            ax.plot(t, wr.signal, color=line_color, linewidth=0.6)

            label = SIGNAL_LABELS[stype_key]
            ax.set_title(
                f"{label} case#{wr.case_id}  [{status}]",
                fontsize=9, fontweight="bold", color=title_color,
            )

            # Metrics text
            dr = wr.domain_result
            metrics = []
            if "hr" in dr and dr["hr"] > 0:
                metrics.append(f"HR={dr['hr']:.0f}")
            if "regularity" in dr and dr.get("regularity", 1.0) < 1.0:
                metrics.append(f"reg={dr['regularity']:.3f}")
            if "autocorr_peak" in dr:
                metrics.append(f"acorr={dr['autocorr_peak']:.3f}")
            if "resp_power_ratio" in dr and dr["resp_power_ratio"] > 0:
                metrics.append(f"resp={dr['resp_power_ratio']:.3f}")
            if "flatline_ratio" in dr and dr.get("flatline_ratio", 0) > 0:
                metrics.append(f"flat={dr['flatline_ratio']:.2f}")
            if dr.get("reason") == "basic_fail":
                br = wr.basic_result
                metrics.append(f"hf={br.get('high_freq_ratio', 0):.3f}")
                metrics.append(f"amp={br.get('amplitude', 0):.2f}")

            if metrics:
                ax.text(
                    0.98, 0.95, "\n".join(metrics),
                    transform=ax.transAxes, fontsize=7,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
                    family="monospace",
                )

            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.tick_params(labelsize=6)
            if row == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=7)

    fig.suptitle(
        f"Real VitalDB Quality Check — Cases {args.cases}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()