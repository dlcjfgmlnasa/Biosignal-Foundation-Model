# -*- coding:utf-8 -*-
"""기존 처리된 zarr 데이터에서 신호별 quality score 분포를 측정한다.

사용법:
  source .venv/Scripts/activate
  PYTHONPATH=. python scripts/measure_quality_stats.py

출력:
  신호 타입별 flatline_ratio, clip_ratio, high_freq_ratio 통계 (mean, std, p50, p95, p99, max)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.parser._common import segment_quality_score

SIGNAL_TYPE_NAMES = {0: "ecg", 1: "abp", 2: "eeg", 3: "ppg", 4: "cvp", 5: "co2", 6: "awp"}

PROCESSED_DIR = Path("datasets/processed")


def load_zarr_data(zarr_path: Path) -> np.ndarray | None:
    """zarr 파일을 로드하여 1D float32로 반환한다."""
    try:
        import zarr
        z = zarr.open(str(zarr_path), mode="r")
        data = np.array(z[:], dtype=np.float32)
        return data.flatten()
    except Exception as exc:
        print(f"  [WARN] {zarr_path.name} 로드 실패: {exc}", file=sys.stderr)
        return None


def load_cache_pt(pt_path: Path) -> np.ndarray | None:
    """cache.pt 파일을 로드하여 1D numpy로 반환한다."""
    try:
        import torch
        tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
        return tensor.numpy().flatten().astype(np.float32)
    except Exception as exc:
        print(f"  [WARN] {pt_path.name} 로드 실패: {exc}", file=sys.stderr)
        return None


def measure_segment_quality(
    data: np.ndarray,
    window_s: float = 30.0,
    sr: float = 100.0,
) -> list[dict[str, float]]:
    """데이터를 window 단위로 잘라 quality score를 측정한다.

    전체 세그먼트 1개로 측정하면 긴 녹음에서 국소 노이즈가 희석되므로
    30초 윈도우로 분할하여 측정한다.
    """
    window_samples = int(window_s * sr)
    scores = []

    # 전체 세그먼트도 측정
    scores.append(segment_quality_score(data))

    # 윈도우별 측정
    n = len(data)
    for start in range(0, n, window_samples):
        end = min(start + window_samples, n)
        if end - start < window_samples // 2:
            break
        scores.append(segment_quality_score(data[start:end]))

    return scores


def main() -> None:
    if not PROCESSED_DIR.exists():
        print(f"ERROR: {PROCESSED_DIR}가 존재하지 않습니다.", file=sys.stderr)
        sys.exit(1)

    jsonl_path = PROCESSED_DIR / "manifest.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path}가 없습니다.", file=sys.stderr)
        sys.exit(1)

    # subject 목록 로드
    subjects = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                subjects.append(json.loads(line))

    print(f"Subject {len(subjects)}명 발견\n")

    # 신호별 quality score 수집
    stats: dict[str, list[dict[str, float]]] = defaultdict(list)
    segment_counts: dict[str, int] = defaultdict(int)

    for subj in subjects:
        manifest_path = PROCESSED_DIR / subj["manifest"]
        if not manifest_path.exists():
            continue

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        subj_dir = manifest_path.parent

        for session in manifest["sessions"]:
            for rec in session["recordings"]:
                signal_type = rec["signal_type"]
                stype_name = SIGNAL_TYPE_NAMES.get(signal_type, f"unk_{signal_type}")
                fname = rec["file"]
                sr = rec.get("sampling_rate", 100.0)

                # cache.pt 우선 시도 (빠름)
                cache_path = subj_dir / f"{fname}.cache.pt"
                zarr_path = subj_dir / fname

                data = None
                if cache_path.exists():
                    data = load_cache_pt(cache_path)
                if data is None and zarr_path.exists():
                    data = load_zarr_data(zarr_path)

                if data is None:
                    continue

                scores = measure_segment_quality(data, window_s=30.0, sr=sr)
                stats[stype_name].extend(scores)
                segment_counts[stype_name] += 1

    # 결과 출력
    print("=" * 90)
    print(f"{'Signal':<8} {'Metric':<18} {'Mean':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8} {'N':>6}")
    print("=" * 90)

    for stype in sorted(stats.keys()):
        scores_list = stats[stype]
        n = len(scores_list)
        n_segs = segment_counts[stype]

        for metric in ["flatline_ratio", "clip_ratio", "high_freq_ratio"]:
            values = np.array([s[metric] for s in scores_list])

            mean = np.mean(values)
            std = np.std(values)
            p50 = np.percentile(values, 50)
            p95 = np.percentile(values, 95)
            p99 = np.percentile(values, 99)
            vmax = np.max(values)

            print(f"{stype:<8} {metric:<18} {mean:>8.4f} {std:>8.4f} {p50:>8.4f} {p95:>8.4f} {p99:>8.4f} {vmax:>8.4f} {n:>6}")

        # pass rate
        pass_rate = np.mean([1.0 if s["pass"] else 0.0 for s in scores_list]) * 100
        print(f"{stype:<8} {'pass_rate(%)':18} {pass_rate:>8.1f}{'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {n_segs:>6} segs")
        print("-" * 90)

    # high_freq_ratio에 대한 신호별 추천 threshold 출력
    print("\n\n신호별 high_freq_ratio 추천 threshold (P99 기반):")
    print("-" * 50)
    for stype in sorted(stats.keys()):
        scores_list = stats[stype]
        hf_values = np.array([s["high_freq_ratio"] for s in scores_list])
        p95 = np.percentile(hf_values, 95)
        p99 = np.percentile(hf_values, 99)
        vmax = np.max(hf_values)
        # 추천: P99 * 1.5 (여유분), 최소 0.5
        recommended = max(p99 * 1.5, 0.5)
        print(f"  {stype:<8}: P95={p95:.4f}  P99={p99:.4f}  Max={vmax:.4f}  → 추천 threshold: {recommended:.2f}")


if __name__ == "__main__":
    main()
