# -*- coding:utf-8 -*-
"""Task 1: Hypotension Prediction - 데이터 준비 스크립트.

미래 5~15분 후 MAP<65 (≥1분 지속) 예측을 위한 (input_window, future_label) 쌍 생성.
4가지 입력 모드: abp, ecg, ppg, ecg_ppg

Label 소스: 항상 ABP (미래 구간의 MAP)
Input 소스: 선택된 signal type의 현재 윈도우

데이터 소스: 로컬 전처리된 .pt 파일 (vitaldb_pt_test/)

사용법:
    # ABP 입력, 5분 후 예측
    python -m downstream.hypotension.prepare_data \
        --data-dir vitaldb_pt_test --input-signals abp --horizon-min 5

    # ECG+PPG 입력, 10분 후 예측
    python -m downstream.hypotension.prepare_data \
        --data-dir vitaldb_pt_test --input-signals ecg ppg --horizon-min 10
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

TARGET_SR: float = 100.0

# signal_type 정수 → 문자열 매핑
SIGNAL_TYPE_MAP: dict[int, str] = {
    0: "ecg",
    1: "abp",
    2: "ppg",
    3: "cvp",
    4: "co2",
    5: "awp",
    6: "pap",
    7: "icp",
}


# ---- 데이터 구조 ----


@dataclass
class ForecastSample:
    """미래 저혈압 예측 샘플."""

    input_signals: dict[str, np.ndarray]  # {"ecg": (win_samples,), ...}
    label: int  # 0=normal, 1=hypotension in future
    label_value: float  # future MAP (mmHg)
    case_id: str
    win_start_sec: float  # input window 시작 (초)
    horizon_sec: float  # prediction horizon (초)


# ---- 로컬 .pt 로더 ----


def _parse_pt_filename(name: str) -> dict | None:
    """파일명에서 메타데이터를 추출한다.

    형식: {subject_id}_S{session}_{signal_name}_{spatial_id}_seg{i}_{j}.pt
    예: VDB_0239_S0_abp_1_seg0_0.pt
    """
    m = re.match(
        r"^(.+?)_S(\d+)_([a-z0-9]+)_(\d+)_seg(\d+)_(\d+)\.pt$",
        name,
    )
    if m is None:
        return None
    return {
        "subject_id": m.group(1),
        "session_id": int(m.group(2)),
        "signal_type": m.group(3),  # 문자열: "ecg", "abp", etc.
        "spatial_id": int(m.group(4)),
        "seg_i": int(m.group(5)),
        "seg_j": int(m.group(6)),
    }


def _load_local_pt_aligned_signals(
    data_dir: str,
    input_signals: list[str],
    min_duration_sec: float = 1200.0,
    max_subjects: int | None = None,
) -> list[dict]:
    """로컬 .pt 디렉토리에서 시간 정렬된 다채널 데이터를 로드한다.

    Parameters
    ----------
    data_dir : vitaldb_pt_test/ 경로.
    input_signals : 입력으로 사용할 signal types (예: ["ecg", "ppg"]).
    min_duration_sec : 최소 유효 신호 길이 (초).
    max_subjects : 최대 subject 수. None이면 전체.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signals": {"abp": array, ...}}
    """
    root = Path(data_dir)
    if not root.is_dir():
        print(f"  ERROR: Data directory not found: {root}")
        return []

    required_types = set(input_signals) | {"abp"}

    subject_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    print(f"  Scanning {len(subject_dirs)} subjects in {root}...")
    cases: list[dict] = []

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name

        # 이 subject의 모든 .pt 파일 파싱
        file_map: dict[tuple[int, int, int], dict[str, Path]] = {}
        # key: (session_id, seg_i, seg_j) → {signal_type_str: path}

        for pt_file in subj_dir.glob("*.pt"):
            meta = _parse_pt_filename(pt_file.name)
            if meta is None:
                continue
            seg_key = (meta["session_id"], meta["seg_i"], meta["seg_j"])
            if seg_key not in file_map:
                file_map[seg_key] = {}
            file_map[seg_key][meta["signal_type"]] = pt_file

        # 필요한 모든 signal type이 있는 세그먼트 찾기
        for seg_key, type_paths in file_map.items():
            available_types = set(type_paths.keys())
            if not required_types.issubset(available_types):
                continue

            # 로드
            signals: dict[str, np.ndarray] = {}
            for stype_str in required_types:
                t = torch.load(type_paths[stype_str], weights_only=True)  # (1, T)
                signals[stype_str] = t.squeeze(0).numpy()  # (T,)

            # 모든 채널을 동일 길이로 자르기
            min_len = min(len(s) for s in signals.values())
            if min_len < int(min_duration_sec * TARGET_SR):
                continue

            signals = {k: v[:min_len] for k, v in signals.items()}
            session_id, seg_i, seg_j = seg_key

            cases.append(
                {
                    "case_id": f"{subject_id}_s{session_id}_seg{seg_i}_{seg_j}",
                    "patient_id": subject_id,
                    "signals": signals,
                }
            )

    print(f"  Loaded {len(cases)} aligned segments with {required_types}")
    return cases


# ---- 윈도우 추출 + 라벨링 ----


def _has_sustained_hypotension(
    future_maps: list[float],
    threshold: float,
    min_consecutive: int,
) -> bool:
    """연속 min_consecutive개 이상 윈도우에서 MAP < threshold인지 확인한다."""
    consecutive = 0
    for m in future_maps:
        if m < threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                return True
        else:
            consecutive = 0
    return False


def extract_forecast_samples(
    cases: list[dict],
    input_signals: list[str],
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    horizon_sec: float = 300.0,
    map_threshold: float = 65.0,
    sustained_sec: float = 60.0,
) -> list[ForecastSample]:
    """시간 정렬된 다채널 데이터에서 (input, future_label) 쌍을 추출한다.

    Parameters
    ----------
    cases : 로드된 케이스 리스트.
    input_signals : 입력으로 사용할 signal types.
    window_sec : 입력 윈도우 길이 (초).
    stride_sec : 슬라이드 보폭 (초).
    horizon_sec : prediction horizon (초). 미래 이 구간 내 MAP<65 발생 여부.
    map_threshold : MAP 미만이면 hypotension.
    sustained_sec : MAP<threshold가 이 시간 이상 지속되어야 positive.

    Returns
    -------
    ForecastSample 리스트.
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    horizon_samples = int(horizon_sec * TARGET_SR)

    # MAP 계산 윈도우: 10초
    map_win_sec = 10.0
    map_win = int(map_win_sec * TARGET_SR)
    # 1분 지속 = 6개 연속 10초 윈도우
    min_consecutive = max(1, int(sustained_sec / map_win_sec))

    # 전체 필요 길이: input window + horizon
    total_needed = win_samples + horizon_samples

    samples: list[ForecastSample] = []

    for case in cases:
        signals = case["signals"]
        abp = signals["abp"]
        n_total = len(abp)

        if n_total < total_needed:
            continue

        for start in range(0, n_total - total_needed + 1, stride_samples):
            # Input window: [start, start + win_samples)
            input_dict = {}
            for stype in input_signals:
                if stype in signals:
                    input_dict[stype] = signals[stype][start : start + win_samples]

            if not input_dict:
                continue

            # Future label: ABP의 [start + win_samples, start + win_samples + horizon_samples) 구간
            future_start = start + win_samples
            future_end = future_start + horizon_samples
            future_abp = abp[future_start:future_end]

            # 미래 구간의 MAP (10초 윈도우별 평균)
            future_maps: list[float] = []
            for j in range(0, len(future_abp) - map_win + 1, map_win):
                w = future_abp[j : j + map_win]
                if not np.isnan(w).any():
                    future_maps.append(float(np.mean(w)))

            if not future_maps:
                continue

            # ≥1분 지속 MAP<65 여부 확인
            label = (
                1
                if _has_sustained_hypotension(
                    future_maps,
                    map_threshold,
                    min_consecutive,
                )
                else 0
            )

            # label_value: 미래 MAP의 최솟값 (참고용)
            min_future_map = min(future_maps)

            samples.append(
                ForecastSample(
                    input_signals=input_dict,
                    label=label,
                    label_value=min_future_map,
                    case_id=case["case_id"],
                    win_start_sec=start / TARGET_SR,
                    horizon_sec=horizon_sec,
                )
            )

    return samples


# ---- 저장 ----


def save_dataset(
    train_samples: list[ForecastSample],
    test_samples: list[ForecastSample],
    input_signals: list[str],
    horizon_sec: float,
    window_sec: float,
    out_dir: str,
) -> Path:
    """ForecastSample 리스트를 .pt로 저장한다."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples: list[ForecastSample]) -> dict:
        if not samples:
            return {
                "signals": {},
                "labels": torch.tensor([]),
                "label_values": torch.tensor([]),
            }

        # 각 signal type별 텐서 생성
        sig_tensors = {}
        for stype in input_signals:
            arrs = [s.input_signals[stype] for s in samples if stype in s.input_signals]
            if arrs:
                sig_tensors[stype] = torch.stack(
                    [torch.from_numpy(a).float() for a in arrs]
                )

        labels = torch.tensor([s.label for s in samples], dtype=torch.long)
        label_values = torch.tensor(
            [s.label_value for s in samples], dtype=torch.float32
        )
        case_ids = [s.case_id for s in samples]

        return {
            "signals": sig_tensors,
            "labels": labels,
            "label_values": label_values,
            "case_ids": case_ids,
        }

    save_dict = {
        "train": _to_tensors(train_samples),
        "test": _to_tensors(test_samples),
        "metadata": {
            "task": "hypotension_forecast",
            "source": "vitaldb_pt",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "map_threshold": 65.0,
            "sustained_sec": 60.0,
            "n_train": len(train_samples),
            "n_test": len(test_samples),
        },
    }

    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    filename = f"task1_hypotension_{mode_str}_h{horizon_min}min.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ---- 통계 출력 ----


def print_stats(
    name: str,
    samples: list[ForecastSample],
) -> None:
    """데이터셋 통계 출력."""
    if not samples:
        print(f"  {name}: 0 samples")
        return

    n_total = len(samples)
    n_hypo = sum(1 for s in samples if s.label == 1)
    n_normal = n_total - n_hypo
    maps = [s.label_value for s in samples]

    print(f"  {name}: {n_total} samples")
    print(f"    Normal:      {n_normal} ({n_normal / n_total * 100:.1f}%)")
    print(f"    Hypotension: {n_hypo} ({n_hypo / n_total * 100:.1f}%)")
    print(
        f"    Future MAP:  [{min(maps):.1f}, {max(maps):.1f}] mmHg, "
        f"mean={np.mean(maps):.1f} +/- {np.std(maps):.1f}"
    )


# ---- 메인 ----


def prepare_hypotension_forecast(
    data_dir: str,
    input_signals: list[str] | None = None,
    max_subjects: int | None = None,
    horizon_min: float = 5.0,
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/hypotension",
) -> Path:
    """Hypotension forecast 데이터를 준비한다.

    Parameters
    ----------
    data_dir : 로컬 .pt 데이터 디렉토리 (vitaldb_pt_test/).
    input_signals : 입력 signal types. None이면 ["abp"].
    max_subjects : 최대 subject 수. None이면 전체.
    horizon_min : prediction horizon (분).
    window_sec : 입력 윈도우 길이 (초).
    stride_sec : 슬라이드 보폭 (초).
    train_ratio : train/test 분할 비율.
    out_dir : 저장 디렉토리.
    """
    if input_signals is None:
        input_signals = ["abp"]

    horizon_sec = horizon_min * 60.0
    # window + horizon + 여유분 (최소 1 stride)
    min_duration_sec = window_sec + horizon_sec + stride_sec
    mode_str = " + ".join(s.upper() for s in input_signals)

    print(f"{'=' * 60}")
    print("  Task 1: Hypotension Forecast")
    print(f"  Data:    {data_dir}")
    print(f"  Input:   {mode_str}")
    print(f"  Horizon: {horizon_min} min")
    print(f"  Window:  {window_sec}s, Stride: {stride_sec}s")
    print("  Label:   MAP<65 sustained >=1min")
    print(f"  Min duration: {min_duration_sec / 60:.1f} min")
    print(f"{'=' * 60}")

    # 1. 데이터 로드
    print("\n[1/4] Loading aligned multi-channel data...")
    cases = _load_local_pt_aligned_signals(
        data_dir,
        input_signals,
        min_duration_sec,
        max_subjects,
    )

    if not cases:
        print("ERROR: No valid cases loaded.", file=sys.stderr)
        sys.exit(1)

    # 2. Train/Test 분할 (patient 단위)
    print(f"\n[2/4] Splitting by patient (ratio={train_ratio})...")
    rng = np.random.default_rng(42)
    patient_ids = list({c["patient_id"] for c in cases})
    rng.shuffle(patient_ids)
    n_train_patients = max(1, int(len(patient_ids) * train_ratio))
    train_patients = set(patient_ids[:n_train_patients])

    train_cases = [c for c in cases if c["patient_id"] in train_patients]
    test_cases = [c for c in cases if c["patient_id"] not in train_patients]
    print(f"  Train: {len(train_cases)} cases ({len(train_patients)} patients)")
    print(
        f"  Test:  {len(test_cases)} cases ({len(patient_ids) - len(train_patients)} patients)"
    )

    # 3. 윈도우 추출 + 라벨링
    print(
        f"\n[3/4] Extracting forecast samples (horizon={horizon_min}min, sustained>=1min)..."
    )
    train_samples = extract_forecast_samples(
        train_cases,
        input_signals,
        window_sec,
        stride_sec,
        horizon_sec,
    )
    test_samples = extract_forecast_samples(
        test_cases,
        input_signals,
        window_sec,
        stride_sec,
        horizon_sec,
    )

    print_stats("Train", train_samples)
    print_stats("Test", test_samples)

    if not train_samples and not test_samples:
        print("ERROR: No samples extracted.", file=sys.stderr)
        sys.exit(1)

    # 4. 저장
    print("\n[4/4] Saving...")
    save_path = save_dataset(
        train_samples,
        test_samples,
        input_signals,
        horizon_sec,
        window_sec,
        out_dir,
    )

    print(f"\n{'=' * 60}")
    print(f"  Done! {save_path}")
    print(f"{'=' * 60}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1: Hypotension Forecast - Data Preparation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Local .pt data directory (e.g. vitaldb_pt_test/)",
    )
    parser.add_argument(
        "--input-signals",
        nargs="+",
        default=["abp"],
        choices=["abp", "ecg", "ppg"],
        help="Input signal types (label always from ABP)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Max number of subjects to load (None=all)",
    )
    parser.add_argument(
        "--horizon-min", type=float, default=5.0, help="Prediction horizon in minutes"
    )
    parser.add_argument(
        "--window-sec", type=float, default=30.0, help="Input window length in seconds"
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=30.0,
        help="Sliding window stride in seconds",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Train/test split ratio"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/downstream/hypotension",
        help="Output directory",
    )
    args = parser.parse_args()

    prepare_hypotension_forecast(
        data_dir=args.data_dir,
        input_signals=args.input_signals,
        max_subjects=args.max_subjects,
        horizon_min=args.horizon_min,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
