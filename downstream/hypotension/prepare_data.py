# -*- coding:utf-8 -*-
"""Task 1: Hypotension Prediction - 데이터 준비 스크립트.

미래 5~15분 후 MAP<65 예측을 위한 (input_window, future_label) 쌍 생성.
4가지 입력 모드: abp, ecg, ppg, ecg_ppg

Label 소스: 항상 ABP (미래 구간의 MAP)
Input 소스: 선택된 signal type의 현재 윈도우

데이터 소스:
  - MIMIC-III Waveform (외부 평가용)
  - VitalDB (내부 평가용)

사용법:
    # MIMIC-III, ECG 입력, 5분 후 예측
    python -m downstream.hypotension.prepare_data \
        --source mimic3 --input-signals ecg --horizon-min 5 --n-cases 5

    # VitalDB, ECG+PPG 입력, 10분 후 예측
    python -m downstream.hypotension.prepare_data \
        --source vitaldb --input-signals ecg ppg --horizon-min 10 --n-cases 10

    # MIMIC-III, ABP 입력 (temporal prediction)
    python -m downstream.hypotension.prepare_data \
        --source mimic3 --input-signals abp --horizon-min 5 --n-cases 5
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

TARGET_SR: float = 100.0


# ---- 데이터 구조 ----


@dataclass
class ForecastSample:
    """미래 저혈압 예측 샘플."""
    input_signals: dict[str, np.ndarray]  # {"ecg": (win_samples,), ...}
    label: int                             # 0=normal, 1=hypotension in future
    label_value: float                     # future MAP (mmHg)
    case_id: str
    win_start_sec: float                   # input window 시작 (초)
    horizon_sec: float                     # prediction horizon (초)


# ---- MIMIC-III 로더 ----


def _load_mimic3_aligned_signals(
    n_cases: int,
    input_signals: list[str],
    min_duration_sec: float = 1200.0,
    manifest_path: str | None = None,
    out_dir: str = "outputs/downstream/mimic3",
) -> list[dict]:
    """MIMIC-III에서 시간 정렬된 다채널 데이터를 로드한다.

    Parameters
    ----------
    min_duration_sec : 최소 유효 신호 길이 (초). window + horizon 이상이어야 함.

    Returns: list of {"case_id": str, "signals": {"abp": array, "ecg": array, ...}}
    """
    import wfdb
    from data.parser.mimic3_waveform import (
        PN_DB, ABP_CHANNEL_NAMES, ECG_CHANNEL_NAMES, PPG_CHANNEL_NAMES,
        scan_abp_records, load_manifest, _apply_pipeline,
        MIMIC3_NATIVE_SR,
    )

    # 필요한 채널 결정 (ABP는 라벨용으로 항상 필요)
    required_types = set(input_signals) | {"abp"}

    # manifest 로드 또는 스캔
    manifest_file = Path(out_dir) / "mimic3_abp_manifest.json"
    if manifest_path and Path(manifest_path).exists():
        records = load_manifest(manifest_path)
    elif manifest_file.exists():
        records = load_manifest(str(manifest_file))
    else:
        print(f"  Scanning for ABP records...")
        records = scan_abp_records(
            max_records=n_cases * 5,
            save_path=str(manifest_file),
        )

    # 필요 채널이 모두 있는 레코드 필터링
    filtered = []
    for r in records:
        has_all = True
        if "ecg" in required_types and not r.has_ecg:
            has_all = False
        if "ppg" in required_types and not r.has_ppg:
            has_all = False
        if has_all:
            filtered.append(r)

    if not filtered:
        print(f"  WARNING: No records with all required signals {required_types}")
        print(f"  Available: {len(records)} ABP records, "
              f"{sum(1 for r in records if r.has_ecg)} with ECG, "
              f"{sum(1 for r in records if r.has_ppg)} with PPG")
        return []

    filtered = filtered[:n_cases]
    print(f"  Found {len(filtered)} records with {required_types}")

    # 각 레코드에서 시간 정렬된 다채널 데이터 추출
    cases = []
    for i, info in enumerate(filtered):
        print(f"  [{i+1}/{len(filtered)}] {info.record_name}...", end=" ")
        t0 = time.time()

        try:
            hdr = wfdb.rdheader(info.record_name, pn_dir=info.pn_dir)
        except Exception as e:
            print(f"SKIP (header error: {e})")
            continue

        if not hasattr(hdr, "seg_name") or not hdr.seg_name:
            print("SKIP (no segments)")
            continue

        # 채널 매핑
        ch_map = {"abp": info.abp_channel}
        if "ecg" in required_types and info.ecg_channel:
            ch_map["ecg"] = info.ecg_channel
        if "ppg" in required_types and info.ppg_channel:
            ch_map["ppg"] = info.ppg_channel

        # 모든 채널이 동시에 존재하는 가장 긴 세그먼트 찾기
        best_seg = None
        best_len = 0

        for seg_name, seg_len in zip(hdr.seg_name, hdr.seg_len):
            if seg_name == "~" or seg_name.endswith("_layout") or seg_len <= 0:
                continue

            try:
                seg_hdr = wfdb.rdheader(seg_name, pn_dir=info.pn_dir)
                if seg_hdr.sig_name is None:
                    continue

                # 모든 필요 채널이 이 세그먼트에 있는지 확인
                all_present = all(
                    ch_name in seg_hdr.sig_name
                    for ch_name in ch_map.values()
                )
                if all_present and seg_len > best_len:
                    best_seg = seg_name
                    best_len = seg_len
            except Exception:
                continue

        if best_seg is None or best_len < int(min_duration_sec * MIMIC3_NATIVE_SR):
            # 최소 20분 이상 필요 (input + horizon)
            print(f"SKIP (no aligned segment >= {min_duration_sec/60:.0f}min)")
            continue

        # 가장 긴 정렬 세그먼트 로드
        try:
            seg = wfdb.rdrecord(best_seg, pn_dir=info.pn_dir)
        except Exception as e:
            print(f"SKIP (read error: {e})")
            continue

        if seg.p_signal is None:
            print("SKIP (no signal)")
            continue

        # 각 채널 추출 + 전처리
        signals = {}
        valid = True
        for stype, ch_name in ch_map.items():
            if ch_name not in seg.sig_name:
                valid = False
                break
            ch_idx = seg.sig_name.index(ch_name)
            raw = seg.p_signal[:, ch_idx].astype(np.float64)

            processed = _apply_pipeline(raw, stype, MIMIC3_NATIVE_SR)
            if processed is None or len(processed) < int(min_duration_sec * TARGET_SR):
                valid = False
                break
            signals[stype] = processed

        if not valid or "abp" not in signals:
            print(f"SKIP (preprocessing failed)")
            continue

        # 모든 채널을 동일 길이로 자르기 (전처리 후 길이가 약간 다를 수 있음)
        min_len = min(len(s) for s in signals.values())
        signals = {k: v[:min_len] for k, v in signals.items()}

        elapsed = time.time() - t0
        dur_min = min_len / TARGET_SR / 60
        print(f"OK ({dur_min:.1f}min, {len(signals)} channels) [{elapsed:.1f}s]")

        cases.append({
            "case_id": info.record_name,
            "patient_id": info.patient_id,
            "signals": signals,
        })

    return cases


# ---- VitalDB 로더 ----


def _load_vitaldb_aligned_signals(
    n_cases: int,
    input_signals: list[str],
    min_duration_sec: float = 1200.0,
    offset_from_end: int = 200,
) -> list[dict]:
    """VitalDB에서 시간 정렬된 다채널 데이터를 로드한다."""
    from downstream.data_utils import (
        load_pilot_cases, PREFERRED_TRACKS,
    )

    required_types = list(set(input_signals) | {"abp"})
    print(f"  Loading {n_cases} VitalDB cases (signals={required_types})...")

    raw_cases = load_pilot_cases(
        n_cases=n_cases,
        offset_from_end=offset_from_end,
        signal_types=required_types,
    )

    cases = []
    for rc in raw_cases:
        # 모든 필요 채널이 있는지 확인
        if not all(st in rc.tracks for st in required_types):
            continue

        # 동일 길이로 자르기
        min_len = min(len(rc.tracks[st]) for st in required_types)
        if min_len < int(min_duration_sec * TARGET_SR):  # 최소 20분
            continue

        signals = {st: rc.tracks[st][:min_len] for st in required_types}
        cases.append({
            "case_id": f"vitaldb_{rc.case_id}",
            "patient_id": str(rc.case_id),
            "signals": signals,
        })

    print(f"  Loaded {len(cases)} cases with all required signals")
    return cases


# ---- 윈도우 추출 + 라벨링 ----


def extract_forecast_samples(
    cases: list[dict],
    input_signals: list[str],
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    horizon_sec: float = 300.0,
    map_threshold: float = 65.0,
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

    Returns
    -------
    ForecastSample 리스트.
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    horizon_samples = int(horizon_sec * TARGET_SR)

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
                    input_dict[stype] = signals[stype][start:start + win_samples]

            if not input_dict:
                continue

            # Future label: ABP의 [start + win_samples, start + win_samples + horizon_samples) 구간
            future_start = start + win_samples
            future_end = future_start + horizon_samples
            future_abp = abp[future_start:future_end]

            # 미래 구간의 MAP (10초 윈도우별 평균의 최솟값)
            map_win = int(10 * TARGET_SR)
            future_maps = []
            for j in range(0, len(future_abp) - map_win + 1, map_win):
                w = future_abp[j:j + map_win]
                if not np.isnan(w).any():
                    future_maps.append(float(np.mean(w)))

            if not future_maps:
                continue

            # 미래 구간에서 MAP<65가 한 번이라도 발생하면 positive
            min_future_map = min(future_maps)
            label = 1 if min_future_map < map_threshold else 0

            samples.append(ForecastSample(
                input_signals=input_dict,
                label=label,
                label_value=min_future_map,
                case_id=case["case_id"],
                win_start_sec=start / TARGET_SR,
                horizon_sec=horizon_sec,
            ))

    return samples


# ---- 저장 ----


def save_dataset(
    train_samples: list[ForecastSample],
    test_samples: list[ForecastSample],
    input_signals: list[str],
    horizon_sec: float,
    window_sec: float,
    source: str,
    out_dir: str,
) -> Path:
    """ForecastSample 리스트를 .pt로 저장한다."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples: list[ForecastSample]) -> dict:
        if not samples:
            return {"signals": {}, "labels": torch.tensor([]), "label_values": torch.tensor([])}

        # 각 signal type별 텐서 생성
        sig_tensors = {}
        for stype in input_signals:
            arrs = [s.input_signals[stype] for s in samples if stype in s.input_signals]
            if arrs:
                sig_tensors[stype] = torch.stack([torch.from_numpy(a).float() for a in arrs])

        labels = torch.tensor([s.label for s in samples], dtype=torch.long)
        label_values = torch.tensor([s.label_value for s in samples], dtype=torch.float32)
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
            "source": source,
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "map_threshold": 65.0,
            "n_train": len(train_samples),
            "n_test": len(test_samples),
        },
    }

    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    filename = f"task1_hypotension_{source}_{mode_str}_h{horizon_min}min.pt"
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
    print(f"    Future MAP:  [{min(maps):.1f}, {max(maps):.1f}] mmHg, "
          f"mean={np.mean(maps):.1f} +/- {np.std(maps):.1f}")


# ---- 메인 ----


def prepare_hypotension_forecast(
    source: str = "mimic3",
    input_signals: list[str] | None = None,
    n_cases: int = 5,
    horizon_min: float = 5.0,
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/hypotension",
    manifest_path: str | None = None,
) -> Path:
    """Hypotension forecast 데이터를 준비한다.

    Parameters
    ----------
    source : "mimic3" 또는 "vitaldb".
    input_signals : 입력 signal types. None이면 ["abp"].
    n_cases : 로드할 케이스 수.
    horizon_min : prediction horizon (분).
    window_sec : 입력 윈도우 길이 (초).
    stride_sec : 슬라이드 보폭 (초).
    train_ratio : train/test 분할 비율.
    out_dir : 저장 디렉토리.
    manifest_path : MIMIC-III manifest 경로.
    """
    if input_signals is None:
        input_signals = ["abp"]

    horizon_sec = horizon_min * 60.0
    # window + horizon + 여유분 (최소 1 stride)
    min_duration_sec = window_sec + horizon_sec + stride_sec
    mode_str = " + ".join(s.upper() for s in input_signals)

    print(f"{'='*60}")
    print(f"  Task 1: Hypotension Forecast")
    print(f"  Source:  {source}")
    print(f"  Input:   {mode_str}")
    print(f"  Horizon: {horizon_min} min")
    print(f"  Window:  {window_sec}s, Stride: {stride_sec}s")
    print(f"  Min duration: {min_duration_sec/60:.1f} min")
    print(f"{'='*60}")

    # 1. 데이터 로드
    print(f"\n[1/4] Loading aligned multi-channel data...")
    if source == "mimic3":
        cases = _load_mimic3_aligned_signals(
            n_cases, input_signals, min_duration_sec, manifest_path,
            out_dir=str(Path(out_dir).parent / "mimic3"),
        )
    elif source == "vitaldb":
        cases = _load_vitaldb_aligned_signals(
            n_cases, input_signals, min_duration_sec,
        )
    else:
        print(f"ERROR: Unknown source '{source}'", file=sys.stderr)
        sys.exit(1)

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
    print(f"  Test:  {len(test_cases)} cases ({len(patient_ids) - len(train_patients)} patients)")

    # 3. 윈도우 추출 + 라벨링
    print(f"\n[3/4] Extracting forecast samples (horizon={horizon_min}min)...")
    train_samples = extract_forecast_samples(
        train_cases, input_signals, window_sec, stride_sec, horizon_sec,
    )
    test_samples = extract_forecast_samples(
        test_cases, input_signals, window_sec, stride_sec, horizon_sec,
    )

    print_stats("Train", train_samples)
    print_stats("Test", test_samples)

    if not train_samples and not test_samples:
        print("ERROR: No samples extracted.", file=sys.stderr)
        sys.exit(1)

    # 4. 저장
    print(f"\n[4/4] Saving...")
    save_path = save_dataset(
        train_samples, test_samples,
        input_signals, horizon_sec, window_sec, source, out_dir,
    )

    print(f"\n{'='*60}")
    print(f"  Done! {save_path}")
    print(f"{'='*60}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1: Hypotension Forecast - Data Preparation",
    )
    parser.add_argument("--source", type=str, default="mimic3",
                        choices=["mimic3", "vitaldb"],
                        help="Data source")
    parser.add_argument("--input-signals", nargs="+", default=["abp"],
                        choices=["abp", "ecg", "ppg"],
                        help="Input signal types (label always from ABP)")
    parser.add_argument("--n-cases", type=int, default=5,
                        help="Number of cases to load")
    parser.add_argument("--horizon-min", type=float, default=5.0,
                        help="Prediction horizon in minutes")
    parser.add_argument("--window-sec", type=float, default=30.0,
                        help="Input window length in seconds")
    parser.add_argument("--stride-sec", type=float, default=30.0,
                        help="Sliding window stride in seconds")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Train/test split ratio")
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/hypotension",
                        help="Output directory")
    parser.add_argument("--manifest", type=str, default=None,
                        help="MIMIC-III manifest JSON path")
    args = parser.parse_args()

    prepare_hypotension_forecast(
        source=args.source,
        input_signals=args.input_signals,
        n_cases=args.n_cases,
        horizon_min=args.horizon_min,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()