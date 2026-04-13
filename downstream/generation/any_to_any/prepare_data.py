# -*- coding:utf-8 -*-
"""Task 8: Any-to-Any Cross-modal — 데이터 준비 스크립트.

다채널 시간 정렬 윈도우를 추출하여 .pt로 저장한다.
run.py에서 오프라인으로 로드하여 평가에 사용.

데이터 소스:
  - VitalDB (수술중 모니터링, 내부 평가)
  - MIMIC-III Waveform (ICU, 외부 평가)

사용법:
    # VitalDB, 기본 4채널 (ecg, abp, ppg, cvp), 10케이스
    python -m downstream.generation.any_to_any.prepare_data \
        --source vitaldb --n-cases 10

    # MIMIC-III, 3채널 (ecg, abp, ppg), 5케이스
    python -m downstream.generation.any_to_any.prepare_data \
        --source mimic3 --signal-types ecg abp ppg --n-cases 5

    # 짧은 윈도우, 좁은 stride
    python -m downstream.generation.any_to_any.prepare_data \
        --source vitaldb --window-sec 10 --stride-sec 5 --n-cases 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


TARGET_SR: float = 100.0


# ---- MIMIC-III 로더 ----


def _load_mimic3_cases(
    n_cases: int,
    signal_types: list[str],
    min_duration_sec: float = 600.0,
    manifest_path: str | None = None,
    out_dir: str = "outputs/downstream/mimic3",
) -> list[dict]:
    """MIMIC-III에서 시간 정렬된 다채널 데이터를 로드한다.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signals": {stype: array}}
    """
    import wfdb
    from data.parser.mimic3_waveform import (
        scan_abp_records,
        load_manifest,
        _apply_pipeline,
        MIMIC3_NATIVE_SR,
    )

    # manifest 로드 또는 스캔
    manifest_file = Path(out_dir) / "mimic3_abp_manifest.json"
    if manifest_path and Path(manifest_path).exists():
        records = load_manifest(manifest_path)
    elif manifest_file.exists():
        records = load_manifest(str(manifest_file))
    else:
        print("  Scanning for ABP records...")
        records = scan_abp_records(
            max_records=n_cases * 5,
            save_path=str(manifest_file),
        )

    # 필요 채널이 모두 있는 레코드 필터링
    filtered = []
    for r in records:
        has_all = True
        if "ecg" in signal_types and not r.has_ecg:
            has_all = False
        if "ppg" in signal_types and not r.has_ppg:
            has_all = False
        if has_all:
            filtered.append(r)

    if not filtered:
        print(f"  WARNING: No records with all required signals {signal_types}")
        return []

    filtered = filtered[:n_cases]
    print(f"  Found {len(filtered)} records with {signal_types}")

    # 각 레코드에서 시간 정렬된 다채널 데이터 추출
    cases = []
    for i, info in enumerate(filtered):
        print(f"  [{i + 1}/{len(filtered)}] {info.record_name}...", end=" ")
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
        ch_map = {}
        if "abp" in signal_types:
            ch_map["abp"] = info.abp_channel
        if "ecg" in signal_types and info.ecg_channel:
            ch_map["ecg"] = info.ecg_channel
        if "ppg" in signal_types and info.ppg_channel:
            ch_map["ppg"] = info.ppg_channel

        if len(ch_map) < len(signal_types):
            print(
                f"SKIP (missing channels: need {signal_types}, have {list(ch_map.keys())})"
            )
            continue

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
                all_present = all(
                    ch_name in seg_hdr.sig_name for ch_name in ch_map.values()
                )
                if all_present and seg_len > best_len:
                    best_seg = seg_name
                    best_len = seg_len
            except Exception:
                continue

        if best_seg is None or best_len < int(min_duration_sec * MIMIC3_NATIVE_SR):
            print(f"SKIP (no aligned segment >= {min_duration_sec / 60:.0f}min)")
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

        if not valid:
            print("SKIP (preprocessing failed)")
            continue

        # 동일 길이로 자르기
        min_len = min(len(s) for s in signals.values())
        signals = {k: v[:min_len] for k, v in signals.items()}

        elapsed = time.time() - t0
        dur_min = min_len / TARGET_SR / 60
        print(f"OK ({dur_min:.1f}min, {len(signals)} ch) [{elapsed:.1f}s]")

        cases.append(
            {
                "case_id": info.record_name,
                "patient_id": info.patient_id,
                "signals": signals,
            }
        )

    return cases


# ---- VitalDB 로더 ----


def _load_vitaldb_cases(
    n_cases: int,
    signal_types: list[str],
    min_duration_sec: float = 600.0,
    offset_from_end: int = 200,
) -> list[dict]:
    """VitalDB에서 시간 정렬된 다채널 데이터를 로드한다.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signals": {stype: array}}
    """
    from downstream.data_utils import load_pilot_cases

    print(f"  Loading {n_cases} VitalDB cases (signals={signal_types})...")
    raw_cases = load_pilot_cases(
        n_cases=n_cases,
        offset_from_end=offset_from_end,
        signal_types=signal_types,
    )

    cases = []
    for rc in raw_cases:
        # 모든 필요 채널이 있는지 확인
        if not all(st in rc.tracks for st in signal_types):
            continue

        # 동일 길이로 자르기
        min_len = min(len(rc.tracks[st]) for st in signal_types)
        if min_len < int(min_duration_sec * TARGET_SR):
            continue

        signals = {st: rc.tracks[st][:min_len] for st in signal_types}
        cases.append(
            {
                "case_id": f"vitaldb_{rc.case_id}",
                "patient_id": str(rc.case_id),
                "signals": signals,
            }
        )

    print(f"  Loaded {len(cases)} cases with all required signals")
    return cases


# ---- 윈도우 추출 ----


def extract_aligned_windows(
    cases: list[dict],
    signal_types: list[str],
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
) -> list[dict]:
    """각 케이스에서 시간 정렬된 다채널 윈도우를 슬라이딩 추출한다.

    Returns
    -------
    list of {"case_id": str, "signals": {stype: array(win_samples,)}}
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)

    windows = []
    for case in cases:
        signals = case["signals"]
        n_total = min(len(signals[st]) for st in signal_types)

        if n_total < win_samples:
            continue

        for start in range(0, n_total - win_samples + 1, stride_samples):
            win = {}
            valid = True
            for st in signal_types:
                seg = signals[st][start : start + win_samples]
                # NaN 체크
                if np.isnan(seg).any():
                    valid = False
                    break
                win[st] = seg

            if valid:
                windows.append(
                    {
                        "case_id": case["case_id"],
                        "patient_id": case["patient_id"],
                        "signals": win,
                    }
                )

    return windows


# ---- 저장 ----


def save_dataset(
    train_windows: list[dict],
    test_windows: list[dict],
    signal_types: list[str],
    window_sec: float,
    source: str,
    out_dir: str,
) -> Path:
    """윈도우 리스트를 .pt로 저장한다."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(windows: list[dict]) -> dict:
        if not windows:
            return {"signals": {}, "case_ids": []}

        sig_tensors = {}
        for st in signal_types:
            arrs = [w["signals"][st] for w in windows]
            sig_tensors[st] = torch.stack([torch.from_numpy(a).float() for a in arrs])
            # (N, win_samples)

        case_ids = [w["case_id"] for w in windows]
        return {"signals": sig_tensors, "case_ids": case_ids}

    save_dict = {
        "train": _to_tensors(train_windows),
        "test": _to_tensors(test_windows),
        "metadata": {
            "task": "any_to_any_cross_modal",
            "source": source,
            "signal_types": signal_types,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "n_train": len(train_windows),
            "n_test": len(test_windows),
        },
    }

    types_str = "_".join(signal_types)
    win_str = int(window_sec)
    filename = f"task8_any_to_any_{source}_{types_str}_w{win_str}s.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ---- 통계 출력 ----


def print_stats(
    name: str,
    windows: list[dict],
    signal_types: list[str],
) -> None:
    """윈도우 통계 출력."""
    if not windows:
        print(f"  {name}: 0 windows")
        return

    n = len(windows)
    case_ids = set(w["case_id"] for w in windows)
    print(f"  {name}: {n} windows from {len(case_ids)} cases")

    for st in signal_types:
        vals = np.concatenate([w["signals"][st] for w in windows])
        print(
            f"    {st:5s}: range=[{vals.min():.2f}, {vals.max():.2f}], "
            f"mean={vals.mean():.2f}, std={vals.std():.2f}"
        )


# ---- 메인 ----


def prepare_any_to_any(
    source: str = "vitaldb",
    signal_types: list[str] | None = None,
    n_cases: int = 10,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/any_to_any",
    manifest_path: str | None = None,
) -> Path:
    """Any-to-Any cross-modal 평가 데이터를 준비한다.

    Parameters
    ----------
    source : "vitaldb" 또는 "mimic3".
    signal_types : 추출할 signal type 목록. None이면 ["ecg", "abp", "ppg", "cvp"].
    n_cases : 로드할 케이스 수.
    window_sec : 윈도우 길이 (초).
    stride_sec : 슬라이드 보폭 (초).
    train_ratio : patient 단위 train/test 분할 비율.
    out_dir : 저장 디렉토리.
    manifest_path : MIMIC-III manifest 경로.
    """
    if signal_types is None:
        signal_types = ["ecg", "abp", "ppg", "cvp"]

    min_duration_sec = window_sec + stride_sec  # 최소 1 윈도우 + 1 stride
    types_str = " + ".join(s.upper() for s in signal_types)

    print(f"{'=' * 60}")
    print("  Task 8: Any-to-Any Cross-modal — Data Preparation")
    print(f"  Source:  {source}")
    print(f"  Signals: {types_str}")
    print(f"  Window:  {window_sec}s, Stride: {stride_sec}s")
    print(f"{'=' * 60}")

    # 1. 데이터 로드
    print("\n[1/4] Loading aligned multi-channel data...")
    if source == "mimic3":
        cases = _load_mimic3_cases(
            n_cases,
            signal_types,
            min_duration_sec,
            manifest_path,
            out_dir=str(Path(out_dir).parent / "mimic3"),
        )
    elif source == "vitaldb":
        cases = _load_vitaldb_cases(
            n_cases,
            signal_types,
            min_duration_sec,
        )
    else:
        print(f"ERROR: Unknown source '{source}'", file=sys.stderr)
        sys.exit(1)

    if not cases:
        print("ERROR: No valid cases loaded.", file=sys.stderr)
        sys.exit(1)

    # 2. Patient 단위 train/test 분할
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

    # 3. 윈도우 추출
    print(
        f"\n[3/4] Extracting aligned windows (window={window_sec}s, stride={stride_sec}s)..."
    )
    train_windows = extract_aligned_windows(
        train_cases, signal_types, window_sec, stride_sec
    )
    test_windows = extract_aligned_windows(
        test_cases, signal_types, window_sec, stride_sec
    )

    print_stats("Train", train_windows, signal_types)
    print_stats("Test", test_windows, signal_types)

    if not train_windows and not test_windows:
        print("ERROR: No windows extracted.", file=sys.stderr)
        sys.exit(1)

    # 4. 저장
    print("\n[4/4] Saving...")
    save_path = save_dataset(
        train_windows,
        test_windows,
        signal_types,
        window_sec,
        source,
        out_dir,
    )

    print(f"\n{'=' * 60}")
    print(f"  Done! {save_path}")
    print(f"  Train: {len(train_windows)} windows, Test: {len(test_windows)} windows")
    print(f"{'=' * 60}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 8: Any-to-Any Cross-modal — Data Preparation",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="vitaldb",
        choices=["mimic3", "vitaldb"],
        help="Data source",
    )
    parser.add_argument(
        "--signal-types",
        nargs="+",
        default=["ecg", "abp", "ppg", "cvp"],
        choices=["ecg", "abp", "ppg", "cvp", "co2", "awp", "pap", "icp"],
        help="Signal types to extract",
    )
    parser.add_argument(
        "--n-cases", type=int, default=10, help="Number of cases to load"
    )
    parser.add_argument(
        "--window-sec", type=float, default=30.0, help="Window length in seconds"
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=15.0,
        help="Sliding window stride in seconds",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train/test split ratio (patient-level)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/downstream/any_to_any",
        help="Output directory",
    )
    parser.add_argument(
        "--manifest", type=str, default=None, help="MIMIC-III manifest JSON path"
    )
    args = parser.parse_args()

    prepare_any_to_any(
        source=args.source,
        signal_types=args.signal_types,
        n_cases=args.n_cases,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
