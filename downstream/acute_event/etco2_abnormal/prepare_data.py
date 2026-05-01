# -*- coding:utf-8 -*-
"""Intraoperative EtCO2 Abnormal Prediction — 데이터 준비 (VitalDB primary).

미래 5~15분 후 EtCO2 abnormal (<35 또는 >45 mmHg, ≥1분 지속) 예측.
Hypocapnia (저탄산혈증) + Hypercapnia (고탄산혈증) 통합 binary label.

Label 소스: EtCO2 trend (raw .vital → ETCO2 1Hz numeric)
Input 소스: parsed .pt 디렉토리의 wave (ECG/ABP/PPG/CO2)

사용법:
    python -m downstream.acute_event.etco2_abnormal.prepare_data \\
        --data-dir <parsed .pt 디렉토리> \\
        --raw-dir <raw vitaldb .vital 디렉토리> \\
        --input-signals co2 ppg \\
        --required-signals ecg ppg abp co2 \\
        --window-secs 60 180 300 600 --horizon-mins 5 10 15 \\
        --out-dir outputs/downstream/etco2_abnormal
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
ETCO2_TRACKS = [
    "Solar8000/ETCO2",         # VitalDB OR 표준 EtCO2 (1Hz numeric)
    "Intellivue/AWAY_CO2_ET",  # K-MIMIC ICU EtCO2
    "Primus/ETCO2",            # 일부 Primus
]
ETCO2_NORMAL_LOW = 35.0   # mmHg, hypocapnia threshold
ETCO2_NORMAL_HIGH = 45.0  # mmHg, hypercapnia threshold


# ---- 데이터 구조 ----


@dataclass
class ForecastSample:
    """미래 EtCO2 abnormal 예측 샘플."""

    input_signals: dict[str, np.ndarray]
    label: int  # 0=normal, 1=abnormal (hypo or hyper) in future
    label_value: float  # future EtCO2 abnormal 정도 (가장 abnormal한 값과 normal 간 거리)
    case_id: str
    win_start_sec: float
    horizon_sec: float


# ---- 로컬 .pt 로더 (hypotension 과 동일) ----


def _parse_pt_filename(name: str) -> dict | None:
    m = re.match(
        r"^(.+?)_S(\d+)_([a-z0-9]+)_(\d+)_seg(\d+)_(\d+)\.pt$",
        name,
    )
    if m is None:
        return None
    return {
        "subject_id": m.group(1),
        "session_id": int(m.group(2)),
        "signal_type": m.group(3),
        "spatial_id": int(m.group(4)),
        "seg_i": int(m.group(5)),
        "seg_j": int(m.group(6)),
    }


def _load_local_pt_aligned_signals(
    data_dir: str,
    input_signals: list[str],
    min_duration_sec: float,
    max_subjects: int | None,
    required_signals: list[str] | None,
) -> list[dict]:
    """parsed .pt 디렉토리에서 시간 정렬된 다채널 segment 로드 (hypotension 동일)."""
    root = Path(data_dir)
    if not root.is_dir():
        print(f"  ERROR: Data directory not found: {root}", file=sys.stderr)
        return []

    if required_signals is not None:
        required_types = set(required_signals) | set(input_signals)
    else:
        required_types = set(input_signals)

    subject_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    print(f"  Scanning {len(subject_dirs)} subjects in {root}...")
    cases: list[dict] = []

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name

        file_map: dict[tuple[int, int, int], dict[str, Path]] = {}
        for pt_file in subj_dir.glob("*.pt"):
            meta = _parse_pt_filename(pt_file.name)
            if meta is None:
                continue
            seg_key = (meta["session_id"], meta["seg_i"], meta["seg_j"])
            file_map.setdefault(seg_key, {})[meta["signal_type"]] = pt_file

        for seg_key, type_paths in file_map.items():
            if not required_types.issubset(set(type_paths.keys())):
                continue

            signals: dict[str, np.ndarray] = {}
            for stype_str in required_types:
                t = torch.load(type_paths[stype_str], weights_only=True)
                signals[stype_str] = t.squeeze(0).numpy()

            min_len = min(len(s) for s in signals.values())
            if min_len < int(min_duration_sec * TARGET_SR):
                continue

            signals = {k: v[:min_len] for k, v in signals.items()}
            session_id, seg_i, seg_j = seg_key

            cases.append({
                "case_id": f"{subject_id}_s{session_id}_seg{seg_i}_{seg_j}",
                "patient_id": subject_id,
                "session_id": session_id,
                "seg_i": seg_i,
                "seg_j": seg_j,
                "signals": signals,
            })

    print(f"  Loaded {len(cases)} aligned segments with {required_types}")
    return cases


# ---- SpO2 trend 추출 (raw .vital 사용) ----


def _load_etco2_trend(
    raw_vital_path: Path,
    target_duration_sec: float,
) -> np.ndarray | None:
    """raw .vital 에서 EtCO2 1Hz trend 추출."""
    try:
        import vitaldb
    except ImportError:
        print("  ERROR: vitaldb library required. pip install vitaldb", file=sys.stderr)
        return None

    if not raw_vital_path.is_file():
        return None
    try:
        vf = vitaldb.VitalFile(str(raw_vital_path))
        avail = vf.get_track_names()
    except Exception:
        return None

    for track in ETCO2_TRACKS:
        if track in avail:
            try:
                arr = vf.to_numpy([track], 1.0)
                if arr is not None and arr.size > 0:
                    return arr.flatten().astype(np.float32)
            except Exception:
                continue
    return None


def _resolve_raw_vital_path(raw_dir: Path, subject_id: str) -> Path | None:
    """parsed subject_id 에서 raw .vital 파일 경로 추론.

    VitalDB OR: VDB_0001 → 0001.vital 또는 vitaldb_open/0001.vital
    """
    digits = "".join(c for c in subject_id if c.isdigit())
    if not digits:
        return None

    # 다양한 naming convention 시도
    candidates = [
        raw_dir / f"{int(digits):04d}.vital",
        raw_dir / f"{int(digits)}.vital",
        raw_dir / f"{subject_id}.vital",
    ]
    # 1.0.0 같은 nested
    for sub in raw_dir.glob("**/*.vital"):
        # filename digit match
        stem_digits = "".join(c for c in sub.stem if c.isdigit())
        if stem_digits and int(stem_digits) == int(digits):
            return sub

    for c in candidates:
        if c.is_file():
            return c
    return None


# ---- 라벨링 ----


def _has_sustained_etco2_abnormal(
    future_etco2: list[float],
    low_thr: float,
    high_thr: float,
    min_consecutive: int,
) -> bool:
    """연속 min_consecutive 개 EtCO2 < low 또는 > high 면 positive."""
    consecutive = 0
    for v in future_etco2:
        if v < low_thr or v > high_thr:
            consecutive += 1
            if consecutive >= min_consecutive:
                return True
        else:
            consecutive = 0
    return False


def extract_forecast_samples(
    cases: list[dict],
    etco2_map: dict[str, np.ndarray],
    input_signals: list[str],
    window_sec: float,
    stride_sec: float,
    horizon_sec: float,
    low_thr: float = ETCO2_NORMAL_LOW,
    high_thr: float = ETCO2_NORMAL_HIGH,
    sustained_sec: float = 60.0,
) -> list[ForecastSample]:
    """SpO2 trend + waveform window 정렬해서 (input, future_label) 쌍 추출.

    SpO2 trend 는 1Hz라 sliding window 의 future 시작 ~ +horizon 구간을
    초 단위로 인덱싱.
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    horizon_samples_target = int(horizon_sec * TARGET_SR)
    total_needed = win_samples + horizon_samples_target

    min_consecutive = max(1, int(sustained_sec))  # 1Hz 라 1초당 1 sample
    samples: list[ForecastSample] = []

    for case in cases:
        signals = case["signals"]
        n_total = min(len(s) for s in signals.values())
        if n_total < total_needed:
            continue

        etco2 = etco2_map.get(case["patient_id"])
        if etco2 is None or len(etco2) == 0:
            continue

        # spo2 (1Hz) 와 wave (100Hz) 의 시간 alignment 가정:
        # parsed segment 가 raw 의 [0..len_sec] 같다고 가정 — 실제로는
        # seg_start_sample 정보가 있어야 정확. 단순화: case-level 로 SpO2 전체 사용
        # (sweep prepare 단계라 정확도 trade-off 수용)

        for start in range(0, n_total - total_needed + 1, stride_samples):
            input_dict = {}
            for stype in input_signals:
                if stype in signals:
                    input_dict[stype] = signals[stype][start : start + win_samples]
            if not input_dict:
                continue

            future_start_sec = (start + win_samples) / TARGET_SR
            future_end_sec = future_start_sec + horizon_sec

            f_start = int(future_start_sec)
            f_end = int(future_end_sec)
            if f_end > len(etco2):
                continue
            future_etco2 = etco2[f_start:f_end]
            future_etco2 = future_etco2[~np.isnan(future_etco2)]
            # EtCO2 가용 범위: 0~80 mmHg 외는 artifact
            future_etco2 = future_etco2[(future_etco2 >= 0.0) & (future_etco2 <= 80.0)]

            if len(future_etco2) < max(1, min_consecutive // 2):
                continue

            label = (
                1
                if _has_sustained_etco2_abnormal(
                    future_etco2.tolist(), low_thr, high_thr, min_consecutive,
                )
                else 0
            )
            # label_value: 가장 abnormal 한 값 (normal range 에서 가장 멀리)
            mid = (low_thr + high_thr) / 2
            dev = np.abs(future_etco2 - mid).max()
            most_abn = future_etco2[np.argmax(np.abs(future_etco2 - mid))]

            samples.append(
                ForecastSample(
                    input_signals=input_dict,
                    label=label,
                    label_value=float(most_abn),
                    case_id=case["case_id"],
                    win_start_sec=start / TARGET_SR,
                    horizon_sec=horizon_sec,
                )
            )

    return samples


# ---- 저장 ----


def save_dataset(
    train_samples: list[ForecastSample],
    val_samples: list[ForecastSample],
    test_samples: list[ForecastSample],
    input_signals: list[str],
    horizon_sec: float,
    window_sec: float,
    out_dir: str,
) -> Path:
    """train/val/test 3-way 저장 (hypotension/aki 와 동일 schema)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples: list[ForecastSample]) -> dict:
        if not samples:
            return {
                "signals": {},
                "labels": torch.tensor([]),
                "label_values": torch.tensor([]),
                "case_ids": [],
            }

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
        "val": _to_tensors(val_samples),
        "test": _to_tensors(test_samples),
        "metadata": {
            "task": "etco2_abnormal_forecast",
            "source": "vitaldb_pt + raw_vital",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "etco2_normal_low": ETCO2_NORMAL_LOW,
            "etco2_normal_high": ETCO2_NORMAL_HIGH,
            "sustained_sec": 60.0,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "n_test": len(test_samples),
        },
    }

    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    win_int = int(window_sec)
    filename = f"task_etco2abn_{mode_str}_w{win_int}s_h{horizon_min}min.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


def print_stats(name: str, samples: list[ForecastSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return
    n = len(samples)
    n_pos = sum(1 for s in samples if s.label == 1)
    vals = [s.label_value for s in samples]
    print(f"  {name}: {n} samples")
    print(f"    Normal:   {n - n_pos} ({(n - n_pos) / n * 100:.1f}%)")
    print(f"    Abnormal: {n_pos} ({n_pos / n * 100:.1f}%)")
    print(
        f"    Future EtCO2 most-abn: [{min(vals):.1f}, {max(vals):.1f}] mmHg, "
        f"mean={np.mean(vals):.1f}"
    )


# ---- 메인 ----


def prepare_etco2_sweep(
    data_dir: str,
    raw_dir: str,
    input_signals: list[str],
    window_secs: list[float],
    horizon_mins: list[float],
    stride_sec: float = 30.0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    max_subjects: int | None = None,
    out_dir: str = "outputs/downstream/etco2_abnormal",
    required_signals: list[str] | None = None,
) -> list[Path]:
    max_window = max(window_secs)
    max_horizon_sec = max(horizon_mins) * 60.0
    min_duration_sec = max_window + max_horizon_sec + stride_sec

    mode_str = " + ".join(s.upper() for s in input_signals)
    req_str = " + ".join(s.upper() for s in required_signals) if required_signals else "auto"
    print(f"\n{'=' * 60}")
    print("  Intraop EtCO2 Abnormal Forecast — Sweep")
    print(f"  Parsed:    {data_dir}")
    print(f"  Raw vital: {raw_dir}")
    print(f"  Input:     {mode_str}")
    print(f"  Required:  {req_str}")
    print(f"  Windows:   {window_secs}")
    print(f"  Horizons:  {horizon_mins}")
    print(f"  Threshold: <{ETCO2_NORMAL_LOW} or >{ETCO2_NORMAL_HIGH} mmHg sustained ≥1min")
    print(f"  Min dur:   {min_duration_sec / 60:.1f} min")
    print(f"{'=' * 60}")

    print("\n[1/4] Loading aligned multi-channel waveform...")
    cases = _load_local_pt_aligned_signals(
        data_dir, input_signals, min_duration_sec, max_subjects,
        required_signals=required_signals,
    )
    if not cases:
        print("ERROR: No valid cases loaded.", file=sys.stderr)
        sys.exit(1)

    print("\n[2/4] Loading EtCO2 trends from raw .vital...")
    raw_root = Path(raw_dir)
    etco2_map: dict[str, np.ndarray] = {}
    unique_pids = sorted({c["patient_id"] for c in cases})
    n_found = 0
    for i, pid in enumerate(unique_pids):
        rv = _resolve_raw_vital_path(raw_root, pid)
        if rv is None:
            continue
        etco2 = _load_etco2_trend(rv, 0)
        if etco2 is not None:
            etco2_map[pid] = etco2
            n_found += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(unique_pids)} processed, EtCO2 found: {n_found}")
    print(f"  EtCO2 trend 보유 환자: {n_found} / {len(unique_pids)}")
    if n_found == 0:
        print("ERROR: No EtCO2 trend found. raw_dir 또는 ETCO2_TRACKS 확인.", file=sys.stderr)
        sys.exit(1)
    cases = [c for c in cases if c["patient_id"] in etco2_map]
    print(f"  EtCO2 가용 case: {len(cases)}")

    print("\n[3/4] Patient-level train/val/test split...")
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0.0:
        raise ValueError(f"train_ratio + val_ratio must be < 1")
    rng = np.random.default_rng(42)
    pids = list({c["patient_id"] for c in cases})
    rng.shuffle(pids)
    n_total = len(pids)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    train_pids = set(pids[:n_train])
    val_pids = set(pids[n_train : n_train + n_val])
    test_pids = set(pids[n_train + n_val :])
    print(f"  Train: {len(train_pids)} / Val: {len(val_pids)} / Test: {len(test_pids)} patients")

    train_cases = [c for c in cases if c["patient_id"] in train_pids]
    val_cases = [c for c in cases if c["patient_id"] in val_pids]
    test_cases = [c for c in cases if c["patient_id"] in test_pids]

    combos = [(w, h) for w in window_secs for h in horizon_mins]
    print(f"\n[4/4] Generating {len(combos)} datasets...")

    saved_paths: list[Path] = []
    for i, (window_sec, horizon_min) in enumerate(combos, 1):
        horizon_sec = horizon_min * 60.0
        print(f"\n  [{i}/{len(combos)}] window={window_sec}s, horizon={horizon_min}min")

        train_s = extract_forecast_samples(
            train_cases, etco2_map, input_signals, window_sec, stride_sec, horizon_sec,
        )
        val_s = extract_forecast_samples(
            val_cases, etco2_map, input_signals, window_sec, stride_sec, horizon_sec,
        )
        test_s = extract_forecast_samples(
            test_cases, etco2_map, input_signals, window_sec, stride_sec, horizon_sec,
        )

        print_stats("    Train", train_s)
        print_stats("    Val", val_s)
        print_stats("    Test", test_s)

        if not (train_s or val_s or test_s):
            print("    SKIP: No samples.")
            continue

        save_path = save_dataset(
            train_s, val_s, test_s, input_signals,
            horizon_sec, window_sec, out_dir,
        )
        saved_paths.append(save_path)

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(saved_paths)}/{len(combos)} datasets saved to {out_dir}")
    print(f"{'=' * 60}")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraop EtCO2 Abnormal Forecast — Data Prep")
    parser.add_argument("--data-dir", required=True, help="parsed .pt 디렉토리")
    parser.add_argument("--raw-dir", required=True, help="raw vitaldb .vital 디렉토리 (EtCO2 라벨 추출용)")
    parser.add_argument("--input-signals", nargs="+", default=["co2"],
                        choices=["abp", "ecg", "ppg", "co2"])
    parser.add_argument("--required-signals", nargs="+", default=None,
                        choices=["abp", "ecg", "ppg", "co2"])
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--horizon-mins", nargs="+", type=float, default=[5.0])
    parser.add_argument("--window-secs", nargs="+", type=float, default=[60.0])
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", default="outputs/downstream/etco2_abnormal")
    args = parser.parse_args()

    prepare_etco2_sweep(
        data_dir=args.data_dir,
        raw_dir=args.raw_dir,
        input_signals=args.input_signals,
        window_secs=args.window_secs,
        horizon_mins=args.horizon_mins,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_subjects=args.max_subjects,
        out_dir=args.out_dir,
        required_signals=args.required_signals,
    )


if __name__ == "__main__":
    main()
