# -*- coding:utf-8 -*-
"""PhysioNet/CinC Challenge 2011 — ECG Signal Quality 파서.

12-lead ECG (500Hz, 10초) → 100Hz .pt 변환.
라벨: acceptable(정상) vs unacceptable(이상/노이즈) 품질 판정.

데이터 구조:
    raw/physionet-challenge-2011/set-a/
    ├── RECORDS           ← 레코드 목록
    ├── REFERENCE.csv     ← 품질 라벨 (존재 시)
    ├── 1002603.hea       ← WFDB 헤더
    ├── 1002603.dat       ← 신호 데이터
    └── ...

라벨 체계 (3-group):
    Group 1 (acceptable):   평균 grade >= 0.70, F grade 최대 1개
    Group 2 (indeterminate): 평균 grade >= 0.70, F grade 2개 이상
    Group 3 (unacceptable): 평균 grade < 0.70

Anomaly detection용: acceptable(0) vs unacceptable(1)

사용법:
    python -m data.parser.physionet2011 \
        --raw-dir datasets/raw/physionet-challenge-2011 \
        --out-dir datasets/processed/signal_quality
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data.parser._common import resample_to_target

TARGET_SR: float = 100.0
NATIVE_SR: float = 500.0

# 12-lead ECG 채널 — 전부 signal_type "ecg"
ECG_LEADS = {"I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"}
SIGNAL_TYPE_INT: int = 0  # ecg


# ── 필터링 (pretraining과 동일) ──────────────────────────────


def _apply_bandpass(data: np.ndarray, lo: float, hi: float, sr: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if hi <= lo:
        return data
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data).astype(data.dtype)


def _apply_notch_filter(
    data: np.ndarray, freq: float, sr: float, Q: float = 30.0
) -> np.ndarray:
    from scipy.signal import filtfilt, iirnotch

    nyq = sr / 2.0
    if freq >= nyq:
        return data
    b, a = iirnotch(freq / nyq, Q)
    return filtfilt(b, a, data).astype(data.dtype)


def preprocess_ecg(signal: np.ndarray, sr: float) -> np.ndarray:
    """ECG 필터링 (pretraining과 동일): NaN 보간 → notch 60Hz → bandpass [0.5, 40Hz]."""
    # NaN 보간
    nan_ratio = float(np.isnan(signal).mean())
    if nan_ratio > 0 and nan_ratio < 1.0:
        nans = np.isnan(signal)
        x = np.arange(len(signal))
        signal = signal.copy()
        signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])

    signal = _apply_notch_filter(signal, 60.0, sr)
    signal = _apply_bandpass(signal, 0.5, 40.0, sr)
    return signal


# ── 라벨 로딩 ────────────────────────────────────────────────


def load_labels(raw_dir: Path) -> dict[str, int]:
    """라벨 파일에서 레코드별 품질 라벨을 로드한다.

    여러 가능한 라벨 파일 형식을 시도:
    1. REFERENCE.csv: record,label
    2. ANSWERS: record,label
    3. .hea 파일 내 주석에서 추출

    Returns
    -------
    {record_name: quality_group (1=acceptable, 2=indeterminate, 3=unacceptable)}
    """
    labels: dict[str, int] = {}

    # 1. REFERENCE.csv 또는 ANSWERS 파일 탐색
    for label_file_name in ["REFERENCE.csv", "ANSWERS", "REFERENCE", "answers.txt"]:
        for label_path in raw_dir.rglob(label_file_name):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        try:
                            labels[name] = int(parts[1].strip())
                        except ValueError:
                            # "acceptable" / "unacceptable" 문자열 처리
                            val = parts[1].strip().lower()
                            if val in ("acceptable", "accept", "1", "true"):
                                labels[name] = 1
                            elif val in ("unacceptable", "unaccept", "3", "false"):
                                labels[name] = 3
                            else:
                                labels[name] = 2  # indeterminate

    # 2. .hea 파일 주석에서 추출
    if not labels:
        for hea_path in raw_dir.rglob("*.hea"):
            with open(hea_path, "r") as f:
                for line in f:
                    if line.startswith("#Quality:"):
                        val = line.split(":")[1].strip().lower()
                        name = hea_path.stem
                        if "acceptable" in val and "un" not in val:
                            labels[name] = 1
                        elif "unacceptable" in val:
                            labels[name] = 3
                        else:
                            labels[name] = 2

    return labels


# ── 파싱 ─────────────────────────────────────────────────────


def parse_record(
    record_path: Path,
    out_dir: Path,
    quality_label: int,
) -> dict | None:
    """단일 WFDB 레코드를 파싱하여 Lead II .pt로 저장한다.

    12-lead 중 Lead II를 주 채널로 사용 (우리 모델의 ECG와 동일).
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    try:
        rec = wfdb.rdrecord(str(record_path))
    except Exception as e:
        print(f"  SKIP {record_path.name}: {e}")
        return None

    if rec.p_signal is None or rec.sig_len == 0:
        return None

    record_name = record_path.name
    fs = float(rec.fs)
    signals_saved: list[dict] = []

    for ch_idx, sig_name in enumerate(rec.sig_name):
        if sig_name not in ECG_LEADS:
            continue

        signal = rec.p_signal[:, ch_idx].astype(np.float64)

        # 필터링 (native SR에서)
        signal = preprocess_ecg(signal, fs)

        # 리샘플링 500Hz → 100Hz
        if fs != TARGET_SR:
            signal = resample_to_target(signal, fs, TARGET_SR)

        tensor = torch.from_numpy(signal).float().unsqueeze(0)  # (1, T)

        out_name = f"{record_name}_ecg_{sig_name}.pt"
        torch.save(tensor, out_dir / out_name)

        signals_saved.append({
            "file": out_name,
            "signal_type": "ecg",
            "signal_type_int": SIGNAL_TYPE_INT,
            "signal_name": sig_name,
            "n_samples": tensor.shape[-1],
            "duration_sec": round(tensor.shape[-1] / TARGET_SR, 2),
        })

    if not signals_saved:
        return None

    # anomaly label: acceptable(1) → 0 (normal), unacceptable(3) → 1 (anomaly)
    anomaly_label = 0 if quality_label == 1 else 1

    return {
        "record": record_name,
        "quality_group": quality_label,
        "anomaly_label": anomaly_label,
        "signals": signals_saved,
        "sampling_rate": TARGET_SR,
        "native_sr": fs,
        "n_leads": len(signals_saved),
        "duration_sec": round(rec.sig_len / rec.fs, 2),
    }


def parse_dataset(
    raw_dir: str,
    out_dir: str,
    max_records: int | None = None,
) -> None:
    """PhysioNet 2011 Challenge 전체 파싱."""
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 라벨 로드
    labels = load_labels(raw_path)
    if labels:
        print(f"Loaded {len(labels)} quality labels")
    else:
        print("  WARNING: No label file found")

    # .hea 파일 탐색
    hea_files = sorted(raw_path.rglob("*.hea"))
    print(f"Found {len(hea_files)} records")

    if max_records is not None:
        hea_files = hea_files[:max_records]

    manifest: list[dict] = []
    quality_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    n_success = 0
    n_skip = 0

    for i, hea_path in enumerate(hea_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{len(hea_files)}] {hea_path.stem}...")

        record_stem = str(hea_path.with_suffix(""))
        record_name = hea_path.stem

        quality_label = labels.get(record_name, 0)  # 0=unknown

        # 이미 파싱된 경우 스킵
        existing_pts = list(out_path.glob(f"{record_name}_*.pt"))
        if existing_pts:
            n_skip += 1
            anomaly_label = 0 if quality_label == 1 else 1
            manifest.append({
                "record": record_name,
                "quality_group": quality_label,
                "anomaly_label": anomaly_label,
                "signals": [{"file": p.name} for p in existing_pts],
                "sampling_rate": TARGET_SR,
            })
            quality_counts[quality_label] = quality_counts.get(quality_label, 0) + 1
            continue

        result = parse_record(Path(record_stem), out_path, quality_label)

        if result is None:
            n_skip += 1
            continue

        manifest.append(result)
        n_success += 1
        quality_counts[quality_label] = quality_counts.get(quality_label, 0) + 1

    # manifest 저장
    n_acceptable = quality_counts.get(1, 0)
    n_indeterminate = quality_counts.get(2, 0)
    n_unacceptable = quality_counts.get(3, 0)
    n_unknown = quality_counts.get(0, 0)

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "dataset": "PhysioNet-Challenge-2011",
            "task": "ECG Signal Quality Assessment",
            "sampling_rate": TARGET_SR,
            "native_sr": NATIVE_SR,
            "n_records": len(manifest),
            "quality_distribution": {
                "acceptable": n_acceptable,
                "indeterminate": n_indeterminate,
                "unacceptable": n_unacceptable,
                "unknown": n_unknown,
            },
            "records": manifest,
        }, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  PhysioNet 2011 Challenge Parsing Complete")
    print(f"  New: {n_success}, Skipped: {n_skip}")
    print(f"  Total: {len(manifest)}")
    print(f"  Acceptable:     {n_acceptable}")
    print(f"  Indeterminate:  {n_indeterminate}")
    print(f"  Unacceptable:   {n_unacceptable}")
    print(f"  Unknown:        {n_unknown}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhysioNet 2011 Challenge — ECG Signal Quality Parser",
    )
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="datasets/processed/signal_quality")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    parse_dataset(raw_dir=args.raw_dir, out_dir=args.out_dir, max_records=args.max_records)


if __name__ == "__main__":
    main()
