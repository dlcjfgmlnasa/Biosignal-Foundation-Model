# -*- coding:utf-8 -*-
"""cardiac arrest prediction — 데이터 준비 (Acute Event Approach 1).

MIMIC-III Waveform + cardiac_arrest cohort CSV →
Cardiac Arrest 조기 경보 예측용 환자 단위 train/test .pt 파일 생성.

Framework: Acute Event Detection (Approach 1, Future Prediction)
  - Hypotension(1.1.1) / ICH(1.1.3) 와 동일한 sliding window framework
  - 600s(10min) input window, 30s stride
  - download_waveforms.py가 risk-set matched anchor를 제공한 후 호출

라벨: cardiac_arrest (0=non-arrest, 1=arrest)

교차 일반화 스토리: 한국 데이터(VitalDB) pretrain → 미국 데이터(MIMIC-III) 평가.

사용법:
    python -m downstream.acute_event.cardiac_arrest.prepare_data \
        --cohort-csv downstream/acute_event/cardiac_arrest/bquxjob_cardiac_arrest_TODO.csv \
        --waveform-dir datasets/raw/mimic3-waveform-cardiac-arrest \
        --out-dir datasets/processed/cardiac_arrest \
        --window-sec 600 --stride-sec 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from data.parser.mimic3_waveform import (
    MIMIC3_NATIVE_SR,
    _apply_pipeline,
)


TARGET_SR: float = 100.0

# WFDB sig_name → signal_type 매핑 (pretrained 채널만)
MIMIC_SIGNAL_MAP: dict[str, str] = {
    "II": "ecg",
    "V": "ecg",
    "ABP": "abp",
    "ART": "abp",
    "PLETH": "ppg",
    "CVP": "cvp",
    "PAP": "pap",
    "ICP": "icp",
}


def load_cohort(cohort_csv: str) -> dict[int, dict]:
    """cohort CSV → {subject_id: patient_info}."""
    patients: dict[int, dict] = {}

    with open(cohort_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["subject_id"])
            cardiac_arrest = int(row.get("cardiac_arrest", "0"))
            patients[sid] = {
                "subject_id": sid,
                "icustay_id": row["icustay_id"],
                "cardiac_arrest": cardiac_arrest,
                "first_arrest_time": row.get("first_arrest_time", ""),
                "sofa_total": row.get("sofa_total", ""),
                "age": row.get("age", ""),
                "gender": row.get("gender", ""),
                "icu_intime": row.get("icu_intime", ""),
                "icu_outtime": row.get("icu_outtime", ""),
            }

    return patients


def parse_waveform_record(
    record_dir: Path,
    record_name: str,
) -> dict[str, np.ndarray] | None:
    """WFDB 레코드에서 신호를 추출한다.

    pretraining과 동일한 전처리 적용 (_apply_pipeline).
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    hea_path = record_dir / f"{record_name}.hea"
    if not hea_path.exists():
        return None

    try:
        rec = wfdb.rdrecord(str(record_dir / record_name))
    except Exception:
        return None

    if rec.p_signal is None or rec.sig_len == 0:
        return None

    fs = float(rec.fs)
    signals: dict[str, np.ndarray] = {}

    for ch_idx, sig_name in enumerate(rec.sig_name):
        sig_type = MIMIC_SIGNAL_MAP.get(sig_name)
        if sig_type is None:
            continue
        if sig_type in signals:
            continue

        signal = rec.p_signal[:, ch_idx].astype(np.float64)
        processed = _apply_pipeline(signal, sig_type, fs)
        if processed is None:
            continue

        signals[sig_type] = processed.astype(np.float32)

    return signals if signals else None


def extract_windows(
    signals: dict[str, np.ndarray],
    window_sec: float,
    stride_sec: float,
    sr: float = TARGET_SR,
) -> list[dict[str, np.ndarray]]:
    """다중 신호에서 고정 길이 윈도우를 추출한다."""
    win_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)

    min_len = min(len(v) for v in signals.values())
    if min_len < win_samples:
        return []

    windows = []
    start = 0
    while start + win_samples <= min_len:
        win = {st: sig[start: start + win_samples] for st, sig in signals.items()}
        windows.append(win)
        start += stride_samples

    return windows


def prepare_dataset(
    cohort_csv: str,
    waveform_dir: str,
    out_dir: str,
    window_sec: float = 600.0,
    stride_sec: float = 30.0,  # 30s sliding (Acute Event 표준 — hypotension/ICH와 일관)
    train_ratio: float = 0.7,
    max_patients: int | None = None,
) -> None:
    """cardiac arrest prediction 데이터셋 준비 (환자 단위 그룹핑)."""
    wf_path = Path(waveform_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Cohort 로드
    patients = load_cohort(cohort_csv)
    print(f"Cohort: {len(patients)} patients")
    print(f"  Cardiac Arrest+: {sum(1 for p in patients.values() if p['cardiac_arrest']==1)}")
    print(f"  Cardiac Arrest-: {sum(1 for p in patients.values() if p['cardiac_arrest']==0)}")

    # 2. Waveform 디렉토리에서 환자별 레코드 탐색
    all_hea = sorted(wf_path.rglob("*.hea"))
    print(f"\nWaveform records found: {len(all_hea)}")

    subject_records: dict[int, list[Path]] = {}
    for hea in all_hea:
        for part in hea.parts:
            if part.startswith("p") and len(part) == 7 and part[1:].isdigit():
                sid = int(part[1:])
                if sid in patients:
                    if sid not in subject_records:
                        subject_records[sid] = []
                    subject_records[sid].append(hea)
                break

    print(f"Matched patients with waveforms: {len(subject_records)}")

    if max_patients is not None:
        pos_sids = [s for s in subject_records if patients[s]["cardiac_arrest"] == 1]
        neg_sids = [s for s in subject_records if patients[s]["cardiac_arrest"] == 0]
        n_pos = min(len(pos_sids), max_patients // 2)
        n_neg = min(len(neg_sids), max_patients - n_pos)
        selected = pos_sids[:n_pos] + neg_sids[:n_neg]
        subject_records = {s: subject_records[s] for s in selected}
        print(f"Limited to {len(subject_records)} patients "
              f"({n_pos} arrest+ + {n_neg} arrest-)")

    # 3. 환자별 윈도우 추출
    patient_data: list[dict] = []

    for i, (sid, hea_files) in enumerate(sorted(subject_records.items())):
        patient = patients[sid]
        patient_windows: list[dict[str, np.ndarray]] = []

        for hea_path in hea_files:
            rec_name = hea_path.stem
            rec_dir = hea_path.parent
            signals = parse_waveform_record(rec_dir, rec_name)
            if signals is None:
                continue

            windows = extract_windows(signals, window_sec, stride_sec)
            patient_windows.extend(windows)

        if patient_windows:
            patient_data.append({
                "subject_id": sid,
                "cardiac_arrest": patient["cardiac_arrest"],
                "windows": patient_windows,
            })

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i + 1}/{len(subject_records)}] subject={sid}, "
                  f"cardiac_arrest={patient['cardiac_arrest']}, "
                  f"windows={len(patient_windows)}")

    if not patient_data:
        print("ERROR: No patients with windows.", file=sys.stderr)
        sys.exit(1)

    n_pos = sum(1 for p in patient_data if p["cardiac_arrest"] == 1)
    n_neg = sum(1 for p in patient_data if p["cardiac_arrest"] == 0)
    total_windows = sum(len(p["windows"]) for p in patient_data)
    avg_windows = total_windows / len(patient_data)
    print(f"\nPatients: {len(patient_data)} (Cardiac Arrest+={n_pos}, Cardiac Arrest-={n_neg})")
    print(f"Total windows: {total_windows}, Avg per patient: {avg_windows:.1f}")

    # 4. Patient-level train/test split
    rng = np.random.default_rng(42)
    sids = [p["subject_id"] for p in patient_data]
    rng.shuffle(sids)
    n_train = max(1, int(len(sids) * train_ratio))
    train_sids = set(sids[:n_train])

    train_patients = [p for p in patient_data if p["subject_id"] in train_sids]
    test_patients = [p for p in patient_data if p["subject_id"] not in train_sids]

    print(f"Split: {len(train_patients)} train, {len(test_patients)} test patients")

    # 5. 환자 단위 .pt 저장
    def _pack_patients(plist: list[dict]) -> list[dict]:
        packed = []
        for p in plist:
            sig_types = sorted(p["windows"][0].keys())
            windows_per_type = {}
            for st in sig_types:
                windows_per_type[st] = torch.stack([
                    torch.from_numpy(w[st]).float() for w in p["windows"]
                ])  # (K, win_samples)
            packed.append({
                "subject_id": p["subject_id"],
                "label": p["cardiac_arrest"],
                "n_windows": len(p["windows"]),
                "signals": windows_per_type,
            })
        return packed

    data = {
        "train": _pack_patients(train_patients),
        "test": _pack_patients(test_patients),
        "metadata": {
            "task": "cardiac_arrest_prediction",
            "source": "MIMIC-III Waveform Matched",
            "label": "cardiac_arrest",
            "aggregation": "patient_level",
            "input_signals": sorted(train_patients[0]["windows"][0].keys()),
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "sampling_rate": TARGET_SR,
            "n_train_patients": len(train_patients),
            "n_test_patients": len(test_patients),
            "n_pos_train": sum(1 for p in train_patients if p["cardiac_arrest"] == 1),
            "n_pos_test": sum(1 for p in test_patients if p["cardiac_arrest"] == 1),
            "train_ratio": train_ratio,
        },
    }

    out_file = out_path / f"cardiac_arrest_w{int(window_sec)}s.pt"
    torch.save(data, out_file)

    n_pos_train = data["metadata"]["n_pos_train"]
    n_pos_test = data["metadata"]["n_pos_test"]
    print(f"\n{'=' * 60}")
    print(f"  cardiac arrest prediction — Data Prepared (Patient-Level)")
    print(f"  Output: {out_file}")
    print(f"  Train: {len(train_patients)} patients (arrest+={n_pos_train})")
    print(f"  Test:  {len(test_patients)} patients (arrest+={n_pos_test})")
    print(f"  Signals: {data['metadata']['input_signals']}")
    print(f"  Window: {window_sec}s, Stride: {stride_sec}s")
    print(f"  Avg windows/patient: {avg_windows:.1f}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="cardiac arrest prediction — Data Preparation"
    )
    parser.add_argument("--cohort-csv", type=str, required=True)
    parser.add_argument("--waveform-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="datasets/processed/cardiac_arrest")
    parser.add_argument("--window-sec", type=float, default=600.0,
                        help="입력 윈도우 길이(초). 기본 600s (10분).")
    parser.add_argument("--stride-sec", type=float, default=30.0,
                        help="sliding window stride(초). 기본 30s "
                             "(Acute Event 표준 — Hypotension/ICH와 일관).")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--max-patients", type=int, default=None)
    args = parser.parse_args()

    prepare_dataset(
        cohort_csv=args.cohort_csv,
        waveform_dir=args.waveform_dir,
        out_dir=args.out_dir,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        max_patients=args.max_patients,
    )


if __name__ == "__main__":
    main()
