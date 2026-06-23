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
import gc
import sys
from pathlib import Path

import numpy as np
import torch

from data.parser.mimic3_waveform import (
    MIMIC3_NATIVE_SR,
    _apply_pipeline,
)
from downstream._kfold_utils import stratified_kfold_patient_splits, summarize_splits
from downstream._save_utils import (
    add_signal_dtype_arg,
    stack_window_dicts_destructive,
)


TARGET_SR: float = 100.0

# WFDB sig_name → signal_type 매핑
# Table 3 (SSOT) #4 Cardiac Arrest: ABP, ECG, PPG, RespImp, CO₂.
# RESP — MEWS/NEWS2 의 RR component, pre-arrest apnea sign (Churpek 2014)
# CO2 — capnography; MIMIC 환자별 가용 시에만 추출(availability-aware).
MIMIC_SIGNAL_MAP: dict[str, str] = {
    "II": "ecg",
    "V": "ecg",
    "ABP": "abp",
    "ART": "abp",
    "PLETH": "ppg",
    # v2: RESP=impedance plethysmography → resp_impedance(7). SSOT: data.spatial_map.
    "RESP": "resp_impedance",
    # CO2 / Capnography (Table 3 #4)
    "CO2": "co2",
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
    n_folds: int = 5,
    max_patients: int | None = None,
    signal_dtype: torch.dtype = torch.float16,
    seed: int = 42,
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

    # 4. Stratified patient-level K-fold CV
    sid_to_label: dict[int, int] = {p["subject_id"]: int(p["cardiac_arrest"]) for p in patient_data}
    patient_to_labels: dict[str, list[int]] = {str(s): [lb] for s, lb in sid_to_label.items()}
    patient_ids = sorted(patient_to_labels.keys())
    splits = stratified_kfold_patient_splits(
        patient_ids, patient_to_labels, n_folds=n_folds, seed=seed,
    )
    print(f"\nStratified {n_folds}-fold patient-level CV:")
    print(summarize_splits(splits, patient_to_labels))

    def _pack_patients(plist: list[dict]) -> list[dict]:
        packed = []
        for p in plist:
            n_w = len(p["windows"])
            windows_per_type = stack_window_dicts_destructive(
                [dict(w) for w in p["windows"]], signal_dtype
            )
            packed.append({
                "subject_id": p["subject_id"],
                "label": p["cardiac_arrest"],
                "n_windows": n_w,
                "signals": windows_per_type,
            })
        return packed

    input_signals_meta = sorted(patient_data[0]["windows"][0].keys())

    PATIENTS_PER_CHUNK = 100

    for fold_idx, (train_sids, val_sids, test_sids) in enumerate(splits):
        # OOM 회피 (Stage 5) — patient-batch chunked save
        for split_name, split_sids in (
            ("train", train_sids), ("val", val_sids), ("test", test_sids),
        ):
            sids_int = {int(s) for s in split_sids}
            split_p_all = [p for p in patient_data if p["subject_id"] in sids_int]
            if not split_p_all:
                continue
            chunk_idx = 0
            total_pat = 0
            total_pos = 0
            for bstart in range(0, len(split_p_all), PATIENTS_PER_CHUNK):
                p_batch = split_p_all[bstart:bstart + PATIENTS_PER_CHUNK]
                n_p = len(p_batch)
                n_pos = sum(1 for p in p_batch if p["cardiac_arrest"] == 1)
                split_packed = _pack_patients(p_batch)
                del p_batch; gc.collect()

                data = {
                    "split": split_name,
                    "data": split_packed,
                    "metadata": {
                        "task": "cardiac_arrest_prediction",
                        "source": "MIMIC-III Waveform Matched",
                        "label": "cardiac_arrest",
                        "aggregation": "patient_level",
                        "input_signals": input_signals_meta,
                        "window_sec": window_sec,
                        "stride_sec": stride_sec,
                        "sampling_rate": TARGET_SR,
                        "fold_idx": fold_idx,
                        "n_folds": n_folds,
                        "split": split_name,
                        "chunk_idx": chunk_idx,
                        "n_patients": n_p,
                        "n_pos": n_pos,
                        "signal_dtype": str(signal_dtype).replace("torch.", ""),
                    },
                }
                fold_suffix = f"_fold{fold_idx}" if n_folds > 1 else ""
                out_file = out_path / (
                    f"cardiac_arrest_w{int(window_sec)}s{fold_suffix}_{split_name}_chunk{chunk_idx}.pt"
                )
                torch.save(data, out_file)
                total_pat += n_p
                total_pos += n_pos
                chunk_idx += 1
                del data, split_packed; gc.collect()
            print(
                f"  Fold {fold_idx} {split_name}: {chunk_idx} chunk(s), "
                f"{total_pat} patients (+={total_pos})"
            )

    print(f"\n{'=' * 60}")
    print(f"  Cardiac Arrest Prediction — Done")
    print(f"  Output dir: {out_path}")
    print(f"  Folds: {n_folds}")
    print(f"  Signals: {input_signals_meta}")
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
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Stratified patient-level K-fold CV (default 5).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int, default=None)
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    prepare_dataset(
        cohort_csv=args.cohort_csv,
        waveform_dir=args.waveform_dir,
        out_dir=args.out_dir,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        n_folds=args.n_folds,
        seed=args.seed,
        max_patients=args.max_patients,
        signal_dtype=dtype_map(args.signal_dtype),
    )


if __name__ == "__main__":
    main()
