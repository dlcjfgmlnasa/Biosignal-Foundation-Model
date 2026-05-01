# -*- coding:utf-8 -*-
"""ICU Mech Vent Need Prediction — 데이터 준비.

MIMIC-III Waveform + vent_need cohort CSV →
ICU 사망 예측용 train/test .pt 파일 생성.

라벨: vent_within_24h (0=생존, 1=사망)

Mech Vent Need 전용 cohort (모든 ICU 유닛 포함, sepsis cohort는 5개 유닛만).
교차 일반화 스토리: 한국 데이터(VitalDB) pretrain → 미국 데이터(MIMIC-III) 평가.

사용법:
    python -m downstream.outcome.vent_need.prepare_data \
        --cohort-csv downstream/classification/vent_need/icu_vent_need_cohort.csv \
        --waveform-dir datasets/raw/mimic3-waveform-vent_need \
        --out-dir datasets/processed/vent_need \
        --window-sec 600 --stride-sec 300
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data.parser.mimic3_waveform import (
    MIMIC3_NATIVE_SR,
    _apply_pipeline,
)


TARGET_SR: float = 100.0

# WFDB sig_name → 우리 signal_type 매핑
# Pretrained model에서 학습한 채널만 사용:
#   ECG: II, V5 (VitalDB SNUADC/ECG_II, ECG_V5)
#   ABP, PPG, CVP, PAP, ICP: 각 1채널
MIMIC_SIGNAL_MAP: dict[str, str] = {
    # ECG — pretrained lead만
    "II": "ecg",
    "V": "ecg",       # MIMIC-III에서 V5 대응 (단일 V lead)
    # ABP
    "ABP": "abp",
    "ART": "abp",
    # PPG
    "PLETH": "ppg",
    # CVP
    "CVP": "cvp",
    # PAP
    "PAP": "pap",
    # ICP
    "ICP": "icp",
}


def load_cohort(cohort_csv: str) -> dict[int, dict]:
    """cohort CSV → {subject_id: patient_info}.

    Mech Vent Need 전용 cohort와 sepsis cohort 모두 지원한다.
    vent_within_24h를 vent_need 라벨로 사용한다.
    """
    patients: dict[int, dict] = {}

    with open(cohort_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["subject_id"])
            vent_need = int(row.get("vent_within_24h", "0"))
            patients[sid] = {
                "subject_id": sid,
                "icustay_id": row["icustay_id"],
                "vent_need": vent_need,
                "first_careunit": row.get("first_careunit", ""),
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
    """WFDB 레코드에서 ECG/ABP/PPG/CVP/PAP/ICP 신호를 추출한다.

    mimic3_waveform.py의 _apply_pipeline을 사용하여
    pretraining과 동일한 전처리 적용.

    Returns
    -------
    {signal_type: 1D ndarray (100Hz)} or None if failed.
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

        # 이미 같은 타입이 있으면 스킵 (첫 번째 우선)
        if sig_type in signals:
            continue

        signal = rec.p_signal[:, ch_idx].astype(np.float64)

        # pretraining과 동일한 전처리 파이프라인 적용
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
    """다중 신호에서 고정 길이 윈도우를 추출한다.

    모든 신호의 공통 길이만큼만 사용하고, window_sec 간격으로 슬라이딩.
    """
    win_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)

    # 공통 길이
    min_len = min(len(v) for v in signals.values())
    if min_len < win_samples:
        return []

    windows = []
    start = 0
    while start + win_samples <= min_len:
        win = {}
        for sig_type, sig in signals.items():
            win[sig_type] = sig[start: start + win_samples]
        windows.append(win)
        start += stride_samples

    return windows


def prepare_dataset(
    cohort_csv: str,
    waveform_dir: str,
    out_dir: str,
    window_sec: float = 600.0,
    stride_sec: float = 300.0,
    train_ratio: float = 0.7,
    max_patients: int | None = None,
) -> None:
    """Mech Vent Need prediction 데이터셋 준비."""
    wf_path = Path(waveform_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Cohort 로드
    patients = load_cohort(cohort_csv)
    print(f"Cohort: {len(patients)} patients")
    print(f"  Deaths: {sum(1 for p in patients.values() if p['vent_need']==1)}")
    print(f"  Survivors: {sum(1 for p in patients.values() if p['vent_need']==0)}")

    # 2. Waveform 디렉토리에서 환자별 레코드 탐색
    all_hea = sorted(wf_path.rglob("*.hea"))
    print(f"\nWaveform records found: {len(all_hea)}")

    # subject_id별 레코드 그룹핑
    subject_records: dict[int, list[Path]] = {}
    for hea in all_hea:
        # 경로에서 subject_id 추출: p00/p000020/p000020-2183-04-28-17-47.hea
        parts = hea.parts
        for part in parts:
            if part.startswith("p") and len(part) == 7 and part[1:].isdigit():
                sid = int(part[1:])
                if sid in patients:
                    if sid not in subject_records:
                        subject_records[sid] = []
                    subject_records[sid].append(hea)
                break

    print(f"Matched patients with waveforms: {len(subject_records)}")

    if max_patients is not None:
        # 사망 환자 우선 포함
        dead_sids = [s for s in subject_records if patients[s]["vent_need"] == 1]
        alive_sids = [s for s in subject_records if patients[s]["vent_need"] == 0]
        n_dead = min(len(dead_sids), max_patients // 2)
        n_alive = min(len(alive_sids), max_patients - n_dead)
        selected_sids = dead_sids[:n_dead] + alive_sids[:n_alive]
        subject_records = {s: subject_records[s] for s in selected_sids}
        print(f"Limited to {len(subject_records)} patients "
              f"({n_dead} dead + {n_alive} alive)")

    # 3. 환자별 윈도우 추출 (환자 단위 그룹핑)
    patient_data: list[dict] = []  # [{subject_id, vent_need, windows: [signals]}]

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
                "vent_need": patient["vent_need"],
                "windows": patient_windows,
            })

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i + 1}/{len(subject_records)}] subject={sid}, "
                  f"vent_need={patient['vent_need']}, "
                  f"windows={len(patient_windows)}")

    if not patient_data:
        print("ERROR: No patients with windows.", file=sys.stderr)
        sys.exit(1)

    n_dead = sum(1 for p in patient_data if p["vent_need"] == 1)
    n_alive = sum(1 for p in patient_data if p["vent_need"] == 0)
    total_windows = sum(len(p["windows"]) for p in patient_data)
    avg_windows = total_windows / len(patient_data)
    print(f"\nPatients: {len(patient_data)} (Dead={n_dead}, Alive={n_alive})")
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

    # 5. 환자 단위로 .pt 저장
    #    각 환자: windows (K, sig_type, win_samples), vent_need label
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
                "vent_need": p["vent_need"],
                "n_windows": len(p["windows"]),
                "signals": windows_per_type,  # {sig_type: (K, win_samples)}
            })
        return packed

    data = {
        "train": _pack_patients(train_patients),
        "test": _pack_patients(test_patients),
        "metadata": {
            "task": "icu_vent_need_prediction",
            "source": "MIMIC-III Waveform Matched",
            "label": "vent_within_24h",
            "aggregation": "patient_level",
            "input_signals": sorted(train_patients[0]["windows"][0].keys()),
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "sampling_rate": TARGET_SR,
            "n_train_patients": len(train_patients),
            "n_test_patients": len(test_patients),
            "n_dead_train": sum(1 for p in train_patients if p["vent_need"] == 1),
            "n_dead_test": sum(1 for p in test_patients if p["vent_need"] == 1),
            "train_ratio": train_ratio,
        },
    }

    out_file = out_path / f"vent_need_w{int(window_sec)}s.pt"
    torch.save(data, out_file)

    n_dead_train = data["metadata"]["n_dead_train"]
    n_dead_test = data["metadata"]["n_dead_test"]
    print(f"\n{'=' * 60}")
    print(f"  ICU Mech Vent Need Prediction — Data Prepared (Patient-Level)")
    print(f"  Output: {out_file}")
    print(f"  Train: {len(train_patients)} patients (dead={n_dead_train})")
    print(f"  Test:  {len(test_patients)} patients (dead={n_dead_test})")
    print(f"  Signals: {data['metadata']['input_signals']}")
    print(f"  Window: {window_sec}s, Stride: {stride_sec}s")
    print(f"  Avg windows/patient: {avg_windows:.1f}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICU Mech Vent Need Prediction — Data Preparation"
    )
    parser.add_argument(
        "--cohort-csv", type=str, required=True,
        help="sepsis3_cohort CSV (vent_within_24h 포함)",
    )
    parser.add_argument(
        "--waveform-dir", type=str, required=True,
        help="다운로드된 MIMIC-III waveform 디렉토리",
    )
    parser.add_argument("--out-dir", type=str, default="datasets/processed/vent_need")
    parser.add_argument("--window-sec", type=float, default=600.0)
    parser.add_argument("--stride-sec", type=float, default=300.0)
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
