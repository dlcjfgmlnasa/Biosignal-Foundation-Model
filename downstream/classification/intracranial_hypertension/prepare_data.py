# -*- coding:utf-8 -*-
"""Intracranial Hypertension Detection — 데이터 준비 (MIMIC-III).

MIMIC-III Waveform에서 ICP 채널이 있는 레코드를 파싱하여
ICP > 20mmHg (≥1분 지속) 예측용 (input_window, future_label) 쌍을 생성한다.

Label 소스: ICP (미래 구간의 평균 ICP > 20mmHg 지속 여부)
Input 소스: ICP + 동시 기록된 ECG, ABP, PPG 등

데이터 소스: MIMIC-III Waveform Matched Subset (PhysioNet)

사용법:
    # Sweep: window × horizon 전체 조합 생성
    python -m downstream.classification.intracranial_hypertension.prepare_data \
        --waveform-dir datasets/raw/mimic3-waveform-ich \
        --window-secs 30 60 300 600 --horizon-mins 5 10 15
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.parser.mimic3_waveform import _apply_pipeline


TARGET_SR: float = 100.0

# ICP 임상 기준
ICP_THRESHOLD: float = 20.0  # mmHg
SUSTAINED_SEC: float = 60.0  # 1분 이상 지속

# MIMIC-III 채널명 → signal_type 매핑 (pretrained 채널만)
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


# ── 데이터 구조 ──────────────────────────────────────────────


@dataclass
class ICHSample:
    """두개내 고혈압 예측 샘플."""

    input_signals: dict[str, np.ndarray]
    label: int  # 0=normal, 1=intracranial hypertension
    label_value: float  # future max ICP (mmHg)
    case_id: str
    win_start_sec: float
    horizon_sec: float


# ── WFDB 파싱 ────────────────────────────────────────────────


def parse_waveform_record(
    record_dir: Path,
    record_name: str,
) -> dict[str, np.ndarray] | None:
    """WFDB 레코드에서 ICP + 기타 신호를 추출한다.

    ICP가 없는 레코드는 None 반환.
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

    # ICP 필수
    if "icp" not in signals:
        return None

    return signals


# ── 환자별 레코드 탐색 ───────────────────────────────────────


def load_patient_signals(
    waveform_dir: Path,
) -> list[dict]:
    """Waveform 디렉토리에서 ICP 포함 레코드를 로드한다.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signals": {type: ndarray}}
    """
    all_hea = sorted(waveform_dir.rglob("*.hea"))
    print(f"  Found {len(all_hea)} .hea files")

    cases: list[dict] = []
    n_no_icp = 0

    for i, hea_path in enumerate(all_hea):
        rec_name = hea_path.stem
        rec_dir = hea_path.parent

        # subject_id 추출
        patient_id = "unknown"
        for part in hea_path.parts:
            if part.startswith("p") and len(part) == 7 and part[1:].isdigit():
                patient_id = part
                break

        signals = parse_waveform_record(rec_dir, rec_name)
        if signals is None:
            n_no_icp += 1
            continue

        cases.append({
            "case_id": f"{patient_id}_{rec_name}",
            "patient_id": patient_id,
            "signals": signals,
        })

        if (i + 1) % 50 == 0 or i == 0:
            sig_types = list(signals.keys())
            dur_sec = len(signals["icp"]) / TARGET_SR
            print(f"    [{i + 1}/{len(all_hea)}] {patient_id}/{rec_name} "
                  f"signals={sig_types} dur={dur_sec:.0f}s")

    print(f"  ICP records: {len(cases)}, skipped (no ICP): {n_no_icp}")
    return cases


# ── 라벨링 ───────────────────────────────────────────────────


def _has_sustained_ich(
    future_icps: list[float],
    threshold: float,
    min_consecutive: int,
) -> bool:
    """연속 min_consecutive개 이상 윈도우에서 ICP > threshold인지 확인한다."""
    consecutive = 0
    for icp_val in future_icps:
        if icp_val > threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                return True
        else:
            consecutive = 0
    return False


# ── 윈도우 추출 ──────────────────────────────────────────────


def extract_forecast_samples(
    cases: list[dict],
    input_signals: list[str],
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    horizon_sec: float = 300.0,
    icp_threshold: float = ICP_THRESHOLD,
    sustained_sec: float = SUSTAINED_SEC,
) -> list[ICHSample]:
    """시간 정렬된 다채널 데이터에서 (input, future_label) 쌍을 추출한다."""
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    horizon_samples = int(horizon_sec * TARGET_SR)

    icp_win_sec = 10.0
    icp_win = int(icp_win_sec * TARGET_SR)
    min_consecutive = max(1, int(sustained_sec / icp_win_sec))

    total_needed = win_samples + horizon_samples
    samples: list[ICHSample] = []

    for case in cases:
        signals = case["signals"]
        icp = signals["icp"]

        # 모든 input signal의 공통 길이
        min_len = min(len(signals[s]) for s in input_signals if s in signals)
        min_len = min(min_len, len(icp))

        if min_len < total_needed:
            continue

        for start in range(0, min_len - total_needed + 1, stride_samples):
            input_dict = {}
            for stype in input_signals:
                if stype in signals:
                    input_dict[stype] = signals[stype][start: start + win_samples]

            if not input_dict:
                continue

            # Future ICP
            future_start = start + win_samples
            future_end = future_start + horizon_samples
            future_icp = icp[future_start:future_end]

            future_icps: list[float] = []
            for j in range(0, len(future_icp) - icp_win + 1, icp_win):
                w = future_icp[j: j + icp_win]
                if not np.isnan(w).any():
                    future_icps.append(float(np.mean(w)))

            if not future_icps:
                continue

            label = (
                1
                if _has_sustained_ich(future_icps, icp_threshold, min_consecutive)
                else 0
            )

            samples.append(
                ICHSample(
                    input_signals=input_dict,
                    label=label,
                    label_value=max(future_icps),
                    case_id=case["case_id"],
                    win_start_sec=start / TARGET_SR,
                    horizon_sec=horizon_sec,
                )
            )

    return samples


# ── 저장 ─────────────────────────────────────────────────────


def save_dataset(
    train_samples: list[ICHSample],
    test_samples: list[ICHSample],
    input_signals: list[str],
    horizon_sec: float,
    window_sec: float,
    out_dir: str,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples: list[ICHSample]) -> dict:
        if not samples:
            return {"signals": {}, "labels": torch.tensor([]),
                    "label_values": torch.tensor([])}

        sig_tensors = {}
        for stype in input_signals:
            arrs = [s.input_signals[stype] for s in samples if stype in s.input_signals]
            if arrs:
                sig_tensors[stype] = torch.stack(
                    [torch.from_numpy(a).float() for a in arrs]
                )

        return {
            "signals": sig_tensors,
            "labels": torch.tensor([s.label for s in samples], dtype=torch.long),
            "label_values": torch.tensor(
                [s.label_value for s in samples], dtype=torch.float32
            ),
            "case_ids": [s.case_id for s in samples],
        }

    save_dict = {
        "train": _to_tensors(train_samples),
        "test": _to_tensors(test_samples),
        "metadata": {
            "task": "intracranial_hypertension_detection",
            "source": "MIMIC-III Waveform",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "icp_threshold": ICP_THRESHOLD,
            "sustained_sec": SUSTAINED_SEC,
            "n_train": len(train_samples),
            "n_test": len(test_samples),
        },
    }

    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    win_int = int(window_sec)
    filename = f"ich_{mode_str}_w{win_int}s_h{horizon_min}min.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ── 통계 출력 ────────────────────────────────────────────────


def print_stats(name: str, samples: list[ICHSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return

    n_total = len(samples)
    n_ich = sum(1 for s in samples if s.label == 1)
    n_normal = n_total - n_ich
    icps = [s.label_value for s in samples]

    print(f"  {name}: {n_total} samples")
    print(f"    Normal: {n_normal} ({n_normal / n_total * 100:.1f}%)")
    print(f"    ICH:    {n_ich} ({n_ich / n_total * 100:.1f}%)")
    print(
        f"    Future ICP max: [{min(icps):.1f}, {max(icps):.1f}] mmHg, "
        f"mean={np.mean(icps):.1f} +/- {np.std(icps):.1f}"
    )


# ── Sweep ────────────────────────────────────────────────────


def prepare_ich_sweep(
    waveform_dir: str,
    input_signals: list[str],
    window_secs: list[float],
    horizon_mins: list[float],
    stride_sec: float = 30.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/intracranial_hypertension",
) -> list[Path]:
    """(window, horizon) 조합을 sweep하여 데이터셋을 생성한다."""
    mode_str = " + ".join(s.upper() for s in input_signals)
    print(f"\n{'=' * 60}")
    print(f"  Intracranial Hypertension Detection - MIMIC-III")
    print(f"  Waveform: {waveform_dir}")
    print(f"  Input:    {mode_str}")
    print(f"  Windows:  {window_secs}")
    print(f"  Horizons: {horizon_mins}")
    print(f"  ICP threshold: {ICP_THRESHOLD} mmHg, sustained: {SUSTAINED_SEC}s")
    print(f"{'=' * 60}")

    # 1. 데이터 로딩
    print("\n[1/3] Loading ICP waveform records...")
    cases = load_patient_signals(Path(waveform_dir))
    if not cases:
        print("ERROR: No ICP records found.", file=sys.stderr)
        sys.exit(1)

    # 2. Patient-level train/test split
    print(f"\n[2/3] Splitting by patient (ratio={train_ratio})...")
    rng = np.random.default_rng(42)
    patient_ids = list({c["patient_id"] for c in cases})
    rng.shuffle(patient_ids)
    n_train = max(1, int(len(patient_ids) * train_ratio))
    train_patients = set(patient_ids[:n_train])

    train_cases = [c for c in cases if c["patient_id"] in train_patients]
    test_cases = [c for c in cases if c["patient_id"] not in train_patients]
    print(f"  Train: {len(train_cases)} records ({len(train_patients)} patients)")
    print(f"  Test:  {len(test_cases)} records "
          f"({len(patient_ids) - n_train} patients)")

    # 3. 조합별 윈도우 추출 + 저장
    combos = [(w, h) for w in window_secs for h in horizon_mins]
    print(f"\n[3/3] Generating {len(combos)} datasets...")

    saved_paths: list[Path] = []
    for i, (window_sec, horizon_min) in enumerate(combos, 1):
        horizon_sec = horizon_min * 60.0
        print(f"\n  [{i}/{len(combos)}] window={window_sec}s, horizon={horizon_min}min")

        train_samples = extract_forecast_samples(
            train_cases, input_signals, window_sec, stride_sec, horizon_sec,
        )
        test_samples = extract_forecast_samples(
            test_cases, input_signals, window_sec, stride_sec, horizon_sec,
        )

        print_stats("    Train", train_samples)
        print_stats("    Test", test_samples)

        if not train_samples and not test_samples:
            print("    SKIP: No samples extracted.")
            continue

        save_path = save_dataset(
            train_samples, test_samples, input_signals,
            horizon_sec, window_sec, out_dir,
        )
        saved_paths.append(save_path)

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(saved_paths)}/{len(combos)} datasets saved to {out_dir}")
    print(f"{'=' * 60}")
    return saved_paths


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intracranial Hypertension Detection - Data Preparation (MIMIC-III)",
    )
    parser.add_argument(
        "--waveform-dir", type=str, required=True,
        help="MIMIC-III waveform directory (ICP 포함 레코드)",
    )
    parser.add_argument(
        "--input-signals", nargs="+", default=["icp"],
        choices=["icp", "ecg", "abp", "ppg", "cvp", "pap"],
        help="Input signal types (label always from ICP)",
    )
    parser.add_argument(
        "--horizon-mins", nargs="+", type=float, default=[5.0],
    )
    parser.add_argument(
        "--window-secs", nargs="+", type=float, default=[30.0],
    )
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--out-dir", type=str,
        default="outputs/downstream/intracranial_hypertension",
    )
    args = parser.parse_args()

    prepare_ich_sweep(
        waveform_dir=args.waveform_dir,
        input_signals=args.input_signals,
        window_secs=args.window_secs,
        horizon_mins=args.horizon_mins,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
