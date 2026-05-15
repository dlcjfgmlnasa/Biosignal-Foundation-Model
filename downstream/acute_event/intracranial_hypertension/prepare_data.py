# -*- coding:utf-8 -*-
"""Intracranial Hypertension Detection — 데이터 준비 (MIMIC-III).

MIMIC-III Waveform에서 ICP 채널이 있는 레코드를 파싱하여
ICP > 20mmHg (≥1분 지속) 예측용 (input_window, future_label) 쌍을 생성한다.

Label 소스: ICP (미래 구간의 평균 ICP > 20mmHg 지속 여부)
Input 소스: ICP + 동시 기록된 ECG, ABP, PPG 등

데이터 소스: MIMIC-III Waveform Matched Subset (PhysioNet)

사용법:
    # Sweep: window × horizon 전체 조합 생성
    python -m downstream.acute_event.intracranial_hypertension.prepare_data \
        --waveform-dir datasets/raw/mimic3-waveform-ich \
        --window-secs 30 60 300 600 --horizon-mins 5 10 15
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.parser.mimic3_waveform import _apply_pipeline
from downstream._kfold_utils import stratified_kfold_patient_splits, summarize_splits


TARGET_SR: float = 100.0

# ICP 임상 기준
ICP_THRESHOLD: float = 20.0  # mmHg
SUSTAINED_SEC: float = 60.0  # 1분 이상 지속

# MIMIC-III 채널명 → signal_type 매핑 (Task #3 ICH: ABP, ECG, PPG, CO2 + ICP for label)
# ICP는 라벨 산출 전용 (input 신호로는 사용하지 않음, --input-signals 로 별도 제어)
MIMIC_SIGNAL_MAP: dict[str, str] = {
    "II": "ecg",
    "V": "ecg",
    "ABP": "abp",
    "ART": "abp",
    "PLETH": "ppg",
    "CO2": "co2",
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
    patient_id: str  # LOSO grouping 필수
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
    import re

    # 마스터 헤더만 선별: 파일명이 p<6자리>-... 형태
    # (segment: <num>_NNNN.hea, layout: <num>_layout.hea 는 제외)
    master_re = re.compile(r"^p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$")
    all_hea = sorted(
        h for h in waveform_dir.rglob("*.hea") if master_re.match(h.stem)
    )
    print(f"  Found {len(all_hea)} master .hea files")

    cases: list[dict] = []
    n_no_icp = 0

    for i, hea_path in enumerate(all_hea):
        rec_name = hea_path.stem
        rec_dir = hea_path.parent

        # subject_id 추출: 디렉토리(p00/p000907) 우선, 없으면 파일명 prefix(p000907)에서
        patient_id = "unknown"
        for part in hea_path.parts:
            if part.startswith("p") and len(part) == 7 and part[1:].isdigit():
                patient_id = part
                break
        if patient_id == "unknown":
            patient_id = rec_name.split("-", 1)[0]  # p000907

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
                    patient_id=case["patient_id"],
                    win_start_sec=start / TARGET_SR,
                    horizon_sec=horizon_sec,
                )
            )

    return samples


# ── 저장 ─────────────────────────────────────────────────────


def _consume_to_tensors_ich(
    samples: list[ICHSample],
    input_signals: list[str],
    signal_dtype: torch.dtype,
) -> dict:
    """ICH samples → packed dict (in-place, samples 비워짐).

    torch.stack 의 2× peak 회피 — 출력 텐서를 미리 할당 + numpy ref pop.
    """
    if not samples:
        return {"signals": {}, "labels": torch.tensor([], dtype=torch.long),
                "label_values": torch.tensor([], dtype=torch.float32),
                "case_ids": [], "subject_ids": []}

    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    label_values = torch.tensor(
        [s.label_value for s in samples], dtype=torch.float32
    )
    case_ids = [s.case_id for s in samples]
    subject_ids = [s.patient_id for s in samples]

    sig_tensors: dict[str, torch.Tensor] = {}
    for stype in input_signals:
        T = next(
            (int(s.input_signals[stype].shape[0])
             for s in samples if stype in s.input_signals),
            None,
        )
        if T is None:
            continue
        n = sum(1 for s in samples if stype in s.input_signals)
        out = torch.empty((n, T), dtype=signal_dtype)
        i = 0
        for s in samples:
            arr = s.input_signals.pop(stype, None)
            if arr is None:
                continue
            out[i].copy_(torch.from_numpy(arr))
            i += 1
        sig_tensors[stype] = out

    return {
        "signals": sig_tensors,
        "labels": labels,
        "label_values": label_values,
        "case_ids": case_ids,
        "subject_ids": subject_ids,
    }


def save_split_dataset(
    split_dict: dict,
    split_name: str,
    input_signals: list[str],
    horizon_sec: float,
    window_sec: float,
    out_dir: str,
    signal_dtype: torch.dtype = torch.float16,
    fold_idx: int | None = None,
    n_folds: int = 1,
    chunk_idx: int | None = None,
) -> Path:
    """Single split chunk packed dict → 별도 .pt (OOM 회피 Stage 5)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_samples = int(split_dict.get("labels", torch.tensor([])).numel())

    save_dict = {
        "split": split_name,
        "data": split_dict,
        "metadata": {
            "task": "intracranial_hypertension_detection",
            "source": "MIMIC-III Waveform",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "icp_threshold": ICP_THRESHOLD,
            "sustained_sec": SUSTAINED_SEC,
            "split": split_name,
            "n_samples": n_samples,
            "fold_idx": fold_idx,
            "n_folds": n_folds,
            "chunk_idx": chunk_idx,
            "signal_dtype": str(signal_dtype).replace("torch.", ""),
        },
    }

    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    win_int = int(window_sec)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    chunk_suffix = f"_chunk{chunk_idx}" if chunk_idx is not None else ""
    filename = (
        f"intracranial_hypertension_{mode_str}_w{win_int}s_h{horizon_min}min"
        f"{fold_suffix}_{split_name}{chunk_suffix}.pt"
    )
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


def _patient_level_ich_labels(
    cases: list[dict],
    icp_threshold: float = ICP_THRESHOLD,
) -> dict[str, list[int]]:
    """환자별 case 레벨 ICH label (stratification 용).

    각 case 의 ICP 시계열에 threshold 초과 샘플 존재 여부를 binary 로 기록.
    """
    patient_to_labels: dict[str, list[int]] = {}
    for case in cases:
        pid = case["patient_id"]
        icp = case.get("signals", {}).get("icp")
        if icp is None or len(icp) == 0:
            patient_to_labels.setdefault(pid, []).append(0)
            continue
        valid = icp[~np.isnan(icp)] if hasattr(icp, "__len__") else icp
        has_high = bool(np.any(valid > icp_threshold)) if len(valid) else False
        patient_to_labels.setdefault(pid, []).append(1 if has_high else 0)
    return patient_to_labels


def prepare_ich_sweep(
    waveform_dir: str,
    input_signals: list[str],
    window_secs: list[float],
    horizon_mins: list[float],
    stride_sec: float = 30.0,
    n_folds: int = 5,
    out_dir: str = "outputs/downstream/intracranial_hypertension",
    split_mode: str = "kfold",
    signal_dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> list[Path]:
    """(window, horizon) 조합 × stratified K-fold CV 데이터셋 생성.

    split_mode:
        "kfold"        : Stratified patient-level K-fold CV (default, n_folds=5).
        "loso_export"  : 모든 샘플을 train에 저장(test 비움) — subject_ids로
                         외부에서 LeaveOneSubjectOut CV 수행용.
    """
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

    # 2. Patient-level split
    patient_ids = sorted({c["patient_id"] for c in cases})
    if split_mode == "loso_export":
        print(f"\n[2/3] LOSO export mode: all {len(patient_ids)} patients in train "
              "(test empty; subject_ids for external LOSO CV).")
        splits = [(set(patient_ids), set(), set())]
    else:
        print(f"\n[2/3] Stratified {n_folds}-fold patient-level CV...")
        patient_to_labels = _patient_level_ich_labels(cases)
        pos_pts = sum(1 for pid in patient_ids if any(patient_to_labels.get(pid, [])))
        print(
            f"  Patient-level positive (any ICP>{ICP_THRESHOLD}): "
            f"{pos_pts}/{len(patient_ids)} "
            f"({100.0 * pos_pts / max(1, len(patient_ids)):.1f}%)"
        )
        splits = stratified_kfold_patient_splits(
            patient_ids, patient_to_labels, n_folds=n_folds, seed=seed,
        )
        print(summarize_splits(splits, patient_to_labels))

    # 3. 조합 × fold 윈도우 추출 + 저장
    combos = [(w, h) for w in window_secs for h in horizon_mins]
    total_runs = len(combos) * len(splits)
    print(f"\n[3/3] Generating {len(combos)} combo × {len(splits)} fold = {total_runs} datasets...")

    saved_paths: list[Path] = []
    run_idx = 0
    for window_sec, horizon_min in combos:
        horizon_sec = horizon_min * 60.0
        for fold_idx, (train_pids, val_pids, test_pids) in enumerate(splits):
            run_idx += 1
            print(
                f"\n  [{run_idx}/{total_runs}] "
                f"window={window_sec}s, horizon={horizon_min}min, fold={fold_idx}"
            )
            train_cases = [c for c in cases if c["patient_id"] in train_pids]
            val_cases = [c for c in cases if c["patient_id"] in val_pids]
            test_cases = [c for c in cases if c["patient_id"] in test_pids]
            # OOM 회피 (Stage 5) — case-batch chunked save
            cur_fold_idx = fold_idx if len(splits) > 1 else None
            CASES_PER_CHUNK = 200

            for split_name, split_cases in (
                ("train", train_cases),
                ("val", val_cases),
                ("test", test_cases),
            ):
                if not split_cases:
                    continue
                chunk_idx = 0
                total = 0
                for batch_start in range(0, len(split_cases), CASES_PER_CHUNK):
                    case_batch = split_cases[batch_start:batch_start + CASES_PER_CHUNK]
                    samples = extract_forecast_samples(
                        case_batch, input_signals, window_sec, stride_sec, horizon_sec,
                    )
                    if not samples:
                        continue
                    packed = _consume_to_tensors_ich(samples, input_signals, signal_dtype)
                    samples.clear()
                    del samples; gc.collect()
                    n_in_chunk = int(packed.get("labels", torch.tensor([])).numel())
                    save_path = save_split_dataset(
                        packed, split_name, input_signals,
                        horizon_sec, window_sec, out_dir,
                        signal_dtype=signal_dtype,
                        fold_idx=cur_fold_idx,
                        n_folds=len(splits),
                        chunk_idx=chunk_idx,
                    )
                    saved_paths.append(save_path)
                    total += n_in_chunk
                    chunk_idx += 1
                    del packed; gc.collect()
                print(f"    {split_name.capitalize()}: {chunk_idx} chunk(s), {total} samples")

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
        "--waveform-dir", type=str,
        default="datasets/raw/mimic3-waveform-ich",
        help="MIMIC-III waveform directory (ICP 포함 레코드)",
    )
    parser.add_argument(
        "--input-signals", nargs="+",
        default=["abp", "ecg", "ppg", "co2"],
        choices=["icp", "ecg", "abp", "ppg", "co2"],
        help="Input signal types (label always from ICP). "
             "Default: ICP + ABP + ECG + PPG (4ch).",
    )
    parser.add_argument(
        "--horizon-mins", nargs="+", type=float, default=[10.0],
    )
    parser.add_argument(
        "--window-secs", nargs="+", type=float, default=[30.0],
    )
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Stratified patient-level K-fold CV (default 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for fold shuffling.",
    )
    parser.add_argument(
        "--split-mode", type=str, default="kfold",
        choices=["kfold", "loso_export"],
        help="'kfold': stratified patient-level K-fold CV. "
             "'loso_export': all in train, external LOSO CV via subject_ids.",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="outputs/downstream/intracranial_hypertension",
    )
    parser.add_argument(
        "--signal-dtype", type=str, default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for waveform tensors. fp16 halves disk/RAM peak; "
             "run.py auto-casts back to fp32 at load time.",
    )
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32}

    prepare_ich_sweep(
        waveform_dir=args.waveform_dir,
        input_signals=args.input_signals,
        window_secs=args.window_secs,
        horizon_mins=args.horizon_mins,
        stride_sec=args.stride_sec,
        n_folds=args.n_folds,
        seed=args.seed,
        out_dir=args.out_dir,
        split_mode=args.split_mode,
        signal_dtype=dtype_map[args.signal_dtype],
    )


if __name__ == "__main__":
    main()
