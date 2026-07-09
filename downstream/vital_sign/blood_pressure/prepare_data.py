# -*- coding:utf-8 -*-
"""Vital-Sign Task: Blood Pressure (SBP/DBP) Estimation — 데이터 준비 스크립트 (VitalDB).

**회귀(regression) task**. 입력 window(PPG, 선택적으로 +ECG)로 **같은 시점**의 수축기/
이완기 혈압(SBP/DBP)을 추정한다. horizon 없음(즉시 추정, cuffless BP).

- Label 소스: 항상 **ABP** — 입력과 동일한 window 의 ABP 파형에서 beat 단위 수축기 peak /
  이완기 trough 를 검출해 window 대표값(중앙값)으로 SBP·DBP 를 산출한다.
  ⚠ ABP 는 **입력에 넣지 않는다**(라벨 소스 → leakage). 입력은 ppg / ecg 만.
- 이 구성은 CSFM·PaPaGei·AnyPPG 등 PPG-FM 의 표준 BP 벤치마크(특히 CSFM 은 VitalDB 사용)
  와 정면 비교되도록 맞춘 것이다. CARMEN downstream 최초의 회귀 task.

품질 게이트 ([[project_downstream_gap_window_policy]]):
  - 입력 window multi-channel valid_ratio ≥ 0.8 (미달 drop, NaN→0 + gap_mask).
  - ABP window 에서 beat 최소 개수·생리 범위(SBP 70–260, DBP 30–150, PP≥10)를 통과해야
    신뢰 가능한 라벨로 채택(미달 drop).

Split: patient-level Stratified K-fold CV (default n_folds=5). Stratification 은 case 별
       고혈압 여부(median SBP ≥ 140)로 잡아 fold 간 BP 분포 균형을 맞춘다.

저장 스키마(회귀): payload 에 회귀 타깃 ``sbp``·``dbp`` (N,) 를 두고, 하위호환·stratify 용
  ``labels``(고혈압 binary)·``label_values``(=sbp) 도 함께 저장한다. 나머지(signals/
  gap_masks/case_ids)와 chunk 파일 규약은 다른 downstream task 와 동일 → 같은 로더 사용.

사용법:
    python -m downstream.vital_sign.blood_pressure.prepare_data \
        --data-dir vitaldb_pt_test --input-signals ppg \
        --window-secs 30 --n-folds 5 \
        --out-dir outputs/downstream/bp_estimation
    # ECG+PPG 입력(멀티모달 FM 비교용)
    python -m downstream.vital_sign.blood_pressure.prepare_data \
        --data-dir vitaldb_pt_test --input-signals ecg ppg --window-secs 30
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from downstream._gap_mask import (
    DEFAULT_VALID_RATIO_THRESHOLD,
    GapStats,
    apply_gap_mask_multichannel,
    compute_valid_ratio,
)
from downstream._kfold_utils import stratified_kfold_patient_splits, summarize_splits
from downstream._save_utils import add_signal_dtype_arg, consume_gap_masks, consume_input_signals
from downstream._seg_intersect import load_aligned_signals_intersection

TARGET_SR: float = 100.0

# 생리 범위 / beat 검출 파라미터.
SBP_RANGE = (70.0, 260.0)
DBP_RANGE = (30.0, 150.0)
MIN_PULSE_PRESSURE = 10.0     # SBP - DBP 최소값
ABP_PHYSIO_RANGE = (20.0, 300.0)  # 이 밖의 ABP 샘플은 artifact → NaN
HYPERTENSION_SBP = 140.0      # stratification 용 고혈압 경계
_MIN_BEATS = 3                # window 당 최소 검출 beat 수


# ---- 데이터 구조 ----


@dataclass
class BPSample:
    """단일 BP 추정 샘플 (input window → 같은 window 의 SBP/DBP)."""

    input_signals: dict[str, np.ndarray]    # {"ppg": (win_samples,), ...} NaN→0
    input_gap_masks: dict[str, np.ndarray]  # 같은 shape bool. True=원본 NaN(gap)
    sbp: float                              # 수축기 혈압 (mmHg)
    dbp: float                              # 이완기 혈압 (mmHg)
    case_id: str
    win_start_sec: float


# ---- ABP → SBP/DBP 추출 ----


def _clean_abp(abp: np.ndarray) -> np.ndarray | None:
    """ABP 를 생리 범위로 clip(범위 밖·NaN 은 NaN) 후 선형보간. 유효 부족이면 None."""
    a = np.asarray(abp, dtype=np.float64).copy()
    a[(a < ABP_PHYSIO_RANGE[0]) | (a > ABP_PHYSIO_RANGE[1])] = np.nan
    good = ~np.isnan(a)
    if good.mean() < 0.7:
        return None
    idx = np.arange(len(a))
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def extract_sbp_dbp(abp_win: np.ndarray, sr: float = TARGET_SR) -> tuple[float, float] | None:
    """ABP window → (SBP, DBP). beat 단위 peak/trough 중앙값. 신뢰 불가 시 None.

    수축기 peak: 최소 간격 0.3s(≈max 200 bpm), prominence 10 mmHg(dicrotic notch 배제).
    이완기 trough: -ABP 의 peak. 생리 범위·pulse pressure sanity 통과해야 채택.
    """
    a = _clean_abp(abp_win)
    if a is None:
        return None
    min_dist = max(1, int(0.3 * sr))
    peaks, _ = find_peaks(a, distance=min_dist, prominence=10.0)
    troughs, _ = find_peaks(-a, distance=min_dist, prominence=10.0)
    if len(peaks) < _MIN_BEATS or len(troughs) < _MIN_BEATS:
        return None
    sbp = float(np.median(a[peaks]))
    dbp = float(np.median(a[troughs]))
    if not (SBP_RANGE[0] <= sbp <= SBP_RANGE[1]):
        return None
    if not (DBP_RANGE[0] <= dbp <= DBP_RANGE[1]):
        return None
    if sbp - dbp < MIN_PULSE_PRESSURE:
        return None
    return sbp, dbp


# ---- 윈도우 추출 + 라벨링 ----


def extract_bp_samples(
    cases: list[dict],
    input_signals: list[str],
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",
) -> list[BPSample]:
    """시간 정렬된 다채널 데이터 → (input window, SBP/DBP) 샘플. horizon 없음."""
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    samples: list[BPSample] = []

    for case in cases:
        signals = case["signals"]
        abp = signals["abp"]
        n_total = len(abp)
        if n_total < win_samples:
            continue

        for start in range(0, n_total - win_samples + 1, stride_samples):
            input_dict = {
                stype: signals[stype][start : start + win_samples]
                for stype in input_signals
                if stype in signals
            }
            if not input_dict:
                continue

            # ── Step 1: 입력 gap-policy window drop ──
            valid_ratio = compute_valid_ratio(list(input_dict.values()))
            if valid_ratio < valid_ratio_threshold:
                if gap_stats is not None:
                    gap_stats.add_drop()
                continue

            # ── Step 2: 같은 window 의 ABP 에서 SBP/DBP 라벨 산출 ──
            bp = extract_sbp_dbp(abp[start : start + win_samples])
            if bp is None:
                if gap_stats is not None:
                    gap_stats.add_drop()
                continue
            sbp, dbp = bp

            # ── Step 3: gap mask + 즉시 float16 캐스팅 ──
            filled_dict, gap_mask_dict = apply_gap_mask_multichannel(
                input_dict, output_dtype=sample_dtype,
            )
            if gap_stats is not None:
                n_total_s = sum(arr.size for arr in filled_dict.values())
                n_gap_s = sum(int(m.sum()) for m in gap_mask_dict.values())
                gap_stats.add_window(n_total_s, n_gap_s)

            samples.append(
                BPSample(
                    input_signals=filled_dict,
                    input_gap_masks=gap_mask_dict,
                    sbp=sbp,
                    dbp=dbp,
                    case_id=case["case_id"],
                    win_start_sec=start / TARGET_SR,
                )
            )

    return samples


# ---- 저장 ----


def pack_samples_to_dict(
    samples: list[BPSample],
    input_signals: list[str],
    signal_dtype: torch.dtype = torch.float16,
) -> dict:
    """BPSample list → packed dict, samples in-place 해제 (peak memory 2× 회피).

    회귀 타깃 sbp·dbp 를 저장하고, 하위호환·stratify 용 labels(고혈압 binary)·
    label_values(=sbp)도 함께 둔다.
    """
    if not samples:
        return {
            "signals": {}, "gap_masks": {},
            "sbp": torch.tensor([]), "dbp": torch.tensor([]),
            "labels": torch.tensor([]), "label_values": torch.tensor([]),
            "case_ids": [],
        }
    sbp = torch.tensor([s.sbp for s in samples], dtype=torch.float32)
    dbp = torch.tensor([s.dbp for s in samples], dtype=torch.float32)
    labels = torch.tensor(
        [1 if s.sbp >= HYPERTENSION_SBP else 0 for s in samples], dtype=torch.long
    )
    case_ids = [s.case_id for s in samples]
    sig_tensors = consume_input_signals(samples, input_signals, signal_dtype)
    gap_tensors = consume_gap_masks(samples, input_signals, attr="input_gap_masks")
    samples.clear()
    gc.collect()
    return {
        "signals": sig_tensors,
        "gap_masks": gap_tensors,
        "sbp": sbp,
        "dbp": dbp,
        "labels": labels,
        "label_values": sbp.clone(),  # 범용 로더용 primary 회귀 타깃
        "case_ids": case_ids,
    }


def save_split_dataset(
    split_dict: dict,
    split_name: str,
    input_signals: list[str],
    window_sec: float,
    out_dir: str,
    signal_dtype: torch.dtype = torch.float16,
    fold_idx: int | None = None,
    n_folds: int = 1,
    chunk_idx: int | None = None,
) -> Path:
    """단일 split chunk 의 packed dict 를 .pt 로 저장 (다른 task 와 동일 규약)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_samples = int(split_dict.get("labels", torch.tensor([])).numel())

    save_dict = {
        "split": split_name,
        "data": split_dict,
        "metadata": {
            "task": "bp_estimation",
            "source": "vitaldb_pt",
            "input_signals": input_signals,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "targets": ["sbp", "dbp"],
            "label_source": "abp_beatwise_median",
            "hypertension_sbp": HYPERTENSION_SBP,
            "gap_policy": "drop+mask",
            "valid_ratio_threshold": DEFAULT_VALID_RATIO_THRESHOLD,
            "split": split_name,
            "n_samples": n_samples,
            "signal_dtype": str(signal_dtype).replace("torch.", ""),
            "fold_idx": fold_idx,
            "n_folds": n_folds,
            "chunk_idx": chunk_idx,
        },
    }

    mode_str = "_".join(input_signals)
    win_int = int(window_sec)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    chunk_suffix = f"_chunk{chunk_idx}" if chunk_idx is not None else ""
    filename = (
        f"bp_{mode_str}_w{win_int}s"
        f"{fold_suffix}_{split_name}{chunk_suffix}.pt"
    )
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {save_path.name} ({file_size_mb:.1f} MB, n={n_samples})")
    return save_path


# ---- 통계 출력 ----


def print_stats(name: str, samples: list[BPSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return
    sbp = np.array([s.sbp for s in samples])
    dbp = np.array([s.dbp for s in samples])
    n_htn = int((sbp >= HYPERTENSION_SBP).sum())
    print(f"  {name}: {len(samples)} samples")
    print(f"    SBP: {sbp.mean():.1f} +/- {sbp.std():.1f} mmHg  [{sbp.min():.0f}, {sbp.max():.0f}]")
    print(f"    DBP: {dbp.mean():.1f} +/- {dbp.std():.1f} mmHg  [{dbp.min():.0f}, {dbp.max():.0f}]")
    print(f"    Hypertensive (SBP>={HYPERTENSION_SBP:.0f}): {n_htn} ({100.0 * n_htn / len(samples):.1f}%)")


# ---- 메인 ----


def _patient_level_htn_labels(cases: list[dict]) -> dict[str, list[int]]:
    """환자별 case 레벨 고혈압 라벨 (stratification 용).

    각 case 의 ABP 전체에서 beat 단위 SBP 중앙값 ≥ HYPERTENSION_SBP → 1.
    라벨 산출 불가(신호 부족)면 0 으로 처리(보수적).
    """
    patient_to_labels: dict[str, list[int]] = {}
    for case in cases:
        pid = case["patient_id"]
        bp = extract_sbp_dbp(case["signals"]["abp"])
        htn = 1 if (bp is not None and bp[0] >= HYPERTENSION_SBP) else 0
        patient_to_labels.setdefault(pid, []).append(htn)
    return patient_to_labels


def prepare_bp_sweep(
    data_dir: str,
    input_signals: list[str],
    window_secs: list[float],
    stride_sec: float = 30.0,
    n_folds: int = 5,
    max_subjects: int | None = None,
    out_dir: str = "outputs/downstream/bp_estimation",
    required_signals: list[str] | None = None,
    signal_dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> list[Path]:
    """window sweep × patient-level Stratified K-fold CV. 각 (w) × fold → train/val/test."""
    max_window = max(window_secs)
    min_duration_sec = max_window + stride_sec

    mode_str = " + ".join(s.upper() for s in input_signals)
    print(f"\n{'=' * 60}")
    print(f"  Vital-Sign Task: Blood Pressure (SBP/DBP) Estimation — {n_folds}-fold CV")
    print(f"  Data:    {data_dir}")
    print(f"  Input:   {mode_str}   (label 소스: ABP, 입력 제외)")
    print(f"  Windows: {window_secs}")
    print(f"  N folds: {n_folds}")
    print(f"{'=' * 60}")

    # ── 1. 데이터 로딩 (1회) — ABP 는 라벨 계산에 필수 → required 자동 포함 ──
    print("\n[1/3] Loading aligned multi-channel data (once)...")
    base_req = set(required_signals if required_signals else input_signals)
    base_req.add("abp")
    cases = load_aligned_signals_intersection(
        data_dir,
        required_signals=sorted(base_req),
        min_duration_sec=min_duration_sec,
        max_subjects=max_subjects,
    )
    if not cases:
        print("ERROR: No valid cases loaded.", file=sys.stderr)
        sys.exit(1)

    # ── 2. Stratified K-fold patient-level 분할 (1회) ──
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    patient_ids = sorted({c["patient_id"] for c in cases})

    print("\n[2/3] Computing patient-level hypertension labels for stratification...")
    patient_to_labels = _patient_level_htn_labels(cases)
    pos_pts = sum(1 for pid in patient_ids if any(patient_to_labels.get(pid, [])))
    print(
        f"  Patient-level hypertensive: {pos_pts}/{len(patient_ids)} "
        f"({100.0 * pos_pts / max(1, len(patient_ids)):.1f}%)"
    )

    if n_folds == 1:
        rng = np.random.default_rng(seed)
        ids = list(patient_ids); rng.shuffle(ids)
        n_total = len(ids)
        n_train = max(1, int(n_total * 0.6))
        n_val = max(1, int(n_total * 0.2))
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
        splits = [(set(ids[:n_train]),
                   set(ids[n_train:n_train + n_val]),
                   set(ids[n_train + n_val:]))]
        print(f"  Single split (n_patients={n_total})")
    else:
        splits = stratified_kfold_patient_splits(
            patient_ids, patient_to_labels, n_folds=n_folds, seed=seed,
        )
        print(f"  Stratified {n_folds}-fold CV (n_patients={len(patient_ids)})")
    print(summarize_splits(splits, patient_to_labels))

    # ── 3. window × fold 추출 + chunked 저장 ──
    total_runs = len(window_secs) * n_folds
    print(f"\n[3/3] Generating {len(window_secs)} window × {n_folds} fold = {total_runs} datasets...")

    saved_paths: list[Path] = []
    run_idx = 0
    for window_sec in window_secs:
        for fold_idx, (train_patients, val_patients, test_patients) in enumerate(splits):
            run_idx += 1
            print(f"\n  [{run_idx}/{total_runs}] window={window_sec}s, fold={fold_idx}")

            train_cases = [c for c in cases if c["patient_id"] in train_patients]
            val_cases = [c for c in cases if c["patient_id"] in val_patients]
            test_cases = [c for c in cases if c["patient_id"] in test_patients]

            gap_stats = GapStats()
            cur_fold_idx = fold_idx if n_folds > 1 else None
            CASES_PER_CHUNK = 200

            for split_name, split_cases in (
                ("train", train_cases),
                ("val", val_cases),
                ("test", test_cases),
            ):
                if not split_cases:
                    continue
                chunk_idx = 0
                total_in_split = 0
                for batch_start in range(0, len(split_cases), CASES_PER_CHUNK):
                    case_batch = split_cases[batch_start:batch_start + CASES_PER_CHUNK]
                    samples = extract_bp_samples(
                        case_batch, input_signals, window_sec, stride_sec,
                        gap_stats=gap_stats,
                    )
                    if not samples:
                        continue
                    packed = pack_samples_to_dict(samples, input_signals, signal_dtype)
                    del samples; gc.collect()
                    n_in_chunk = int(packed.get("labels", torch.tensor([])).numel())
                    save_path = save_split_dataset(
                        packed, split_name, input_signals, window_sec, out_dir,
                        signal_dtype=signal_dtype,
                        fold_idx=cur_fold_idx,
                        n_folds=n_folds,
                        chunk_idx=chunk_idx,
                    )
                    saved_paths.append(save_path)
                    total_in_split += n_in_chunk
                    chunk_idx += 1
                    del packed; gc.collect()
                print(f"    {split_name.capitalize()}: {chunk_idx} chunk(s), {total_in_split} total samples")

            print(gap_stats.summary())

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(saved_paths)} chunk files saved to {out_dir}")
    print(f"{'=' * 60}")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vital-Sign Task: Blood Pressure (SBP/DBP) Estimation - Data Preparation (VitalDB)",
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Local .pt data directory (e.g. vitaldb_pt_test/)")
    parser.add_argument("--input-signals", nargs="+", default=["ppg"],
                        choices=["ecg", "ppg"],
                        help="Input signal types (label 은 항상 ABP 에서, ABP 는 입력 불가)")
    parser.add_argument("--window-secs", nargs="+", type=float, default=[30.0],
                        help="Input window lengths in seconds (e.g. 10 30)")
    parser.add_argument("--stride-sec", type=float, default=30.0,
                        help="Sliding window stride in seconds (기본 = 비중첩)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Patient-level K-fold CV (default 5). 1 = single 60/20/20 split.")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Max number of subjects to load (None=all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for fold/split shuffling.")
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/bp_estimation",
                        help="Output directory")
    parser.add_argument("--required-signals", nargs="+", default=None,
                        choices=["ecg", "ppg", "abp"],
                        help="Signals required for loading (paired comparison). "
                        "e.g. --required-signals ecg ppg abp → 모든 조합 동일 환자.")
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    prepare_bp_sweep(
        data_dir=args.data_dir,
        input_signals=args.input_signals,
        window_secs=args.window_secs,
        stride_sec=args.stride_sec,
        n_folds=args.n_folds,
        max_subjects=args.max_subjects,
        seed=args.seed,
        out_dir=args.out_dir,
        required_signals=args.required_signals,
        signal_dtype=dtype_map(args.signal_dtype),
    )


if __name__ == "__main__":
    main()
