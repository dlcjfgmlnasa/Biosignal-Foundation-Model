# -*- coding:utf-8 -*-
"""Postoperative AKI Prediction — 데이터 준비 (VitalDB).

VitalDB intraop waveform + clinical_data.csv + lab_data.csv → 환자 단위 .pt 파일.

라벨 (KDIGO Cr 기준; VitalDB 공식 예제 mbp_aki.ipynb / xgb_aki.ipynb 따름):
  Stage 1: peak postop Cr ≥ 1.5× preop Cr  OR  Δ ≥ 0.3 mg/dL
  Stage 2: peak postop Cr ≥ 2.0× preop Cr
  Stage 3: peak postop Cr ≥ 3.0× preop Cr  OR  peak ≥ 4.0 mg/dL
  Binary: AKI = stage ≥ 1
  Postop window: opend < dt < opend + 7 days (KDIGO 표준)

VitalDB CSV 스키마 (확인됨):
  clinical_data.csv : caseid, preop_cr, opend (seconds), aneend, ...
  lab_data.csv      : caseid, dt (seconds, case file 시작 기준), name, result
                      creatinine은 name == 'cr' (소문자)
  ※ dt는 수술 시작이 아니라 *case 시작* 기준이므로 반드시 opend로 postop 구간 잘라야 함

입력 신호: K-MIMIC pretrain overlap (ABP, ECG, PPG, CVP)
  → EEG/AWP/CO2/PAP/ICP는 K-MIMIC overlap 부분만 채택

사용법:
    python -m downstream.outcome.aki.prepare_data \
        --data-dir <vitaldb 파싱 .pt 디렉토리> \
        --clinical-csv clinical_data.csv \
        --lab-csv lab_data.csv \
        --window-sec 600 --stride-sec 300 \
        --out-dir datasets/processed/aki
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from downstream._gap_mask import (
    DEFAULT_VALID_RATIO_THRESHOLD,
    GapStats,
    apply_gap_mask_multichannel,
    compute_valid_ratio,
)
from downstream._kfold_utils import stratified_kfold_patient_splits, summarize_splits
from downstream._save_utils import (
    add_signal_dtype_arg,
    stack_window_dicts_destructive,
)


TARGET_SR: float = 100.0

DEFAULT_INPUT_SIGNALS: list[str] = ["abp", "ecg", "ppg", "cvp"]


# ---- VitalDB 임상 라벨 로딩 ----


@dataclass
class CaseLabel:
    """한 case의 AKI 라벨 + 메타."""

    case_id: int
    subject_id: str  # "VDB_xxxx"
    preop_cr: float  # mg/dL
    peak_postop_cr: float  # mg/dL (within max_postop_days)
    abs_increase: float  # peak_postop_cr - preop_cr
    ratio: float  # peak_postop_cr / preop_cr
    aki_stage: int  # 0/1/2/3
    aki_binary: int  # 0 or 1


def load_preop_and_opend(
    clinical_csv: str,
) -> dict[int, tuple[float, float]]:
    """clinical_data.csv → {caseid: (preop_cr, opend_sec)}.

    필수 컬럼: caseid, preop_cr, opend
      opend: case file 시작 기준 수술 종료 시각 (초). dt 비교용 anchor.
      ※ aneend도 가능하나 공식 xgb_aki 예제는 opend 사용 → 일관성 위해 opend 채택.
    """
    out: dict[int, tuple[float, float]] = {}
    with open(clinical_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for col in ("caseid", "preop_cr", "opend"):
            if col not in reader.fieldnames:
                print(
                    f"ERROR: clinical CSV must have 'caseid', 'preop_cr', 'opend'. "
                    f"Found: {reader.fieldnames}",
                    file=sys.stderr,
                )
                sys.exit(1)
        for row in reader:
            try:
                caseid = int(row["caseid"])
                preop_cr = float(row["preop_cr"])
                opend = float(row["opend"])
            except (ValueError, TypeError):
                continue
            if preop_cr <= 0 or preop_cr > 20:  # 비정상 값 제외
                continue
            if opend <= 0:
                continue
            out[caseid] = (preop_cr, opend)
    return out


def load_postop_peak_cr(
    lab_csv: str,
    case_to_opend: dict[int, float],
    max_postop_days: float = 7.0,
) -> dict[int, tuple[float, float]]:
    """lab_data.csv → {caseid: (peak postop Cr, hours after opend)}.

    필수 컬럼: caseid, dt, name, result
      dt: case file 시작 기준 (초). 수술 시작 아님!
      name: VitalDB는 creatinine을 'cr' (소문자)로 기록 (공식 예제 확인).
      result: 수치 (mg/dL)

    Postop 구간: opend < dt < opend + max_postop_days*86400 (KDIGO 7일 표준)
    """
    by_case: dict[int, list[tuple[float, float]]] = defaultdict(list)
    max_postop_sec = max_postop_days * 86400.0

    with open(lab_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for col in ("caseid", "dt", "name", "result"):
            if col not in reader.fieldnames:
                print(
                    f"ERROR: lab CSV must have caseid/dt/name/result. "
                    f"Found: {reader.fieldnames}",
                    file=sys.stderr,
                )
                sys.exit(1)

        for row in reader:
            if str(row["name"]).strip().lower() != "cr":
                continue
            try:
                caseid = int(row["caseid"])
            except (ValueError, TypeError):
                continue
            if caseid not in case_to_opend:
                continue
            try:
                dt_sec = float(row["dt"])
                cr = float(row["result"])
            except (ValueError, TypeError):
                continue
            if cr <= 0 or cr > 20:
                continue
            opend = case_to_opend[caseid]
            if dt_sec <= opend or dt_sec > opend + max_postop_sec:
                continue
            by_case[caseid].append((dt_sec - opend, cr))  # opend 기준 hours

    peak: dict[int, tuple[float, float]] = {}
    for caseid, entries in by_case.items():
        peak_cr = max(cr for _, cr in entries)
        peak_offset_sec = next(off for off, cr in entries if cr == peak_cr)
        peak[caseid] = (peak_cr, peak_offset_sec / 3600.0)
    return peak


def compute_kdigo_stage(
    preop_cr: float,
    peak_cr: float,
    abs_increase_threshold: float = 0.3,
) -> int:
    """KDIGO Cr 기준만으로 AKI stage 계산 (UO 기준 미사용).

    Returns: 0 (no AKI) / 1 / 2 / 3
    """
    ratio = peak_cr / preop_cr
    inc = peak_cr - preop_cr

    # KDIGO: 먼저 AKI(급성 손상) 정의를 충족해야 한다.
    #   acute rise ≥0.3 mg/dL OR ≥1.5× baseline.
    # 이 조건 없이 peak_cr≥4.0 만으로 Stage 3 을 매기면, baseline 이 높은
    # 안정적 CKD 환자(예: preop_cr=4.2, ratio≈1.0)가 AKI 로 over-call 된다.
    # KDIGO 의 "increase to ≥4.0 mg/dL" 도 AKI 충족을 전제로 한 Stage 3 기준이다.
    is_aki = ratio >= 1.5 or inc >= abs_increase_threshold
    if not is_aki:
        return 0

    # AKI 확정 후 중증도 분류:
    # Stage 3: ≥3.0× baseline OR peak ≥4.0 mg/dL
    if ratio >= 3.0 or peak_cr >= 4.0:
        return 3
    # Stage 2: 2.0-2.9× baseline
    if ratio >= 2.0:
        return 2
    # Stage 1: 1.5-1.9× baseline OR ≥0.3 mg/dL absolute increase
    return 1


def build_aki_labels(
    clinical_csv: str,
    lab_csv: str,
    max_postop_days: float = 7.0,
    abs_increase_threshold: float = 0.3,
) -> dict[str, CaseLabel]:
    """clinical + lab CSV → {subject_id: CaseLabel}.

    subject_id 포맷은 vitaldb 파서와 동일: ``VDB_{caseid:04d}``.
    """
    print(f"  Loading preop_cr + opend: {clinical_csv}")
    case_meta = load_preop_and_opend(clinical_csv)
    print(f"    {len(case_meta)} cases with valid preop_cr and opend")

    print(f"  Loading postop creatinine (name=='cr', dt > opend): {lab_csv}")
    peak_map = load_postop_peak_cr(
        lab_csv,
        {cid: opend for cid, (_, opend) in case_meta.items()},
        max_postop_days=max_postop_days,
    )
    print(f"    {len(peak_map)} cases with postop creatinine measurement")

    labels: dict[str, CaseLabel] = {}
    for caseid, (preop_cr, _opend) in case_meta.items():
        if caseid not in peak_map:
            continue
        peak_cr, _peak_h = peak_map[caseid]
        stage = compute_kdigo_stage(preop_cr, peak_cr, abs_increase_threshold)
        labels[f"VDB_{caseid:04d}"] = CaseLabel(
            case_id=caseid,
            subject_id=f"VDB_{caseid:04d}",
            preop_cr=preop_cr,
            peak_postop_cr=peak_cr,
            abs_increase=peak_cr - preop_cr,
            ratio=peak_cr / preop_cr,
            aki_stage=stage,
            aki_binary=int(stage >= 1),
        )

    n_aki = sum(1 for v in labels.values() if v.aki_binary == 1)
    print(
        f"  AKI labels built: {len(labels)} cases "
        f"(AKI={n_aki}, no-AKI={len(labels) - n_aki}, "
        f"prevalence={n_aki / max(len(labels), 1) * 100:.1f}%)"
    )
    by_stage = {s: sum(1 for v in labels.values() if v.aki_stage == s) for s in range(4)}
    print(f"  Stage distribution: {by_stage}")
    return labels


# ---- 파싱된 .pt waveform 로더 ----


def _parse_pt_filename(name: str) -> dict | None:
    """vitaldb 파서 출력 파일명에서 메타 추출.

    형식: ``{subject_id}_S{session}_{signal_name}_{spatial}_seg{i}_{j}.pt``
    """
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


def load_aligned_signals_for_subject(
    subj_dir: Path,
    required_signals: set[str],
) -> list[dict[str, np.ndarray]]:
    """한 subject의 시간 정렬된 다채널 segment들을 로드.

    같은 (session_id, seg_i, seg_j) 키에 모든 required_signals가 있을 때만 채택.
    각 segment는 모든 신호의 최소 길이로 잘라 정렬.

    Returns
    -------
    list of {"abp": (T,), "ecg": (T,), ...} ndarray dicts
    """
    file_map: dict[tuple[int, int, int], dict[str, Path]] = defaultdict(dict)
    for pt in subj_dir.glob("*.pt"):
        meta = _parse_pt_filename(pt.name)
        if meta is None:
            continue
        if meta["signal_type"] not in required_signals:
            continue
        key = (meta["session_id"], meta["seg_i"], meta["seg_j"])
        file_map[key][meta["signal_type"]] = pt

    segments: list[dict[str, np.ndarray]] = []
    for _seg_key, type_paths in file_map.items():
        if not required_signals.issubset(type_paths.keys()):
            continue
        sigs: dict[str, np.ndarray] = {}
        for stype, path in type_paths.items():
            t = torch.load(path, weights_only=True)  # (1, T) 또는 (T,)
            sigs[stype] = t.squeeze(0).numpy() if t.ndim == 2 else t.numpy()
        min_len = min(len(s) for s in sigs.values())
        sigs = {k: v[:min_len].astype(np.float32) for k, v in sigs.items()}
        segments.append(sigs)
    return segments


def _max_consecutive_nan(arr: np.ndarray) -> int:
    """배열에서 연속 NaN의 최장 길이 (samples)."""
    is_nan = np.isnan(arr)
    if not is_nan.any():
        return 0
    # 연속 구간 길이 계산
    diff = np.diff(np.concatenate([[0], is_nan.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int((ends - starts).max()) if len(starts) else 0


def _linear_interpolate_nan(arr: np.ndarray) -> np.ndarray:
    """짧은 NaN 구간을 선형 보간 (양 끝 NaN은 forward/back fill)."""
    out = arr.astype(np.float32, copy=True)
    is_nan = np.isnan(out)
    if not is_nan.any():
        return out
    idx = np.arange(len(out))
    valid = ~is_nan
    if valid.sum() == 0:
        return out  # 전부 NaN — interpolation 불가
    out[is_nan] = np.interp(idx[is_nan], idx[valid], out[valid])
    return out


def extract_windows(
    signals: dict[str, np.ndarray],
    window_sec: float,
    stride_sec: float,
    sr: float = TARGET_SR,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",  # OOM 회피
) -> list[tuple[float, dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """다채널 sliding window 추출 + gap drop+mask 정책.

    정책 ([[project_downstream_gap_window_policy]]):
      Step 1 — multi-channel valid_ratio < threshold 면 window drop
      Step 2 — 통과 window 의 NaN 위치 → 0 fill + bool gap_mask

    이전 interpolation 정책 폐기 — pretrain mask_token 과 일관성 위해 [MASK] 교체.

    Returns
    -------
    list of (start_sec, signals_filled, gap_masks).
    """
    win_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    min_len = min(len(v) for v in signals.values())
    if min_len < win_samples:
        return []

    out: list[tuple[float, dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    start = 0
    while start + win_samples <= min_len:
        win = {k: v[start: start + win_samples] for k, v in signals.items()}

        # Step 1: gap-policy window drop
        valid_ratio = compute_valid_ratio(list(win.values()))
        if valid_ratio < valid_ratio_threshold:
            if gap_stats is not None:
                gap_stats.add_drop()
            start += stride_samples
            continue

        # Step 2: gap mask + 즉시 float16 캐스팅 (메모리 50% 절감)
        filled, gap_mask = apply_gap_mask_multichannel(win, output_dtype=sample_dtype)
        if gap_stats is not None:
            n_total_s = sum(arr.size for arr in filled.values())
            n_gap_s = sum(int(m.sum()) for m in gap_mask.values())
            gap_stats.add_window(n_total_s, n_gap_s)

        out.append((start / sr, filled, gap_mask))
        start += stride_samples
    return out


# ---- 메인 ----


def prepare_aki_dataset(
    data_dir: str,
    clinical_csv: str,
    lab_csv: str,
    out_dir: str,
    input_signals: list[str],
    window_sec: float = 600.0,
    stride_sec: float = 300.0,
    n_folds: int = 5,
    label_mode: str = "binary",
    max_postop_days: float = 7.0,
    max_subjects: int | None = None,
    required_signals: list[str] | None = None,
    min_windows_per_patient: int = 3,
    signal_dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> None:
    """AKI prediction 데이터셋을 패치(환자) 단위로 빌드."""
    if label_mode not in {"binary", "stage"}:
        print(f"ERROR: label-mode must be 'binary' or 'stage', got {label_mode}")
        sys.exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # paired comparison: required_signals 기준으로 cohort + window 결정,
    # 출력 .pt에는 input_signals만 저장. None이면 input_signals와 동일.
    if required_signals is None:
        required_set = set(input_signals)
    else:
        required_set = set(required_signals) | set(input_signals)
    input_set = set(input_signals)

    print(f"\n{'=' * 60}")
    print(f"  Postoperative AKI Prediction — Data Preparation")
    print(f"  Data dir:    {data_dir}")
    print(f"  Clinical:    {clinical_csv}")
    print(f"  Lab:         {lab_csv}")
    print(f"  Inputs:      {sorted(input_set)}")
    print(f"  Required:    {sorted(required_set)}")
    print(f"  Window:      {window_sec}s, Stride: {stride_sec}s")
    print(f"  Label mode:  {label_mode}")
    print(f"  Postop win:  {max_postop_days} days")
    print(f"{'=' * 60}\n")

    # ── 1. AKI 라벨 빌드 ──
    print("[1/4] Building AKI labels...")
    labels = build_aki_labels(
        clinical_csv, lab_csv, max_postop_days=max_postop_days
    )
    if not labels:
        print("ERROR: no AKI labels built (check CSV columns).", file=sys.stderr)
        sys.exit(1)

    # ── 2. 파싱된 .pt에서 라벨 매칭 가능한 subject 디렉토리 검색 ──
    print(f"\n[2/4] Scanning waveform subject dirs in {data_dir}...")
    root = Path(data_dir)
    if not root.is_dir():
        print(f"ERROR: data dir not found: {root}", file=sys.stderr)
        sys.exit(1)

    subject_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    matched_dirs = [d for d in subject_dirs if d.name in labels]
    print(
        f"  Found {len(subject_dirs)} subject dirs, "
        f"{len(matched_dirs)} matched with AKI labels"
    )
    if not matched_dirs:
        print(
            "ERROR: no overlap between waveform subjects and AKI labels. "
            "Check subject_id format (expected VDB_xxxx).",
            file=sys.stderr,
        )
        sys.exit(1)

    if max_subjects is not None:
        # AKI/non-AKI 양쪽에서 비례 샘플링
        aki_dirs = [d for d in matched_dirs if labels[d.name].aki_binary == 1]
        non_dirs = [d for d in matched_dirs if labels[d.name].aki_binary == 0]
        n_aki = min(len(aki_dirs), max_subjects // 2)
        n_non = min(len(non_dirs), max_subjects - n_aki)
        matched_dirs = aki_dirs[:n_aki] + non_dirs[:n_non]
        print(f"  Limited to {len(matched_dirs)} ({n_aki} AKI + {n_non} non-AKI)")

    # ── 3. 각 subject별로 윈도우 추출 ──
    # required_set 기준으로 cohort + NaN policy → paired comparison 일관성
    # NaN 정책: ratio ≤ 5%, max gap ≤ 1s, 통과 시 선형 보간
    print(f"\n[3/4] Extracting windows from {len(matched_dirs)} subjects...")
    print(
        f"  NaN policy: ratio<=5% AND max_gap<=1s, "
        f"passing windows linearly interpolated"
    )
    patient_data: list[dict] = []
    gap_stats_global = GapStats()

    for i, subj_dir in enumerate(matched_dirs):
        label = labels[subj_dir.name]
        segments = load_aligned_signals_for_subject(subj_dir, required_set)

        # 각 segment 내에서 sliding window 추출. seg_offset_sec로 segment간 시간 분리.
        # gap drop+mask 정책 적용 (NaN → 0 fill + bool gap_mask).
        windowed: list[tuple[float, dict[str, np.ndarray], dict[str, np.ndarray]]] = []
        seg_offset_sec = 0.0
        for seg in segments:
            seg_len_sec = (
                min(len(v) for v in seg.values()) / TARGET_SR if seg else 0.0
            )
            for rel_sec, win, gap_mask in extract_windows(
                seg, window_sec, stride_sec, gap_stats=gap_stats_global,
            ):
                windowed.append((seg_offset_sec + rel_sec, win, gap_mask))
            # segment끼리 인접 윈도우로 보이지 않도록 offset 누적 + window_sec margin
            seg_offset_sec += seg_len_sec + window_sec

        if len(windowed) < min_windows_per_patient:
            continue

        # 출력은 input_signals만 — required ⊃ input일 수 있음
        if input_set != required_set:
            windowed = [
                (
                    t,
                    {st: w[st] for st in w.keys() if st in input_set},
                    {st: gm[st] for st in gm.keys() if st in input_set},
                )
                for t, w, gm in windowed
            ]

        target = label.aki_stage if label_mode == "stage" else label.aki_binary
        patient_data.append({
            "subject_id": subj_dir.name,
            "case_id": label.case_id,
            "label": target,
            "preop_cr": label.preop_cr,
            "peak_postop_cr": label.peak_postop_cr,
            "start_secs": [t for t, _, _ in windowed],
            "windows": [w for _, w, _ in windowed],
            "gap_masks": [gm for _, _, gm in windowed],
        })

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"  [{i + 1}/{len(matched_dirs)}] {subj_dir.name}: "
                f"label={target}, n_windows={len(windowed)}"
            )

    if not patient_data:
        print("ERROR: no patients with extractable windows.", file=sys.stderr)
        sys.exit(1)

    # 통계
    if label_mode == "binary":
        n_pos = sum(1 for p in patient_data if p["label"] == 1)
        print(
            f"\n  Patients with windows: {len(patient_data)} "
            f"(AKI={n_pos}, no-AKI={len(patient_data) - n_pos})"
        )
    else:
        by_stage = {s: sum(1 for p in patient_data if p["label"] == s) for s in range(4)}
        print(f"\n  Patients with windows: {len(patient_data)}, stages: {by_stage}")
    total_windows = sum(len(p["windows"]) for p in patient_data)
    print(
        f"  Total windows: {total_windows}, "
        f"avg/patient: {total_windows / len(patient_data):.1f}"
    )

    # ── 4. Stratified patient-level K-fold CV + 저장 ──
    print(f"\n[4/4] Stratified {n_folds}-fold patient-level CV...")
    sid_to_label: dict[str, int] = {p["subject_id"]: int(p["label"]) for p in patient_data}
    patient_to_labels: dict[str, list[int]] = {sid: [lb] for sid, lb in sid_to_label.items()}
    patient_ids = sorted(patient_to_labels.keys())
    splits = stratified_kfold_patient_splits(
        patient_ids, patient_to_labels, n_folds=n_folds, seed=seed,
    )
    print(summarize_splits(splits, patient_to_labels))
    print(gap_stats_global.summary())

    def _pack(plist: list[dict]) -> list[dict]:
        packed = []
        for p in plist:
            n_w = len(p["windows"])
            sig_tensors = stack_window_dicts_destructive(
                p["windows"], signal_dtype
            )
            # gap_masks 도 stack (bool, K x win_samples)
            gap_tensors: dict[str, torch.Tensor] = {}
            for stype in input_set:
                arrs = [gm[stype] for gm in p["gap_masks"] if stype in gm]
                if arrs:
                    gap_tensors[stype] = torch.from_numpy(np.stack(arrs)).bool()
            start_secs = torch.tensor(p["start_secs"], dtype=torch.float32)
            packed.append({
                "subject_id": p["subject_id"],
                "case_id": p["case_id"],
                "label": p["label"],
                "preop_cr": p["preop_cr"],
                "peak_postop_cr": p["peak_postop_cr"],
                "n_windows": n_w,
                "signals": sig_tensors,
                "gap_masks": gap_tensors,
                "start_secs": start_secs,
            })
        return packed

    sig_str = "_".join(sorted(input_signals))

    def _shallow_copy(plist):
        return [
            {
                **p,
                "windows": [dict(w) for w in p["windows"]],
                "gap_masks": [dict(gm) for gm in p["gap_masks"]],
            }
            for p in plist
        ]

    PATIENTS_PER_CHUNK = 100

    for fold_idx, (train_sids, val_sids, test_sids) in enumerate(splits):
        print(f"\n  [Fold {fold_idx}] Patient-batch chunked save (Stage 5)...")
        for split_name, split_sids in (
            ("train", train_sids), ("val", val_sids), ("test", test_sids),
        ):
            split_p_all = [p for p in patient_data if p["subject_id"] in split_sids]
            if not split_p_all:
                continue
            chunk_idx = 0
            total_pat = 0
            total_pos = 0
            for bstart in range(0, len(split_p_all), PATIENTS_PER_CHUNK):
                p_batch = split_p_all[bstart:bstart + PATIENTS_PER_CHUNK]
                n_p = len(p_batch)
                n_pos = sum(1 for p in p_batch if p["label"] == 1)
                split_packed = _pack(_shallow_copy(p_batch))
                del p_batch; gc.collect()

                save_dict = {
                    "split": split_name,
                    "data": split_packed,
                    "metadata": {
                        "task": "postop_aki_prediction",
                        "source": "VitalDB intraop waveform + clinical/lab",
                        "label_mode": label_mode,
                        "kdigo_definition": (
                            "Stage based on postop peak Cr / preop Cr ratio "
                            "(≥1.5 = stage1, ≥2.0 = stage2, ≥3.0 or ≥4.0 mg/dL = stage3) "
                            "or absolute increase ≥0.3 mg/dL for stage 1."
                        ),
                        "input_signals": sorted(input_set),
                        "required_signals": sorted(required_set),
                        "window_sec": window_sec,
                        "stride_sec": stride_sec,
                        "sampling_rate": TARGET_SR,
                        "max_postop_days": max_postop_days,
                        "fold_idx": fold_idx,
                        "n_folds": n_folds,
                        "split": split_name,
                        "chunk_idx": chunk_idx,
                        "n_patients": n_p,
                        "n_pos": n_pos if label_mode == "binary" else None,
                        "signal_dtype": str(signal_dtype).replace("torch.", ""),
                    },
                }
                fold_suffix = f"_fold{fold_idx}" if n_folds > 1 else ""
                out_file = (
                    out_path / f"aki_{label_mode}_{sig_str}_w{int(window_sec)}s"
                    f"{fold_suffix}_{split_name}_chunk{chunk_idx}.pt"
                )
                torch.save(save_dict, out_file)
                total_pat += n_p
                total_pos += n_pos if label_mode == "binary" else 0
                chunk_idx += 1
                del save_dict, split_packed; gc.collect()
            print(
                f"    {split_name}: {chunk_idx} chunk(s), {total_pat} patients (+={total_pos})"
            )

    print(f"\n{'=' * 60}")
    print(f"  Done: {n_folds} fold(s) saved to {out_path}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Postop AKI Prediction — Data Prep")
    parser.add_argument(
        "--data-dir", required=True,
        help="파싱된 VitalDB .pt 디렉토리 (per-subject dirs containing *.pt)",
    )
    parser.add_argument(
        "--clinical-csv", required=True,
        help="VitalDB clinical_info CSV (caseid, preop_cr 컬럼 필수)",
    )
    parser.add_argument(
        "--lab-csv", required=True,
        help="VitalDB lab CSV (caseid, dt, name, result 컬럼 필수)",
    )
    parser.add_argument(
        "--input-signals", nargs="+", default=DEFAULT_INPUT_SIGNALS,
        help="입력 신호 (K-MIMIC pretrain overlap: abp ecg ppg cvp)",
    )
    parser.add_argument(
        "--required-signals", nargs="+", default=None,
        help="Paired comparison용 required cohort 신호. "
        "지정 시 모든 sweep이 동일 환자/윈도우 풀 사용. "
        "예: --required-signals abp ecg ppg",
    )
    parser.add_argument("--window-sec", type=float, default=600.0)
    parser.add_argument("--stride-sec", type=float, default=300.0)
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Stratified patient-level K-fold CV (default 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for fold shuffling.",
    )
    parser.add_argument(
        "--label-mode", choices=["binary", "stage"], default="binary",
        help="binary=AKI vs no, stage=KDIGO 0/1/2/3",
    )
    parser.add_argument(
        "--max-postop-days", type=float, default=7.0,
        help="postop AKI 정의 윈도우 (일). KDIGO 기본 7일.",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--min-windows-per-patient",
        type=int,
        default=3,
        help="환자당 최소 윈도우 수. 미만 환자는 cohort에서 제외.",
    )
    parser.add_argument(
        "--out-dir", default="datasets/processed/aki",
    )
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    prepare_aki_dataset(
        data_dir=args.data_dir,
        clinical_csv=args.clinical_csv,
        lab_csv=args.lab_csv,
        out_dir=args.out_dir,
        input_signals=args.input_signals,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        n_folds=args.n_folds,
        seed=args.seed,
        label_mode=args.label_mode,
        max_postop_days=args.max_postop_days,
        max_subjects=args.max_subjects,
        required_signals=args.required_signals,
        min_windows_per_patient=args.min_windows_per_patient,
        signal_dtype=dtype_map(args.signal_dtype),
    )


if __name__ == "__main__":
    main()
