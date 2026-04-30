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
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


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

    # Stage 3: ≥3.0× baseline OR ≥4.0 mg/dL
    if ratio >= 3.0 or peak_cr >= 4.0:
        return 3
    # Stage 2: 2.0-2.9× baseline
    if ratio >= 2.0:
        return 2
    # Stage 1: 1.5-1.9× baseline OR ≥0.3 mg/dL absolute increase
    if ratio >= 1.5 or inc >= abs_increase_threshold:
        return 1
    return 0


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


def extract_windows(
    signals: dict[str, np.ndarray],
    window_sec: float,
    stride_sec: float,
    sr: float = TARGET_SR,
) -> list[dict[str, np.ndarray]]:
    """다채널 신호에서 sliding window 추출. NaN 포함 윈도우는 제외."""
    win_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    min_len = min(len(v) for v in signals.values())
    if min_len < win_samples:
        return []

    out: list[dict[str, np.ndarray]] = []
    start = 0
    while start + win_samples <= min_len:
        win = {k: v[start: start + win_samples] for k, v in signals.items()}
        if not any(np.isnan(arr).any() for arr in win.values()):
            out.append(win)
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
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    label_mode: str = "binary",
    max_postop_days: float = 7.0,
    max_subjects: int | None = None,
    required_signals: list[str] | None = None,
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
    # required_set 기준으로 cohort + NaN-free window 결정 → paired comparison 일관성
    print(f"\n[3/4] Extracting windows from {len(matched_dirs)} subjects...")
    patient_data: list[dict] = []

    for i, subj_dir in enumerate(matched_dirs):
        label = labels[subj_dir.name]
        segments = load_aligned_signals_for_subject(subj_dir, required_set)

        windows: list[dict[str, np.ndarray]] = []
        for seg in segments:
            windows.extend(extract_windows(seg, window_sec, stride_sec))

        if not windows:
            continue

        # 출력은 input_signals만 — required ⊃ input일 수 있음
        if input_set != required_set:
            windows = [
                {st: w[st] for st in w.keys() if st in input_set}
                for w in windows
            ]

        target = label.aki_stage if label_mode == "stage" else label.aki_binary
        patient_data.append({
            "subject_id": subj_dir.name,
            "case_id": label.case_id,
            "label": target,
            "preop_cr": label.preop_cr,
            "peak_postop_cr": label.peak_postop_cr,
            "windows": windows,
        })

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"  [{i + 1}/{len(matched_dirs)}] {subj_dir.name}: "
                f"label={target}, n_windows={len(windows)}"
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

    # ── 4. Patient-level train/val/test 3-way split + 저장 ──
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1, got {train_ratio} + {val_ratio}"
        )
    print(
        f"\n[4/4] Patient-level split "
        f"(train={train_ratio}, val={val_ratio}, test={test_ratio:.2f}) and save..."
    )
    rng = np.random.default_rng(42)
    sids = [p["subject_id"] for p in patient_data]
    rng.shuffle(sids)
    n_total = len(sids)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    train_sids = set(sids[:n_train])
    val_sids = set(sids[n_train : n_train + n_val])
    test_sids = set(sids[n_train + n_val :])

    train_p = [p for p in patient_data if p["subject_id"] in train_sids]
    val_p = [p for p in patient_data if p["subject_id"] in val_sids]
    test_p = [p for p in patient_data if p["subject_id"] in test_sids]

    def _pack(plist: list[dict]) -> list[dict]:
        packed = []
        for p in plist:
            sig_types = sorted(p["windows"][0].keys())
            sig_tensors = {
                st: torch.stack([torch.from_numpy(w[st]).float() for w in p["windows"]])
                for st in sig_types
            }
            packed.append({
                "subject_id": p["subject_id"],
                "case_id": p["case_id"],
                "label": p["label"],
                "preop_cr": p["preop_cr"],
                "peak_postop_cr": p["peak_postop_cr"],
                "n_windows": len(p["windows"]),
                "signals": sig_tensors,  # {sig_type: (K, win_samples)}
            })
        return packed

    save_dict = {
        "train": _pack(train_p),
        "val": _pack(val_p),
        "test": _pack(test_p),
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
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "n_train_patients": len(train_p),
            "n_val_patients": len(val_p),
            "n_test_patients": len(test_p),
        },
    }
    if label_mode == "binary":
        save_dict["metadata"]["n_pos_train"] = sum(1 for p in train_p if p["label"] == 1)
        save_dict["metadata"]["n_pos_val"] = sum(1 for p in val_p if p["label"] == 1)
        save_dict["metadata"]["n_pos_test"] = sum(1 for p in test_p if p["label"] == 1)

    sig_str = "_".join(sorted(input_signals))
    out_file = out_path / f"aki_{label_mode}_{sig_str}_w{int(window_sec)}s.pt"
    torch.save(save_dict, out_file)

    print(f"\n{'=' * 60}")
    print(f"  Saved: {out_file}")
    print(f"  Train: {len(train_p)} patients")
    print(f"  Val:   {len(val_p)} patients")
    print(f"  Test:  {len(test_p)} patients")
    if label_mode == "binary":
        print(
            f"  Class balance — train: AKI={save_dict['metadata']['n_pos_train']}, "
            f"val: AKI={save_dict['metadata']['n_pos_val']}, "
            f"test: AKI={save_dict['metadata']['n_pos_test']}"
        )
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
        "--train-ratio",
        type=float,
        default=0.6,
        help="Train split ratio (patient-level). Default 0.6.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Val split ratio (patient-level). Test = 1 - train - val. Default 0.2.",
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
        "--out-dir", default="datasets/processed/aki",
    )
    args = parser.parse_args()

    prepare_aki_dataset(
        data_dir=args.data_dir,
        clinical_csv=args.clinical_csv,
        lab_csv=args.lab_csv,
        out_dir=args.out_dir,
        input_signals=args.input_signals,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        label_mode=args.label_mode,
        max_postop_days=args.max_postop_days,
        max_subjects=args.max_subjects,
        required_signals=args.required_signals,
    )


if __name__ == "__main__":
    main()
