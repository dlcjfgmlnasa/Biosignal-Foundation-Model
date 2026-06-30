# -*- coding:utf-8 -*-
"""Imminent Cardiac Arrest Detection — 데이터 준비 (MIMIC-III, window-level).

`download_waveforms.py` 가 만든 ``download_manifest.json`` 을 읽어
**anchor-relative window-level** (input_window, label) 쌍을 생성한다.
IOH(hypotension)·ICH(intracranial_hypertension) 와 동일한 sliding-window /
horizon-sweep / stratified K-fold 구조 — 라벨 규칙만 다르다.

라벨 규칙 (Approach 1, Future Prediction):
  - center = onset (arrest+ : first_arrest_time) 또는 anchor (arrest- : risk-set
    matched anchor_time). download 가 negative 에 시간-컨텍스트 매칭된 anchor 부여.
  - 윈도우 종료 절대시각 t_end 의 onset 까지 잔여시간 tte = (center − t_end).
  - horizon h 에 대해 ``h ≤ tte ≤ h + max_lead`` 인 윈도우만 emit
    (= onset 보다 최소 h, 최대 h+max_lead 앞에서 끝나는 윈도우 → leakage 없음).
  - label = 환자 cardiac_arrest (1=arrest+, 0=arrest-). negative 윈도우도 동일
    band 에서 추출 → positive 와 시간 컨텍스트 일치(risk-set 보존).

Label 소스: download_manifest.json (환자별 center + 다운로드된 레코드 목록)
Input 소스: MIMIC waveform — ECG / PPG (비침습; 전 채널 동시 valid 구간만 사용)

데이터 소스: MIMIC-III Waveform Matched Subset (PhysioNet)
교차 일반화: 한국 데이터(VitalDB) pretrain → 미국 데이터(MIMIC-III) 평가.

사용법:
    # Canonical (10분 window, 5/15/30분 horizon) + 5-fold CV
    python -m downstream.acute_event.cardiac_arrest.prepare_data \
        --manifest datasets/raw/mimic3-waveform-cardiac-arrest/download_manifest.json \
        --waveform-dir datasets/raw/mimic3-waveform-cardiac-arrest \
        --input-signals ecg ppg \
        --window-secs 600 --horizon-mins 5 15 30 \
        --out-dir datasets/processed/cardiac_arrest
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from data.parser._common import resample_to_target
from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    _apply_filter,
    _apply_median_filter,
    _apply_range_check,
    _detect_electrocautery,
    _extract_nan_free_segments,
)
from downstream._gap_mask import (
    DEFAULT_VALID_RATIO_THRESHOLD,
    GapStats,
    apply_gap_mask_multichannel,
    compute_valid_ratio,
)
from downstream._kfold_utils import stratified_kfold_patient_splits, summarize_splits
from downstream._save_utils import add_signal_dtype_arg, consume_gap_masks
from downstream.acute_event.cardiac_arrest.download_waveforms import (
    parse_datetime_str,
)


TARGET_SR: float = 100.0

# WFDB sig_name → signal_type 매핑
# Table 3 (SSOT) #4 Cardiac Arrest: ABP, ECG, PPG, RespImp, CO₂.
# RESP — MEWS/NEWS2 의 RR component, pre-arrest apnea sign (Churpek 2014)
# CO2 — capnography. ⚠️ 새 reader 는 input_signals 전 채널이 **동시 valid** 한 구간만
#   span 으로 만든다(intersection). RespImp/CO2 를 input 에 넣으면 5-신호 동시 가용
#   레코드가 희귀해 cohort 가 크게 축소될 수 있음 → 기본은 항상 가용한 ECG/ABP/PPG.
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


# ── 데이터 구조 ──────────────────────────────────────────────


@dataclass
class CAWindowSample:
    """곧 일어날 cardiac arrest 예측용 window-level 샘플."""

    input_signals: dict[str, np.ndarray]
    input_gap_masks: dict[str, np.ndarray]  # bool, True=원본이 NaN (gap)
    label: int  # 0=arrest-, 1=arrest+
    label_value: float  # tte (window end → center, 분)
    case_id: str
    patient_id: str  # patient-level fold/cluster grouping
    win_start_sec: float  # record start 기준 윈도우 시작 (초)
    horizon_sec: float


# ── WFDB 파싱 ────────────────────────────────────────────────


def _record_base_datetime(rec, record_name: str) -> datetime | None:
    """레코드 시작 절대시각: WFDB 헤더 base_date+base_time(초 해상도) 우선,
    없으면 파일명 pXXXXXX-YYYY-MM-DD-HH-MM(분 해상도) 폴백."""
    bd = getattr(rec, "base_date", None)
    bt = getattr(rec, "base_time", None)
    if bd is not None and bt is not None:
        try:
            return datetime(
                bd.year, bd.month, bd.day,
                bt.hour, bt.minute, getattr(bt, "second", 0),
                getattr(bt, "microsecond", 0),
            )
        except Exception:
            pass
    m = re.match(r"p\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", record_name)
    if m is None:
        return None
    try:
        return datetime(*(int(m.group(i)) for i in range(1, 6)))
    except ValueError:
        return None


def _process_channel_span(
    marked: np.ndarray,   # (L,) native-sr, NaN at invalid
    start: int,
    length: int,
    cfg,
    native_sr: float,
) -> np.ndarray:
    """공통 valid span [start:start+length] 에 median+filter+resample 적용 → 100Hz.

    (_apply_pipeline 의 Step4-5 와 동일 primitive — 다만 채널이 **공통 span** 을 공유해
    타임라인 정렬이 보존된다.)
    """
    seg = marked[start:start + length].astype(np.float64)
    if cfg.median_kernel > 0:
        seg = _apply_median_filter(seg, kernel_size=cfg.median_kernel)
    seg = _apply_filter(seg, cfg, native_sr)
    if native_sr != TARGET_SR:
        seg = resample_to_target(seg, orig_sr=native_sr, target_sr=TARGET_SR)
    return seg.astype(np.float32)


def parse_record_aligned_spans(
    record_dir: Path,
    record_name: str,
    channels: list[str],
) -> list[tuple[datetime, dict[str, np.ndarray]]]:
    """WFDB 레코드 → **채널 정렬·타임라인 보존** aligned span 리스트.

    `_apply_pipeline` 은 채널마다 *가장 긴 NaN-free 세그먼트* 만 남기고 시작 offset 을
    버려, (1) sample0 ≠ rec_start 로 절대시간이 어긋나고 (2) 채널마다 offset 이 달라
    window 가 채널별로 다른 wall-clock 구간을 가리키는 버그가 있다. CA 는 onset 까지의
    절대 잔여시간으로 라벨링하므로 이 두 가지가 모두 치명적이다. 이 함수는 그 대신:

      1. 채널마다 range-check + spike → 원본 타임라인 위 NaN 마킹(전 채널 동일 length/fs).
      2. 전 channels 동시 valid 한 sample 만 valid_all (intersection).
      3. valid_all 의 연속 span(≥60s) 마다 각 채널을 **같은 [start:start+len]** 으로
         median+filter+resample → 100Hz (채널 정렬 보존).
      4. span 절대 시작시각 = base_dt + start/native_sr (offset 정확 반영).

    모든 `channels` 가 레코드에 존재해야 정렬 가능 — 하나라도 없으면 빈 리스트.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    hea_path = record_dir / f"{record_name}.hea"
    if not hea_path.exists():
        return []
    try:
        rec = wfdb.rdrecord(str(record_dir / record_name))
    except Exception:
        return []
    if rec.p_signal is None or rec.sig_len == 0:
        return []

    native_sr = float(rec.fs)
    base_dt = _record_base_datetime(rec, record_name)
    if base_dt is None:
        return []

    # WFDB sig_name → signal_type, 첫 등장 컬럼만
    type_to_col: dict[str, int] = {}
    for ch_idx, sig_name in enumerate(rec.sig_name):
        st = MIMIC_SIGNAL_MAP.get(sig_name)
        if st is not None and st not in type_to_col:
            type_to_col[st] = ch_idx
    if not all(c in type_to_col for c in channels):
        return []  # 모든 input channel 이 있어야 정렬 가능(intersection cohort)

    # 1. 채널별 range-check + spike → 마킹 + valid_all intersection
    L = int(rec.sig_len)
    valid_all = np.ones(L, dtype=bool)
    marked: dict[str, np.ndarray] = {}
    cfgs: dict[str, object] = {}
    for c in channels:
        cfg = SIGNAL_CONFIGS.get(c)
        if cfg is None:
            return []
        cfgs[c] = cfg
        data = rec.p_signal[:, type_to_col[c]].astype(np.float64)
        if cfg.valid_range is not None:
            data, _ = _apply_range_check(data, cfg.valid_range)
        if cfg.spike_detection:
            data, _ = _detect_electrocautery(
                data, native_sr, threshold_std=cfg.spike_threshold_std,
            )
        marked[c] = data
        valid_all &= ~np.isnan(data)

    # 2. valid_all 연속 span(≥60s) — proxy 로 기존 추출기 재사용
    min_samples = int(60.0 * native_sr)
    proxy = np.where(valid_all, 0.0, np.nan)
    spans = _extract_nan_free_segments(proxy, min_samples)
    if not spans:
        return []

    # 3. span 마다 채널 정렬 처리
    out: list[tuple[datetime, dict[str, np.ndarray]]] = []
    for start, seg in spans:
        length = len(seg)
        sig: dict[str, np.ndarray] = {
            c: _process_channel_span(marked[c], start, length, cfgs[c], native_sr)
            for c in channels
        }
        # resample 반올림으로 채널 간 1 sample 차 가능 → 최소 길이로 정렬
        m = min(len(v) for v in sig.values())
        sig = {k: v[:m] for k, v in sig.items()}
        span_start_dt = base_dt + timedelta(seconds=start / native_sr)
        out.append((span_start_dt, sig))
    return out


# ── 환자 로딩 (manifest-driven) ──────────────────────────────


def _patient_center(patient: dict) -> datetime | None:
    """환자의 anchor 시각: arrest+ → first_arrest_time, arrest- → anchor_time."""
    if int(patient.get("cardiac_arrest", 0)) == 1:
        return parse_datetime_str(patient.get("first_arrest_time", ""))
    # risk-set matched anchor (download 가 부여) → 없으면 icu_intime fallback
    return parse_datetime_str(
        patient.get("anchor_time") or patient.get("icu_intime", "")
    )


def load_patients_from_manifest(
    manifest_path: str,
    waveform_dir: str,
    input_signals: list[str],
) -> list[dict]:
    """download_manifest.json + 다운로드된 waveform → 환자별 **정렬 span** 로드.

    각 레코드를 parse_record_aligned_spans 로 처리해, input_signals 전 채널이 동시
    valid 한 구간만 채널 정렬된 100Hz span 으로 만든다. 모든 span 이 input 채널을
    전부 가지므로 packed tensor 가 ragged 해지지 않고(IOH/ICH stack 규약 충족),
    span 의 절대 시작시각이 정확해 onset-relative 라벨링에 누수가 없다.

    Returns
    -------
    list of {
        "patient_id": "p000020", "subject_id": int, "label": int,
        "center": datetime,
        "records": [(span_start_dt, signals_dict), ...],   # 채널 정렬 span
    }
    """
    channels = sorted(set(input_signals))

    mpath = Path(manifest_path)
    if not mpath.is_file():
        print(f"  ERROR: manifest not found: {mpath}", file=sys.stderr)
        return []
    with open(mpath, "r") as f:
        manifest = json.load(f)
    patients_meta = manifest.get("patients", [])
    print(f"  Manifest: {len(patients_meta)} patients "
          f"(horizons={manifest.get('horizons')}, pre_hours={manifest.get('pre_hours')})")
    print(f"  Align channels (모두 동시 valid 필요): {channels}")

    wf_root = Path(waveform_dir)
    patients: list[dict] = []
    n_no_center = 0
    n_no_spans = 0
    n_rec_with_spans = 0

    for pm in patients_meta:
        sid = int(pm["subject_id"])
        patient_id = f"p{sid:06d}"
        center = _patient_center(pm)
        if center is None:
            n_no_center += 1
            continue

        spans: list[tuple[datetime, dict]] = []
        for rec_path in pm.get("waveform_records", []):
            parts = rec_path.strip().split("/")
            if len(parts) < 3:
                continue
            rec_dir = wf_root / parts[0] / parts[1]
            rec_name = parts[-1]
            rec_spans = parse_record_aligned_spans(rec_dir, rec_name, channels)
            if rec_spans:
                n_rec_with_spans += 1
                spans.extend(rec_spans)

        if not spans:
            n_no_spans += 1
            continue

        patients.append({
            "patient_id": patient_id,
            "subject_id": sid,
            "label": int(pm.get("cardiac_arrest", 0)),
            "center": center,
            "records": spans,
        })

    n_pos = sum(1 for p in patients if p["label"] == 1)
    print(
        f"  Loaded {len(patients)} patients "
        f"(arrest+={n_pos}, arrest-={len(patients) - n_pos}); "
        f"skipped: no_center={n_no_center}, no_aligned_spans={n_no_spans}; "
        f"records yielding spans={n_rec_with_spans}"
    )
    return patients


# ── 윈도우 추출 ──────────────────────────────────────────────


def extract_ca_window_samples(
    patients: list[dict],
    input_signals: list[str],
    window_sec: float = 600.0,
    stride_sec: float = 30.0,
    horizon_sec: float = 900.0,
    max_lead_sec: float = 600.0,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",  # OOM 회피
) -> list[CAWindowSample]:
    """anchor-relative window-level (input, label) 쌍을 추출한다.

    윈도우 종료 절대시각 t_end 의 center 까지 잔여시간 tte = center − t_end.
    ``horizon_sec ≤ tte ≤ horizon_sec + max_lead_sec`` 인 윈도우만 emit:
      - tte ≥ horizon  → 윈도우가 onset 보다 최소 h 앞에서 끝남 (leakage 없음)
      - tte ≤ horizon + max_lead → 너무 이른 윈도우 배제 (pre-arrest band 제한)

    Gap policy (project_downstream_gap_window_policy):
      Step 1 — input window 의 multi-channel valid_ratio < threshold → drop
      Step 2 — 통과 window 의 NaN 위치는 0 으로 채우고 gap_mask boolean 저장.
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    samples: list[CAWindowSample] = []

    for patient in patients:
        center = patient["center"]
        pid = patient["patient_id"]
        label = int(patient["label"])

        for span_idx, (span_start, signals) in enumerate(patient["records"]):
            avail = [s for s in input_signals if s in signals]
            if not avail:
                continue
            min_len = min(len(signals[s]) for s in avail)
            if min_len < win_samples:
                continue

            for start in range(0, min_len - win_samples + 1, stride_samples):
                end = start + win_samples
                # 윈도우 종료 절대시각 → center 까지 잔여시간(초). span_start 는
                # parse_record_aligned_spans 가 offset 까지 반영한 정확한 절대시각.
                t_end = span_start + timedelta(seconds=end / TARGET_SR)
                tte = (center - t_end).total_seconds()
                if tte < horizon_sec or tte > horizon_sec + max_lead_sec:
                    continue

                input_dict = {
                    s: signals[s][start:end] for s in input_signals if s in signals
                }
                if not input_dict:
                    continue

                # Step 1: gap-policy window drop (multi-channel valid_ratio)
                valid_ratio = compute_valid_ratio(list(input_dict.values()))
                if valid_ratio < valid_ratio_threshold:
                    if gap_stats is not None:
                        gap_stats.add_drop()
                    continue

                # Step 2: gap mask + 즉시 dtype 캐스팅 (OOM 회피)
                filled_dict, gap_mask_dict = apply_gap_mask_multichannel(
                    input_dict, output_dtype=sample_dtype,
                )
                if gap_stats is not None:
                    n_total_s = sum(arr.size for arr in filled_dict.values())
                    n_gap_s = sum(int(m.sum()) for m in gap_mask_dict.values())
                    gap_stats.add_window(n_total_s, n_gap_s)

                samples.append(
                    CAWindowSample(
                        input_signals=filled_dict,
                        input_gap_masks=gap_mask_dict,
                        label=label,
                        label_value=tte / 60.0,
                        case_id=f"{pid}_span{span_idx}_w{start}",
                        patient_id=pid,
                        win_start_sec=start / TARGET_SR,
                        horizon_sec=horizon_sec,
                    )
                )

    return samples


# ── 저장 ─────────────────────────────────────────────────────


def _consume_to_tensors_ca(
    samples: list[CAWindowSample],
    input_signals: list[str],
    signal_dtype: torch.dtype,
) -> dict:
    """CA samples → packed dict (in-place, samples 비워짐). torch.stack 2× peak 회피."""
    if not samples:
        return {"signals": {}, "gap_masks": {},
                "labels": torch.tensor([], dtype=torch.long),
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

    gap_tensors = consume_gap_masks(samples, input_signals, attr="input_gap_masks")

    return {
        "signals": sig_tensors,
        "gap_masks": gap_tensors,
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
    max_lead_sec: float,
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
            "task": "cardiac_arrest_prediction",
            "source": "MIMIC-III Waveform Matched",
            "label": "cardiac_arrest",
            "aggregation": "window_level",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "max_lead_sec": max_lead_sec,
            "sampling_rate": TARGET_SR,
            "negative_strategy": "risk-set matching (time-from-admission)",
            "split": split_name,
            "n_samples": n_samples,
            "fold_idx": fold_idx,
            "n_folds": n_folds,
            "chunk_idx": chunk_idx,
            "signal_dtype": str(signal_dtype).replace("torch.", ""),
            "valid_ratio_threshold": DEFAULT_VALID_RATIO_THRESHOLD,
        },
    }

    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    win_int = int(window_sec)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    chunk_suffix = f"_chunk{chunk_idx}" if chunk_idx is not None else ""
    filename = (
        f"cardiac_arrest_{mode_str}_w{win_int}s_h{horizon_min}min"
        f"{fold_suffix}_{split_name}{chunk_suffix}.pt"
    )
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {save_path.name} ({file_size_mb:.1f} MB, n={n_samples})")
    return save_path


# ── 통계 출력 ────────────────────────────────────────────────


def print_stats(name: str, samples: list[CAWindowSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return
    n_total = len(samples)
    n_pos = sum(1 for s in samples if s.label == 1)
    ttes = [s.label_value for s in samples]
    print(f"  {name}: {n_total} samples")
    print(f"    arrest+: {n_pos} ({n_pos / n_total * 100:.1f}%)")
    print(
        f"    time-to-center: [{min(ttes):.1f}, {max(ttes):.1f}] min, "
        f"mean={np.mean(ttes):.1f} +/- {np.std(ttes):.1f}"
    )


# ── Sweep ────────────────────────────────────────────────────


def prepare_ca_sweep(
    manifest_path: str,
    waveform_dir: str,
    input_signals: list[str],
    window_secs: list[float],
    horizon_mins: list[float],
    stride_sec: float = 30.0,
    max_lead_sec: float = 600.0,
    n_folds: int = 5,
    out_dir: str = "datasets/processed/cardiac_arrest",
    signal_dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> list[Path]:
    """(window, horizon) 조합 × stratified K-fold CV 데이터셋 생성 (window-level)."""
    mode_str = " + ".join(s.upper() for s in input_signals)
    print(f"\n{'=' * 60}")
    print(f"  Imminent Cardiac Arrest Detection — MIMIC-III (window-level)")
    print(f"  Manifest: {manifest_path}")
    print(f"  Input:    {mode_str}")
    print(f"  Windows:  {window_secs}")
    print(f"  Horizons: {horizon_mins} min")
    print(f"  Max lead: {max_lead_sec / 60:.1f} min (윈도우 종료 band)")
    print(f"{'=' * 60}")

    # 1. 데이터 로딩 — input_signals 전 채널 동시 valid 구간만 정렬 span 으로 로드
    print("\n[1/3] Loading patients + aligned waveform spans from manifest...")
    patients = load_patients_from_manifest(
        manifest_path, waveform_dir, input_signals,
    )
    if not patients:
        print("ERROR: No patients loaded from manifest.", file=sys.stderr)
        sys.exit(1)

    # 2. Stratified patient-level K-fold CV
    patient_ids = sorted({p["patient_id"] for p in patients})
    patient_to_labels: dict[str, list[int]] = {
        p["patient_id"]: [int(p["label"])] for p in patients
    }
    pos_pts = sum(1 for pid in patient_ids if any(patient_to_labels.get(pid, [])))
    print(f"\n[2/3] Stratified {n_folds}-fold patient-level CV "
          f"(arrest+ patients: {pos_pts}/{len(patient_ids)})...")
    splits = stratified_kfold_patient_splits(
        patient_ids, patient_to_labels, n_folds=n_folds, seed=seed,
    )
    print(summarize_splits(splits, patient_to_labels))

    # 3. 조합 × fold 윈도우 추출 + 저장
    combos = [(w, h) for w in window_secs for h in horizon_mins]
    total_runs = len(combos) * len(splits)
    print(f"\n[3/3] Generating {len(combos)} combo × {len(splits)} fold "
          f"= {total_runs} datasets...")

    saved_paths: list[Path] = []
    sample_dtype_str = str(signal_dtype).replace("torch.", "")
    PATIENTS_PER_CHUNK = 200
    run_idx = 0
    for window_sec, horizon_min in combos:
        horizon_sec = horizon_min * 60.0
        for fold_idx, (train_pids, val_pids, test_pids) in enumerate(splits):
            run_idx += 1
            print(
                f"\n  [{run_idx}/{total_runs}] "
                f"window={window_sec}s, horizon={horizon_min}min, fold={fold_idx}"
            )
            cur_fold_idx = fold_idx if len(splits) > 1 else None

            for split_name, split_pids in (
                ("train", train_pids),
                ("val", val_pids),
                ("test", test_pids),
            ):
                split_patients = [p for p in patients if p["patient_id"] in split_pids]
                if not split_patients:
                    continue
                gap_stats = GapStats()
                chunk_idx = 0
                total = 0
                total_pos = 0
                for bstart in range(0, len(split_patients), PATIENTS_PER_CHUNK):
                    p_batch = split_patients[bstart:bstart + PATIENTS_PER_CHUNK]
                    samples = extract_ca_window_samples(
                        p_batch, input_signals, window_sec, stride_sec, horizon_sec,
                        max_lead_sec=max_lead_sec,
                        gap_stats=gap_stats,
                        sample_dtype=sample_dtype_str,
                    )
                    if not samples:
                        continue
                    n_pos = sum(1 for s in samples if s.label == 1)
                    packed = _consume_to_tensors_ca(samples, input_signals, signal_dtype)
                    samples.clear()
                    del samples; gc.collect()
                    n_in_chunk = int(packed.get("labels", torch.tensor([])).numel())
                    save_path = save_split_dataset(
                        packed, split_name, input_signals,
                        horizon_sec, window_sec, out_dir,
                        max_lead_sec=max_lead_sec,
                        signal_dtype=signal_dtype,
                        fold_idx=cur_fold_idx,
                        n_folds=len(splits),
                        chunk_idx=chunk_idx,
                    )
                    saved_paths.append(save_path)
                    total += n_in_chunk
                    total_pos += n_pos
                    chunk_idx += 1
                    del packed; gc.collect()
                print(f"    {split_name.capitalize()}: {chunk_idx} chunk(s), "
                      f"{total} windows (+={total_pos})")
                print(gap_stats.summary())

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(saved_paths)} files saved across "
          f"{len(combos)} combo × {len(splits)} fold")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")
    return saved_paths


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Imminent Cardiac Arrest Detection — Data Preparation "
                    "(MIMIC-III, window-level)",
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="download_manifest.json (download_waveforms.py 산출). "
             "환자별 center(first_arrest_time/anchor_time) + 레코드 목록.",
    )
    parser.add_argument(
        "--waveform-dir", type=str, required=True,
        help="다운로드된 MIMIC-III waveform 루트 (manifest 의 record path 와 합쳐짐).",
    )
    parser.add_argument(
        "--input-signals", nargs="+",
        default=["ecg", "ppg"],
        choices=["ecg", "abp", "ppg", "resp_impedance", "co2"],
        help="Input signal types. Default: ECG + PPG (비침습, 코호트 넓음). "
             "전 채널 동시 valid 구간만 span 으로 사용(intersection) — ABP/CO2 등 "
             "추가 시 동시 가용 레코드 희귀로 cohort 축소될 수 있음.",
    )
    parser.add_argument(
        "--horizon-mins", nargs="+", type=float, default=[5.0, 15.0, 30.0],
        help="Prediction horizons (분). Default 5/15/30 (Acute Event scale).",
    )
    parser.add_argument(
        "--window-secs", nargs="+", type=float, default=[600.0],
        help="Input window 길이(초). Default 600 (10분).",
    )
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument(
        "--max-lead-sec", type=float, default=600.0,
        help="윈도우 종료 band: horizon h 에 대해 h ≤ tte ≤ h+max_lead 인 윈도우만 "
             "채택. Default 600s(10분) — download pre window 와 정합.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Stratified patient-level K-fold CV (default 5).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=str, default="datasets/processed/cardiac_arrest",
    )
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    prepare_ca_sweep(
        manifest_path=args.manifest,
        waveform_dir=args.waveform_dir,
        input_signals=args.input_signals,
        window_secs=args.window_secs,
        horizon_mins=args.horizon_mins,
        stride_sec=args.stride_sec,
        max_lead_sec=args.max_lead_sec,
        n_folds=args.n_folds,
        seed=args.seed,
        out_dir=args.out_dir,
        signal_dtype=dtype_map(args.signal_dtype),
    )


if __name__ == "__main__":
    main()
