# -*- coding:utf-8 -*-
"""Intraoperative Hypoxemia Prediction — 데이터 준비 (VitalDB primary).

미래 horizon 구간 SpO2 < 92% (≥1분 지속) 예측을 위한 (input_window, future_label) 쌍 생성.
SpO2 임계 92%: BTF/WHO 중재 수준이자 수술중 hypoxemia 예측 표준(Lundberg 2018, ≤92).

Label 소스: SpO2 trend (raw .vital → vitaldb library 로 PLETH_SPO2 1Hz 추출)
Input 소스: parsed .pt 디렉토리의 wave (ECG/PPG/CO2/AWP/ABP)

Hypotension prepare_data.py 와 같은 sweep 구조 (window×horizon×signal-combo, paired comparison).

시간축 정렬 (2026-07-10 수정):
    parser 는 `.vital` 을 dtstart 부터 100Hz 로 리샘플한 뒤 NaN-free segment 로 쪼개고,
    각 recording 의 `start_sample`(=dtstart 로부터의 100Hz 절대 인덱스)을 manifest 에
    기록한다. SpO2 trend 도 `to_numpy(track, 1.0)` 로 **같은 dtstart 원점**의 1Hz 격자다.
    따라서 case-내 offset `start` 의 절대 시각(초) = (start_sample + start) / 100.
    구 코드는 segment 가 raw t=0 에서 시작한다고 가정해 라벨이 통째로 어긋났다.

Baseline 누수 가드:
    예측 시점에 이미 SpO2 < threshold 인 윈도우는 기본 제외(`--exclude-already-low`).
    포함하면 "이미 저산소 → 계속 저산소" 지속성만으로 양성이 맞아 성능이 부풀려진다
    (IOH 의 '현재-저혈압 제외' 와 같은 계열, Yang et al. BJA 2025).

사용법:
    python -m downstream.acute_event.hypoxemia.prepare_data \\
        --data-dir <parsed .pt 디렉토리> \\
        --raw-dir <raw vitaldb .vital 디렉토리> \\
        --input-signals ecg ppg co2 awp \\
        --window-secs 300 --horizon-mins 5 10 15 \\
        --out-dir outputs/downstream/hypoxemia
"""

from __future__ import annotations

import argparse
import gc
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from downstream._save_utils import add_signal_dtype_arg, consume_gap_masks, consume_input_signals
from downstream._seg_intersect import load_aligned_signals_intersection

TARGET_SR: float = 100.0
SPO2_TRACKS = [
    "Solar8000/PLETH_SPO2",       # VitalDB OR 표준 (1Hz numeric)
    "Intellivue/PLETH_SAT_O2",    # K-MIMIC ICU
    "Solar8000/PLETH_SAT_O2",
]


# ---- 데이터 구조 ----


@dataclass
class ForecastSample:
    """미래 hypoxemia 예측 샘플."""

    input_signals: dict[str, np.ndarray]   # NaN → 0 fill
    input_gap_masks: dict[str, np.ndarray] # bool, True=원본이 NaN (gap)
    label: int  # 0=normal, 1=hypoxemia in future
    label_value: float  # future SpO2 최솟값 (%)
    case_id: str
    win_start_sec: float
    horizon_sec: float


# NOTE: 구 filename-seg_key 로더(_parse_pt_filename / _load_local_pt_aligned_signals)는
#   제거했다. manifest 의 start_sample 을 싣지 않아 SpO2 라벨 정렬이 불가능하고,
#   sweep 은 downstream._seg_intersect.load_aligned_signals_intersection 만 쓴다.


# ---- SpO2 trend 추출 (raw .vital 사용) ----


def _load_spo2_trend(raw_vital_path: Path) -> np.ndarray | None:
    """raw .vital 에서 SpO2 1Hz trend 추출 (dtstart 원점).

    SpO2 track 만 지정해 로드한다 — 전체 track 파싱은 네트워크 마운트에서 수십 배
    느리다. 여러 track 후보를 순차 시도하고 첫 발견 사용.
    Returns: (T,) 1Hz SpO2 array 또는 None.
    """
    try:
        import vitaldb
    except ImportError:
        print("  ERROR: vitaldb library required. pip install vitaldb", file=sys.stderr)
        return None

    if not raw_vital_path.is_file():
        return None
    try:
        vf = vitaldb.VitalFile(str(raw_vital_path), track_names=SPO2_TRACKS)
        avail = set(vf.get_track_names())
    except Exception:
        return None

    for track in SPO2_TRACKS:
        if track in avail:
            try:
                # interval=1.0 → dtstart 기준 1Hz 격자. parser 의 start_sample 과 동일 원점.
                arr = vf.to_numpy([track], 1.0)
                if arr is not None and arr.size > 0:
                    return arr.flatten().astype(np.float32)
            except Exception:
                continue
    return None


def build_raw_vital_index(raw_dir: Path) -> dict[int, Path]:
    """raw .vital 트리를 **한 번만** 순회해 {case 번호: 경로} 인덱스 구성.

    구 코드는 subject 마다 `raw_dir.glob("**/*.vital")` 를 다시 돌아, 네트워크 마운트
    에서 O(N_subject × N_file) 디렉토리 순회가 발생했다.
    """
    index: dict[int, Path] = {}
    for p in raw_dir.glob("**/*.vital"):
        digits = "".join(c for c in p.stem if c.isdigit())
        if not digits:
            continue
        index.setdefault(int(digits), p)
    return index


def _resolve_raw_vital_path(index: dict[int, Path], subject_id: str) -> Path | None:
    """parsed subject_id(VDB_0001) → raw .vital 경로 (사전 구축 인덱스 조회)."""
    digits = "".join(c for c in subject_id if c.isdigit())
    if not digits:
        return None
    return index.get(int(digits))


# ---- 라벨링 ----


def _has_sustained_hypoxemia(
    future_spo2: list[float],
    threshold: float,
    min_consecutive: int,
) -> bool:
    """연속 min_consecutive 개 SpO2 < threshold 면 positive."""
    consecutive = 0
    for s in future_spo2:
        if s < threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                return True
        else:
            consecutive = 0
    return False


def extract_forecast_samples(
    cases: list[dict],
    spo2_map: dict[str, np.ndarray],  # patient_id → SpO2 trend (1Hz)
    input_signals: list[str],
    window_sec: float,
    stride_sec: float,
    horizon_sec: float,
    spo2_threshold: float = 92.0,
    sustained_sec: float = 60.0,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",  # OOM 회피
    exclude_already_low: bool = True,
    baseline_sec: float = 30.0,
    drop_counts: dict[str, int] | None = None,
    allow_tail_windows: bool = False,
) -> list[ForecastSample]:
    """SpO2 trend + waveform window 정렬해서 (input, future_label) 쌍 추출.

    시간 정렬: case["start_sample"] 은 dtstart 로부터의 100Hz 절대 인덱스이고 SpO2
    trend 도 dtstart 원점 1Hz 이므로, 윈도우 끝의 절대 초 =
    (start_sample + start + win_samples) / TARGET_SR 로 직접 인덱싱한다.

    exclude_already_low: 예측 시점 baseline SpO2 (윈도우 끝 직전 baseline_sec 의
    중앙값)가 이미 threshold 미만이면 그 윈도우를 버린다. 지속성만으로 양성이
    맞아떨어지는 누수를 막는다.
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    horizon_samples_target = int(horizon_sec * TARGET_SR)
    # 라벨은 SpO2 trend(전체 recording)에서 오므로 파형이 horizon 만큼 더 있을 필요는
    # 없다. 기본값은 보수적으로 파형까지 요구(구 동작 유지). allow_tail_windows=True 면
    # case 말미 windows 를 회복한다 — 라벨은 trend 범위 가드(out_of_trend)가 지킨다.
    total_needed = win_samples if allow_tail_windows else win_samples + horizon_samples_target

    min_consecutive = max(1, int(sustained_sec))  # 1Hz 라 1초당 1 sample
    samples: list[ForecastSample] = []

    def _bump(key: str) -> None:
        if drop_counts is not None:
            drop_counts[key] = drop_counts.get(key, 0) + 1

    for case in cases:
        signals = case["signals"]
        n_total = min(len(s) for s in signals.values())
        if n_total < total_needed:
            continue

        spo2 = spo2_map.get(case["patient_id"])
        if spo2 is None or len(spo2) == 0:
            continue

        # dtstart 원점 정렬: parser manifest 의 start_sample(100Hz 절대 인덱스).
        # _seg_intersect 가 case 에 실어 준다. 없으면 정렬 불가 → case skip.
        if "start_sample" not in case:
            _bump("no_start_sample")
            continue
        seg_off_samples = int(case["start_sample"])

        for start in range(0, n_total - total_needed + 1, stride_samples):
            input_dict = {}
            for stype in input_signals:
                if stype in signals:
                    input_dict[stype] = signals[stype][start : start + win_samples]
            if not input_dict:
                continue

            # Step 1: gap-policy window drop
            valid_ratio = compute_valid_ratio(list(input_dict.values()))
            if valid_ratio < valid_ratio_threshold:
                if gap_stats is not None:
                    gap_stats.add_drop()
                _bump("gap")
                continue

            win_end_sec = (seg_off_samples + start + win_samples) / TARGET_SR
            future_start_sec = win_end_sec
            future_end_sec = future_start_sec + horizon_sec

            # 아래 drop 들은 gap policy 가 아니라 **라벨 가용성/설계** 사유다.
            # gap_stats 에 넣으면 논문에 보고할 "gap 으로 버린 비율"이 부풀려진다
            # → drop_counts 로만 집계한다.
            f_start = int(round(future_start_sec))
            f_end = int(round(future_end_sec))
            if f_start < 0 or f_end > len(spo2):
                _bump("out_of_trend")
                continue

            # 예측 시점 baseline: 이미 저산소면 지속성 누수 → drop
            if exclude_already_low:
                b_start = max(0, f_start - int(baseline_sec))
                base = spo2[b_start:f_start]
                base = base[~np.isnan(base)]
                base = base[(base >= 50.0) & (base <= 100.0)]
                if base.size == 0:
                    _bump("no_baseline")
                    continue
                if float(np.median(base)) < spo2_threshold:
                    _bump("already_low")
                    continue

            future_spo2 = spo2[f_start:f_end]
            future_spo2 = future_spo2[~np.isnan(future_spo2)]
            # SpO2 가용 범위: 50~100 외는 artifact
            future_spo2 = future_spo2[(future_spo2 >= 50.0) & (future_spo2 <= 100.0)]

            if len(future_spo2) < max(1, min_consecutive // 2):
                _bump("no_future_spo2")
                continue

            label = (
                1
                if _has_sustained_hypoxemia(
                    future_spo2.tolist(), spo2_threshold, min_consecutive,
                )
                else 0
            )
            min_future_spo2 = float(future_spo2.min())

            # Step 2: gap mask + 즉시 float16 캐스팅
            filled_dict, gap_mask_dict = apply_gap_mask_multichannel(
                input_dict, output_dtype=sample_dtype,
            )
            if gap_stats is not None:
                n_total_s = sum(arr.size for arr in filled_dict.values())
                n_gap_s = sum(int(m.sum()) for m in gap_mask_dict.values())
                gap_stats.add_window(n_total_s, n_gap_s)

            samples.append(
                ForecastSample(
                    input_signals=filled_dict,
                    input_gap_masks=gap_mask_dict,
                    label=label,
                    label_value=min_future_spo2,
                    case_id=case["case_id"],
                    win_start_sec=start / TARGET_SR,
                    horizon_sec=horizon_sec,
                )
            )

    return samples


# ---- 저장 ----


def pack_samples_to_dict(
    samples: list[ForecastSample],
    input_signals: list[str],
    signal_dtype: torch.dtype = torch.float16,
) -> dict:
    """ForecastSample list → packed dict, samples in-place 해제 (OOM 회피)."""
    if not samples:
        return {
            "signals": {},
            "gap_masks": {},
            "labels": torch.tensor([]),
            "label_values": torch.tensor([]),
            "case_ids": [],
        }
    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    label_values = torch.tensor(
        [s.label_value for s in samples], dtype=torch.float32
    )
    case_ids = [s.case_id for s in samples]
    sig_tensors = consume_input_signals(samples, input_signals, signal_dtype)
    gap_tensors = consume_gap_masks(samples, input_signals, attr="input_gap_masks")
    samples.clear()
    gc.collect()
    return {
        "signals": sig_tensors,
        "gap_masks": gap_tensors,
        "labels": labels,
        "label_values": label_values,
        "case_ids": case_ids,
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
    spo2_threshold: float = 92.0,
    sustained_sec: float = 60.0,
    exclude_already_low: bool = True,
    baseline_sec: float = 30.0,
) -> Path:
    """Single split chunk packed dict → 별도 .pt (OOM 회피 Stage 5)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_samples = int(split_dict.get("labels", torch.tensor([])).numel())

    save_dict = {
        "split": split_name,
        "data": split_dict,
        "metadata": {
            "task": "hypoxemia_forecast",
            "source": "vitaldb_pt + raw_vital",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "spo2_threshold": spo2_threshold,
            "sustained_sec": sustained_sec,
            "exclude_already_low": exclude_already_low,
            "baseline_sec": baseline_sec,
            "label_alignment": "dtstart-absolute (start_sample offset)",
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
    horizon_min = int(horizon_sec / 60)
    win_int = int(window_sec)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    chunk_suffix = f"_chunk{chunk_idx}" if chunk_idx is not None else ""
    filename = (
        f"hypoxemia_{mode_str}_w{win_int}s_h{horizon_min}min"
        f"{fold_suffix}_{split_name}{chunk_suffix}.pt"
    )
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {save_path.name} ({file_size_mb:.1f} MB, n={n_samples})")
    return save_path


def print_stats(name: str, samples: list[ForecastSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return
    n = len(samples)
    n_pos = sum(1 for s in samples if s.label == 1)
    spo2_vals = [s.label_value for s in samples]
    print(f"  {name}: {n} samples")
    print(f"    Normal:    {n - n_pos} ({(n - n_pos) / n * 100:.1f}%)")
    print(f"    Hypoxemia: {n_pos} ({n_pos / n * 100:.1f}%)")
    print(
        f"    Future SpO2: [{min(spo2_vals):.1f}, {max(spo2_vals):.1f}] %, "
        f"mean={np.mean(spo2_vals):.1f} +/- {np.std(spo2_vals):.1f}"
    )


# ---- 메인 ----


def _patient_level_hypoxemia_labels(
    cases: list[dict],
    spo2_map: dict[str, np.ndarray],
    spo2_threshold: float = 92.0,
) -> dict[str, list[int]]:
    """환자별 case 레벨 hypoxemia label (stratification 용).

    각 case 가 **실제로 덮는 시간 구간**의 SpO2 에 < threshold 샘플이 있는지 기록한다.
    (구 코드는 recording 전체를 봐서, case 구간 밖 저산소까지 양성으로 셌다.)
    """
    patient_to_labels: dict[str, list[int]] = {}
    for case in cases:
        pid = case["patient_id"]
        if pid not in spo2_map:
            continue
        sp = spo2_map[pid]
        if sp.size == 0:
            continue
        off = int(case.get("start_sample", 0))
        n_wave = min(len(s) for s in case["signals"].values())
        s0 = max(0, int(off / TARGET_SR))
        s1 = min(len(sp), int((off + n_wave) / TARGET_SR))
        span = sp[s0:s1] if s1 > s0 else sp[:0]
        valid = span[~np.isnan(span)]
        valid = valid[(valid >= 50.0) & (valid <= 100.0)]
        has_hypo = bool(np.any(valid < spo2_threshold)) if valid.size else False
        patient_to_labels.setdefault(pid, []).append(1 if has_hypo else 0)
    return patient_to_labels


def prepare_hypoxemia_sweep(
    data_dir: str,
    raw_dir: str,
    input_signals: list[str],
    window_secs: list[float],
    horizon_mins: list[float],
    stride_sec: float = 30.0,
    n_folds: int = 5,
    max_subjects: int | None = None,
    out_dir: str = "outputs/downstream/hypoxemia",
    required_signals: list[str] | None = None,
    signal_dtype: torch.dtype = torch.float16,
    seed: int = 42,
    spo2_threshold: float = 92.0,
    sustained_sec: float = 60.0,
    exclude_already_low: bool = True,
    baseline_sec: float = 30.0,
    workers: int = 16,
    allow_tail_windows: bool = False,
) -> list[Path]:
    max_window = max(window_secs)
    max_horizon_sec = max(horizon_mins) * 60.0
    if allow_tail_windows:
        min_duration_sec = max_window + stride_sec
    else:
        min_duration_sec = max_window + max_horizon_sec + stride_sec

    mode_str = " + ".join(s.upper() for s in input_signals)
    req_str = " + ".join(s.upper() for s in required_signals) if required_signals else "auto"
    print(f"\n{'=' * 60}")
    print("  Intraop Hypoxemia Forecast — Sweep")
    print(f"  Parsed:    {data_dir}")
    print(f"  Raw vital: {raw_dir}")
    print(f"  Input:     {mode_str}")
    print(f"  Required:  {req_str}")
    print(f"  Windows:   {window_secs}")
    print(f"  Horizons:  {horizon_mins}")
    print(f"  Min dur:   {min_duration_sec / 60:.1f} min")
    print(f"  Label:     SpO2 < {spo2_threshold:.0f}% sustained >= {sustained_sec:.0f}s")
    print(f"  Baseline:  exclude_already_low={exclude_already_low} (median of last {baseline_sec:.0f}s)")
    print(f"{'=' * 60}")

    # required 가 input 을 포함하지 않으면, 로더가 그 신호를 아예 안 읽어 입력 채널이
    # 조용히 사라진다 (consume_input_signals 는 없는 stype 을 그냥 건너뜀).
    if required_signals is not None:
        missing = sorted(set(input_signals) - set(required_signals))
        if missing:
            print(
                f"ERROR: --input-signals 가 --required-signals 의 부분집합이어야 합니다. "
                f"required 에 빠진 입력: {missing}",
                file=sys.stderr,
            )
            sys.exit(1)

    print("\n[1/4] Loading aligned multi-channel waveform (manifest intersection)...")
    # Manifest 기반 시간선 intersection — filename seg_key 매칭 대비 cohort 13-124× 회복
    base_req = set(required_signals if required_signals else input_signals)
    cases = load_aligned_signals_intersection(
        data_dir,
        required_signals=sorted(base_req),
        min_duration_sec=min_duration_sec,
        max_subjects=max_subjects,
    )
    if not cases:
        print(
            f"ERROR: No valid cases loaded.\n"
            f"  요구한 연속 교집합 길이 = {min_duration_sec:.0f}s "
            f"(= window {max_window:.0f} + horizon {max_horizon_sec:.0f} + stride {stride_sec:.0f}).\n"
            f"  원인 진단:\n"
            f"    python -m downstream.acute_event.hypoxemia.inspect_cohort "
            f"--data-dir {data_dir} --max-subjects 300\n"
            f"  완화 옵션: --allow-tail-windows (min_dur → window+stride), "
            f"또는 --required-signals 를 좁히기.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 정렬 전제: 한 subject = 한 .vital = 한 session. 여러 session 이면 start_sample
    # 원점이 session 마다 달라 단일 SpO2 trend 와 맞출 수 없다 → 해당 subject drop.
    sessions_by_pid: dict[str, set[str]] = {}
    for c in cases:
        sessions_by_pid.setdefault(c["patient_id"], set()).add(str(c.get("session_id", "")))
    multi_sess = {p for p, s in sessions_by_pid.items() if len(s) > 1}
    if multi_sess:
        print(f"  [WARN] multi-session subject {len(multi_sess)} 명 제외 (start_sample 원점 모호)")
        cases = [c for c in cases if c["patient_id"] not in multi_sess]

    print("\n[2/4] Loading SpO2 trends from raw .vital...")
    raw_root = Path(raw_dir)
    print("  indexing raw .vital tree (1-pass)...")
    raw_index = build_raw_vital_index(raw_root)
    print(f"  raw .vital 파일: {len(raw_index)}")

    spo2_map: dict[str, np.ndarray] = {}
    unique_pids = sorted({c["patient_id"] for c in cases})

    def _one(pid: str) -> tuple[str, np.ndarray | None]:
        rv = _resolve_raw_vital_path(raw_index, pid)
        if rv is None:
            return pid, None
        return pid, _load_spo2_trend(rv)

    # 네트워크 마운트 I/O — ThreadPool 병렬 (memory: feedback_network_mounted_storage)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, pid) for pid in unique_pids]
        for fut in as_completed(futs):
            pid, spo2 = fut.result()
            done += 1
            if spo2 is not None and spo2.size > 0:
                spo2_map[pid] = spo2
            if done % 200 == 0:
                print(f"  {done}/{len(unique_pids)} processed, SpO2 found: {len(spo2_map)}")

    n_found = len(spo2_map)
    print(f"  SpO2 trend 보유 환자: {n_found} / {len(unique_pids)}")
    if n_found == 0:
        print("ERROR: No SpO2 trend found. raw_dir 또는 SPO2_TRACKS 확인.", file=sys.stderr)
        sys.exit(1)
    cases = [c for c in cases if c["patient_id"] in spo2_map]
    print(f"  SpO2 가용 case: {len(cases)}")

    print(f"\n[3/4] Stratified {n_folds}-fold patient-level CV split...")
    patient_ids = sorted({c["patient_id"] for c in cases})
    patient_to_labels = _patient_level_hypoxemia_labels(
        cases, spo2_map, spo2_threshold=spo2_threshold,
    )
    pos_pts = sum(1 for pid in patient_ids if any(patient_to_labels.get(pid, [])))
    print(
        f"  Patient-level positive (any SpO2<{int(spo2_threshold)}%): {pos_pts}/{len(patient_ids)} "
        f"({100.0 * pos_pts / max(1, len(patient_ids)):.1f}%)"
    )
    splits = stratified_kfold_patient_splits(
        patient_ids, patient_to_labels, n_folds=n_folds, seed=seed,
    )
    print(summarize_splits(splits, patient_to_labels))

    combos = [(w, h) for w in window_secs for h in horizon_mins]
    total_runs = len(combos) * n_folds
    print(f"\n[4/4] Generating {len(combos)} combo × {n_folds} fold = {total_runs} datasets...")

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
            gap_stats = GapStats()
            drop_counts: dict[str, int] = {}
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
                total = 0
                total_pos = 0
                for batch_start in range(0, len(split_cases), CASES_PER_CHUNK):
                    case_batch = split_cases[batch_start:batch_start + CASES_PER_CHUNK]
                    samples = extract_forecast_samples(
                        case_batch, spo2_map, input_signals, window_sec, stride_sec, horizon_sec,
                        gap_stats=gap_stats,
                        spo2_threshold=spo2_threshold, sustained_sec=sustained_sec,
                        exclude_already_low=exclude_already_low,
                        baseline_sec=baseline_sec,
                        drop_counts=drop_counts,
                        allow_tail_windows=allow_tail_windows,
                    )
                    if not samples:
                        continue
                    n_pos = sum(1 for s in samples if s.label == 1)
                    packed = pack_samples_to_dict(samples, input_signals, signal_dtype)
                    del samples; gc.collect()
                    n_in_chunk = int(packed.get("labels", torch.tensor([])).numel())
                    save_path = save_split_dataset(
                        packed, split_name, input_signals,
                        horizon_sec, window_sec, out_dir,
                        signal_dtype=signal_dtype,
                        fold_idx=cur_fold_idx,
                        n_folds=n_folds,
                        chunk_idx=chunk_idx,
                        spo2_threshold=spo2_threshold,
                        sustained_sec=sustained_sec,
                        exclude_already_low=exclude_already_low,
                        baseline_sec=baseline_sec,
                    )
                    saved_paths.append(save_path)
                    total += n_in_chunk
                    total_pos += n_pos
                    chunk_idx += 1
                    del packed; gc.collect()
                prev = f"{100.0 * total_pos / total:.2f}%" if total else "n/a"
                print(f"    {split_name.capitalize()}: {chunk_idx} chunk(s), "
                      f"{total} samples (+={total_pos}, prevalence={prev})")

            print(gap_stats.summary())
            if drop_counts:
                summary = ", ".join(f"{k}={v}" for k, v in sorted(drop_counts.items()))
                print(f"    window drops: {summary}")

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(saved_paths)}/{len(combos)} datasets saved to {out_dir}")
    print(f"{'=' * 60}")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraop Hypoxemia Forecast — Data Prep")
    parser.add_argument("--data-dir", required=True, help="parsed .pt 디렉토리")
    parser.add_argument("--raw-dir", required=True, help="raw vitaldb .vital 디렉토리 (SpO2 라벨 추출용)")
    # 입력 조합은 코호트 크기와 trade-off: co2/awp 를 required 로 두면 capnography +
    # 기도압이 있는 전신마취 케이스로 좁혀진다. --max-subjects 로 먼저 N 을 재 볼 것.
    parser.add_argument("--input-signals", nargs="+", default=["ecg", "ppg", "co2", "awp"],
                        choices=["abp", "ecg", "ppg", "co2", "awp"])
    parser.add_argument("--required-signals", nargs="+", default=None,
                        choices=["abp", "ecg", "ppg", "co2", "awp"])
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--horizon-mins", nargs="+", type=float, default=[5.0, 10.0, 15.0])
    parser.add_argument("--window-secs", nargs="+", type=float, default=[300.0])
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Stratified patient-level K-fold CV (default 5).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16,
                        help="SpO2 trend 로딩 ThreadPool worker 수 (네트워크 마운트 I/O).")
    parser.add_argument("--out-dir", default="outputs/downstream/hypoxemia")
    parser.add_argument("--spo2-threshold", type=float, default=92.0,
                        help="hypoxemia 임계 SpO2(%). default 92 (Lundberg 2018·WHO ≤92, 수술중 예측 표준)")
    parser.add_argument("--sustained-sec", type=float, default=60.0,
                        help="positive 로 인정할 SpO2<threshold 연속 지속(초). "
                             "완화 시 10~30 권장 (양성↑)")
    parser.add_argument("--include-already-low", action="store_true",
                        help="예측 시점에 이미 SpO2<threshold 인 윈도우도 포함(기본 제외). "
                             "포함하면 지속성만으로 양성이 맞아 성능이 부풀려진다.")
    parser.add_argument("--baseline-sec", type=float, default=30.0,
                        help="baseline SpO2 판정 구간(초, 윈도우 끝 직전 중앙값).")
    parser.add_argument("--allow-tail-windows", action="store_true",
                        help="파형이 horizon 만큼 남지 않은 case 말미 윈도우도 사용. "
                             "라벨은 SpO2 trend 에서 오므로 유효하며 코호트가 늘어난다. "
                             "기본 off(구 동작 유지 — 다른 task 와 표본 구성 일관).")
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    prepare_hypoxemia_sweep(
        data_dir=args.data_dir,
        raw_dir=args.raw_dir,
        input_signals=args.input_signals,
        window_secs=args.window_secs,
        horizon_mins=args.horizon_mins,
        stride_sec=args.stride_sec,
        n_folds=args.n_folds,
        seed=args.seed,
        max_subjects=args.max_subjects,
        out_dir=args.out_dir,
        required_signals=args.required_signals,
        signal_dtype=dtype_map(args.signal_dtype),
        spo2_threshold=args.spo2_threshold,
        sustained_sec=args.sustained_sec,
        exclude_already_low=not args.include_already_low,
        baseline_sec=args.baseline_sec,
        workers=args.workers,
        allow_tail_windows=args.allow_tail_windows,
    )


if __name__ == "__main__":
    main()
