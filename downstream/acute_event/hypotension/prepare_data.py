# -*- coding:utf-8 -*-
"""Task 1: Hypotension Prediction - 데이터 준비 스크립트.

미래 5~15분 후 MAP<65 (≥1분 지속) 예측을 위한 (input_window, future_label) 쌍 생성.
Label 소스: 항상 ABP (미래 구간의 MAP)
Input 소스: 선택된 signal type 의 현재 윈도우

Split: patient-level K-fold CV (clinical AI 표준, default n_folds=5)
       각 fold i — train (n_folds-2 folds) / val (1 fold) / test (1 fold).
       n_folds=5 면 60/20/20 ratio, 모든 환자가 정확히 1번씩 test.

사용법:
    # Canonical + 5-fold CV (default)
    python -m downstream.acute_event.hypotension.prepare_data \
        --data-dir vitaldb_pt_test --input-signals ecg ppg abp \
        --window-secs 300 --horizon-mins 15

    # Sweep ablation + 5-fold CV (appendix 용)
    python -m downstream.acute_event.hypotension.prepare_data \
        --data-dir vitaldb_pt_test --input-signals ecg ppg abp \
        --window-secs 60 180 300 600 --horizon-mins 5 10 15

    # Single split (legacy 60/20/20)
    python -m downstream.acute_event.hypotension.prepare_data \
        --data-dir vitaldb_pt_test --input-signals ecg ppg abp \
        --window-secs 300 --horizon-mins 15 --n-folds 1
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
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

# signal_type 정수 → 문자열 매핑
SIGNAL_TYPE_MAP: dict[int, str] = {
    0: "ecg",
    1: "abp",
    2: "ppg",
    3: "cvp",
    4: "co2",
    5: "awp",
    6: "pap",
    7: "icp",
}


# ---- 데이터 구조 ----


@dataclass
class ForecastSample:
    """미래 저혈압 예측 샘플."""

    input_signals: dict[str, np.ndarray]  # {"ecg": (win_samples,), ...}  NaN 은 0 으로 채워짐
    input_gap_masks: dict[str, np.ndarray]  # 같은 shape, bool. True=원본이 NaN (gap)
    label: int  # 0=normal, 1=hypotension in future
    label_value: float  # future MAP (mmHg)
    case_id: str
    win_start_sec: float  # input window 시작 (초)
    horizon_sec: float  # prediction horizon (초)


# ---- 로컬 .pt 로더 ----


def _parse_pt_filename(name: str) -> dict | None:
    """파일명에서 메타데이터를 추출한다.

    형식: {subject_id}_S{session}_{signal_name}_{spatial_id}_seg{i}_{j}.pt
    예: VDB_0239_S0_abp_1_seg0_0.pt
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
        "signal_type": m.group(3),  # 문자열: "ecg", "abp", etc.
        "spatial_id": int(m.group(4)),
        "seg_i": int(m.group(5)),
        "seg_j": int(m.group(6)),
    }


def _load_local_pt_aligned_signals(
    data_dir: str,
    input_signals: list[str],
    min_duration_sec: float = 1200.0,
    max_subjects: int | None = None,
    required_signals: list[str] | None = None,
) -> list[dict]:
    """로컬 .pt 디렉토리에서 시간 정렬된 다채널 데이터를 로드한다.

    Parameters
    ----------
    data_dir : vitaldb_pt_test/ 경로.
    input_signals : 입력으로 사용할 signal types (예: ["ecg", "ppg"]).
    min_duration_sec : 최소 유효 신호 길이 (초).
    max_subjects : 최대 subject 수. None이면 전체.
    required_signals : 로딩 시 반드시 존재해야 하는 signal types.
        None이면 input_signals + abp. Paired comparison을 위해
        모든 조합에서 동일 환자를 사용하려면 ["ecg", "ppg", "abp"] 지정.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signals": {"abp": array, ...}}
    """
    root = Path(data_dir)
    if not root.is_dir():
        print(f"  ERROR: Data directory not found: {root}")
        return []

    if required_signals is not None:
        required_types = set(required_signals) | {"abp"}
    else:
        required_types = set(input_signals) | {"abp"}

    subject_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    print(f"  Scanning {len(subject_dirs)} subjects in {root}...")
    cases: list[dict] = []

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name

        # 이 subject의 모든 .pt 파일 파싱
        file_map: dict[tuple[int, int, int], dict[str, Path]] = {}
        # key: (session_id, seg_i, seg_j) → {signal_type_str: path}

        for pt_file in subj_dir.glob("*.pt"):
            meta = _parse_pt_filename(pt_file.name)
            if meta is None:
                continue
            seg_key = (meta["session_id"], meta["seg_i"], meta["seg_j"])
            if seg_key not in file_map:
                file_map[seg_key] = {}
            file_map[seg_key][meta["signal_type"]] = pt_file

        # 필요한 모든 signal type이 있는 세그먼트 찾기
        for seg_key, type_paths in file_map.items():
            available_types = set(type_paths.keys())
            if not required_types.issubset(available_types):
                continue

            # 로드
            signals: dict[str, np.ndarray] = {}
            for stype_str in required_types:
                t = torch.load(type_paths[stype_str], weights_only=True)  # (1, T)
                signals[stype_str] = t.squeeze(0).numpy()  # (T,)

            # 모든 채널을 동일 길이로 자르기
            min_len = min(len(s) for s in signals.values())
            if min_len < int(min_duration_sec * TARGET_SR):
                continue

            signals = {k: v[:min_len] for k, v in signals.items()}
            session_id, seg_i, seg_j = seg_key

            cases.append(
                {
                    "case_id": f"{subject_id}_s{session_id}_seg{seg_i}_{seg_j}",
                    "patient_id": subject_id,
                    "signals": signals,
                }
            )

    print(f"  Loaded {len(cases)} aligned segments with {required_types}")
    return cases


# ---- 윈도우 추출 + 라벨링 ----


def _has_sustained_hypotension(
    future_maps: list[float],
    threshold: float,
    min_consecutive: int,
) -> bool:
    """연속 min_consecutive개 이상 윈도우에서 MAP < threshold인지 확인한다."""
    consecutive = 0
    for m in future_maps:
        if m < threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                return True
        else:
            consecutive = 0
    return False


def extract_forecast_samples(
    cases: list[dict],
    input_signals: list[str],
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    horizon_sec: float = 300.0,
    map_threshold: float = 65.0,
    sustained_sec: float = 60.0,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",  # OOM 회피: sample 누적 시 즉시 float16 캐스팅
) -> list[ForecastSample]:
    """시간 정렬된 다채널 데이터에서 (input, future_label) 쌍을 추출한다.

    Parameters
    ----------
    cases : 로드된 케이스 리스트.
    input_signals : 입력으로 사용할 signal types.
    window_sec : 입력 윈도우 길이 (초).
    stride_sec : 슬라이드 보폭 (초).
    horizon_sec : prediction horizon (초). 미래 이 구간 내 MAP<65 발생 여부.
    map_threshold : MAP 미만이면 hypotension.
    sustained_sec : MAP<threshold가 이 시간 이상 지속되어야 positive.

    Returns
    -------
    ForecastSample 리스트.
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    horizon_samples = int(horizon_sec * TARGET_SR)

    # MAP 계산 윈도우: 10초
    map_win_sec = 10.0
    map_win = int(map_win_sec * TARGET_SR)
    # 1분 지속 = 6개 연속 10초 윈도우
    min_consecutive = max(1, int(sustained_sec / map_win_sec))

    # 전체 필요 길이: input window + horizon
    total_needed = win_samples + horizon_samples

    samples: list[ForecastSample] = []

    for case in cases:
        signals = case["signals"]
        abp = signals["abp"]
        n_total = len(abp)

        if n_total < total_needed:
            continue

        for start in range(0, n_total - total_needed + 1, stride_samples):
            # Input window: [start, start + win_samples)
            input_dict = {}
            for stype in input_signals:
                if stype in signals:
                    input_dict[stype] = signals[stype][start : start + win_samples]

            if not input_dict:
                continue

            # ── Step 1: gap-policy window drop ([[project_downstream_gap_window_policy]]) ──
            #   input window 의 multi-channel valid_ratio < threshold → drop
            valid_ratio = compute_valid_ratio(list(input_dict.values()))
            if valid_ratio < valid_ratio_threshold:
                if gap_stats is not None:
                    gap_stats.add_drop()
                continue

            # Future label: ABP의 [start + win_samples, start + win_samples + horizon_samples) 구간
            future_start = start + win_samples
            future_end = future_start + horizon_samples
            future_abp = abp[future_start:future_end]

            # 미래 구간의 MAP (10초 윈도우별 평균)
            # MAP <30 또는 >200 mmHg 윈도우는 artifact로 제외 (docs/downstream_tasks.md:260)
            future_maps: list[float] = []
            for j in range(0, len(future_abp) - map_win + 1, map_win):
                w = future_abp[j : j + map_win]
                if np.isnan(w).any():
                    continue
                m = float(np.mean(w))
                if m < 30.0 or m > 200.0:
                    continue
                future_maps.append(m)

            # artifact 제거 후 최소 min_consecutive의 절반은 남아야 신뢰 가능
            if len(future_maps) < max(1, min_consecutive // 2):
                if gap_stats is not None:
                    gap_stats.add_drop()
                continue

            # ≥1분 지속 MAP<65 여부 확인
            label = (
                1
                if _has_sustained_hypotension(
                    future_maps,
                    map_threshold,
                    min_consecutive,
                )
                else 0
            )

            # label_value: 미래 MAP의 최솟값 (참고용)
            min_future_map = min(future_maps)

            # ── Step 2: gap mask 적용 + 즉시 float16 캐스팅 (메모리 50% 절감) ──
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
                    label_value=min_future_map,
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
    """ForecastSample list → packed dict, samples in-place 해제.

    OOM 회피: 호출 후 samples 가 모두 비워져 caller 가 즉시 free 가능.
    consume_input_signals + consume_gap_masks 가 numpy array 를 pop 하면서
    in-place 로 tensor 에 복사 → peak memory 2× 회피.
    """
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


def save_packed_dataset(
    train_dict: dict,
    val_dict: dict,
    test_dict: dict,
    input_signals: list[str],
    horizon_sec: float,
    window_sec: float,
    out_dir: str,
    signal_dtype: torch.dtype = torch.float16,
    fold_idx: int | None = None,
    n_folds: int = 1,
) -> Path:
    """Pre-packed dict 들을 .pt 로 저장 (train/val/test 3-way).

    pack_samples_to_dict 와 짝 — caller 가 split 별 순차 pack 후 호출.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_train = int(train_dict.get("labels", torch.tensor([])).numel())
    n_val = int(val_dict.get("labels", torch.tensor([])).numel())
    n_test = int(test_dict.get("labels", torch.tensor([])).numel())

    save_dict = {
        "train": train_dict,
        "val": val_dict,
        "test": test_dict,
        "metadata": {
            "task": "hypotension_forecast",
            "source": "vitaldb_pt",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "map_threshold": 65.0,
            "sustained_sec": 60.0,
            "gap_policy": "drop+mask",
            "valid_ratio_threshold": DEFAULT_VALID_RATIO_THRESHOLD,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "signal_dtype": str(signal_dtype).replace("torch.", ""),
            "fold_idx": fold_idx,
            "n_folds": n_folds,
        },
    }

    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    win_int = int(window_sec)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    filename = f"hypotension_{mode_str}_w{win_int}s_h{horizon_min}min{fold_suffix}.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ---- 통계 출력 ----


def print_stats(
    name: str,
    samples: list[ForecastSample],
) -> None:
    """데이터셋 통계 출력."""
    if not samples:
        print(f"  {name}: 0 samples")
        return

    n_total = len(samples)
    n_hypo = sum(1 for s in samples if s.label == 1)
    n_normal = n_total - n_hypo
    maps = [s.label_value for s in samples]

    print(f"  {name}: {n_total} samples")
    print(f"    Normal:      {n_normal} ({n_normal / n_total * 100:.1f}%)")
    print(f"    Hypotension: {n_hypo} ({n_hypo / n_total * 100:.1f}%)")
    print(
        f"    Future MAP:  [{min(maps):.1f}, {max(maps):.1f}] mmHg, "
        f"mean={np.mean(maps):.1f} +/- {np.std(maps):.1f}"
    )


# ---- 메인 ----


def _patient_level_hypo_labels(
    cases: list[dict],
    map_threshold: float = 65.0,
    sustained_sec: float = 60.0,
) -> dict[str, list[int]]:
    """환자별 case 레벨 hypotension 라벨 (stratification 용).

    각 case 의 ABP 신호 전체를 스캔하여 ≥1분 지속 MAP<65 episode 존재 여부를 binary 로 기록.
    동일 환자의 여러 case 들이 모두 모여 patient_to_labels[pid] = [0, 1, 0, ...] 형태로 반환.

    Note: prediction horizon 과 무관한 patient-level positivity → 모든 (w, h) combo 에서
    동일한 fold 분할이 가능.
    """
    map_win_sec = 10.0
    map_win = int(map_win_sec * TARGET_SR)
    min_consecutive = max(1, int(sustained_sec / map_win_sec))

    patient_to_labels: dict[str, list[int]] = {}
    for case in cases:
        pid = case["patient_id"]
        abp = case["signals"]["abp"]
        # 10s MAP windows 전체
        future_maps: list[float] = []
        for j in range(0, len(abp) - map_win + 1, map_win):
            w = abp[j : j + map_win]
            if np.isnan(w).any():
                continue
            m = float(np.mean(w))
            if m < 30.0 or m > 200.0:
                continue
            future_maps.append(m)
        has_hypo = _has_sustained_hypotension(
            future_maps, map_threshold, min_consecutive
        ) if future_maps else False
        patient_to_labels.setdefault(pid, []).append(1 if has_hypo else 0)
    return patient_to_labels


def prepare_hypotension_sweep(
    data_dir: str,
    input_signals: list[str],
    window_secs: list[float],
    horizon_mins: list[float],
    stride_sec: float = 30.0,
    n_folds: int = 5,
    max_subjects: int | None = None,
    out_dir: str = "outputs/downstream/hypotension",
    required_signals: list[str] | None = None,
    signal_dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> list[Path]:
    """(window, horizon) 조합 sweep + K-fold patient-level CV.

    각 (w, h) 조합 × 각 fold 마다 train/val/test .pt 1개 저장.
    n_folds=1 이면 단일 split (60/20/20).
    """
    # ── 1. 데이터 로딩 (1회) ──
    # 가장 긴 window + horizon 기준으로 min_duration 설정
    max_window = max(window_secs)
    max_horizon_sec = max(horizon_mins) * 60.0
    min_duration_sec = max_window + max_horizon_sec + stride_sec

    mode_str = " + ".join(s.upper() for s in input_signals)
    req_str = " + ".join(s.upper() for s in required_signals) if required_signals else "auto"
    print(f"\n{'=' * 60}")
    print(f"  Task 1: Hypotension Forecast — Sweep × {n_folds}-fold CV")
    print(f"  Data:    {data_dir}")
    print(f"  Input:   {mode_str}")
    print(f"  Required: {req_str}")
    print(f"  Windows: {window_secs}")
    print(f"  Horizons: {horizon_mins}")
    print(f"  N folds: {n_folds}")
    print(f"  Min duration: {min_duration_sec / 60:.1f} min")
    print(f"{'=' * 60}")

    print("\n[1/3] Loading aligned multi-channel data (once)...")
    # Manifest 기반 시간선 intersection 으로 로드
    # (filename seg_key 매칭 대비 cohort 13-124× 회복 검증됨 — 2026-05-14)
    # ABP 는 MAP label 계산에 필수 → required 에 자동 포함
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

    # Patient-level hypotension label 계산 (stratification 용)
    print(f"\n[2/3] Computing patient-level hypotension labels for stratification...")
    patient_to_labels = _patient_level_hypo_labels(cases)
    pos_pts = sum(
        1 for pid in patient_ids if any(patient_to_labels.get(pid, []))
    )
    print(
        f"  Patient-level positive: {pos_pts}/{len(patient_ids)} "
        f"({100.0 * pos_pts / max(1, len(patient_ids)):.1f}%)"
    )

    if n_folds == 1:
        # 단일 split (60/20/20) backward compat — non-stratified
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

    # ── 3. 조합 × fold 윈도우 추출 + 저장 ──
    combos = [(w, h) for w in window_secs for h in horizon_mins]
    total_runs = len(combos) * n_folds
    print(f"\n[3/3] Generating {len(combos)} combo × {n_folds} fold = {total_runs} datasets...")

    saved_paths: list[Path] = []
    run_idx = 0
    for window_sec, horizon_min in combos:
        horizon_sec = horizon_min * 60.0
        for fold_idx, (train_patients, val_patients, test_patients) in enumerate(splits):
            run_idx += 1
            print(
                f"\n  [{run_idx}/{total_runs}] "
                f"window={window_sec}s, horizon={horizon_min}min, fold={fold_idx}"
            )

            train_cases = [c for c in cases if c["patient_id"] in train_patients]
            val_cases = [c for c in cases if c["patient_id"] in val_patients]
            test_cases = [c for c in cases if c["patient_id"] in test_patients]

            # OOM 회피 — split 별 순차 extract → pack → free
            # 이전: 3 samples list 동시 보유 (~66 GB) + tensor 빌드 → ~100 GB
            # 현재: 1 samples list + 이미 packed 된 tensor → peak ~50 GB
            gap_stats = GapStats()

            print_stats("    Train (extracting)", [])
            train_samples = extract_forecast_samples(
                train_cases, input_signals, window_sec, stride_sec, horizon_sec,
                gap_stats=gap_stats,
            )
            print_stats("    Train", train_samples)
            train_dict = pack_samples_to_dict(train_samples, input_signals, signal_dtype)
            del train_samples; gc.collect()

            val_samples = extract_forecast_samples(
                val_cases, input_signals, window_sec, stride_sec, horizon_sec,
                gap_stats=gap_stats,
            )
            print_stats("    Val", val_samples)
            val_dict = pack_samples_to_dict(val_samples, input_signals, signal_dtype)
            del val_samples; gc.collect()

            test_samples = extract_forecast_samples(
                test_cases, input_signals, window_sec, stride_sec, horizon_sec,
                gap_stats=gap_stats,
            )
            print_stats("    Test", test_samples)
            test_dict = pack_samples_to_dict(test_samples, input_signals, signal_dtype)
            del test_samples; gc.collect()

            print(gap_stats.summary())

            n_total = (
                int(train_dict.get("labels", torch.tensor([])).numel())
                + int(val_dict.get("labels", torch.tensor([])).numel())
                + int(test_dict.get("labels", torch.tensor([])).numel())
            )
            if n_total == 0:
                print("    SKIP: No samples extracted.")
                continue

            save_path = save_packed_dataset(
                train_dict, val_dict, test_dict, input_signals,
                horizon_sec, window_sec, out_dir,
                signal_dtype=signal_dtype,
                fold_idx=fold_idx if n_folds > 1 else None,
                n_folds=n_folds,
            )
            saved_paths.append(save_path)
            del train_dict, val_dict, test_dict
            gc.collect()

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(saved_paths)}/{len(combos)} datasets saved to {out_dir}")
    print(f"{'=' * 60}")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1: Hypotension Forecast - Data Preparation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Local .pt data directory (e.g. vitaldb_pt_test/)",
    )
    parser.add_argument(
        "--input-signals",
        nargs="+",
        default=["abp"],
        choices=["abp", "ecg", "ppg"],
        help="Input signal types (label always from ABP)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Max number of subjects to load (None=all)",
    )
    parser.add_argument(
        "--horizon-mins",
        nargs="+",
        type=float,
        default=[5.0],
        help="Prediction horizons in minutes (e.g. 5 10 15)",
    )
    parser.add_argument(
        "--window-secs",
        nargs="+",
        type=float,
        default=[30.0],
        help="Input window lengths in seconds (e.g. 30 60 300 600)",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=30.0,
        help="Sliding window stride in seconds",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Patient-level K-fold CV (default 5). Use 1 for single 60/20/20 split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for fold/split shuffling.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/downstream/hypotension",
        help="Output directory",
    )
    parser.add_argument(
        "--required-signals",
        nargs="+",
        default=None,
        choices=["abp", "ecg", "ppg"],
        help="Signals required for loading (paired comparison). "
        "e.g. --required-signals ecg ppg abp ensures all combos use same patients.",
    )
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    prepare_hypotension_sweep(
        data_dir=args.data_dir,
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
    )


if __name__ == "__main__":
    main()
