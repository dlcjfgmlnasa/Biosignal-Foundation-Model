# -*- coding:utf-8 -*-
"""Vital Sign Forecasting - 데이터 준비 스크립트.

임의 신호의 과거 waveform으로 미래 waveform을 예측하는 task.
model.generate() (autoregressive) API를 평가한다.

지원 신호: ecg, abp, ppg (MIMIC-III 가용 신호)
데이터 소스: MIMIC-III Waveform (external)

사용법:
    # ECG forecasting (5 cases)
    python -m downstream.generation.intra_modal_forecast.prepare_data --signal-type ecg --n-cases 5

    # ABP forecasting
    python -m downstream.generation.intra_modal_forecast.prepare_data --signal-type abp --n-cases 5

    # 시각화 포함
    python -m downstream.generation.intra_modal_forecast.prepare_data --signal-type ecg --n-cases 5 --visualize
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from downstream._save_utils import (
    add_signal_dtype_arg,
    stack_attr_destructive,
)
from downstream._kfold_utils import (
    stratified_kfold_patient_splits,
    summarize_splits,
)

TARGET_SR: float = 100.0

# Task #13 Waveform Forecasting: All modality.
# Main paper: ECG, ABP, PPG, CO2 (universal coverage).
# Appendix: RESP_Impedance, AWP, CVP, ICP, PAP (rare/specialized cohorts).
# v2: RESP → resp_impedance(8). RESP_Flow(9)는 다운스트림 데이터 소스 부재
# (MIMIC-III=impedance only, VitalDB Open=resp 없음, flow는 K-MIMIC pretrain 전용)
# 이므로 forecast 타깃에서 제외. SSOT: data.spatial_map.SIGNAL_KEY_TO_TYPE.
MAIN_SIGNAL_TYPES = ["ecg", "abp", "ppg", "co2"]
APPENDIX_SIGNAL_TYPES = ["resp_impedance", "awp", "cvp", "icp", "pap"]
ALL_SIGNAL_TYPES = MAIN_SIGNAL_TYPES + APPENDIX_SIGNAL_TYPES


# ---- 데이터 구조 ----


@dataclass
class ForecastSample:
    """Forecasting 샘플."""
    context: np.ndarray     # (context_samples,) 과거 waveform
    target: np.ndarray      # (target_samples,) 미래 waveform (ground truth)
    signal_type: str
    case_id: str
    win_start_sec: float


@dataclass
class MultiInputForecastSample:
    """Multi-input forecasting 샘플 (context: 여러 signal, target: 단일 signal)."""
    context: np.ndarray       # (n_context_signals, context_samples) 과거 멀티채널
    target: np.ndarray        # (target_samples,) 미래 target signal
    context_signal_types: list[str]   # 각 context 채널의 signal type
    target_signal_type: str
    case_id: str
    win_start_sec: float


# ---- MIMIC-III 로더 ----


def _load_mimic3_signal(
    n_cases: int,
    signal_type: str,
) -> list[dict]:
    """MIMIC-III에서 단일 signal type 데이터를 로드한다."""
    from data.parser.mimic3_waveform import (
        scan_abp_records, load_and_preprocess_record,
    )

    if signal_type not in ("abp", "ecg", "ppg"):
        # Task #13 main 4 (incl. co2) 및 appendix 5: 단일 채널 single-input 은
        # 현재 scan_abp_records (ABP-필수) 경로만 지원. CO2/RESP/AWP/CVP/ICP/PAP 는
        # --multi-input 모드 + ICH waveform_dir 경유 추출 필요.
        print(
            f"  WARNING: single-input MIMIC-III loader supports abp/ecg/ppg only "
            f"(got {signal_type}). For Task #13 appendix modalities, use --multi-input "
            f"with --waveform-dir pointing at the ICH-style multi-channel extraction."
        )
        return []

    records = scan_abp_records(max_records=n_cases * 5, verbose=False)

    # 필요 채널 필터
    filtered = []
    for r in records:
        if signal_type == "abp" and r.has_abp:
            filtered.append(r)
        elif signal_type == "ecg" and r.has_ecg:
            filtered.append(r)
        elif signal_type == "ppg" and r.has_ppg:
            filtered.append(r)

    filtered = filtered[:n_cases]

    results = []
    for i, info in enumerate(filtered):
        case = load_and_preprocess_record(info, [signal_type])
        if case and signal_type in case.signals:
            sig = case.signals[signal_type]
            if len(sig) >= int(60 * TARGET_SR):
                results.append({
                    "case_id": info.record_name,
                    "signal": sig,
                })

    print(f"  Loaded {len(results)} cases with {signal_type.upper()} from MIMIC-III")
    return results


# ---- VitalDB 로더 ----


def _load_vitaldb_signal(
    n_cases: int,
    signal_type: str,
    offset_from_end: int = 200,
    vital_dir: str | None = None,
) -> list[dict]:
    """VitalDB에서 단일 signal type 의 연속 waveform 을 로드한다.

    cross_modal._load_vitaldb_cases 와 동일하게 ``load_pilot_cases`` 를 사용하되,
    forecast 는 단일 채널의 연속 waveform 에서 (context, target) 쌍을 뽑으므로
    해당 signal_type 트랙만 가져와 ``{"case_id", "signal"}`` 형태로 반환한다
    (``_load_mimic3_signal`` 출력 규약과 동일).

    Returns
    -------
    list of {"case_id": str, "signal": np.ndarray}
    """
    from downstream.data_utils import load_pilot_cases

    print(f"  Loading {n_cases} VitalDB cases (signal={signal_type})...")
    raw_cases = load_pilot_cases(
        n_cases=n_cases,
        offset_from_end=offset_from_end,
        signal_types=[signal_type],
        vital_dir=vital_dir,
    )

    min_samples = int(60 * TARGET_SR)
    results: list[dict] = []
    for rc in raw_cases:
        if signal_type not in rc.tracks:
            continue
        sig = np.asarray(rc.tracks[signal_type], dtype=np.float64).ravel()
        if len(sig) < min_samples:
            continue
        results.append({
            "case_id": f"vitaldb_{rc.case_id}",
            "signal": sig,
        })

    print(f"  Loaded {len(results)} cases with {signal_type.upper()} from VitalDB")
    return results


# ---- 윈도우 추출 ----


def _load_mimic3_multi_input(
    waveform_dir: str,
    context_signals: list[str],
    target_signal: str,
) -> list[dict]:
    """ICH waveform 디렉토리에서 context_signals + target_signal을 모두 가진
    레코드 목록을 로드한다.

    Returns
    -------
    list of {"case_id": str, "signals": {type: ndarray}}
    """
    from downstream.acute_event.intracranial_hypertension.prepare_data import (
        load_patient_signals,
    )

    cases = load_patient_signals(Path(waveform_dir))

    needed = set(context_signals) | {target_signal}
    filtered = []
    for c in cases:
        if needed.issubset(c["signals"].keys()):
            filtered.append(c)

    print(
        f"  Filtered {len(filtered)} / {len(cases)} cases with "
        f"all of {sorted(needed)}"
    )
    return filtered


def extract_multi_input_samples(
    cases: list[dict],
    context_signals: list[str],
    target_signal: str,
    context_sec: float = 30.0,
    target_sec: float = 10.0,
    stride_sec: float = 15.0,
) -> list[MultiInputForecastSample]:
    """Multi-input paired windows: context = 멀티채널, target = 단일채널."""
    ctx_samples = int(context_sec * TARGET_SR)
    tgt_samples = int(target_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    total_needed = ctx_samples + tgt_samples

    samples: list[MultiInputForecastSample] = []

    for case in cases:
        sigs = case["signals"]
        # 모든 채널의 길이 정렬 (min 기준)
        min_len = min(len(sigs[s]) for s in context_signals + [target_signal])
        if min_len < total_needed:
            continue

        for start in range(0, min_len - total_needed + 1, stride_samples):
            ctx_stack = np.stack(
                [sigs[s][start:start + ctx_samples] for s in context_signals],
                axis=0,
            )  # (n_context_signals, ctx_samples)
            tgt = sigs[target_signal][
                start + ctx_samples:start + ctx_samples + tgt_samples
            ]

            if np.isnan(ctx_stack).any() or np.isnan(tgt).any():
                continue

            samples.append(MultiInputForecastSample(
                context=ctx_stack,
                target=tgt,
                context_signal_types=list(context_signals),
                target_signal_type=target_signal,
                case_id=case["case_id"],
                win_start_sec=start / TARGET_SR,
            ))

    return samples


def save_multi_input_dataset(
    train_samples: list[MultiInputForecastSample],
    test_samples: list[MultiInputForecastSample],
    context_signals: list[str],
    target_signal: str,
    context_sec: float,
    target_sec: float,
    source: str,
    out_dir: str,
    signal_dtype: torch.dtype = torch.float16,
    chunk_windows: int = 2000,
    val_samples: list[MultiInputForecastSample] | None = None,
    fold_idx: int | None = None,
    n_folds: int = 1,
) -> list[Path]:
    """Multi-input (context=멀티채널, target=단일채널) 쌍을 split별 chunk 저장.

    파일명:
      - 단일: ``forecasting_multi_{source}_{ctx_tag}_to_{target}_ctx{C}s_tgt{T}s_{split}_chunk{C}.pt``
      - K-fold: ``..._tgt{T}s_fold{F}_{split}_chunk{C}.pt``

    context 는 2D (n_context, ctx_T) 이므로 chunk 텐서는 (n, n_context, ctx_T).
    ``fold_idx`` 지정 시 파일명에 ``_fold{F}``, ``val_samples`` 있으면 3 split 저장.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ctx_tag = "+".join(context_signals)
    base = (
        f"forecasting_multi_{source}_{ctx_tag}_to_{target_signal}"
        f"_ctx{int(context_sec)}s_tgt{int(target_sec)}s"
    )
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""

    split_items: list[tuple[str, list[MultiInputForecastSample]]] = [
        ("train", train_samples)
    ]
    if val_samples is not None:
        split_items.append(("val", val_samples))
    split_items.append(("test", test_samples))

    saved: list[Path] = []
    for split_name, samples in split_items:
        if not samples:
            continue
        chunk_idx = 0
        total = 0
        for bstart in range(0, len(samples), chunk_windows):
            batch = samples[bstart:bstart + chunk_windows]
            n = len(batch)
            case_ids = [s.case_id for s in batch]
            # context 는 다차원 — stack_attr_destructive(1D 가정) 대신 직접 build.
            ctx = torch.empty(
                (n,) + batch[0].context.shape, dtype=signal_dtype
            )  # (n, n_context, ctx_T)
            for i, s in enumerate(batch):
                ctx[i].copy_(torch.from_numpy(s.context))
                s.context = None
            tgt = stack_attr_destructive(batch, "target", signal_dtype)  # (n, tgt_T)

            chunk = {
                "split": split_name,
                "context": ctx,
                "target": tgt,
                "case_ids": case_ids,
                "metadata": {
                    "task": "multi_input_forecasting",
                    "source": source,
                    "context_signal_types": list(context_signals),
                    "target_signal_type": target_signal,
                    "context_sec": context_sec,
                    "target_sec": target_sec,
                    "sampling_rate": TARGET_SR,
                    "fold_idx": fold_idx,
                    "n_folds": n_folds,
                    "split": split_name,
                    "chunk_idx": chunk_idx,
                    "n_windows": n,
                    "signal_dtype": str(signal_dtype).replace("torch.", ""),
                },
            }
            out_file = (
                out_path / f"{base}{fold_suffix}_{split_name}_chunk{chunk_idx}.pt"
            )
            torch.save(chunk, out_file)
            file_size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"  Saved: {out_file.name} ({n} samples, {file_size_mb:.2f} MB)")
            saved.append(out_file)
            total += n
            chunk_idx += 1
            del batch, ctx, tgt, chunk
            gc.collect()
        print(f"  {split_name}: {chunk_idx} chunk(s), {total} samples")

    train_samples.clear()
    test_samples.clear()
    if val_samples is not None:
        val_samples.clear()
    gc.collect()
    return saved


def extract_forecast_samples(
    cases: list[dict],
    signal_type: str,
    context_sec: float = 30.0,
    target_sec: float = 10.0,
    stride_sec: float = 15.0,
) -> list[ForecastSample]:
    """과거(context) + 미래(target) 윈도우 쌍을 추출한다."""
    ctx_samples = int(context_sec * TARGET_SR)
    tgt_samples = int(target_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    total_needed = ctx_samples + tgt_samples

    samples: list[ForecastSample] = []

    for case in cases:
        sig = case["signal"]
        if len(sig) < total_needed:
            continue

        for start in range(0, len(sig) - total_needed + 1, stride_samples):
            context = sig[start:start + ctx_samples]
            target = sig[start + ctx_samples:start + ctx_samples + tgt_samples]

            if np.isnan(context).any() or np.isnan(target).any():
                continue

            samples.append(ForecastSample(
                context=context,
                target=target,
                signal_type=signal_type,
                case_id=case["case_id"],
                win_start_sec=start / TARGET_SR,
            ))

    return samples


# ---- 저장 ----


def save_dataset(
    train_samples: list[ForecastSample],
    test_samples: list[ForecastSample],
    signal_type: str,
    context_sec: float,
    target_sec: float,
    source: str,
    out_dir: str,
    signal_dtype: torch.dtype = torch.float16,
    chunk_windows: int = 2000,
    val_samples: list[ForecastSample] | None = None,
    fold_idx: int | None = None,
    n_folds: int = 1,
) -> list[Path]:
    """(context, target) 쌍을 split별 chunk .pt로 저장 (sepsis chunk 패턴).

    파일명:
      - 단일(fold 미사용): ``forecasting_{source}_{signal}_ctx{C}s_tgt{T}s_{split}_chunk{C}.pt``
      - K-fold:           ``forecasting_{source}_{signal}_ctx{C}s_tgt{T}s_fold{F}_{split}_chunk{C}.pt``

    각 chunk 는 최대 ``chunk_windows`` 개 샘플을 담고, chunk 단위로 텐서를
    destructive build → save → ``del + gc.collect()`` 한다. ``fold_idx`` 지정 시
    파일명에 ``_fold{F}`` 가 붙고, ``val_samples`` 가 있으면 3 split 을 저장한다.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base = (
        f"forecasting_{source}_{signal_type}"
        f"_ctx{int(context_sec)}s_tgt{int(target_sec)}s"
    )
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""

    split_items: list[tuple[str, list[ForecastSample]]] = [("train", train_samples)]
    if val_samples is not None:
        split_items.append(("val", val_samples))
    split_items.append(("test", test_samples))

    saved: list[Path] = []
    for split_name, samples in split_items:
        if not samples:
            continue
        chunk_idx = 0
        total = 0
        for bstart in range(0, len(samples), chunk_windows):
            batch = samples[bstart:bstart + chunk_windows]
            n = len(batch)
            case_ids = [s.case_id for s in batch]
            ctx = stack_attr_destructive(batch, "context", signal_dtype)  # (n, ctx_T)
            tgt = stack_attr_destructive(batch, "target", signal_dtype)   # (n, tgt_T)

            chunk = {
                "split": split_name,
                "context": ctx,
                "target": tgt,
                "case_ids": case_ids,
                "metadata": {
                    "task": "vital_sign_forecasting",
                    "source": source,
                    "signal_type": signal_type,
                    "context_sec": context_sec,
                    "target_sec": target_sec,
                    "sampling_rate": TARGET_SR,
                    "fold_idx": fold_idx,
                    "n_folds": n_folds,
                    "split": split_name,
                    "chunk_idx": chunk_idx,
                    "n_windows": n,
                    "signal_dtype": str(signal_dtype).replace("torch.", ""),
                },
            }
            out_file = (
                out_path / f"{base}{fold_suffix}_{split_name}_chunk{chunk_idx}.pt"
            )
            torch.save(chunk, out_file)
            file_size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"  Saved: {out_file.name} ({n} samples, {file_size_mb:.2f} MB)")
            saved.append(out_file)
            total += n
            chunk_idx += 1
            del batch, ctx, tgt, chunk
            gc.collect()
        print(f"  {split_name}: {chunk_idx} chunk(s), {total} samples")

    train_samples.clear()
    test_samples.clear()
    if val_samples is not None:
        val_samples.clear()
    gc.collect()
    return saved


# ---- 통계 ----


def print_stats(name: str, samples: list[ForecastSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return

    n = len(samples)
    ctx_dur = len(samples[0].context) / TARGET_SR
    tgt_dur = len(samples[0].target) / TARGET_SR
    print(f"  {name}: {n} samples (context={ctx_dur:.0f}s, target={tgt_dur:.0f}s)")


# ---- 시각화 ----


def _visualize(
    samples: list[ForecastSample],
    signal_type: str,
    context_sec: float,
    target_sec: float,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping.")
        return

    print("\nGenerating visualizations...")

    n_show = min(4, len(samples))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), squeeze=False)

    for i in range(n_show):
        ax = axes[i, 0]
        s = samples[i]

        ctx_t = np.arange(len(s.context)) / TARGET_SR
        tgt_t = np.arange(len(s.target)) / TARGET_SR + context_sec

        ax.plot(ctx_t, s.context, color="steelblue", linewidth=0.6, label="Context (input)")
        ax.plot(tgt_t, s.target, color="tab:red", linewidth=0.6, label="Target (to predict)")
        ax.axvline(context_sec, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel(f"{signal_type.upper()}")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, context_sec + target_sec)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(f"Vital Sign Forecasting - {signal_type.upper()} "
                 f"(context={context_sec:.0f}s -> predict {target_sec:.0f}s)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / f"forecasting_{signal_type}_examples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---- 메인 ----


def prepare_forecasting(
    source: str = "mimic3",
    signal_type: str = "ecg",
    n_cases: int = 10,
    context_sec: float = 30.0,
    target_sec: float = 10.0,
    stride_sec: float = 15.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/forecasting",
    visualize: bool = False,
    signal_dtype: torch.dtype = torch.float16,
    chunk_windows: int = 2000,
    n_folds: int = 1,
    seed: int = 42,
    vital_dir: str | None = None,
) -> list[Path]:
    # "all"이면 모든 signal type에 대해 반복
    if signal_type == "all":
        stypes = ALL_SIGNAL_TYPES
    else:
        stypes = [signal_type]

    saved_paths = []
    for stype in stypes:
        print(f"\n{'='*60}")
        print(f"  Vital Sign Forecasting: {stype.upper()}")
        print(f"  Source: {source}")
        print(f"  Context: {context_sec}s -> Predict: {target_sec}s")
        print(f"{'='*60}")

        # 1. 데이터 로드
        print(f"\n[1/4] Loading {stype.upper()} data...")
        if source == "mimic3":
            cases = _load_mimic3_signal(n_cases, stype)
        elif source == "vitaldb":
            cases = _load_vitaldb_signal(n_cases, stype, vital_dir=vital_dir)
        else:
            print(f"ERROR: Unknown source '{source}'", file=sys.stderr)
            continue

        if not cases:
            print(f"  SKIP: No valid {stype.upper()} data")
            continue

        # ── K-fold CV (n_folds>=2): case(=patient)-level group K-fold ──
        if n_folds >= 2:
            # generation = no label → 더미 uniform label 로 plain group K-fold.
            # forecast case 는 case_id 가 곧 환자/레코드 단위 → 그룹 키로 사용.
            print(f"\n[2-4/4] Patient(case)-level {n_folds}-fold CV...")
            case_ids = sorted({str(c["case_id"]) for c in cases})
            patient_to_labels = {pid: [0] for pid in case_ids}
            splits = stratified_kfold_patient_splits(
                case_ids, patient_to_labels, n_folds=n_folds, seed=seed,
            )
            print(summarize_splits(splits))

            for fold_idx, (train_p, val_p, test_p) in enumerate(splits):
                print(f"\n  Fold {fold_idx}: extract samples + save...")
                fold_samples: dict[str, list[ForecastSample]] = {}
                for split_name, pset in (
                    ("train", train_p), ("val", val_p), ("test", test_p),
                ):
                    split_cases = [c for c in cases if str(c["case_id"]) in pset]
                    fold_samples[split_name] = extract_forecast_samples(
                        split_cases, stype, context_sec, target_sec, stride_sec,
                    )
                if not any(fold_samples.values()):
                    print(f"    WARN: fold {fold_idx} no samples — skip.")
                    continue
                paths = save_dataset(
                    fold_samples["train"], fold_samples["test"],
                    stype, context_sec, target_sec, source, out_dir,
                    signal_dtype=signal_dtype,
                    chunk_windows=chunk_windows,
                    val_samples=fold_samples["val"],
                    fold_idx=fold_idx,
                    n_folds=n_folds,
                )
                saved_paths.extend(paths)
            continue

        # ── 단일 train/test (n_folds==1, back-compat) ───────────────
        # 2. Train/Test 분할 (case-level)
        print(f"\n[2/4] Splitting (ratio={train_ratio})...")
        rng = np.random.default_rng(seed)
        indices = list(range(len(cases)))
        rng.shuffle(indices)
        n_train = max(1, int(len(cases) * train_ratio))
        train_cases = [cases[i] for i in indices[:n_train]]
        test_cases = [cases[i] for i in indices[n_train:]]
        print(f"  Train: {len(train_cases)} cases, Test: {len(test_cases)} cases")

        # 3. 윈도우 추출
        print(f"\n[3/4] Extracting forecast samples...")
        train_samples = extract_forecast_samples(
            train_cases, stype, context_sec, target_sec, stride_sec,
        )
        test_samples = extract_forecast_samples(
            test_cases, stype, context_sec, target_sec, stride_sec,
        )
        print_stats("Train", train_samples)
        print_stats("Test", test_samples)

        if not train_samples and not test_samples:
            print(f"  SKIP: No samples for {stype.upper()}")
            continue

        # (옵션) 시각화 — save 가 샘플을 destructive 하게 비우므로 먼저 수행.
        if visualize:
            all_samples = train_samples + test_samples
            _visualize(all_samples, stype, context_sec, target_sec, Path(out_dir))

        # 4. 저장 (split별 chunk)
        print(f"\n[4/4] Saving...")
        paths = save_dataset(
            train_samples, test_samples,
            stype, context_sec, target_sec, source, out_dir,
            signal_dtype=signal_dtype,
            chunk_windows=chunk_windows,
        )
        saved_paths.extend(paths)

    if saved_paths:
        print(f"\n{'='*60}")
        print(f"  Forecasting data ready: {len(saved_paths)} signal type(s)")
        for p in saved_paths:
            print(f"    {p}")
        print(f"{'='*60}")

    return saved_paths


def prepare_multi_input_forecasting(
    waveform_dir: str,
    context_signals: list[str],
    target_signal: str,
    context_sec: float = 30.0,
    target_sec: float = 10.0,
    stride_sec: float = 15.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/forecasting",
    source: str = "mimic3",
    signal_dtype: torch.dtype = torch.float16,
    chunk_windows: int = 2000,
    n_folds: int = 1,
    seed: int = 42,
) -> list[Path]:
    """ABP+ICP → ICP 같은 multi-input forecasting 데이터셋 구축."""
    print(f"\n{'='*60}")
    print(f"  Multi-Input Forecasting: {'+'.join(context_signals)} → {target_signal}")
    print(f"  Source: {source} waveform dir: {waveform_dir}")
    print(f"  Context: {context_sec}s, Target: {target_sec}s")
    print(f"{'='*60}")

    cases = _load_mimic3_multi_input(waveform_dir, context_signals, target_signal)
    if not cases:
        print("  SKIP: No cases with all required signals")
        return []

    # ── K-fold CV (n_folds>=2): case(=patient)-level group K-fold ──
    if n_folds >= 2:
        print(f"\n  Patient(case)-level {n_folds}-fold CV...")
        case_ids = sorted({str(c["case_id"]) for c in cases})
        patient_to_labels = {pid: [0] for pid in case_ids}
        splits = stratified_kfold_patient_splits(
            case_ids, patient_to_labels, n_folds=n_folds, seed=seed,
        )
        print(summarize_splits(splits))

        saved_paths: list[Path] = []
        for fold_idx, (train_p, val_p, test_p) in enumerate(splits):
            print(f"\n  Fold {fold_idx}: extract samples + save...")
            fold_samples: dict[str, list[MultiInputForecastSample]] = {}
            for split_name, pset in (
                ("train", train_p), ("val", val_p), ("test", test_p),
            ):
                split_cases = [c for c in cases if str(c["case_id"]) in pset]
                fold_samples[split_name] = extract_multi_input_samples(
                    split_cases, context_signals, target_signal,
                    context_sec, target_sec, stride_sec,
                )
            if not any(fold_samples.values()):
                print(f"    WARN: fold {fold_idx} no samples — skip.")
                continue
            paths = save_multi_input_dataset(
                fold_samples["train"], fold_samples["test"],
                context_signals, target_signal,
                context_sec, target_sec, source, out_dir,
                signal_dtype=signal_dtype,
                chunk_windows=chunk_windows,
                val_samples=fold_samples["val"],
                fold_idx=fold_idx,
                n_folds=n_folds,
            )
            saved_paths.extend(paths)
        return saved_paths

    # ── 단일 train/test (n_folds==1, back-compat) ───────────────
    # Train/Test split (case-level group)
    rng = np.random.default_rng(seed)
    indices = list(range(len(cases)))
    rng.shuffle(indices)
    n_train = max(1, int(len(cases) * train_ratio))
    train_cases = [cases[i] for i in indices[:n_train]]
    test_cases = [cases[i] for i in indices[n_train:]]
    print(f"  Train: {len(train_cases)}, Test: {len(test_cases)}")

    train_samples = extract_multi_input_samples(
        train_cases, context_signals, target_signal,
        context_sec, target_sec, stride_sec,
    )
    test_samples = extract_multi_input_samples(
        test_cases, context_signals, target_signal,
        context_sec, target_sec, stride_sec,
    )
    print(f"  Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

    if not train_samples and not test_samples:
        print("  SKIP: No valid samples extracted")
        return []

    return save_multi_input_dataset(
        train_samples, test_samples,
        context_signals, target_signal,
        context_sec, target_sec, source, out_dir,
        signal_dtype=signal_dtype,
        chunk_windows=chunk_windows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vital Sign Forecasting - Data Preparation",
    )
    parser.add_argument("--vital-dir", type=str, default=None,
                        help="로컬 VitalDB Open .vital 디렉토리 (source=vitaldb 시 API 대신 사용)")
    parser.add_argument("--source", type=str, default="mimic3",
                        choices=["mimic3", "vitaldb"],
                        help="Data source. vitaldb 는 single-input 만 지원 "
                             "(load_pilot_cases 경유 연속 waveform).")
    parser.add_argument("--signal-type", type=str, default="ecg",
                        choices=ALL_SIGNAL_TYPES + ["all"],
                        help="Signal type to forecast ('all' for all types, "
                             "single-input only)")
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--context-sec", type=float, default=30.0,
                        help="Context window length (seconds)")
    parser.add_argument("--target-sec", type=float, default=10.0,
                        help="Target prediction length (seconds)")
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/forecasting")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--chunk-windows", type=int, default=2000,
        help="Samples per split chunk file (sepsis-style chunked save, "
             "lowers peak RAM/disk). Default 2000.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Patient(case)-level K-fold CV. >=2 면 fold별 train/val/test chunk "
             "저장 (파일명에 _fold{F}). 1 이면 단일 train/test (back-compat). "
             "Default 5.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Patient split RNG seed.",
    )

    # Multi-input forecasting 플래그
    parser.add_argument(
        "--multi-input", action="store_true",
        help="Multi-input 모드: --context-channels + --target-channel 지정",
    )
    parser.add_argument(
        "--context-channels", type=str, nargs="+", default=None,
        help="Context로 사용할 signal types (예: --context-channels abp icp)",
    )
    parser.add_argument(
        "--target-channel", type=str, default=None,
        help="Target signal type (예: --target-channel icp)",
    )
    parser.add_argument(
        "--waveform-dir", type=str,
        default="datasets/raw/mimic3-waveform-ich",
        help="MIMIC-III waveform directory for multi-input (ICH download 결과)",
    )
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()
    sig_dtype = dtype_map(args.signal_dtype)

    if args.multi_input:
        if not args.context_channels or not args.target_channel:
            parser.error(
                "--multi-input 모드는 --context-channels 와 --target-channel 필요"
            )
        prepare_multi_input_forecasting(
            waveform_dir=args.waveform_dir,
            context_signals=args.context_channels,
            target_signal=args.target_channel,
            context_sec=args.context_sec,
            target_sec=args.target_sec,
            stride_sec=args.stride_sec,
            train_ratio=args.train_ratio,
            out_dir=args.out_dir,
            source=args.source,
            signal_dtype=sig_dtype,
            chunk_windows=args.chunk_windows,
            n_folds=args.n_folds,
            seed=args.seed,
        )
    else:
        prepare_forecasting(
            source=args.source,
            signal_type=args.signal_type,
            n_cases=args.n_cases,
            context_sec=args.context_sec,
            target_sec=args.target_sec,
            stride_sec=args.stride_sec,
            train_ratio=args.train_ratio,
            out_dir=args.out_dir,
            visualize=args.visualize,
            vital_dir=args.vital_dir,
            signal_dtype=sig_dtype,
            chunk_windows=args.chunk_windows,
            n_folds=args.n_folds,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
