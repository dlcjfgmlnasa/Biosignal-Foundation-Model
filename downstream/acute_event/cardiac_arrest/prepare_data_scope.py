# -*- coding:utf-8 -*-
"""Imminent Cardiac Arrest / Mortality Detection — 데이터 준비 (SCOPE, window-level).

기존 MIMIC-III 기반 `prepare_data.py` 의 **소스 교체판**. 신호 소스만
SCOPE(SNUH ICU, `.vital`)로 바꾸고, 윈도우 추출 / horizon band / stratified
K-fold / gap policy / 저장 로직은 `prepare_data.py` 의 것을 그대로 재사용한다.

데이터셋: SCOPE (KHDP `icu-cardiac-arrest`)
  - 4,517 ICU admission `.vital` (파일명 = 9자리 case_id)
  - 트랙: `Signals/ECG_II` (ECG II, 500Hz) + `Signals/PLETH` (PPG, 125Hz)
  - 길이: admission 당 24–48h, **outcome 시점을 2100-01-01 00:00:00 UTC 에 anchor**
  - metadata: `metadata_v1.0.xlsx` (outcome ∈ {Discharge, Death, Cardiac Arrest})

MIMIC 대비 단순화 / 차이:
  - **center = 고정 anchor (2100-01-01)** — 모든 케이스 공통. MIMIC 의 Δt risk-set
    매칭(`assign_negative_anchors`)이 불필요: SCOPE 는 모든 케이스가 outcome 직전
    24–48h 만 담고 같은 시점에 정렬돼 있어, `extract_ca_window_samples` 의 horizon
    band 로직이 positive/negative 의 lead-time 매칭을 자동 수행한다.
  - **입력 = ECG + PPG 2채널뿐** (SCOPE 에 ABP/CO2/RespImp 없음).
  - **fold split = case_id 단위** (metadata 에 환자-stay 연결 ID 없음 → 같은 환자가
    여러 admission 을 가져도 연결 불가. 경미한 cross-admission leakage caveat).
  - **정렬보존 전처리**: `_apply_pipeline`(최장 NaN-free 세그먼트만 반환)은 anchor·
    채널 정렬을 깨므로 사용하지 않고, 전체 band 타임라인을 유지한 채 range→fill→
    filter→resample 하고 gap 은 NaN 으로 남겨 gap policy 가 처리하게 한다.

Task 분리 (각자 독립 binary, 서로 안 엮임):
  - ``--task arrest`` : pos=Cardiac Arrest(74) / neg=Discharge(4095) / Death 제외
  - ``--task death``  : pos=Death(348)        / neg=Discharge(4095) / Cardiac Arrest 제외

2-pass (네트워크 마운트 스토리지 I/O 최적화):
  - Pass 1: 각 .vital 을 1회만 로딩 → anchor 직전 band 만 전처리 → case 캐시(.pt) 저장.
  - Pass 2: 캐시에서 stratified K-fold × horizon 별 윈도우 추출 → chunk .pt 저장.

사용법:
    python -m downstream.acute_event.cardiac_arrest.prepare_data_scope \
        --metadata datasets/icu-cardiac-arrest/1.0/metadata_v1.0.xlsx \
        --vital-dir datasets/icu-cardiac-arrest/1.0 \
        --task arrest \
        --input-signals ecg ppg \
        --window-secs 600 --horizon-mins 5 15 30 \
        --out-dir datasets/processed/scope_cardiac_arrest \
        --workers 8
"""

from __future__ import annotations

import argparse
import gc
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from data.parser._common import resample_to_target
from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    _apply_filter,
    _apply_median_filter,
    _apply_notch_filter,
    _apply_range_check,
    _fill_short_nan_gaps,
)
from downstream._kfold_utils import (
    stratified_kfold_patient_splits,
    summarize_splits,
)
from downstream._save_utils import add_signal_dtype_arg
from downstream.acute_event.cardiac_arrest.prepare_data import (
    TARGET_SR,
    GapStats,
    _consume_to_tensors_ca,
    extract_ca_window_samples,
)


# outcome 를 anchor 한 고정 reference: 2100-01-01 00:00:00 UTC (POSIX 4102444800).
# .vital dtend 가 이 값 근처(±수초~1h). 모든 케이스 공통 anchor.
OUTCOME_ANCHOR_TS: float = 4102444800.0
OUTCOME_ANCHOR_DT: datetime = datetime.fromtimestamp(OUTCOME_ANCHOR_TS, tz=timezone.utc)

# SCOPE .vital 트랙 → signal_type 매핑
SCOPE_TRACK_MAP: dict[str, str] = {
    "Signals/ECG_II": "ecg",
    "Signals/PLETH": "ppg",
}

# outcome 라벨 (metadata 'outcome' 컬럼 값, 공백 strip 후 비교)
OUTCOME_DISCHARGE = "Discharge"
OUTCOME_DEATH = "Death"
OUTCOME_ARREST = "Cardiac Arrest"

# task → (positive outcome, 제외 outcome). negative 는 항상 Discharge.
TASK_SPEC: dict[str, tuple[str, str]] = {
    "arrest": (OUTCOME_ARREST, OUTCOME_DEATH),
    "death": (OUTCOME_DEATH, OUTCOME_ARREST),
}


# ── 코호트 구성 (metadata xlsx) ──────────────────────────────


def build_scope_cohort(
    metadata_path: str,
    vital_dir: str,
    task: str,
) -> list[dict]:
    """metadata_v1.0.xlsx → task 별 코호트 case 리스트.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "label": int, "vital_path": Path}
        label 1=positive, 0=Discharge(negative). 파일 부재 case 는 제외.
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas 필요. pip install pandas openpyxl", file=sys.stderr)
        sys.exit(1)

    if task not in TASK_SPEC:
        print(f"ERROR: unknown task '{task}' (arrest|death)", file=sys.stderr)
        sys.exit(1)
    pos_outcome, excl_outcome = TASK_SPEC[task]

    df = pd.read_excel(metadata_path)
    if "case_id" not in df.columns or "outcome" not in df.columns:
        print(f"ERROR: metadata 에 case_id/outcome 컬럼 없음: {list(df.columns)}",
              file=sys.stderr)
        sys.exit(1)

    df["outcome"] = df["outcome"].astype(str).str.strip()
    vdir = Path(vital_dir)

    cohort: list[dict] = []
    n_excl = n_missing_file = 0
    for _, row in df.iterrows():
        outcome = row["outcome"]
        if outcome == pos_outcome:
            label = 1
        elif outcome == OUTCOME_DISCHARGE:
            label = 0
        else:
            n_excl += 1  # 제외 그룹 (다른 outcome)
            continue

        # case_id → 파일명. 9자리 정수(예: 100008162) → "{case_id}.vital".
        raw = row["case_id"]
        try:
            cid = str(int(raw))
        except (ValueError, TypeError):
            cid = str(raw).strip()
        vpath = vdir / f"{cid}.vital"
        if not vpath.is_file():
            n_missing_file += 1
            continue

        cohort.append({
            "case_id": cid,
            "patient_id": cid,  # patient-stay 연결 ID 부재 → case_id 단위 split
            "label": label,
            "vital_path": vpath,
        })

    n_pos = sum(c["label"] for c in cohort)
    print(f"  Cohort [{task}]: {len(cohort)} cases "
          f"(pos={n_pos}, neg={len(cohort) - n_pos}); "
          f"excluded({excl_outcome})={n_excl}, missing_file={n_missing_file}")
    return cohort


# ── Pass 1: .vital → anchor 직전 band 캐시 ────────────────────


def _process_band(
    raw: np.ndarray,
    cfg,
    native_sr: float,
) -> np.ndarray | None:
    """band(원본 SR) → 정렬보존 전처리 → 100Hz. gap 은 NaN 으로 유지.

    `_apply_pipeline` 과 달리 최장 세그먼트 추출을 하지 않아 타임라인 길이/정렬이
    보존된다 (ECG·PPG 채널 정렬, anchor 정렬 유지). NaN 위치는 보존돼 downstream
    gap policy(valid_ratio drop + gap_mask)가 처리한다.
    """
    raw = np.asarray(raw, dtype=np.float64).flatten()
    if raw.size == 0:
        return None

    # chunk 경계 micro-NaN(<100ms) forward-fill, 진짜 disconnect 는 NaN 유지
    raw = _fill_short_nan_gaps(raw, max(1, int(0.1 * native_sr)))

    if cfg.valid_range is not None:
        raw, _ = _apply_range_check(raw, cfg.valid_range)

    nan_mask = np.isnan(raw)
    if nan_mask.all():
        return None

    # 필터링은 NaN 불가 → 보간 채움으로 임시 메움 후 필터, 이후 gap 복원
    filled = raw.copy()
    if nan_mask.any():
        idx = np.arange(len(filled))
        good = ~nan_mask
        filled[nan_mask] = np.interp(idx[nan_mask], idx[good], filled[good])

    if cfg.median_kernel > 0:
        filled = _apply_median_filter(filled, kernel_size=cfg.median_kernel)
    if cfg.notch_freq is not None:
        filled = _apply_notch_filter(filled, freq=cfg.notch_freq, sr=native_sr)
    filled = _apply_filter(filled, cfg, native_sr)

    if native_sr != TARGET_SR:
        out = resample_to_target(filled, orig_sr=native_sr, target_sr=TARGET_SR)
        # gap mask 를 target grid 로 nearest 매핑
        n_out = len(out)
        src_idx = np.clip(
            np.round(np.arange(n_out) * native_sr / TARGET_SR).astype(int),
            0, len(nan_mask) - 1,
        )
        out_mask = nan_mask[src_idx]
    else:
        out = np.asarray(filled, dtype=np.float64)
        out_mask = nan_mask

    out = out.astype(np.float32)
    out[out_mask] = np.nan  # long-gap 복원 → NaN
    return out


def load_and_cache_case(
    case: dict,
    input_signals: list[str],
    band_sec: float,
    cache_dir: Path,
) -> dict | None:
    """단일 .vital → anchor 직전 band 전처리 → case 캐시(.pt) 저장.

    캐시: {ecg, ppg: float32@100Hz (NaN=gap), band_start_ts, label, case_id}.
    Returns 캐시 메타(없으면 None). 이미 캐시 존재 시 재사용.
    """
    import vitaldb

    cid = case["case_id"]
    cache_path = cache_dir / f"{cid}.pt"
    if cache_path.is_file():
        try:
            meta = torch.load(cache_path, map_location="cpu", weights_only=False)
            return {"case_id": cid, "label": int(case["label"]),
                    "band_start_ts": meta["band_start_ts"],
                    "cache_path": cache_path}
        except Exception:
            pass  # 손상 캐시 → 재생성

    try:
        vf = vitaldb.VitalFile(str(case["vital_path"]))
    except Exception as exc:
        print(f"    [WARN] {cid}: load 실패 {exc}", file=sys.stderr)
        return None
    if vf.dtstart is None:
        return None
    dtstart = float(vf.dtstart)

    t0 = OUTCOME_ANCHOR_TS - band_sec  # band 시작 절대시각
    t1 = OUTCOME_ANCHOR_TS             # anchor (post-anchor tail 배제)

    sigs: dict[str, np.ndarray] = {}
    band_start_ts_target: float | None = None
    for track, stype in SCOPE_TRACK_MAP.items():
        if stype not in input_signals:
            continue
        cfg = SIGNAL_CONFIGS.get(stype)
        if cfg is None:
            continue
        try:
            trk = vf.find_track(track)
            native_sr = trk.srate if trk is not None and trk.srate > 0 else 0.0
            if native_sr <= 0:
                continue
            full = vf.to_numpy(track, interval=1.0 / native_sr)
        except Exception as exc:
            print(f"    [WARN] {cid}/{track}: {exc}", file=sys.stderr)
            continue
        if full is None or len(full) == 0:
            continue
        full = full.flatten()

        # band 절대시각 → native 인덱스 슬라이스
        i0 = max(0, int(round((t0 - dtstart) * native_sr)))
        i1 = min(len(full), int(round((t1 - dtstart) * native_sr)))
        if i1 - i0 < int(60 * native_sr):  # 최소 60s 미만이면 사용 불가
            continue
        band = full[i0:i1]
        del full

        processed = _process_band(band, cfg, native_sr)
        if processed is None:
            continue

        # 실제 band 시작 절대시각 (clip 반영) → target grid
        actual_start_ts = dtstart + i0 / native_sr
        if band_start_ts_target is None:
            band_start_ts_target = actual_start_ts
        else:
            band_start_ts_target = max(band_start_ts_target, actual_start_ts)
        sigs[stype] = processed

    if not sigs or band_start_ts_target is None:
        return None

    # 채널 간 길이/시작 정렬: 공통 시작(max start) 기준 trim
    aligned: dict[str, np.ndarray] = {}
    for stype, arr in sigs.items():
        # 각 채널 시작이 band_start_ts_target 보다 이르면 앞부분 잘라 정렬
        # (개별 채널 actual_start 차이 무시 가능 수준 — 모두 t0 기준 슬라이스라 동일).
        aligned[stype] = arr
    min_len = min(len(a) for a in aligned.values())
    aligned = {s: a[:min_len] for s, a in aligned.items()}

    save_obj = {
        "band_start_ts": band_start_ts_target,
        "label": int(case["label"]),
        "case_id": cid,
        **aligned,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(save_obj, cache_path)
    return {"case_id": cid, "label": int(case["label"]),
            "band_start_ts": band_start_ts_target, "cache_path": cache_path}


def run_pass1(
    cohort: list[dict],
    input_signals: list[str],
    band_sec: float,
    cache_dir: Path,
    workers: int = 8,
) -> list[dict]:
    """모든 case 의 band 캐시 생성 (ThreadPool 병렬, 네트워크 I/O 최적화)."""
    print(f"\n[Pass 1] band 캐시 생성: {len(cohort)} cases "
          f"(band={band_sec / 60:.1f}min, workers={workers}) → {cache_dir}")
    metas: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(load_and_cache_case, c, input_signals, band_sec, cache_dir): c
            for c in cohort
        }
        for fut in as_completed(futs):
            done += 1
            try:
                m = fut.result()
            except Exception as exc:
                print(f"    [WARN] case 실패: {exc}", file=sys.stderr)
                m = None
            if m is not None:
                metas.append(m)
            if done % 200 == 0:
                print(f"    {done}/{len(cohort)} 처리 (cached={len(metas)})")
    n_pos = sum(m["label"] for m in metas)
    print(f"  Pass 1 완료: {len(metas)}/{len(cohort)} cached "
          f"(pos={n_pos}, neg={len(metas) - n_pos})")
    return metas


# ── Pass 2: 캐시 → fold × horizon 윈도잉 → 저장 ───────────────


def _meta_to_patient(meta: dict, input_signals: list[str]) -> dict | None:
    """case 캐시 → extract_ca_window_samples 가 먹는 patient dict.

    records = [(rec_start_dt, {ecg, ppg})], center = 고정 anchor.
    """
    try:
        obj = torch.load(meta["cache_path"], map_location="cpu", weights_only=False)
    except Exception:
        return None
    signals: dict[str, np.ndarray] = {}
    for s in input_signals:
        arr = obj.get(s)
        if arr is None:
            continue
        signals[s] = np.asarray(arr, dtype=np.float32)
    if not signals:
        return None
    rec_start_dt = datetime.fromtimestamp(float(obj["band_start_ts"]), tz=timezone.utc)
    return {
        "patient_id": meta["case_id"],
        "subject_id": meta["case_id"],
        "label": int(meta["label"]),
        "center": OUTCOME_ANCHOR_DT,
        "records": [(rec_start_dt, signals)],
    }


def _save_scope_split(
    split_dict: dict,
    split_name: str,
    task: str,
    input_signals: list[str],
    horizon_sec: float,
    window_sec: float,
    out_dir: str,
    max_lead_sec: float,
    signal_dtype: torch.dtype,
    fold_idx: int | None,
    n_folds: int,
    chunk_idx: int,
) -> Path:
    """SCOPE split chunk → .pt (source 메타만 SCOPE 로 다르게)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_samples = int(split_dict.get("labels", torch.tensor([])).numel())
    save_dict = {
        "split": split_name,
        "data": split_dict,
        "metadata": {
            "task": f"scope_{task}_prediction",
            "source": "SCOPE (SNUH ICU, KHDP icu-cardiac-arrest)",
            "label": "cardiac_arrest" if task == "arrest" else "mortality",
            "aggregation": "window_level",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "max_lead_sec": max_lead_sec,
            "sampling_rate": TARGET_SR,
            "negative_strategy": "anchor-aligned (fixed outcome anchor, lead-time band)",
            "split": split_name,
            "n_samples": n_samples,
            "fold_idx": fold_idx,
            "n_folds": n_folds,
            "chunk_idx": chunk_idx,
            "signal_dtype": str(signal_dtype).replace("torch.", ""),
            "split_unit": "case_id",
        },
    }
    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    win_int = int(window_sec)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    filename = (
        f"scope_{task}_{mode_str}_w{win_int}s_h{horizon_min}min"
        f"{fold_suffix}_{split_name}_chunk{chunk_idx}.pt"
    )
    save_path = out_path / filename
    torch.save(save_dict, save_path)
    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"      Saved: {save_path.name} ({size_mb:.1f} MB, n={n_samples})")
    return save_path


def run_pass2(
    metas: list[dict],
    task: str,
    input_signals: list[str],
    window_secs: list[float],
    horizon_mins: list[float],
    stride_sec: float,
    max_lead_sec: float,
    n_folds: int,
    out_dir: str,
    signal_dtype: torch.dtype,
    seed: int,
    cases_per_chunk: int = 200,
) -> list[Path]:
    """캐시 메타 → stratified case-level K-fold × (w,h) 윈도우 추출 → 저장."""
    case_ids = sorted({m["case_id"] for m in metas})
    case_to_labels = {m["case_id"]: [int(m["label"])] for m in metas}
    meta_by_case = {m["case_id"]: m for m in metas}
    pos_cases = sum(1 for c in case_ids if any(case_to_labels.get(c, [])))

    print(f"\n[Pass 2] Stratified {n_folds}-fold (case-level) "
          f"(pos cases: {pos_cases}/{len(case_ids)})")
    splits = stratified_kfold_patient_splits(
        case_ids, case_to_labels, n_folds=n_folds, seed=seed,
    )
    print(summarize_splits(splits, case_to_labels))

    combos = [(w, h) for w in window_secs for h in horizon_mins]
    saved: list[Path] = []
    sample_dtype_str = str(signal_dtype).replace("torch.", "")

    for window_sec, horizon_min in combos:
        horizon_sec = horizon_min * 60.0
        print(f"\n  === window={window_sec}s, horizon={horizon_min}min ===")
        for fold_idx, (train_c, val_c, test_c) in enumerate(splits):
            cur_fold = fold_idx if len(splits) > 1 else None
            for split_name, split_cases in (
                ("train", train_c), ("val", val_c), ("test", test_c),
            ):
                split_metas = [meta_by_case[c] for c in split_cases if c in meta_by_case]
                if not split_metas:
                    continue
                gap_stats = GapStats()
                chunk_idx = 0
                total = total_pos = 0
                buf: list = []
                n_cases_in_buf = 0

                def _flush(buf_samples, cidx):
                    nonlocal total, total_pos
                    if not buf_samples:
                        return cidx
                    n_pos = sum(1 for s in buf_samples if s.label == 1)
                    packed = _consume_to_tensors_ca(
                        buf_samples, input_signals, signal_dtype)
                    n_in = int(packed.get("labels", torch.tensor([])).numel())
                    sp = _save_scope_split(
                        packed, split_name, task, input_signals,
                        horizon_sec, window_sec, out_dir,
                        max_lead_sec=max_lead_sec, signal_dtype=signal_dtype,
                        fold_idx=cur_fold, n_folds=len(splits), chunk_idx=cidx,
                    )
                    saved.append(sp)
                    total += n_in
                    total_pos += n_pos
                    del packed
                    gc.collect()
                    return cidx + 1

                for m in split_metas:
                    patient = _meta_to_patient(m, input_signals)
                    if patient is None:
                        continue
                    samples = extract_ca_window_samples(
                        [patient], input_signals, window_sec, stride_sec,
                        horizon_sec, max_lead_sec=max_lead_sec,
                        gap_stats=gap_stats, sample_dtype=sample_dtype_str,
                    )
                    buf.extend(samples)
                    n_cases_in_buf += 1
                    if n_cases_in_buf >= cases_per_chunk:
                        chunk_idx = _flush(buf, chunk_idx)
                        buf = []
                        n_cases_in_buf = 0
                chunk_idx = _flush(buf, chunk_idx)
                buf = []
                print(f"    [fold{fold_idx}/{split_name}] {chunk_idx} chunk(s), "
                      f"{total} windows (+={total_pos})")
                print(gap_stats.summary())

    print(f"\n  Pass 2 완료: {len(saved)} files → {out_dir}")
    return saved


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCOPE Cardiac Arrest / Mortality — Data Preparation (window-level)",
    )
    parser.add_argument("--metadata", type=str, required=True,
                        help="metadata_v1.0.xlsx 경로")
    parser.add_argument("--vital-dir", type=str, required=True,
                        help=".vital 파일 디렉토리 (파일명=case_id)")
    parser.add_argument("--task", type=str, default="arrest",
                        choices=["arrest", "death"],
                        help="arrest: pos=Cardiac Arrest / death: pos=Death "
                             "(둘 다 neg=Discharge, 나머지 outcome 제외)")
    parser.add_argument("--input-signals", nargs="+", default=["ecg", "ppg"],
                        choices=["ecg", "ppg"],
                        help="SCOPE 가용 = ECG+PPG (2채널)")
    parser.add_argument("--horizon-mins", nargs="+", type=float,
                        default=[5.0, 15.0, 30.0])
    parser.add_argument("--window-secs", nargs="+", type=float, default=[600.0])
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument("--max-lead-sec", type=float, default=600.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str,
                        default="datasets/processed/scope_cardiac_arrest")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="band 캐시 디렉토리 (기본: <out-dir>/_band_cache/<task>)")
    parser.add_argument("--workers", type=int, default=8)
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    # band: 모든 horizon+window+max_lead 를 포괄하는 anchor 직전 구간 + 버퍼
    band_sec = (
        max(args.horizon_mins) * 60.0
        + args.max_lead_sec
        + max(args.window_secs)
        + 60.0
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        Path(args.out_dir) / "_band_cache" / args.task
    )

    print(f"{'=' * 64}")
    print(f"  SCOPE Cardiac Arrest / Mortality — Data Preparation")
    print(f"  Task:     {args.task}  (pos={TASK_SPEC[args.task][0]}, neg=Discharge)")
    print(f"  Input:    {' + '.join(s.upper() for s in args.input_signals)}")
    print(f"  Windows:  {args.window_secs}s / Horizons: {args.horizon_mins}min")
    print(f"  Band:     {band_sec / 60:.1f}min (anchor 직전 로딩 구간)")
    print(f"{'=' * 64}")

    cohort = build_scope_cohort(args.metadata, args.vital_dir, args.task)
    if not cohort:
        print("ERROR: 코호트 비어 있음.", file=sys.stderr)
        sys.exit(1)

    metas = run_pass1(cohort, args.input_signals, band_sec, cache_dir,
                      workers=args.workers)
    if not metas:
        print("ERROR: 캐시된 case 없음.", file=sys.stderr)
        sys.exit(1)

    run_pass2(
        metas, args.task, args.input_signals,
        args.window_secs, args.horizon_mins, args.stride_sec,
        args.max_lead_sec, args.n_folds, args.out_dir,
        dtype_map(args.signal_dtype), args.seed,
    )


if __name__ == "__main__":
    main()
