# -*- coding:utf-8 -*-
"""Outcome Prediction — Cardiac Arrest (patient-level, SCOPE) — 데이터 준비.

acute_event 의 window-level `prepare_data_scope.py` 와 **같은 SCOPE .vital 머신러리**
(build_scope_cohort / TASK_SPEC / OUTCOME_ANCHOR_* / _process_band / load_and_cache_case
/ run_pass1)를 self-contained 복사해 재사용한다(서버 코드 드리프트 회피). 차이는
Pass-2 가 **환자 단위(patient-level)** 라는 것뿐이다:

  - 환자 = SCOPE case (arrest → label 1, discharge → label 0, **death 제외**).
  - 입력 = ECG + PPG (2채널, 100Hz).
  - **Lead time = 12h**: 예측은 outcome 12h 전에 이뤄진다 → 입력 신호는 반드시
    **(outcome_anchor − 12h)** 이하에서 끝난다. 마지막 12h 는 lead gap (제외, leakage 없음).
  - **History = 고정 12h band** [anchor−24h, anchor−12h] (confound 제거: 모든 환자
    동일 길이). 5분(300s) window, stride 300s (non-overlap) → window start ≥ anchor−24h
    AND window end ≤ anchor−12h. max_windows=144 (=12h/5min) 로 환자마다 균일.
  - 환자 자격 = 이 band 내 valid window ≥ 1 개 (≥24h 기록 필요, SCOPE 24-48h 라 대부분 통과).
  - Aggregation 은 `downstream.aggregator` (TransformerAggregator / encode_patient_windows)
    를 mortality 와 동일 포맷으로 재사용 → 공유 aggregator/eval 무수정.
  - Patient-level stratified 5-fold CV. **자연 유병률(balance 안 함).**

⚠ COHORT SURVIVAL LOGGING (필수): arrest / discharge 환자 중 12h-lead 필터를 통과하는
   수, 기록 부족으로 drop 되는 수, 환자별 window 수 통계를 반드시 출력한다 (arrest
   코호트가 ~74 뿐이라 몇 명이 생존하는지 확인이 결정적).

저장 포맷은 mortality (`downstream/outcome/mortality/prepare_data.py`) 와 **동일**하다:
  patient entry = {"subject_id": int, "label": int, "n_windows": int,
                   "signals": {sig_type: (K, T)}}
  per-(fold,split,chunk) 로 `{"split","data":[patient...],"metadata":{...}}` 저장 →
  `load_prepared_split_chunked` 가 그대로 읽는다.

사용법:
    python -m downstream.outcome.cardiac_arrest.prepare_data_scope \
        --metadata datasets/icu-cardiac-arrest/1.0/metadata_v1.0.xlsx \
        --vital-dir datasets/icu-cardiac-arrest/1.0 \
        --task arrest --input-signals ecg ppg \
        --window-sec 300 --stride-sec 300 --lead-sec 43200 --history-sec 43200 \
        --max-windows 144 \
        --out-dir datasets/processed/scope_cardiac_arrest_outcome --workers 8
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
from downstream._gap_mask import (
    DEFAULT_VALID_RATIO_THRESHOLD,
    GapStats,
    apply_gap_mask_multichannel,
    compute_valid_ratio,
)
from downstream._kfold_utils import (
    stratified_kfold_patient_splits,
    summarize_splits,
)
from downstream._save_utils import (
    add_signal_dtype_arg,
    stack_window_dicts_destructive,
)

# 100Hz 통일 (pretraining 과 동일).
TARGET_SR: float = 100.0

# outcome 를 anchor 한 고정 reference: 2100-01-01 00:00:00 UTC (POSIX 4102444800).
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
# arrest: pos=Cardiac Arrest / neg=Discharge / Death 제외 (이 outcome task 기본).
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
        label 1=positive(event), 0=Discharge(negative). 파일 부재 case 는 제외.
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
            n_excl += 1  # 제외 그룹 (예: death)
            continue

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
    gap policy(valid_ratio drop + NaN→0 fill)가 처리한다.
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

    band_sec 는 cutoff(anchor−lead) 이전 history 전체를 담도록 크게(48h) 잡는다 —
    SCOPE 는 어차피 24-48h 뿐이라 가용분만 캐시된다.
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

        actual_start_ts = dtstart + i0 / native_sr
        if band_start_ts_target is None:
            band_start_ts_target = actual_start_ts
        else:
            band_start_ts_target = max(band_start_ts_target, actual_start_ts)
        sigs[stype] = processed

    if not sigs or band_start_ts_target is None:
        return None

    # 채널 간 길이 정렬 (모두 t0 기준 슬라이스라 공통 시작)
    min_len = min(len(a) for a in sigs.values())
    aligned = {s: a[:min_len] for s, a in sigs.items()}

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
          f"(band={band_sec / 3600:.1f}h, workers={workers}) → {cache_dir}")
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


# ── Pass 2: 캐시 → 환자 단위 window 추출 (12h-lead) ───────────


def _meta_to_patient(meta: dict, input_signals: list[str]) -> dict | None:
    """case 캐시 → patient dict (records + 고정 anchor center)."""
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


def extract_patient_windows(
    patient: dict,
    input_signals: list[str],
    window_sec: float,
    stride_sec: float,
    lead_sec: float,
    history_sec: float,
    max_windows: int,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",
    collect: bool = True,
) -> tuple[list[dict] | None, int, int]:
    """한 환자의 **고정 12h history band** [anchor−(lead+history), anchor−lead] window 추출.

    confound 제거: 가변 history 대신 모든 환자에게 동일한 고정 band 를 쓴다. window 는
    **start ≥ anchor−(lead_sec+history_sec)** 이고 **end ≤ anchor−lead_sec** 인 것만
    유지한다 (lead 12h gap + history 12h). 이때 max 는 history/stride = 12h/5min = 144.
    gap policy: valid_ratio < threshold → drop, 통과분 NaN→0 fill. cutoff 에 가장 가까운
    최근 max_windows 개만 유지 (t_end 오름차순의 뒤쪽).

    collect=False 면 실제 신호를 채우지 않고 카운트만 반환 (survival 집계용).

    Returns
    -------
    (windows|None, raw_count, kept_count)
        windows — collect=True 면 [{sig_type: np.ndarray(T,)}] (t_end 오름차순), 아니면 None.
        raw_count — band·gap 통과 window 총수 (cap 이전).
        kept_count — min(raw_count, max_windows).
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    center = patient["center"]
    # 고정 band 경계: tte_end ∈ [lead, lead+history−window] (window 통째로 band 내).
    tte_end_max = lead_sec + history_sec - window_sec

    kept: list[tuple[float, dict | None]] = []  # (t_end_ts, window|None)
    for _rec_idx, (rec_start, signals) in enumerate(patient["records"]):
        # 모든 입력 신호가 동시에 있는 레코드만 사용.
        if any(s not in signals for s in input_signals):
            continue
        min_len = min(len(signals[s]) for s in input_signals)
        if min_len < win_samples:
            continue

        for start in range(0, min_len - win_samples + 1, stride_samples):
            end = start + win_samples
            t_end = rec_start + timedelta(seconds=end / TARGET_SR)
            tte = (center - t_end).total_seconds()
            # 고정 band: 입력은 [anchor−24h, anchor−12h] 안에 완전히 들어와야 함.
            #   end ≤ anchor−lead (tte ≥ lead)  AND  start ≥ anchor−(lead+history)
            #   (start 조건 ⇔ tte ≤ lead+history−window).
            if tte < lead_sec or tte > tte_end_max:
                continue

            input_dict = {s: signals[s][start:end] for s in input_signals}
            valid_ratio = compute_valid_ratio(list(input_dict.values()))
            if valid_ratio < valid_ratio_threshold:
                if gap_stats is not None:
                    gap_stats.add_drop()
                continue

            if collect:
                filled_dict, gap_mask_dict = apply_gap_mask_multichannel(
                    input_dict, output_dtype=sample_dtype,
                )
                if gap_stats is not None:
                    n_total_s = sum(arr.size for arr in filled_dict.values())
                    n_gap_s = sum(int(m.sum()) for m in gap_mask_dict.values())
                    gap_stats.add_window(n_total_s, n_gap_s)
                kept.append(
                    (t_end.timestamp(),
                     {s: filled_dict[s] for s in input_signals})
                )
            else:
                if gap_stats is not None:
                    gap_stats.add_window(0, 0)
                kept.append((t_end.timestamp(), None))

    kept.sort(key=lambda x: x[0])  # t_end 오름차순 (과거 → cutoff 근접)
    raw_count = len(kept)
    if raw_count > max_windows:
        kept = kept[-max_windows:]  # cutoff 에 가장 가까운 최근 max_windows 개
    kept_count = len(kept)
    windows = [w for (_t, w) in kept] if collect else None
    return windows, raw_count, kept_count


def _survey_cohort(
    metas: list[dict],
    input_signals: list[str],
    window_sec: float,
    stride_sec: float,
    lead_sec: float,
    history_sec: float,
    max_windows: int,
    valid_ratio_threshold: float,
) -> dict[str, int]:
    """고정-band 생존 코호트 집계 (필수 logging) → 자격 있는 case→label.

    각 label(arrest=1 / discharge=0) 별로 출력:
      - 생존(band 내 valid window ≥1) 수, 기록 부족 drop 수,
      - 환자별 window 수 통계(고정 band 라 ~max_windows 근방으로 균일해야 함),
      - **pre-anchor recording 길이(시간) 분포**(outcome 간 균형 확인용).
    """
    print(f"\n[Survey] 고정 band [anchor−{(lead_sec + history_sec) / 3600:.0f}h, "
          f"anchor−{lead_sec / 3600:.0f}h] 생존 코호트 집계 ({len(metas)} cases)")
    kept_counts: dict[int, list[int]] = {0: [], 1: []}
    dur_hours: dict[int, list[float]] = {0: [], 1: []}       # 자격 환자 recording 길이
    dur_hours_all: dict[int, list[float]] = {0: [], 1: []}   # 전체(drop 포함) recording 길이
    n_dropped: dict[int, int] = {0: 0, 1: 0}
    n_cap_hit: dict[int, int] = {0: 0, 1: 0}
    qualified: dict[str, int] = {}

    for i, m in enumerate(metas):
        label = int(m["label"])
        patient = _meta_to_patient(m, input_signals)
        if patient is None:
            n_dropped[label] += 1
            continue
        # pre-anchor 가용 기록 길이(시간): band_start(=rec_start) → anchor.
        rec_start, _sig = patient["records"][0]
        avail_h = (OUTCOME_ANCHOR_DT - rec_start).total_seconds() / 3600.0
        dur_hours_all[label].append(avail_h)

        _w, raw_count, kept_count = extract_patient_windows(
            patient, input_signals, window_sec, stride_sec, lead_sec,
            history_sec, max_windows, valid_ratio_threshold, collect=False,
        )
        del patient
        if kept_count >= 1:
            qualified[m["case_id"]] = label
            kept_counts[label].append(kept_count)
            dur_hours[label].append(avail_h)
            if raw_count > max_windows:
                n_cap_hit[label] += 1
        else:
            n_dropped[label] += 1  # 기록 부족(band [24h,12h] 내 valid window 없음)
        if (i + 1) % 200 == 0:
            print(f"    surveyed {i + 1}/{len(metas)}")

    def _istats(vals: list[int]) -> str:
        if not vals:
            return "n/a"
        a = np.asarray(vals)
        return (f"min={a.min()} median={int(np.median(a))} "
                f"mean={a.mean():.1f} max={a.max()}")

    def _fstats(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        a = np.asarray(vals)
        return (f"min={a.min():.1f}h median={np.median(a):.1f}h "
                f"mean={a.mean():.1f}h max={a.max():.1f}h")

    print("\n" + "=" * 64)
    print("  COHORT SURVIVAL (fixed 12h history band, 12h lead)")
    print("=" * 64)
    for label, name in ((1, "arrest  (label 1)"), (0, "discharge(label 0)")):
        n_ok = len(kept_counts[label])
        print(f"  {name}: survived={n_ok}  "
              f"dropped(insufficient recording)={n_dropped[label]}")
        print(f"      window-count/patient: {_istats(kept_counts[label])}  "
              f"(hit cap {max_windows}: {n_cap_hit[label]})")
        print(f"      pre-anchor recording (qualified): {_fstats(dur_hours[label])}")
        print(f"      pre-anchor recording (all cases):  {_fstats(dur_hours_all[label])}")
    n_pos = len(kept_counts[1])
    n_neg = len(kept_counts[0])
    print("-" * 64)
    print(f"  QUALIFIED total: {len(qualified)} patients "
          f"(pos={n_pos}, neg={n_neg}, "
          f"prevalence={n_pos / max(1, len(qualified)):.3f})")
    print("=" * 64)
    return qualified


def _save_patient_split(
    packed: list[dict],
    split_name: str,
    task: str,
    input_signals: list[str],
    window_sec: float,
    stride_sec: float,
    lead_sec: float,
    history_sec: float,
    max_windows: int,
    out_dir: str,
    signal_dtype: torch.dtype,
    fold_idx: int | None,
    n_folds: int,
    chunk_idx: int,
    valid_ratio_threshold: float,
) -> Path:
    """patient-level split chunk → .pt (mortality 저장 포맷과 동일)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_pat = len(packed)
    n_pos = sum(int(p["label"]) for p in packed)
    lead_h = lead_sec / 3600.0
    band_lo_h = (lead_sec + history_sec) / 3600.0
    save_dict = {
        "split": split_name,
        "data": packed,
        "metadata": {
            "task": "scope_cardiac_arrest_outcome",
            "source": "SCOPE (SNUH ICU, KHDP icu-cardiac-arrest)",
            "label": "label",
            "aggregation": "patient_level",
            "input_signals": input_signals,
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "lead_sec": lead_sec,
            "history_sec": history_sec,
            "window_band": f"[anchor-{band_lo_h:.0f}h, anchor-{lead_h:.0f}h]",
            "window_policy": (
                f"valid_ratio>={valid_ratio_threshold} else drop; NaN->0 fill; "
                f"fixed band start>=anchor-{band_lo_h:.0f}h and "
                f"end<=anchor-{lead_h:.0f}h (12h lead gap excluded)"
            ),
            "max_windows": max_windows,
            "sampling_rate": TARGET_SR,
            "negative_strategy": (
                "cross-patient (pos=arrest patient, neg=Discharge patient; "
                "death excluded); natural prevalence"
            ),
            "split": split_name,
            "n_patients": n_pat,
            "n_pos": n_pos,
            "fold_idx": fold_idx,
            "n_folds": n_folds,
            "chunk_idx": chunk_idx,
            "signal_dtype": str(signal_dtype).replace("torch.", ""),
            "split_unit": "case_id",
        },
    }
    win_int = int(window_sec)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    filename = (
        f"cardiac_arrest_w{win_int}s{fold_suffix}_{split_name}_chunk{chunk_idx}.pt"
    )
    save_path = out_path / filename
    torch.save(save_dict, save_path)
    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"      Saved: {save_path.name} ({size_mb:.1f} MB, patients={n_pat}, "
          f"pos={n_pos})")
    return save_path


def run_pass2(
    metas: list[dict],
    qualified: dict[str, int],
    task: str,
    input_signals: list[str],
    window_sec: float,
    stride_sec: float,
    lead_sec: float,
    history_sec: float,
    max_windows: int,
    n_folds: int,
    out_dir: str,
    signal_dtype: torch.dtype,
    seed: int,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    patients_per_chunk: int = 100,
) -> list[Path]:
    """qualified case → stratified patient-level K-fold → 환자 단위 chunk 저장.

    streaming: 각 (fold,split) 을 case 단위로 순회하며 patients_per_chunk 만큼 버퍼링
    후 pack→save→free (전체 코호트를 RAM 에 올리지 않음).
    """
    meta_by_case = {m["case_id"]: m for m in metas}
    case_ids = sorted(qualified.keys())
    case_to_labels = {c: [int(qualified[c])] for c in case_ids}
    n_pos = sum(1 for c in case_ids if qualified[c] == 1)

    print(f"\n[Pass 2] Stratified {n_folds}-fold (patient-level) "
          f"qualified={len(case_ids)} (pos={n_pos}, neg={len(case_ids) - n_pos})")
    splits = stratified_kfold_patient_splits(
        case_ids, case_to_labels, n_folds=n_folds, seed=seed,
    )
    print(summarize_splits(splits, case_to_labels))

    sample_dtype_str = str(signal_dtype).replace("torch.", "")
    input_signals_sorted = sorted(input_signals)  # stack_window_dicts 는 key 정렬
    saved: list[Path] = []

    for fold_idx, (train_c, val_c, test_c) in enumerate(splits):
        cur_fold = fold_idx if len(splits) > 1 else None
        for split_name, split_cases in (
            ("train", train_c), ("val", val_c), ("test", test_c),
        ):
            gap_stats = GapStats()
            chunk_idx = 0
            total_pat = total_pos = 0
            buf: list[dict] = []

            def _flush(buf_p: list[dict], cidx: int) -> int:
                nonlocal total_pat, total_pos
                if not buf_p:
                    return cidx
                packed: list[dict] = []
                for pd_ in buf_p:
                    n_w = len(pd_["windows"])
                    windows_per_type = stack_window_dicts_destructive(
                        [dict(w) for w in pd_["windows"]], signal_dtype,
                    )
                    packed.append({
                        "subject_id": int(pd_["case_id"]),
                        "label": int(pd_["label"]),
                        "n_windows": n_w,
                        "signals": windows_per_type,
                    })
                sp = _save_patient_split(
                    packed, split_name, task, input_signals_sorted,
                    window_sec, stride_sec, lead_sec, history_sec, max_windows,
                    out_dir, signal_dtype, cur_fold, len(splits), cidx,
                    valid_ratio_threshold,
                )
                saved.append(sp)
                total_pat += len(packed)
                total_pos += sum(int(p["label"]) for p in packed)
                del packed
                gc.collect()
                return cidx + 1

            for c in sorted(split_cases):
                m = meta_by_case.get(c)
                if m is None:
                    continue
                patient = _meta_to_patient(m, input_signals)
                if patient is None:
                    continue
                windows, _raw, _kept = extract_patient_windows(
                    patient, input_signals, window_sec, stride_sec, lead_sec,
                    history_sec, max_windows, valid_ratio_threshold,
                    gap_stats=gap_stats, sample_dtype=sample_dtype_str, collect=True,
                )
                del patient
                if not windows:
                    continue
                buf.append({"case_id": c, "label": int(m["label"]),
                            "windows": windows})
                if len(buf) >= patients_per_chunk:
                    chunk_idx = _flush(buf, chunk_idx)
                    buf = []
            chunk_idx = _flush(buf, chunk_idx)
            buf = []
            print(f"    [fold{fold_idx}/{split_name}] {chunk_idx} chunk(s), "
                  f"{total_pat} patients (pos={total_pos})")
            print(gap_stats.summary())

    print(f"\n  Pass 2 완료: {len(saved)} files → {out_dir}")
    return saved


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCOPE Cardiac Arrest — Outcome Prediction (patient-level, 12h lead)",
    )
    parser.add_argument("--metadata", type=str, required=True,
                        help="metadata_v1.0.xlsx 경로")
    parser.add_argument("--vital-dir", type=str, required=True,
                        help=".vital 파일 디렉토리 (파일명=case_id)")
    parser.add_argument("--task", type=str, default="arrest",
                        choices=["arrest", "death"],
                        help="arrest: pos=Cardiac Arrest / neg=Discharge / Death 제외")
    parser.add_argument("--input-signals", nargs="+", default=["ecg", "ppg"],
                        choices=["ecg", "ppg"],
                        help="SCOPE 가용 = ECG+PPG (2채널)")
    parser.add_argument("--window-sec", type=float, default=300.0)
    parser.add_argument("--stride-sec", type=float, default=300.0)
    parser.add_argument("--lead-sec", type=float, default=43200.0,
                        help="lead time(초). 입력은 anchor−lead 이하에서 끝남 (기본 12h).")
    parser.add_argument("--history-sec", type=float, default=43200.0,
                        help="고정 history band 길이(초). window 는 "
                             "[anchor−(lead+history), anchor−lead] 안에만 (기본 12h).")
    parser.add_argument("--max-windows", type=int, default=144,
                        help="환자당 최대 window (고정 band=12h/5min=144).")
    parser.add_argument("--band-hours", type=float, default=48.0,
                        help="anchor 직전 캐시 band(시간). 고정 history band(≥24h) 포함.")
    parser.add_argument("--valid-ratio", type=float,
                        default=DEFAULT_VALID_RATIO_THRESHOLD,
                        help="window valid(non-NaN) 비율 임계 (미만이면 drop).")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str,
                        default="datasets/processed/scope_cardiac_arrest_outcome")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="band 캐시 디렉토리 (기본: <out-dir>/_band_cache/<task>)")
    parser.add_argument("--patients-per-chunk", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    band_sec = args.band_hours * 3600.0
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        Path(args.out_dir) / "_band_cache" / args.task
    )

    print(f"{'=' * 64}")
    print("  SCOPE Cardiac Arrest — Outcome Prediction (patient-level)")
    print(f"  Task:     {args.task}  (pos={TASK_SPEC[args.task][0]}, neg=Discharge, "
          f"Death 제외)")
    print(f"  Input:    {' + '.join(s.upper() for s in args.input_signals)}")
    print(f"  Window:   {args.window_sec}s / stride {args.stride_sec}s / "
          f"max_windows {args.max_windows}")
    print(f"  Lead:     {args.lead_sec / 3600:.1f}h  History band: "
          f"[anchor−{(args.lead_sec + args.history_sec) / 3600:.0f}h, "
          f"anchor−{args.lead_sec / 3600:.0f}h] (fixed)")
    print(f"  Cache band: {band_sec / 3600:.1f}h (anchor 직전)")
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

    qualified = _survey_cohort(
        metas, args.input_signals, args.window_sec, args.stride_sec,
        args.lead_sec, args.history_sec, args.max_windows, args.valid_ratio,
    )
    if not qualified:
        print("ERROR: 고정 band 필터 통과 환자 없음.", file=sys.stderr)
        sys.exit(1)

    run_pass2(
        metas, qualified, args.task, args.input_signals,
        args.window_sec, args.stride_sec, args.lead_sec, args.history_sec,
        args.max_windows, args.n_folds, args.out_dir,
        dtype_map(args.signal_dtype), args.seed,
        valid_ratio_threshold=args.valid_ratio,
        patients_per_chunk=args.patients_per_chunk,
    )


if __name__ == "__main__":
    main()
