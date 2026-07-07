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

Negative 구성 (``--neg-mode``, 2026-07-05 추가):
  - ``cross-patient`` (기본, SCOPE 논문 정합): pos=event 환자·neg=Discharge 환자.
    "arrest-환자 vs discharge-환자" 구분(who). horizon band 로 lead-time 매칭.
  - ``within-patient`` (같은 arrest 환자 임박 경보): **arrest 환자만** 사용, 같은 환자
    안에서 **pos=임박(horizon band, arrest 5~15분 전)·neg=비-임박(tte≥12h)·회색지대 drop**.
    환자 중증도(who) 대신 arrest 로의 가속(when) 학습. band 를 24h 로 확장(12h+ neg 확보),
    neg 는 pos×10 로 캡(자연 ~100:1 완화). ⚠ 74 환자뿐 → CI 넓음(환자 단위 bootstrap),
    SCOPE 논문(cross-patient)과 직접 비교 불가 — 우리 고유 imminence task.

2-pass (네트워크 마운트 스토리지 I/O 최적화):
  - Pass 1: 각 .vital 을 1회만 로딩 → anchor 직전 band 만 전처리 → case 캐시(.pt) 저장.
  - Pass 2: 캐시에서 stratified K-fold × horizon 별 윈도우 추출 → chunk .pt 저장.

사용법:
    # cross-patient (기본, SCOPE 논문 정합)
    python -m downstream.acute_event.cardiac_arrest.prepare_data_scope \
        --metadata datasets/icu-cardiac-arrest/1.0/metadata_v1.0.xlsx \
        --vital-dir datasets/icu-cardiac-arrest/1.0 \
        --task arrest --input-signals ecg ppg \
        --window-secs 600 --horizon-mins 5 15 30 \
        --out-dir datasets/processed/scope_cardiac_arrest --workers 8

    # within-patient (같은 arrest 환자 임박 경보)
    python -m downstream.acute_event.cardiac_arrest.prepare_data_scope \
        ... --neg-mode within-patient \
        --within-neg-min-hours 12 --within-neg-band-hours 24 --within-neg-per-pos 10 \
        --out-dir datasets/processed/scope_cardiac_arrest_within
"""

from __future__ import annotations

import argparse
import gc
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
from downstream._save_utils import add_signal_dtype_arg, consume_gap_masks

# 100Hz 통일 (pretraining 과 동일). prepare_data.py 의 동명 상수와 동일 값.
TARGET_SR: float = 100.0


# ── window 추출/패킹 (drift 방지 위해 self-contained — 커밋된 유틸만 의존) ─────
# NOTE: MIMIC prepare_data.py 의 extract_ca_window_samples / _consume_to_tensors_ca
#       와 알고리즘 동일. 서버 코드 드리프트(prepare_data.py 미커밋)에 영향받지
#       않도록 SCOPE 모듈에 내장한다.


@dataclass
class CAWindowSample:
    """곧 일어날 acute event(arrest/death) 예측용 window-level 샘플."""

    input_signals: dict[str, np.ndarray]
    input_gap_masks: dict[str, np.ndarray]  # bool, True=원본이 NaN (gap)
    label: int  # 0=neg(Discharge), 1=pos(event)
    label_value: float  # tte (window end → center, 분)
    case_id: str
    patient_id: str
    win_start_sec: float
    horizon_sec: float


def extract_ca_window_samples(
    patients: list[dict],
    input_signals: list[str],
    window_sec: float = 600.0,
    stride_sec: float = 30.0,
    horizon_sec: float = 900.0,
    max_lead_sec: float = 600.0,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",
    neg_mode: str = "cross-patient",
    within_neg_min_sec: float = 43200.0,
    pos_gap_sec: float = 300.0,
    pos_horizon_sec: float = 86400.0,
    neg_min_sec: float = 86400.0,
    neg_max_sec: float = 172800.0,
) -> list[CAWindowSample]:
    """anchor-relative window-level (input, label) 쌍 추출.

    윈도우 종료 절대시각 t_end 의 center 까지 잔여시간 tte = center − t_end.

    neg_mode:
      - "cross-patient" (기본, SCOPE 논문 정합): ``horizon ≤ tte ≤ horizon+max_lead``
        윈도우만 emit, **라벨 = 케이스 라벨**(arrest 환자=1, discharge 환자=0). 즉
        arrest-환자 vs discharge-환자 구분(who).
      - "within-patient" (같은 arrest 환자 임박 경보): **arrest 환자만**(upstream 필터)
        에서 tte 로 **window-level 라벨** — ``horizon ≤ tte ≤ horizon+max_lead``=임박
        (label 1), ``tte ≥ within_neg_min_sec``(기본 12h)=비-임박(label 0), 그 사이
        회색지대=drop. → 환자 중증도(who) 대신 arrest 로의 가속(when) 학습.

    두 모드 공통: onset 보다 최소 horizon 앞에서 끝남(leakage 없음). Gap policy:
    valid_ratio < threshold → drop, 통과분 NaN→0 + gap_mask 저장.
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    samples: list[CAWindowSample] = []

    for patient in patients:
        center = patient["center"]
        pid = patient["patient_id"]
        label = int(patient["label"])

        for rec_idx, (rec_start, signals) in enumerate(patient["records"]):
            # 모든 입력 신호(ecg+ppg)가 동시에 있는 레코드만 사용. 일부만 있으면
            # window 가 신호 결여 → labels/signals 길이·정렬 불일치 유발하므로 제외.
            if any(s not in signals for s in input_signals):
                continue
            min_len = min(len(signals[s]) for s in input_signals)
            if min_len < win_samples:
                continue

            for start in range(0, min_len - win_samples + 1, stride_samples):
                end = start + win_samples
                t_end = rec_start + timedelta(seconds=end / TARGET_SR)
                tte = (center - t_end).total_seconds()

                # ── window 라벨 결정 (neg_mode 별) ──
                if neg_mode == "scope":
                    # SCOPE 논문 정합 비대칭 band.
                    #   양성(arrest): arrest 직전 [pos_gap, pos_gap+pos_horizon]
                    #     (논문 "last 24h before arrest" + leakage 회피 gap).
                    #   음성(discharge): 퇴실 [neg_min, neg_max] 전 (방어적 default
                    #     24–48h = 48h 창의 이른 부분, 논문 "first 24h of admission" 의도).
                    if label == 1:
                        if pos_gap_sec <= tte <= pos_gap_sec + pos_horizon_sec:
                            win_label = 1
                        else:
                            continue
                    else:
                        if neg_min_sec <= tte <= neg_max_sec:
                            win_label = 0
                        else:
                            continue
                elif neg_mode == "within-patient":
                    # arrest 환자만 들어옴(upstream 필터). tte 로 window-level 라벨.
                    if horizon_sec <= tte <= horizon_sec + max_lead_sec:
                        win_label = 1                      # 임박(positive)
                    elif tte >= within_neg_min_sec:
                        win_label = 0                      # 비-임박(negative, 12h+)
                    else:
                        continue                           # 회색지대 → drop
                else:  # cross-patient (기존): 케이스 라벨, 임박 밴드만
                    if tte < horizon_sec or tte > horizon_sec + max_lead_sec:
                        continue
                    win_label = label

                input_dict = {
                    s: signals[s][start:end] for s in input_signals if s in signals
                }
                # 모든 입력 신호가 다 있어야 함(위 레코드 가드로 보장되나 belt-and-suspenders).
                if len(input_dict) != len(input_signals):
                    if gap_stats is not None:
                        gap_stats.add_drop()
                    continue

                valid_ratio = compute_valid_ratio(list(input_dict.values()))
                if valid_ratio < valid_ratio_threshold:
                    if gap_stats is not None:
                        gap_stats.add_drop()
                    continue

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
                        label=win_label,
                        label_value=tte / 60.0,
                        case_id=f"{pid}_rec{rec_idx}_w{start}",
                        patient_id=pid,
                        win_start_sec=start / TARGET_SR,
                        horizon_sec=horizon_sec,
                    )
                )

    return samples


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
        # labels/case_ids 와 정렬·길이 일치 위해 반드시 전체 N개로 쌓는다.
        # (extract_ca_window_samples 가 모든 입력신호 동시존재를 보장) 결여 시 즉시 에러.
        n = len(samples)
        out = torch.empty((n, T), dtype=signal_dtype)
        for i, s in enumerate(samples):
            arr = s.input_signals.pop(stype, None)
            if arr is None:
                raise ValueError(
                    f"sample {i} missing signal '{stype}' — extraction 이 모든 "
                    f"input_signals 동시존재를 보장해야 함 (labels/signals 정렬 붕괴 방지)"
                )
            out[i].copy_(torch.from_numpy(arr))
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


def _flag_true(v) -> bool:
    """metadata flag 컬럼(문자열/숫자 혼재) → bool. '1'/'Y'/'True'/1/1.0 등을 참."""
    import math

    if v is None:
        return False
    if isinstance(v, (int, float)):
        try:
            return not math.isnan(float(v)) and int(v) != 0
        except (ValueError, TypeError):
            return False
    return str(v).strip().lower() in {"1", "y", "yes", "true", "t"}


def build_scope_cohort(
    metadata_path: str,
    vital_dir: str,
    task: str,
    exclude_death_post: bool = False,
    exclude_arrest_pre: bool = False,
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
    n_excl = n_missing_file = n_flag_excl = 0
    for _, row in df.iterrows():
        outcome = row["outcome"]
        if outcome == pos_outcome:
            label = 1
        elif outcome == OUTCOME_DISCHARGE:
            label = 0
        else:
            n_excl += 1  # 제외 그룹 (다른 outcome)
            continue

        # 컬럼 기반 제외 (rigorous arm). death_post: 퇴실 24h내 사망 → "건강한 대조군"
        # 아님(음성 오염). arrest_pre: 입원 24h전 arrest → 생리 이미 손상.
        if exclude_death_post and _flag_true(row.get("death_post")):
            n_flag_excl += 1
            continue
        if exclude_arrest_pre and _flag_true(row.get("arrest_pre")):
            n_flag_excl += 1
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
            # ECMO/VAD 는 파형 크게 변형 → shortcut 위험. stratify/보고용 태그.
            "circ_support": _flag_true(row.get("circulatory_support")),
        })

    n_pos = sum(c["label"] for c in cohort)
    n_circ = sum(1 for c in cohort if c.get("circ_support"))
    print(f"  Cohort [{task}]: {len(cohort)} cases "
          f"(pos={n_pos}, neg={len(cohort) - n_pos}); "
          f"excluded({excl_outcome})={n_excl}, missing_file={n_missing_file}, "
          f"flag_excluded={n_flag_excl}, circ_support={n_circ}")
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
    neg_mode: str = "cross-patient",
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
            "neg_mode": neg_mode,
            "negative_strategy": (
                "within-patient (arrest-only; pos=imminent horizon band, "
                "neg=non-imminent far band, same patient)"
                if neg_mode == "within-patient"
                else "scope-repro (SCOPE Lee et al. 정합; pos=event patient pre-event "
                     "horizon, neg=Discharge patient earlier band; asymmetric)"
                if neg_mode == "scope"
                else "cross-patient (pos=event patient, neg=Discharge patient; "
                     "fixed outcome anchor, lead-time band)"
            ),
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
    train_neg_cap_per_case: int = 0,
    neg_mode: str = "cross-patient",
    within_neg_min_sec: float = 43200.0,
    within_neg_per_pos: int = 10,
    pos_gap_sec: float = 300.0,
    pos_horizon_sec: float = 86400.0,
    neg_min_sec: float = 86400.0,
    neg_max_sec: float = 172800.0,
) -> list[Path]:
    """캐시 메타 → stratified case-level K-fold × (w,h) 윈도우 추출 → 저장.

    cross-patient: train_neg_cap_per_case > 0 이면 **train split 의 음성(Discharge)
      case 당 window 를 무작위 subsample** (val/test 자연분포 유지).
    within-patient: arrest 환자 안에서 pos(임박)/neg(비-임박) window 라벨. 자연
      imbalance 가 ~100:1(10min pos vs 수 h neg)로 극심하므로, ``within_neg_per_pos``
      > 0 이면 **환자별 neg 를 pos 수 × 이 배수로 subsample**(전 split 동일 —
      within-patient 유병률은 밴드로 설계된 값이라 자연분포 개념이 없음). pos 없는
      환자는 제외.
    """
    neg_cap_rng = np.random.default_rng(seed + 1)
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
                        neg_mode=neg_mode,
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
                        neg_mode=neg_mode, within_neg_min_sec=within_neg_min_sec,
                        pos_gap_sec=pos_gap_sec, pos_horizon_sec=pos_horizon_sec,
                        neg_min_sec=neg_min_sec, neg_max_sec=neg_max_sec,
                    )
                    if neg_mode == "within-patient":
                        # 환자별 pos(임박)/neg(비-임박). pos 없으면 제외, neg 는
                        # pos×within_neg_per_pos 로 캡 (전 split 동일 — 설계 유병률).
                        pos_s = [s for s in samples if s.label == 1]
                        neg_s = [s for s in samples if s.label == 0]
                        if not pos_s:
                            samples = []
                        else:
                            cap = within_neg_per_pos * len(pos_s)
                            if within_neg_per_pos > 0 and len(neg_s) > cap:
                                keep = neg_cap_rng.permutation(len(neg_s))[:cap]
                                neg_s = [neg_s[j] for j in sorted(keep.tolist())]
                            samples = pos_s + neg_s
                    elif (split_name == "train" and train_neg_cap_per_case > 0
                            and int(m["label"]) == 0
                            and len(samples) > train_neg_cap_per_case):
                        # cross-patient: train 음성 case 만 캡 (val/test 불변).
                        keep = neg_cap_rng.permutation(len(samples))[:train_neg_cap_per_case]
                        samples = [samples[j] for j in sorted(keep.tolist())]
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
    parser.add_argument(
        "--train-neg-cap-per-case", type=int, default=0,
        help="[cross-patient] train split 음성(Discharge) case 당 window 최대 개수"
        "(무작위 subsample). 0=캡 없음. val/test 자연분포 유지. 극불균형 완화용.",
    )
    parser.add_argument(
        "--neg-mode", type=str, default="cross-patient",
        choices=["cross-patient", "within-patient", "scope"],
        help="cross-patient(기본): pos=arrest환자·neg=discharge환자 (같은 lead-time band). "
        "within-patient(같은 arrest환자 임박): arrest환자만, pos=임박·neg=비-임박(12h+). "
        "scope(논문 Lee et al. 정합, 비대칭 band): pos=arrest 직전 pos_horizon, "
        "neg=discharge 퇴실 neg_tte_min~max 전(방어적 24–48h).",
    )
    # ── scope mode (SCOPE 논문 정합 재현) 전용 ──
    parser.add_argument(
        "--pos-gap-min", type=float, default=5.0,
        help="[scope] 양성(arrest) lead-time gap(분). window 끝~arrest 최소 간격 "
        "(leakage 회피). 기본 5min.",
    )
    parser.add_argument(
        "--pos-horizon-h", type=float, default=24.0,
        help="[scope] 양성 horizon(시간). arrest 직전 이 시간 내 window = 양성. "
        "논문 = 24h. 파일명 h{분}min 에 반영.",
    )
    parser.add_argument(
        "--neg-tte-min-h", type=float, default=24.0,
        help="[scope] 음성(discharge) tte 하한(시간). 방어적 default 24h "
        "(퇴실 직전 안정구간 shortcut 회피).",
    )
    parser.add_argument(
        "--neg-tte-max-h", type=float, default=48.0,
        help="[scope] 음성 tte 상한(시간). 기본 48h (48h 창의 이른 24h = 논문 "
        "'first 24h of admission' 의도).",
    )
    parser.add_argument(
        "--exclude-death-post", action="store_true",
        help="[rigorous] 퇴실 24h내 사망(death_post) discharge 를 음성에서 제외 "
        "(대조군 오염 제거). 기본 off(논문 정합).",
    )
    parser.add_argument(
        "--exclude-arrest-pre", action="store_true",
        help="[rigorous] 입원 24h전 arrest(arrest_pre) case 제외. 기본 off.",
    )
    parser.add_argument(
        "--within-neg-min-hours", type=float, default=12.0,
        help="[within-patient] 비-임박 negative 의 최소 tte(시간). 기본 12h "
        "(이보다 가까우면 전조 오염 소지 → 회색지대 drop).",
    )
    parser.add_argument(
        "--within-neg-band-hours", type=float, default=24.0,
        help="[within-patient] band 로딩 범위(시간). 이 범위까지 negative 확보. 기본 24h.",
    )
    parser.add_argument(
        "--within-neg-per-pos", type=int, default=10,
        help="[within-patient] 환자별 neg 를 pos 수 × 이 배수로 캡(전 split). 0=캡 없음. "
        "자연 imbalance(~100:1) 완화용. 기본 10.",
    )
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    within = args.neg_mode == "within-patient"
    scope = args.neg_mode == "scope"
    within_neg_min_sec = args.within_neg_min_hours * 3600.0
    # scope 비대칭 band 파라미터
    pos_gap_sec = args.pos_gap_min * 60.0
    pos_horizon_sec = args.pos_horizon_h * 3600.0
    neg_min_sec = args.neg_tte_min_h * 3600.0
    neg_max_sec = args.neg_tte_max_h * 3600.0
    # scope 는 horizon_mins 대신 pos_horizon 을 쓴다 → 파일명/루프용 단일값으로 대체.
    horizon_mins = args.horizon_mins
    if scope:
        horizon_mins = [args.pos_horizon_h * 60.0]  # 파일명 h{1440}min = 24h

    # band: cross-patient=horizon+lead+window. within-patient=within_neg_band_hours.
    # scope=양성 pos_horizon 과 음성 neg_max 중 더 먼 것 + window 커버.
    if within:
        band_sec = args.within_neg_band_hours * 3600.0 + max(args.window_secs) + 60.0
    elif scope:
        band_sec = (
            max(pos_gap_sec + pos_horizon_sec, neg_max_sec)
            + max(args.window_secs)
            + 60.0
        )
    else:
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
    print(f"  Task:     {args.task}  (pos={TASK_SPEC[args.task][0]})")
    if within:
        nm_desc = (f"  (arrest-only, neg tte≥{args.within_neg_min_hours:.0f}h, "
                   f"neg≤{args.within_neg_per_pos}×pos)")
    elif scope:
        nm_desc = (f"  (SCOPE-repro: pos=arrest {args.pos_gap_min:.0f}min–"
                   f"{args.pos_horizon_h:.0f}h, neg=discharge {args.neg_tte_min_h:.0f}–"
                   f"{args.neg_tte_max_h:.0f}h; excl death_post={args.exclude_death_post} "
                   f"arrest_pre={args.exclude_arrest_pre})")
    else:
        nm_desc = "  (neg=Discharge)"
    print(f"  Neg-mode: {args.neg_mode}{nm_desc}")
    print(f"  Input:    {' + '.join(s.upper() for s in args.input_signals)}")
    print(f"  Windows:  {args.window_secs}s / Horizons: {horizon_mins}min "
          f"({'scope: pos_horizon=' + str(args.pos_horizon_h) + 'h' if scope else 'band'})")
    print(f"  Band:     {band_sec / 60:.1f}min (anchor 직전 로딩 구간)")
    print(f"{'=' * 64}")

    cohort = build_scope_cohort(
        args.metadata, args.vital_dir, args.task,
        exclude_death_post=args.exclude_death_post,
        exclude_arrest_pre=args.exclude_arrest_pre,
    )
    if not cohort:
        print("ERROR: 코호트 비어 있음.", file=sys.stderr)
        sys.exit(1)

    if within:
        # within-patient: arrest(양성) 환자만 — neg 는 같은 환자의 비-임박 구간.
        n_before = len(cohort)
        cohort = [c for c in cohort if int(c["label"]) == 1]
        print(f"  [within-patient] arrest 환자만 유지: {n_before} → {len(cohort)} cases")
        if not cohort:
            print("ERROR: within-patient 인데 arrest(양성) case 가 없음.", file=sys.stderr)
            sys.exit(1)

    metas = run_pass1(cohort, args.input_signals, band_sec, cache_dir,
                      workers=args.workers)
    if not metas:
        print("ERROR: 캐시된 case 없음.", file=sys.stderr)
        sys.exit(1)

    run_pass2(
        metas, args.task, args.input_signals,
        args.window_secs, horizon_mins, args.stride_sec,
        args.max_lead_sec, args.n_folds, args.out_dir,
        dtype_map(args.signal_dtype), args.seed,
        train_neg_cap_per_case=args.train_neg_cap_per_case,
        neg_mode=args.neg_mode, within_neg_min_sec=within_neg_min_sec,
        within_neg_per_pos=args.within_neg_per_pos,
        pos_gap_sec=pos_gap_sec, pos_horizon_sec=pos_horizon_sec,
        neg_min_sec=neg_min_sec, neg_max_sec=neg_max_sec,
    )


if __name__ == "__main__":
    main()
