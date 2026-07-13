# -*- coding:utf-8 -*-
"""Massive Transfusion Prediction — 데이터 준비 (SNUH OR, VitalDB `.vital`, window-level).

`study_population.csv` 의 case-level label(`masstf`) + onset 시각(`correct_start_time`)
을 읽어, 해당 case 의 raw `.vital` 파형을 직접 파싱해 **anchor-relative window-level**
(input_window, label) 쌍을 만든다. Cardiac Arrest / IOH / ICH 와 동일한 sliding-window
/ horizon-sweep / stratified K-fold 구조 — 소스가 MIMIC WFDB 가 아니라 raw `.vital`,
라벨이 `masstf` 라는 점만 다르다.

라벨 규칙 (Future Prediction, 5/10/15 min horizon):
  - center(anchor):
      · masstf+ (양성): correct_start_time = 대량수혈 시작 시각(onset).
      · masstf- (음성): onset 이 없으므로 risk-set matching 으로 pseudo-anchor 부여.
        양성들의 "opstart → onset" 소요시간 분포에서 하나를 뽑아
        anchor = opstart + sampled_offset. 양성과 동일한 "수술 진행 시점" 분포를
        갖게 해, 모델이 '수술 몇 분째' 라는 trivial shortcut 으로 분리하지 못하게 함.
  - 윈도우 종료 절대시각 t_end 의 center 까지 잔여시간 tte = center − t_end.
  - horizon h 에 대해 ``h ≤ tte ≤ h + max_lead`` 인 윈도우만 emit
    (= onset 보다 최소 h, 최대 h+max_lead 앞에서 끝나는 윈도우 → leakage 없음).
  - label = case 의 masstf (1=양성, 0=음성). 음성 윈도우도 동일 band 에서 추출.

시각 정합 (검증 완료 2026-07-13):
  - `.vital` 내부 timeline = UTC epoch (`VitalFile.dtstart`). 로컬 KST = UTC + 9h.
  - `correct_start_time`/`opstart` 등 CSV 시각은 **로컬 KST wall-clock** ('Z' 접미는
    실제 UTC 가 아니라 형식상 붙은 것 — opstart≤onset≤opend 로 로컬 정합 확인).
    → KST(+9)로 해석해 epoch 로 변환하면 `.vital` timeline 과 정확히 맞음.

Input 소스: SNUADC 파형 — ABP(ART) / PPG(PLETH). 전 채널이 동시 valid 한 구간만
  채널 정렬 span 으로 사용(intersection). 대량수혈은 실혈성 저혈압 + 말초관류 저하로
  나타나므로 ABP(혈압 파형) + PPG(맥파·관류) 가 핵심. (ECG 등도 --input-signals 로 선택 가능)

음성 서브샘플링: 음성 case 는 17,570개(양성 416개)로, 전량(≈374GB) 파싱은 비현실적.
  `--neg-per-pos` 배(기본 3)만 seeded 샘플로 파싱한다. → prevalence 는 인위적이므로
  AUROC 를 주지표로 보고하고, true prevalence(≈2.3%)를 병기한다.

사용법:
    # Canonical (10분 window, 5/10/15분 horizon) + 5-fold CV, 음성 3배
    python -m downstream.acute_event.massive_transfusion.prepare_data \
        --study-csv F:/Massive_Transfusion_Raw/study_population.csv \
        --raw-dir   F:/Massive_Transfusion_Raw \
        --input-signals abp ppg \
        --window-secs 600 --horizon-mins 5 10 15 \
        --neg-per-pos 3 --workers 8 \
        --out-dir F:/Massive_Transfusion_Downstream
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
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


TARGET_SR: float = 100.0

# CSV 로컬 시각(KST) → epoch 변환용 고정 오프셋 (한국 표준시, DST 없음).
KST = timezone(timedelta(hours=9))

# signal_type → `.vital` 트랙 후보 (선호 순). 첫 존재 트랙 1개를 채널로 사용.
# SNUADC(OR) 우선 → Solar8000/Intellivue(대체 장비) 순.
CHANNEL_TRACK_CANDIDATES: dict[str, list[str]] = {
    "ecg": [
        "SNUADC/ECG_II", "Solar8000/ECG_II", "SNUADCM/ECG_II",
        "Intellivue/ECG_II", "Intellivue/ECG_II_WAV",
        "SNUADC/ECG_I", "SNUADC/ECG_V5",
    ],
    "abp": [
        "SNUADC/ART", "SNUADCM/ART",
        "Intellivue/ART", "Intellivue/ABP", "SNUADC/FEM",
    ],
    "ppg": [
        "SNUADC/PLETH", "Solar8000/PLETH", "SNUADCM/PLETH", "Intellivue/PLETH",
    ],
    "cvp": ["SNUADC/CVP", "Solar8000/CVP", "Intellivue/CVP"],
    "co2": ["Primus/CO2", "Solar8000/CO2", "Datex-Ohmeda/CO2", "Intellivue/CO2"],
    "awp": ["Primus/AWP", "Solar8000/AWP", "Datex-Ohmeda/AWP", "Intellivue/AWP_WAV"],
}


# ── 데이터 구조 ──────────────────────────────────────────────


@dataclass
class MTWindowSample:
    """대량수혈 예측용 window-level 샘플."""

    input_signals: dict[str, np.ndarray]
    input_gap_masks: dict[str, np.ndarray]  # bool, True=원본이 NaN (gap)
    label: int  # 0=masstf-, 1=masstf+
    label_value: float  # tte (window end → center, 분)
    case_id: str
    patient_id: str  # patient-level fold/cluster grouping (= fileid stem)
    win_start_sec: float  # span 시작 기준 윈도우 시작 (초)
    horizon_sec: float


@dataclass
class CaseMeta:
    """study_population 한 row 의 label + anchor 메타 (신호 파싱 전)."""

    patient_id: str
    vital_path: Path
    label: int
    center_epoch: float  # onset(양성) 또는 risk-set anchor(음성)의 epoch(초)


# ── 시각 파싱 ────────────────────────────────────────────────


def parse_local_epoch(s: str | None) -> float | None:
    """CSV 로컬(KST) 시각 문자열 → epoch(초). 실패/빈값 → None.

    허용 형식: "2018-02-27 15:40:00", "2018-02-27 14:42",
              "2018-02-27T17:45:00Z"(Z 는 형식상 접미 — 로컬로 취급).
    """
    if s is None:
        return None
    s = s.strip()
    if not s or s.lower() in ("nan", "nat", "none", "null"):
        return None
    s = s.replace("T", " ").rstrip("Z").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=KST).timestamp()
        except ValueError:
            continue
    return None


# ── `.vital` 파싱 (채널 정렬 span) ───────────────────────────


def _select_tracks(available: set[str], channels: list[str]) -> dict[str, str] | None:
    """각 채널의 선호 트랙 1개를 고른다. 하나라도 없으면 None(정렬 불가)."""
    chosen: dict[str, str] = {}
    for c in channels:
        for cand in CHANNEL_TRACK_CANDIDATES.get(c, []):
            if cand in available:
                chosen[c] = cand
                break
        if c not in chosen:
            return None
    return chosen


def parse_vital_aligned_spans(
    vital_path: Path,
    channels: list[str],
) -> list[tuple[float, dict[str, np.ndarray]]]:
    """`.vital` 레코드 → **채널 정렬·타임라인 보존** aligned span 리스트.

    Cardiac Arrest 의 `parse_record_aligned_spans`(MIMIC WFDB) 를 `.vital` 용으로 이식:
      1. 선택 트랙을 공통 grid(최대 native SR, 보통 500Hz)로 동시 추출(NaN-padded).
      2. 채널별 range-check + spike → 원본 grid 위 NaN 마킹, 전 채널 동시 valid 만
         valid_all(intersection).
      3. valid_all 연속 span(≥60s) 마다 각 채널을 **같은 [start:start+len]** 으로
         median+notch+filter+resample → 100Hz (채널 정렬 보존).
      4. span 절대 시작 epoch = dtstart + start/grid_sr.

    모든 `channels` 가 존재해야 정렬 가능 — 하나라도 없으면 빈 리스트.
    """
    try:
        import vitaldb
    except ImportError:
        print("ERROR: vitaldb 패키지 필요. pip install vitaldb", file=sys.stderr)
        sys.exit(1)

    try:
        vf = vitaldb.VitalFile(str(vital_path))
    except Exception:
        return []

    available = set(vf.get_track_names())
    chosen = _select_tracks(available, channels)
    if chosen is None:
        return []

    # 공통 grid SR = 선택 트랙들의 최대 native SR (없으면 500).
    srates: list[float] = []
    for t in chosen.values():
        trk = vf.find_track(t)
        sr = getattr(trk, "srate", 0.0) if trk is not None else 0.0
        if sr and sr > 0:
            srates.append(float(sr))
    grid_sr = max(srates) if srates else 500.0

    t0 = getattr(vf, "dtstart", None)
    if t0 is None:
        return []
    t0 = float(t0)

    # 공통 grid 로 전 채널 동시 추출 → (L, n_ch), 채널 정렬 보장.
    tracks_list = [chosen[c] for c in channels]
    try:
        mat = np.asarray(vf.to_numpy(tracks_list, interval=1.0 / grid_sr))
    except Exception:
        return []
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    if mat.shape[0] == 0 or mat.shape[1] != len(channels):
        return []
    L = mat.shape[0]

    # 1. 채널별 range-check + spike → 마킹 + valid_all intersection
    valid_all = np.ones(L, dtype=bool)
    marked: dict[str, np.ndarray] = {}
    cfgs: dict[str, object] = {}
    for i, c in enumerate(channels):
        cfg = SIGNAL_CONFIGS.get(c)
        if cfg is None:
            return []
        cfgs[c] = cfg
        data = mat[:, i].astype(np.float64)
        if cfg.valid_range is not None:
            data, _ = _apply_range_check(data, cfg.valid_range)
        if cfg.spike_detection:
            data, _ = _detect_electrocautery(
                data, grid_sr, threshold_std=cfg.spike_threshold_std,
            )
        marked[c] = data
        valid_all &= ~np.isnan(data)

    # 2. valid_all 연속 span(≥60s) — proxy 로 기존 추출기 재사용
    min_samples = int(60.0 * grid_sr)
    proxy = np.where(valid_all, 0.0, np.nan)
    spans = _extract_nan_free_segments(proxy, min_samples)
    if not spans:
        return []

    # 3. span 마다 채널 정렬 처리 (median → notch → filter → resample)
    out: list[tuple[float, dict[str, np.ndarray]]] = []
    for start, seg in spans:
        length = len(seg)
        sig: dict[str, np.ndarray] = {}
        for c in channels:
            cfg = cfgs[c]
            s = marked[c][start:start + length].astype(np.float64)
            if cfg.median_kernel > 0:
                s = _apply_median_filter(s, kernel_size=cfg.median_kernel)
            if cfg.notch_freq is not None:
                s = _apply_notch_filter(s, freq=cfg.notch_freq, sr=grid_sr)
            s = _apply_filter(s, cfg, grid_sr)
            if grid_sr != TARGET_SR:
                s = resample_to_target(s, orig_sr=grid_sr, target_sr=TARGET_SR)
            sig[c] = s.astype(np.float32)
        # resample 반올림으로 채널 간 1 sample 차 가능 → 최소 길이로 정렬
        m = min(len(v) for v in sig.values())
        sig = {k: v[:m] for k, v in sig.items()}
        span_start_epoch = t0 + start / grid_sr
        out.append((span_start_epoch, sig))
    return out


# ── 윈도우 추출 ──────────────────────────────────────────────


def extract_windows_from_spans(
    spans: list[tuple[float, dict[str, np.ndarray]]],
    center_epoch: float,
    label: int,
    patient_id: str,
    input_signals: list[str],
    window_sec: float,
    stride_sec: float,
    horizon_sec: float,
    max_lead_sec: float,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",
) -> list[MTWindowSample]:
    """한 case 의 정렬 span 들에서 anchor-relative band window 를 추출한다.

    윈도우 종료 절대 epoch 의 center 까지 잔여시간 tte = center − t_end.
    ``horizon_sec ≤ tte ≤ horizon_sec + max_lead_sec`` 인 윈도우만 emit(leakage 없음).
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    samples: list[MTWindowSample] = []

    for span_idx, (span_start, signals) in enumerate(spans):
        if any(s not in signals for s in input_signals):
            continue
        min_len = min(len(signals[s]) for s in input_signals)
        if min_len < win_samples:
            continue

        for start in range(0, min_len - win_samples + 1, stride_samples):
            end = start + win_samples
            t_end = span_start + end / TARGET_SR
            tte = center_epoch - t_end
            if tte < horizon_sec or tte > horizon_sec + max_lead_sec:
                continue

            input_dict = {
                s: signals[s][start:end] for s in input_signals if s in signals
            }
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
                n_gap_s = sum(int(mm.sum()) for mm in gap_mask_dict.values())
                gap_stats.add_window(n_total_s, n_gap_s)

            samples.append(
                MTWindowSample(
                    input_signals=filled_dict,
                    input_gap_masks=gap_mask_dict,
                    label=label,
                    label_value=tte / 60.0,
                    case_id=f"{patient_id}_span{span_idx}_w{start}",
                    patient_id=patient_id,
                    win_start_sec=start / TARGET_SR,
                    horizon_sec=horizon_sec,
                )
            )

    return samples


def process_one_case(
    case: CaseMeta,
    input_signals: list[str],
    window_sec: float,
    horizons_sec: list[float],
    stride_sec: float,
    max_lead_sec: float,
    valid_ratio_threshold: float,
    sample_dtype: str,
) -> dict[float, list[MTWindowSample]]:
    """단일 case: `.vital` 파싱(1회) → 모든 horizon band window 추출.

    Returns
    -------
    {horizon_sec: [MTWindowSample, ...]}  (span 은 추출 직후 폐기 → 메모리 최소).
    """
    spans = parse_vital_aligned_spans(case.vital_path, input_signals)
    result: dict[float, list[MTWindowSample]] = {h: [] for h in horizons_sec}
    if not spans:
        return result
    for h in horizons_sec:
        result[h] = extract_windows_from_spans(
            spans, case.center_epoch, case.label, case.patient_id,
            input_signals, window_sec, stride_sec, h, max_lead_sec,
            valid_ratio_threshold=valid_ratio_threshold,
            sample_dtype=sample_dtype,
        )
    del spans
    return result


# ── multiprocessing worker ───────────────────────────────────

_MP_ARGS: dict = {}


def _worker(case: CaseMeta) -> tuple[str, int, dict[float, list[MTWindowSample]]]:
    """Pool worker: (patient_id, label, {horizon: samples})."""
    res = process_one_case(
        case,
        _MP_ARGS["input_signals"],
        _MP_ARGS["window_sec"],
        _MP_ARGS["horizons_sec"],
        _MP_ARGS["stride_sec"],
        _MP_ARGS["max_lead_sec"],
        _MP_ARGS["valid_ratio_threshold"],
        _MP_ARGS["sample_dtype"],
    )
    return case.patient_id, case.label, res


def _worker_init(mp_args: dict) -> None:
    global _MP_ARGS
    _MP_ARGS = mp_args


# ── case 로딩 (study_population) ─────────────────────────────


def load_case_metas(
    study_csv: str,
    raw_dir: str,
    neg_per_pos: float,
    max_neg_cases: int | None,
    seed: int,
    max_pos_cases: int | None = None,
) -> tuple[list[CaseMeta], float]:
    """study_population.csv → CaseMeta 리스트 (신호 파싱 전).

    양성(masstf=1)은 전부, 음성은 neg_per_pos 배(seeded)만 채택.
    음성 anchor 는 양성의 (onset − opstart) 분포에서 뽑아 opstart 에 더한다(risk-set).
    실제 `.vital` 이 존재하는 row 만 유지.

    Returns
    -------
    (metas, population_prevalence)
        population_prevalence = 전체 study_population 의 masstf+ 비율(≈2.3%, 서브샘플 무관).
    """
    raw = Path(raw_dir)
    pos_rows: list[dict] = []
    neg_rows: list[dict] = []

    with open(study_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = (row.get("fileid") or "").strip()
            if not fid.endswith(".vital"):
                continue
            masstf = (row.get("masstf") or "").strip()
            try:
                is_pos = int(float(masstf)) == 1
            except ValueError:
                continue
            (pos_rows if is_pos else neg_rows).append(row)

    n_total_rows = len(pos_rows) + len(neg_rows)
    population_prevalence = len(pos_rows) / max(1, n_total_rows)
    print(f"  study_population: {len(pos_rows)} 양성(masstf+), {len(neg_rows)} 음성 "
          f"(population prevalence {100 * population_prevalence:.2f}%)")

    # 양성 onset-from-opstart 분포 (risk-set anchor 소스)
    pos_offsets: list[float] = []
    metas: list[CaseMeta] = []
    n_pos_used = 0
    n_pos_skip = 0
    for row in pos_rows:
        if max_pos_cases is not None and n_pos_used >= max_pos_cases:
            break
        fid = row["fileid"].strip()
        vpath = raw / fid
        onset = parse_local_epoch(row.get("correct_start_time"))
        opstart = parse_local_epoch(row.get("opstart"))
        if onset is None or not vpath.is_file():
            n_pos_skip += 1
            continue
        if opstart is not None and onset > opstart:
            pos_offsets.append(onset - opstart)
        metas.append(CaseMeta(
            patient_id=Path(fid).stem, vital_path=vpath, label=1,
            center_epoch=onset,
        ))
        n_pos_used += 1

    if not pos_offsets:
        # onset-opstart 파생 실패 시 보수적 기본값(수술 시작 2시간 후).
        pos_offsets = [7200.0]

    # 음성 서브샘플 (seeded)
    rng = np.random.default_rng(seed)
    n_target = int(round(neg_per_pos * n_pos_used))
    if max_neg_cases is not None:
        n_target = min(n_target, max_neg_cases)
    n_target = min(n_target, len(neg_rows))
    neg_idx = rng.permutation(len(neg_rows))[:n_target]
    offsets_arr = np.asarray(pos_offsets)

    n_neg_used = 0
    n_neg_skip = 0
    for j in neg_idx:
        row = neg_rows[j]
        fid = row["fileid"].strip()
        vpath = raw / fid
        opstart = parse_local_epoch(row.get("opstart"))
        if opstart is None or not vpath.is_file():
            n_neg_skip += 1
            continue
        anchor = opstart + float(rng.choice(offsets_arr))
        metas.append(CaseMeta(
            patient_id=Path(fid).stem, vital_path=vpath, label=0,
            center_epoch=anchor,
        ))
        n_neg_used += 1

    print(
        f"  채택 case: 양성 {n_pos_used}(skip {n_pos_skip}), "
        f"음성 {n_neg_used}/{n_target} 목표(skip {n_neg_skip}); "
        f"risk-set offset N={len(pos_offsets)} "
        f"(median {np.median(offsets_arr) / 60:.0f}min)"
    )
    return metas, population_prevalence


# ── 저장 (cardiac_arrest 규약과 동일 포맷) ────────────────────


def _consume_to_tensors(
    samples: list[MTWindowSample],
    input_signals: list[str],
    signal_dtype: torch.dtype,
) -> dict:
    """samples → packed dict (in-place, samples 비워짐)."""
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
        n = len(samples)
        out = torch.empty((n, T), dtype=signal_dtype)
        for i, s in enumerate(samples):
            arr = s.input_signals.pop(stype, None)
            if arr is None:
                raise ValueError(
                    f"sample {i} missing signal '{stype}' — extraction 이 모든 "
                    f"input_signals 동시존재를 보장해야 함"
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
    dataset_prevalence: float | None = None,
    population_prevalence: float | None = None,
) -> Path:
    """Single split chunk packed dict → 별도 .pt."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_samples = int(split_dict.get("labels", torch.tensor([])).numel())

    save_dict = {
        "split": split_name,
        "data": split_dict,
        "metadata": {
            "task": "massive_transfusion_prediction",
            "source": "SNUH OR (VitalDB .vital)",
            "label": "masstf",
            "aggregation": "window_level",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "max_lead_sec": max_lead_sec,
            "sampling_rate": TARGET_SR,
            "negative_strategy": "risk-set matching (time-from-opstart)",
            "dataset_prevalence": dataset_prevalence,       # 서브샘플 case-level 양성비
            "population_prevalence": population_prevalence,  # 전체 study_pop 실제(≈0.023)
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
        f"massive_transfusion_{mode_str}_w{win_int}s_h{horizon_min}min"
        f"{fold_suffix}_{split_name}{chunk_suffix}.pt"
    )
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {save_path.name} ({file_size_mb:.1f} MB, n={n_samples})")
    return save_path


# ── Sweep ────────────────────────────────────────────────────


def prepare_mt_sweep(
    study_csv: str,
    raw_dir: str,
    input_signals: list[str],
    window_secs: list[float],
    horizon_mins: list[float],
    stride_sec: float = 30.0,
    max_lead_sec: float = 300.0,
    neg_per_pos: float = 3.0,
    max_neg_cases: int | None = None,
    max_pos_cases: int | None = None,
    n_folds: int = 5,
    workers: int = 1,
    out_dir: str = "F:/Massive_Transfusion_Downstream",
    signal_dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> list[Path]:
    """(window, horizon) 조합 × stratified K-fold CV 데이터셋 생성 (window-level)."""
    mode_str = " + ".join(s.upper() for s in input_signals)
    print(f"\n{'=' * 62}")
    print(f"  Massive Transfusion Prediction — SNUH OR (window-level)")
    print(f"  Study CSV: {study_csv}")
    print(f"  Raw dir:   {raw_dir}")
    print(f"  Input:     {mode_str}")
    print(f"  Windows:   {window_secs}s  Horizons: {horizon_mins} min")
    print(f"  Max lead:  {max_lead_sec / 60:.1f} min (윈도우 종료 band)")
    print(f"  Neg/Pos:   {neg_per_pos}x  Workers: {workers}")
    print(f"  Output:    {out_dir}")
    print(f"{'=' * 62}")

    # 1. case 메타 로딩 (label + anchor, 신호 파싱 전)
    print("\n[1/4] Loading case labels + risk-set anchors from study_population...")
    metas, population_prev = load_case_metas(
        study_csv, raw_dir, neg_per_pos, max_neg_cases, seed,
        max_pos_cases=max_pos_cases,
    )
    if not metas:
        print("ERROR: No cases loaded.", file=sys.stderr)
        sys.exit(1)

    n_pos_cases = sum(1 for m in metas if m.label == 1)
    # 서브샘플 case-level prevalence (신호 무관). population_prev 는 전체 대비 실제값(≈2.3%).
    dataset_prev = n_pos_cases / max(1, len(metas))

    # 2. `.vital` 파싱(1회) → case 별 shard 를 **디스크에 즉시 기록**하고 메모리 해제.
    #    (모든 window 를 부모 RAM 에 쌓으면 OOM — 32GB 기기에서 부모만 ~15GB 로 커짐.
    #     대신 case 별로 파싱→pack→shard 저장→free 하여 부모 메모리를 상수로 bound.)
    import shutil

    horizons_sec = [h * 60.0 for h in horizon_mins]
    window_sec = window_secs[0]
    if len(window_secs) > 1:
        print(f"  [note] window-secs 여러 개 지정 — 첫 값 {window_sec}s 만 사용"
              f"(파싱 1회 재사용 위해).", file=sys.stderr)

    print(f"\n[2/4] Parsing {len(metas)} `.vital` files → per-case shards "
          f"(window={window_sec}s, horizons={horizon_mins}min)...")

    mp_args = {
        "input_signals": input_signals,
        "window_sec": window_sec,
        "horizons_sec": horizons_sec,
        "stride_sec": stride_sec,
        "max_lead_sec": max_lead_sec,
        "valid_ratio_threshold": DEFAULT_VALID_RATIO_THRESHOLD,
        "sample_dtype": str(signal_dtype).replace("torch.", ""),
    }

    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    shard_dir = Path(out_dir) / "_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # horizon 별 shard 인덱스: {horizon_sec: [(patient_id, label, shard_path), ...]}
    shard_index: dict[float, list[tuple[str, int, Path]]] = {h: [] for h in horizons_sec}
    patient_to_label: dict[str, int] = {}
    win_counts: dict[float, tuple[int, int]] = {h: (0, 0) for h in horizons_sec}  # (n, n_pos)
    n_cases_with_windows = 0

    def _write_shards(pid: str, label: int, res: dict[float, list[MTWindowSample]]) -> None:
        nonlocal n_cases_with_windows
        got = False
        for h, s_list in res.items():
            if not s_list:
                continue
            n_pos = sum(1 for s in s_list if s.label == 1)
            packed = _consume_to_tensors(s_list, input_signals, signal_dtype)
            n = int(packed.get("labels", torch.tensor([])).numel())
            if n == 0:
                continue
            sp = shard_dir / f"{pid}_h{int(h / 60)}.pt"
            torch.save(packed, sp)
            shard_index[h].append((pid, label, sp))
            c_n, c_p = win_counts[h]
            win_counts[h] = (c_n + n, c_p + n_pos)
            got = True
            del packed
        if got:
            patient_to_label[pid] = label
            n_cases_with_windows += 1

    if workers > 1:
        from multiprocessing import Pool
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(mp_args,)) as pool:
            it = pool.imap_unordered(_worker, metas, chunksize=1)
            if _have_tqdm:
                it = tqdm(it, total=len(metas), desc="Parsing .vital", unit="case")
            for pid, label, res in it:
                _write_shards(pid, label, res)
                del res
    else:
        _worker_init(mp_args)
        it = tqdm(metas, desc="Parsing .vital", unit="case") if _have_tqdm else metas
        for case in it:
            pid, label, res = _worker(case)
            _write_shards(pid, label, res)
            del res

    for h in horizons_sec:
        n, n_pos = win_counts[h]
        print(f"    h={int(h / 60)}min: {n} windows (+={n_pos})")
    print(f"  cases yielding ≥1 window: {n_cases_with_windows}/{len(metas)}")

    # 3. Stratified patient-level K-fold CV (label 기준)
    patient_ids = sorted(patient_to_label.keys())
    patient_to_labels = {pid: [patient_to_label[pid]] for pid in patient_ids}
    pos_pts = sum(1 for pid in patient_ids if patient_to_label[pid] == 1)
    print(f"\n[3/4] Stratified {n_folds}-fold patient-level CV "
          f"(masstf+ patients: {pos_pts}/{len(patient_ids)})...")
    if pos_pts < n_folds or (len(patient_ids) - pos_pts) < n_folds:
        print(f"  WARNING: fold 당 양성/음성 부족 가능 (pos={pos_pts}, "
              f"neg={len(patient_ids) - pos_pts}, folds={n_folds}).", file=sys.stderr)
    splits = stratified_kfold_patient_splits(
        patient_ids, patient_to_labels, n_folds=n_folds, seed=seed,
    )
    print(summarize_splits(splits, patient_to_labels))

    # 4. horizon × fold × split → shard 스트리밍 병합 → chunk 저장.
    #    한 번에 chunk 1개(≤WIN_PER_CHUNK window)만 버퍼링 → 부모 메모리 bound.
    print(f"\n[4/4] Assembling datasets from shards "
          f"({len(horizons_sec)} horizon × {len(splits)} fold)...")
    saved_paths: list[Path] = []
    WIN_PER_CHUNK = 8000  # chunk 당 window 상한 (~3GB float16 이내)

    for h in horizons_sec:
        horizon_min = int(h / 60)
        pid_to_shard = {pid: sp for (pid, _lab, sp) in shard_index[h]}

        for fold_idx, (train_pids, val_pids, test_pids) in enumerate(splits):
            cur_fold = fold_idx if len(splits) > 1 else None
            for split_name, split_pids in (
                ("train", train_pids), ("val", val_pids), ("test", test_pids),
            ):
                buf: dict = {
                    "sig": {s: [] for s in input_signals},
                    "gap": {s: [] for s in input_signals},
                    "lab": [], "lv": [], "cid": [], "sid": [],
                }
                buffered = 0
                chunk_idx = 0
                total = 0
                total_pos = 0

                def _flush() -> None:
                    nonlocal buffered, chunk_idx
                    if buffered == 0:
                        return
                    packed = {
                        "signals": {s: torch.cat(buf["sig"][s], 0)
                                    for s in input_signals if buf["sig"][s]},
                        "gap_masks": {s: torch.cat(buf["gap"][s], 0)
                                      for s in input_signals if buf["gap"][s]},
                        "labels": torch.cat(buf["lab"]),
                        "label_values": torch.cat(buf["lv"]),
                        "case_ids": list(buf["cid"]),
                        "subject_ids": list(buf["sid"]),
                    }
                    sp = save_split_dataset(
                        packed, split_name, input_signals, h, window_sec, out_dir,
                        max_lead_sec=max_lead_sec, signal_dtype=signal_dtype,
                        fold_idx=cur_fold, n_folds=len(splits), chunk_idx=chunk_idx,
                        dataset_prevalence=dataset_prev,
                        population_prevalence=population_prev,
                    )
                    saved_paths.append(sp)
                    chunk_idx += 1
                    for s in input_signals:
                        buf["sig"][s].clear()
                        buf["gap"][s].clear()
                    buf["lab"].clear(); buf["lv"].clear()
                    buf["cid"].clear(); buf["sid"].clear()
                    buffered = 0
                    del packed; gc.collect()

                for pid in sorted(split_pids):
                    sp = pid_to_shard.get(pid)
                    if sp is None:
                        continue
                    shard = torch.load(sp, weights_only=False)
                    n = int(shard["labels"].numel())
                    if n == 0:
                        continue
                    for s in input_signals:
                        if s in shard["signals"]:
                            buf["sig"][s].append(shard["signals"][s])
                        if s in shard["gap_masks"]:
                            buf["gap"][s].append(shard["gap_masks"][s])
                    buf["lab"].append(shard["labels"])
                    buf["lv"].append(shard["label_values"])
                    buf["cid"].extend(shard["case_ids"])
                    buf["sid"].extend(shard["subject_ids"])
                    buffered += n
                    total += n
                    total_pos += int(shard["labels"].sum().item())
                    if buffered >= WIN_PER_CHUNK:
                        _flush()
                _flush()
                if total > 0:
                    print(f"    h={horizon_min}min fold{fold_idx} {split_name}: "
                          f"{chunk_idx} chunk(s), {total} windows (+={total_pos})")

    # shard 임시파일 정리
    shutil.rmtree(shard_dir, ignore_errors=True)

    print(f"\n{'=' * 62}")
    print(f"  Done! {len(saved_paths)} files → {out_dir}")
    print(f"  Population prevalence≈{100 * population_prev:.2f}% (양성416/전체17986); "
          f"서브샘플 dataset prevalence≈{100 * dataset_prev:.1f}% (AUROC 주지표).")
    print(f"{'=' * 62}")
    return saved_paths


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    # Windows 콘솔(cp949)에서도 유니코드(—, ≥, × 등) 출력이 죽지 않도록 UTF-8 재설정.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Massive Transfusion Prediction — Data Preparation "
                    "(SNUH OR, VitalDB .vital, window-level)",
    )
    parser.add_argument(
        "--study-csv", type=str, required=True,
        help="study_population.csv (masstf 라벨 + correct_start_time onset).",
    )
    parser.add_argument(
        "--raw-dir", type=str, required=True,
        help="`.vital` 파일 루트 (study_population.fileid 와 합쳐짐).",
    )
    parser.add_argument(
        "--input-signals", nargs="+", default=["abp", "ppg"],
        choices=["ecg", "abp", "ppg", "cvp", "co2", "awp"],
        help="Input signal types. Default: ABP + PPG (실혈성 저혈압 + 말초관류 저하). "
             "전 채널 동시 valid 구간만 span 으로 사용(intersection).",
    )
    parser.add_argument(
        "--horizon-mins", nargs="+", type=float, default=[5.0, 10.0, 15.0],
        help="Prediction horizons (분). Default 5/10/15.",
    )
    parser.add_argument(
        "--window-secs", nargs="+", type=float, default=[600.0],
        help="Input window 길이(초). Default 600 (10분). 여러 개면 첫 값만 사용.",
    )
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument(
        "--max-lead-sec", type=float, default=300.0,
        help="윈도우 종료 band: h ≤ tte ≤ h+max_lead. Default 300s(5분).",
    )
    parser.add_argument(
        "--neg-per-pos", type=float, default=3.0,
        help="음성 case 서브샘플 배수(양성 대비). Default 3.",
    )
    parser.add_argument(
        "--max-neg-cases", type=int, default=None,
        help="음성 case 절대 상한 (None=neg-per-pos 만 적용).",
    )
    parser.add_argument(
        "--max-pos-cases", type=int, default=None,
        help="양성 case 상한 (None=전량 416). smoke test/디버깅용.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Stratified patient-level K-fold CV (default 5).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="`.vital` 파싱 병렬 worker 수 (기본 1). CPU-heavy 라 8+ 권장.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=str, default="F:/Massive_Transfusion_Downstream",
    )
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    prepare_mt_sweep(
        study_csv=args.study_csv,
        raw_dir=args.raw_dir,
        input_signals=args.input_signals,
        window_secs=args.window_secs,
        horizon_mins=args.horizon_mins,
        stride_sec=args.stride_sec,
        max_lead_sec=args.max_lead_sec,
        neg_per_pos=args.neg_per_pos,
        max_neg_cases=args.max_neg_cases,
        max_pos_cases=args.max_pos_cases,
        n_folds=args.n_folds,
        workers=args.workers,
        seed=args.seed,
        out_dir=args.out_dir,
        signal_dtype=dtype_map(args.signal_dtype),
    )


if __name__ == "__main__":
    main()
