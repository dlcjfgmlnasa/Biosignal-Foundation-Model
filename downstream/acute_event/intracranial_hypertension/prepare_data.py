# -*- coding:utf-8 -*-
"""Intracranial Hypertension Detection — 데이터 준비 (MIMIC-III).

MIMIC-III Waveform에서 ICP 채널이 있는 레코드를 파싱하여
ICP > 20mmHg (≥1분 지속) 예측용 (input_window, future_label) 쌍을 생성한다.
  (임계 20mmHg = intracranial hypertension 정의 Czosnyka & Pickard 2004, 논문 Table 3
   기준. BTF Carney 2017 치료 임계 22 보다 낮춰 희소 양성 표본 확보.)

Label 소스: ICP (미래 구간의 평균 ICP > 20mmHg 지속 여부)
Input 소스: ICP + 동시 기록된 ECG, ABP, PPG 등
표본 구성 (--sampling-mode, IOH 와 동일 원칙; 공통: 현재-ICH 제외 = 예측 시점 ICP<20 만):
  - unbiased(기본): dense sliding, 미래 sustained ICP>20 = positive, 나머지(15~20 경계
    포함) 전부 negative. sparse/clean-neg 없음 → 실제 forward 평가(현실적).
  - biased(supplementary, 선행 crisis/non-crisis epoch 방식): clean-neg(미래 valid ICP 전부
    <15, Czosnyka 2004)만 negative + 경계(15~20) 제외 + sparse 20분 → 과대평가.

데이터 소스: MIMIC-III Waveform Matched Subset (PhysioNet)

사용법:
    # Sweep: window × horizon 전체 조합 생성
    python -m downstream.acute_event.intracranial_hypertension.prepare_data \
        --waveform-dir datasets/raw/mimic3-waveform-ich \
        --window-secs 30 60 300 600 --horizon-mins 5 10 15
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

from data.parser._common import resample_to_target
from data.parser.mimic3_waveform import _apply_pipeline
from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    _apply_filter,
    _apply_median_filter,
    _apply_range_check,
    _detect_electrocautery,
)
from downstream._gap_mask import (
    DEFAULT_VALID_RATIO_THRESHOLD,
    GapStats,
    apply_gap_mask_multichannel,
    compute_valid_ratio,
)
from downstream._kfold_utils import stratified_kfold_patient_splits, summarize_splits
from downstream._save_utils import consume_gap_masks


TARGET_SR: float = 100.0

# ICP 임상 기준
ICP_THRESHOLD: float = 20.0  # mmHg (sustained >1min). intracranial hypertension 정의
#                              (Czosnyka & Pickard 2004; 논문 Table 3 기준). 정상 ICP 7-15mmHg.
#                              (BTF 4판 Carney 2017 치료 임계 22 보다 낮춰 희소 양성 표본 확보.)
SUSTAINED_SEC: float = 60.0  # 1분 이상 지속 (구 10s-평균 sustained 모드 잔존 인자)

# ── Güiza 식 이벤트 정의 (anchored 샘플링용) ──────────────────
#   ICP 를 1분 median 블록으로 요약(아티팩트 견고) → 5블록 연속(=5분) median > 20
#   이면 ICP crisis 로 확정. run 첫 블록 = onset. (Güiza et al. Crit Care Med 2013)
BLOCK_SEC: float = 60.0     # 1분 median 블록
EVENT_CONSEC: int = 5       # 5블록 연속(=5분) > threshold = crisis 확정

# MIMIC-III 채널명 → signal_type 매핑 (Task #3 ICH: ABP, ECG, PPG, CO2 + ICP for label)
# ICP는 라벨 산출 전용 (input 신호로는 사용하지 않음, --input-signals 로 별도 제어)
MIMIC_SIGNAL_MAP: dict[str, str] = {
    "II": "ecg",
    "V": "ecg",
    "ABP": "abp",
    "ART": "abp",
    "PLETH": "ppg",
    "CO2": "co2",
    "ICP": "icp",
}


# ── 데이터 구조 ──────────────────────────────────────────────


@dataclass
class ICHSample:
    """두개내 고혈압 예측 샘플."""

    input_signals: dict[str, np.ndarray]
    input_gap_masks: dict[str, np.ndarray]  # bool, True=원본이 NaN (gap)
    label: int  # 0=normal, 1=intracranial hypertension
    label_value: float  # future max ICP (mmHg)
    case_id: str
    patient_id: str  # LOSO grouping 필수
    win_start_sec: float
    horizon_sec: float


# ── WFDB 파싱 ────────────────────────────────────────────────


def parse_waveform_record(
    record_dir: Path,
    record_name: str,
) -> dict[str, np.ndarray] | None:
    """WFDB 레코드에서 ICP + 기타 신호를 추출한다.

    ICP가 없는 레코드는 None 반환.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    hea_path = record_dir / f"{record_name}.hea"
    if not hea_path.exists():
        return None

    try:
        rec = wfdb.rdrecord(str(record_dir / record_name))
    except Exception:
        return None

    if rec.p_signal is None or rec.sig_len == 0:
        return None

    fs = float(rec.fs)
    signals: dict[str, np.ndarray] = {}

    for ch_idx, sig_name in enumerate(rec.sig_name):
        sig_type = MIMIC_SIGNAL_MAP.get(sig_name)
        if sig_type is None:
            continue
        if sig_type in signals:
            continue

        signal = rec.p_signal[:, ch_idx].astype(np.float64)
        processed = _apply_pipeline(signal, sig_type, fs)
        if processed is None:
            continue

        signals[sig_type] = processed.astype(np.float32)

    # ICP 필수
    if "icp" not in signals:
        return None

    return signals


# ── 정렬(aligned) 파싱: ABP+ICP 를 공통 시간축에서 세그먼트 단위로 ──────
#   parse_waveform_record 는 채널별로 각자의 최장 NaN-free 세그먼트를 뽑아
#   (offset 폐기) 채널 간 시간축이 어긋날 수 있다(multi-modal 정렬 버그). anchored
#   prediction 은 ICP 라벨과 ABP 입력이 같은 시각이어야 하므로, 채널 공통 valid 구간
#   을 교집합해 세그먼트별로 정렬 처리한다(불연속 concat 금지 — 세그먼트 분리).


def _contiguous_runs(valid: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """valid(bool) 에서 연속 True 구간 [s, e) 중 길이 ≥ min_samples 인 것."""
    diff = np.diff(valid.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_samples]


def _process_channel_segment(
    raw_seg: np.ndarray, stype: str, native_sr: float,
) -> np.ndarray | None:
    """NaN-free 원시 세그먼트에 median→filter→resample 적용 (정렬·길이 보존).

    _apply_pipeline 의 Step 4-5 와 동일하나 NaN-free 세그먼트 추출은 이미 공통
    교집합에서 끝났으므로 생략 → 채널 간 정렬 유지.
    """
    cfg = SIGNAL_CONFIGS.get(stype)
    if cfg is None:
        return None
    seg = raw_seg
    if cfg.median_kernel > 0:
        seg = _apply_median_filter(seg, kernel_size=cfg.median_kernel)
    seg = _apply_filter(seg, cfg, native_sr)
    if native_sr != TARGET_SR:
        seg = resample_to_target(seg, orig_sr=native_sr, target_sr=TARGET_SR)
    return seg.astype(np.float32)


def parse_aligned_segments(
    record_dir: Path,
    record_name: str,
    input_signals: list[str],
    min_seg_sec: float,
) -> list[dict[str, np.ndarray]]:
    """레코드에서 input_signals(+icp) 를 정렬된 연속 세그먼트로 반환한다.

    반환 각 원소: {stype: ndarray(TARGET_SR)} — 모든 채널이 **동일 길이·동일 시각**.
    ICP 는 라벨 산출용으로 항상 포함. 공통 valid 구간이 min_seg_sec 미만이면 제외.
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

    fs = float(rec.fs)
    needed = set(input_signals) | {"icp"}

    # 원시 채널 수집 (같은 record → 모든 채널 동일 fs·동일 길이 = 정렬됨)
    raw: dict[str, np.ndarray] = {}
    for ch_idx, sname in enumerate(rec.sig_name):
        st = MIMIC_SIGNAL_MAP.get(sname)
        if st is None or st not in needed or st in raw:
            continue
        raw[st] = rec.p_signal[:, ch_idx].astype(np.float64)
    if not needed.issubset(raw):
        return []

    # range check + spike → NaN (채널별, 길이 보존)
    clean: dict[str, np.ndarray] = {}
    for st, arr in raw.items():
        cfg = SIGNAL_CONFIGS.get(st)
        a = arr
        if cfg is not None and cfg.valid_range is not None:
            a, _ = _apply_range_check(a, cfg.valid_range)
        if cfg is not None and cfg.spike_detection:
            a, _ = _detect_electrocautery(a, fs, threshold_std=cfg.spike_threshold_std)
        clean[st] = a

    # 공통 valid 마스크 (모든 채널이 동시에 유효한 구간만)
    n = min(len(a) for a in clean.values())
    common_valid = np.ones(n, dtype=bool)
    for a in clean.values():
        common_valid &= ~np.isnan(a[:n])

    min_samples_native = int(min_seg_sec * fs)
    segments: list[dict[str, np.ndarray]] = []
    for s, e in _contiguous_runs(common_valid, min_samples_native):
        seg: dict[str, np.ndarray] = {}
        ok = True
        for st, a in clean.items():
            proc = _process_channel_segment(a[s:e], st, fs)
            if proc is None or proc.size == 0:
                ok = False
                break
            seg[st] = proc
        if not ok:
            continue
        # resample 반올림 차이 방어 — 채널 공통 최소 길이로 절단 (정렬 유지)
        L = min(len(v) for v in seg.values())
        segments.append({st: v[:L] for st, v in seg.items()})
    return segments


# ── Güiza 라벨 엔진 (1분 median onset) ───────────────────────


def _icp_median_blocks(icp: np.ndarray) -> np.ndarray:
    """100Hz ICP → 1분 median 블록 timeline (Güiza). 세그먼트는 NaN-free 전제."""
    block_samps = max(1, int(round(BLOCK_SEC * TARGET_SR)))  # 6000
    n_blocks = len(icp) // block_samps
    if n_blocks == 0:
        return np.empty(0, dtype=np.float32)
    trimmed = icp[: n_blocks * block_samps].reshape(n_blocks, block_samps)
    return np.median(trimmed, axis=1).astype(np.float32)


def _find_onsets(blocks: np.ndarray, threshold: float, consec: int) -> list[int]:
    """5블록 연속 median>threshold 인 crisis run 의 onset(첫 블록) index 목록.

    run 이 consec 에 도달하는 순간 run_start 를 onset 으로 확정(ICP 가 처음 임계를
    넘은 시점). 한 run 은 onset 하나만 → 진행 중 위기 중복 계수 차단.
    (scan_positives.find_onsets 와 동일 정의; 순환 import 회피 위해 복제.)
    """
    onsets: list[int] = []
    run = 0
    run_start = 0
    for i, v in enumerate(blocks):
        if not np.isnan(v) and v > threshold:
            if run == 0:
                run_start = i
            run += 1
            if run == consec:
                onsets.append(run_start)
        else:
            run = 0
    return onsets


# ── 환자별 레코드 탐색 ───────────────────────────────────────


def load_patient_signals(
    waveform_dir: Path,
    required_signals: list[str] | None = None,
) -> list[dict]:
    """Waveform 디렉토리에서 ICP 포함 레코드를 로드한다.

    required_signals 지정 시 그 channel set 이 **모두** 존재하는 record 만 유지
    (VitalDB task / arrhythmia 패턴과 정합 — sample 마다 동일 채널 보장).
    ICP 는 라벨 산출에 항상 필요하므로 자동으로 required 에 포함.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signals": {type: ndarray}}
    """
    import re

    if required_signals is not None:
        required_set = set(required_signals) | {"icp"}
    else:
        required_set = {"icp"}

    # 마스터 헤더만 선별: 파일명이 p<6자리>-... 형태
    # (segment: <num>_NNNN.hea, layout: <num>_layout.hea 는 제외)
    master_re = re.compile(r"^p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$")
    all_hea = sorted(
        h for h in waveform_dir.rglob("*.hea") if master_re.match(h.stem)
    )
    print(f"  Found {len(all_hea)} master .hea files")
    print(f"  Required signals: {sorted(required_set)}")

    cases: list[dict] = []
    n_no_icp = 0
    n_missing_required = 0

    for i, hea_path in enumerate(all_hea):
        rec_name = hea_path.stem
        rec_dir = hea_path.parent

        # subject_id 추출: 디렉토리(p00/p000907) 우선, 없으면 파일명 prefix(p000907)에서
        patient_id = "unknown"
        for part in hea_path.parts:
            if part.startswith("p") and len(part) == 7 and part[1:].isdigit():
                patient_id = part
                break
        if patient_id == "unknown":
            patient_id = rec_name.split("-", 1)[0]  # p000907

        signals = parse_waveform_record(rec_dir, rec_name)
        if signals is None:
            n_no_icp += 1
            continue

        if not required_set.issubset(set(signals.keys())):
            n_missing_required += 1
            continue

        cases.append({
            "case_id": f"{patient_id}_{rec_name}",
            "patient_id": patient_id,
            "signals": signals,
        })

        if (i + 1) % 50 == 0 or i == 0:
            sig_types = list(signals.keys())
            dur_sec = len(signals["icp"]) / TARGET_SR
            print(f"    [{i + 1}/{len(all_hea)}] {patient_id}/{rec_name} "
                  f"signals={sig_types} dur={dur_sec:.0f}s")

    print(
        f"  ICP records: {len(cases)}, "
        f"skipped (no ICP): {n_no_icp}, "
        f"skipped (missing required {sorted(required_set - {'icp'})}): "
        f"{n_missing_required}"
    )
    return cases


# ── 레코드 열거 (scandir flat-numeric / rglob matched) ───────

_SEG_RE = re.compile(r"_\d{4}$")
_MASTER_NUM_RE = re.compile(r"^\d+$")
_MASTER_MATCHED_RE = re.compile(r"^p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$")


def _patient_id_of(stem: str, parts: tuple[str, ...]) -> str:
    """matched(pXXXXXX) 는 subject dir/prefix, numeric 은 record stem 을 pid 로."""
    for part in parts:
        if part.startswith("p") and len(part) == 7 and part[1:].isdigit():
            return part
    m = _MASTER_MATCHED_RE.match(stem)
    if m:
        return stem.split("-", 1)[0]
    return stem  # numeric general mimic3wdb: record = group


def list_records_meta(
    waveform_dir: Path, scan_dir: bool,
) -> list[tuple[str, Path, str]]:
    """(patient_id, record_dir, record_name) 목록.

    scan_dir=True: waveform_dir 최상위를 scandir(재귀·stat 없음)해 마스터 헤더 직접
      열거 — flat numeric(3008477.hea/_0001.dat, 일반 mimic3wdb) 레이아웃 대응.
    scan_dir=False: rglob 로 matched(pXXXXXX-...) 마스터 열거(네트워크 마운트 느림).
    """
    import os

    out: list[tuple[str, Path, str]] = []
    if scan_dir:
        with os.scandir(waveform_dir) as it:
            for entry in it:
                if not entry.name.endswith(".hea"):
                    continue
                stem = entry.name[:-4]
                if stem.endswith("_layout") or _SEG_RE.search(stem):
                    continue
                if _MASTER_MATCHED_RE.match(stem) or _MASTER_NUM_RE.match(stem):
                    out.append((_patient_id_of(stem, ()), waveform_dir, stem))
    else:
        for hea in waveform_dir.rglob("*.hea"):
            if not _MASTER_MATCHED_RE.match(hea.stem):
                continue
            out.append((_patient_id_of(hea.stem, hea.parts), hea.parent, hea.stem))
    return sorted(out)


def load_aligned_cases(
    waveform_dir: Path,
    input_signals: list[str],
    min_seg_sec: float,
    scan_dir: bool = True,
) -> list[dict]:
    """정렬 세그먼트 단위 case 리스트.

    각 case = 한 연속 세그먼트 {case_id, patient_id, signals{stype: ndarray100}} —
    모든 채널 동일 길이·동일 시각. 한 레코드가 여러 세그먼트면 같은 patient_id 로
    복수 case 생성(case_id 에 _segK 접미).
    """
    records = list_records_meta(waveform_dir, scan_dir=scan_dir)
    print(f"  Found {len(records)} master records "
          f"({'scandir' if scan_dir else 'rglob'})")
    print(f"  Input signals (aligned, +icp label): {sorted(set(input_signals) | {'icp'})}")

    cases: list[dict] = []
    n_no_sig = 0
    n_seg = 0
    for i, (pid, rec_dir, rec_name) in enumerate(records):
        segs = parse_aligned_segments(rec_dir, rec_name, input_signals, min_seg_sec)
        if not segs:
            n_no_sig += 1
            continue
        for k, seg in enumerate(segs):
            n_seg += 1
            cases.append({
                "case_id": f"{pid}_{rec_name}_seg{k}",
                "patient_id": pid,
                "signals": seg,
            })
        if (i + 1) % 50 == 0 or i == 0:
            dur = sum(len(s["icp"]) for s in segs) / TARGET_SR
            print(f"    [{i + 1}/{len(records)}] {pid}/{rec_name} "
                  f"segs={len(segs)} dur={dur:.0f}s")

    print(f"  Aligned cases (segments): {n_seg} from "
          f"{len(records) - n_no_sig} records "
          f"(skipped no-abp/icp/short: {n_no_sig})")
    return cases


# ── 라벨링 ───────────────────────────────────────────────────


def _has_sustained_ich(
    future_icps: list[float],
    threshold: float,
    min_consecutive: int,
) -> bool:
    """연속 min_consecutive개 이상 윈도우에서 ICP > threshold인지 확인한다."""
    consecutive = 0
    for icp_val in future_icps:
        if icp_val > threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                return True
        else:
            consecutive = 0
    return False


# ── 윈도우 추출 (anchored: Güiza onset + 균일 stride 음성) ────


def extract_anchored_samples(
    cases: list[dict],
    input_signals: list[str],
    window_sec: float = 1200.0,
    horizon_sec: float = 300.0,
    neg_stride_sec: float = 600.0,
    icp_threshold: float = ICP_THRESHOLD,
    event_consec: int = EVENT_CONSEC,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",
    pos_mode: str = "dense",
    pos_stride_sec: float = 60.0,
) -> list[ICHSample]:
    """정렬 세그먼트 case 들에서 (input, label) 쌍을 추출한다.

    ``pos_mode`` (양성 sampling):
      "dense" (기본): pos_stride 간격 dense sliding — 예측구간 [win_end, win_end+H)
        안에 crisis onset 이 있으면 양성. horizon 이 길수록 양성 window 가 늘어 →
        prevalence 가 horizon 따라 증가(IOH 정합, scan dense 모드와 동일 정의).
      "anchored" (레거시): crisis onset 마다 정확히 horizon 앞에서 끝나는 window
        1개(중복 제거·onset 보존). onset 수와 무관하게 horizon 무관 flat prevalence.
    두 모드 공통: 예측 시점(입력 끝) baseline median ≤ threshold 여야 함(현재-ICH 제외).

    음성: crisis 없는 구간을 neg_stride_sec 균일 stride 로 추출(경계 15~20 유지 →
      정직). horizon 에 onset 이 없고 baseline ≤ threshold 인 window. dense 양성 +
      strided 음성이라 음성 수는 ~horizon 무관 → prevalence 가 양성(horizon 의존)으로
      결정된다. neg_stride 는 prevalence 다이얼(600s ≈ IOH 스케일 3.9/7/9.7%).

    ICP 라벨 판정은 1분 median 블록 그리드에서, 입력 window 추출은 그 블록 경계에
    정렬된 100Hz 구간에서 이뤄진다(block b → sample [b*6000, ...)).
    """
    block_samps = max(1, int(round(BLOCK_SEC * TARGET_SR)))  # 6000
    win_b = max(1, int(round(window_sec / BLOCK_SEC)))
    hor_b = max(1, int(round(horizon_sec / BLOCK_SEC)))
    neg_b = max(1, int(round(neg_stride_sec / BLOCK_SEC)))
    pos_b = max(1, int(round(pos_stride_sec / BLOCK_SEC)))
    win_samples = int(window_sec * TARGET_SR)
    total_needed_b = win_b + hor_b

    samples: list[ICHSample] = []

    for case in cases:
        signals = case["signals"]
        if "icp" not in signals:
            continue
        icp = signals["icp"]
        blocks = _icp_median_blocks(icp)
        L = len(blocks)
        if L < total_needed_b:
            continue
        onsets = _find_onsets(blocks, icp_threshold, event_consec)
        onset_set = set(onsets)

        def _emit(start_block: int, label: int) -> None:
            start = start_block * block_samps
            input_dict: dict[str, np.ndarray] = {}
            for st in input_signals:
                arr = signals.get(st)
                if arr is None or start + win_samples > len(arr):
                    return  # 채널 길이 부족 → window 불가
                input_dict[st] = arr[start: start + win_samples]
            if not input_dict:
                return
            valid_ratio = compute_valid_ratio(list(input_dict.values()))
            if valid_ratio < valid_ratio_threshold:
                if gap_stats is not None:
                    gap_stats.add_drop()
                return
            win_end_b = start_block + win_b
            fut = blocks[win_end_b: win_end_b + hor_b]
            fut = fut[~np.isnan(fut)]
            label_value = float(fut.max()) if fut.size else float("nan")
            filled_dict, gap_mask_dict = apply_gap_mask_multichannel(
                input_dict, output_dtype=sample_dtype,
            )
            if gap_stats is not None:
                n_total_s = sum(a.size for a in filled_dict.values())
                n_gap_s = sum(int(m.sum()) for m in gap_mask_dict.values())
                gap_stats.add_window(n_total_s, n_gap_s)
            samples.append(
                ICHSample(
                    input_signals=filled_dict,
                    input_gap_masks=gap_mask_dict,
                    label=label,
                    label_value=label_value,
                    case_id=case["case_id"],
                    patient_id=case["patient_id"],
                    win_start_sec=start_block * BLOCK_SEC,
                    horizon_sec=horizon_sec,
                )
            )

        # ── 양성 ──
        if pos_mode == "dense":
            # dense sliding(pos_stride): 예측구간 안에 onset 이 있고 현재-ICH 아니면 양성.
            # horizon 길수록 band 가 커져 양성 window ↑ → prevalence ↑.
            for start_block in range(0, L - total_needed_b + 1, pos_b):
                win_end_b = start_block + win_b
                base = blocks[win_end_b - 1]           # 예측 시점 baseline
                if np.isnan(base) or float(base) > icp_threshold:
                    continue                            # 현재-ICH/무효 → 제외
                if any(win_end_b <= o < win_end_b + hor_b for o in onset_set):
                    _emit(start_block, 1)
        else:  # anchored (레거시): onset 당 1개, horizon 끝 블록 == onset
            for o in onsets:
                win_end_b = o - hor_b + 1
                start_block = win_end_b - win_b
                if start_block < 0:
                    continue
                base = blocks[win_end_b - 1]          # 예측 시점 baseline
                if np.isnan(base) or float(base) > icp_threshold:
                    if gap_stats is not None:
                        gap_stats.add_drop()         # 현재-ICH → 제외
                    continue
                _emit(start_block, 1)

        # ── 음성: crisis-free 균일 stride ──
        for start_block in range(0, L - total_needed_b + 1, neg_b):
            win_end_b = start_block + win_b
            base = blocks[win_end_b - 1]
            if np.isnan(base) or float(base) > icp_threshold:
                continue
            if any(win_end_b <= o < win_end_b + hor_b for o in onset_set):
                continue                              # horizon 에 onset → 음성 아님
            _emit(start_block, 0)

    return samples


# ── 윈도우 추출 (구 dense/biased 10s-평균 모드) ───────────────


def extract_forecast_samples(
    cases: list[dict],
    input_signals: list[str],
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    horizon_sec: float = 300.0,
    icp_threshold: float = ICP_THRESHOLD,
    neg_icp_threshold: float = 15.0,        # biased 모드 clean-neg: 미래 내내 ICP<이 값(정상)
    sustained_sec: float = SUSTAINED_SEC,
    min_sample_gap_sec: float = 1200.0,     # biased 모드 sparse 간격(20분)
    sampling_mode: str = "unbiased",        # unbiased(현실적·기본) | biased(과대평가·선행비교)
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",  # OOM 회피
    task_mode: str = "prediction",  # prediction(미래 horizon IH) | detection(동시 window IH, aICP식)
) -> list[ICHSample]:
    """시간 정렬된 다채널 데이터에서 (input, future_label) 쌍을 추출한다.

    task_mode:
      "prediction" — 입력 window 끝 이후 horizon 구간의 IH 를 예측. 현재-ICH(입력
        끝 baseline ICP>임계) 제외 → 신규 발생 예측(기존 canonical).
      "detection"  — 입력 window 와 동시(same-window)의 IH 를 탐지(aICP식). 현재-ICH
        제외 없음, horizon 무시. ICP 는 입력에 없고 라벨로만 쓰여 circularity 차단.

    Gap policy (project_downstream_gap_window_policy):
      Step 1 — input window 의 multi-channel valid_ratio < threshold → window drop
      Step 2 — 통과 window 의 NaN 위치는 0 으로 채우고 gap_mask boolean 저장.
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    horizon_samples = int(horizon_sec * TARGET_SR)
    is_detection = task_mode == "detection"

    icp_win_sec = 10.0
    icp_win = int(icp_win_sec * TARGET_SR)
    min_consecutive = max(1, int(sustained_sec / icp_win_sec))
    min_gap_samples = int(min_sample_gap_sec * TARGET_SR)
    # 라벨 구간 내 기대 10초-윈도우 수 (detection=입력 window, prediction=horizon)
    label_span = win_samples if is_detection else horizon_samples
    n_future_win = max(1, label_span // icp_win)
    # sustained 요건(연속 bucket 수)은 라벨 구간 bucket 수를 넘을 수 없다. 넘으면 영구
    # 음성이 되므로 clamp — 특히 짧은 window(예: 10s detection = bucket 1개)에서 필수.
    min_consecutive = min(min_consecutive, n_future_win)

    total_needed = win_samples if is_detection else win_samples + horizon_samples
    samples: list[ICHSample] = []

    for case in cases:
        signals = case["signals"]
        icp = signals["icp"]

        # 모든 input signal의 공통 길이
        min_len = min(len(signals[s]) for s in input_signals if s in signals)
        min_len = min(min_len, len(icp))

        if min_len < total_needed:
            continue

        # sparse sampling: case 내 class 별 마지막 채택 start (간격 강제용)
        last_accepted: dict[int, int | None] = {0: None, 1: None}

        for start in range(0, min_len - total_needed + 1, stride_samples):
            input_dict = {}
            for stype in input_signals:
                if stype in signals:
                    input_dict[stype] = signals[stype][start: start + win_samples]

            if not input_dict:
                continue

            # Step 1: gap-policy window drop (multi-channel valid_ratio)
            valid_ratio = compute_valid_ratio(list(input_dict.values()))
            if valid_ratio < valid_ratio_threshold:
                if gap_stats is not None:
                    gap_stats.add_drop()
                continue

            if is_detection:
                # ── detection(aICP식): 동시(same-window) ICP 로 라벨. 현재-ICH 제외 없음. ──
                #   입력 ABP+ECG 만으로 "이 구간에 두개내 고혈압이 있는가"를 탐지.
                #   ICP 는 입력에 없고 라벨로만 쓰여 circularity 차단.
                label_icp = icp[start: start + win_samples]
            else:
                # ── 현재-ICH 제외 (HPI 표준): 예측 시점(T=입력 끝)에 이미 ICP>threshold 면 drop. ──
                #   진행 중 두개내압 상승의 연속(trivial positive)을 배제, 신규 발생만 예측.
                #   입력 끝 30초 baseline ICP 평균이 threshold 이상(또는 확인 불가)이면 버린다.
                base_start = max(start, start + win_samples - 3 * icp_win)
                baseline_icp = icp[base_start: start + win_samples]
                base_valid = baseline_icp[~np.isnan(baseline_icp)]
                if base_valid.size < icp_win or float(np.mean(base_valid)) >= icp_threshold:
                    if gap_stats is not None:
                        gap_stats.add_drop()
                    continue
                label_icp = icp[start + win_samples: start + win_samples + horizon_samples]

            # 10s sub-window NaN-free 평균 (prediction=미래 horizon / detection=동시 window)
            future_icps: list[float] = []
            for j in range(0, len(label_icp) - icp_win + 1, icp_win):
                w = label_icp[j: j + icp_win]
                if not np.isnan(w).any():
                    future_icps.append(float(np.mean(w)))

            if not future_icps:
                if gap_stats is not None:
                    gap_stats.add_drop()
                continue

            # ── 라벨 결정 ──
            #   positive: 미래 horizon 내 ≥1분 지속 ICP>threshold (신규 두개내압 상승;
            #     기본 20 = Czosnyka 2004). 현재-ICH 제외는 두 모드 공통(위에서 이미 적용).
            #   unbiased(기본): dense sliding, positive 외 나머지(15~20 경계 포함) 전부
            #     negative. sparse/clean-neg 없음 → 실제 forward 평가(현실적).
            #   biased(선행 crisis/non-crisis epoch 방식, supplementary): clean-neg(미래 valid
            #     ICP 전부 <15, Czosnyka 2004)만 negative + 경계(15~20) 제외 + sparse → 과대평가.
            is_pos = _has_sustained_ich(future_icps, icp_threshold, min_consecutive)
            if sampling_mode == "unbiased":
                label = 1 if is_pos else 0
            else:  # biased
                is_clean_neg = (
                    len(future_icps) >= int(0.8 * n_future_win)
                    and all(m < neg_icp_threshold for m in future_icps)
                )
                if is_pos:
                    label = 1
                elif is_clean_neg:
                    label = 0
                else:
                    if gap_stats is not None:
                        gap_stats.add_drop()
                    continue
                # sparse sampling: 같은 case·class 최소 간격(20분) — biased 만.
                prev = last_accepted[label]
                if prev is not None and (start - prev) < min_gap_samples:
                    continue
                last_accepted[label] = start

            # Step 2: gap mask + 즉시 dtype 캐스팅 (OOM 회피)
            filled_dict, gap_mask_dict = apply_gap_mask_multichannel(
                input_dict, output_dtype=sample_dtype,
            )
            if gap_stats is not None:
                n_total_s = sum(arr.size for arr in filled_dict.values())
                n_gap_s = sum(int(m.sum()) for m in gap_mask_dict.values())
                gap_stats.add_window(n_total_s, n_gap_s)

            samples.append(
                ICHSample(
                    input_signals=filled_dict,
                    input_gap_masks=gap_mask_dict,
                    label=label,
                    label_value=max(future_icps),
                    case_id=case["case_id"],
                    patient_id=case["patient_id"],
                    win_start_sec=start / TARGET_SR,
                    horizon_sec=(0.0 if is_detection else horizon_sec),
                )
            )

    return samples


# ── 저장 ─────────────────────────────────────────────────────


def _consume_to_tensors_ich(
    samples: list[ICHSample],
    input_signals: list[str],
    signal_dtype: torch.dtype,
) -> dict:
    """ICH samples → packed dict (in-place, samples 비워짐).

    torch.stack 의 2× peak 회피 — 출력 텐서를 미리 할당 + numpy ref pop.
    """
    if not samples:
        return {"signals": {}, "gap_masks": {},
                "labels": torch.tensor([], dtype=torch.long),
                "label_values": torch.tensor([], dtype=torch.float32),
                "win_start_sec": torch.tensor([], dtype=torch.float32),
                "case_ids": [], "subject_ids": []}

    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    label_values = torch.tensor(
        [s.label_value for s in samples], dtype=torch.float32
    )
    # win_start_sec 보존: (subject_id, win_start_sec, label_value, y_score) 로 환자 내
    # 시간축 ICP 추적(정적 환자분류 아닌 동적 IH 탐지)을 사후 분석·plot 하기 위함.
    win_start_sec = torch.tensor(
        [s.win_start_sec for s in samples], dtype=torch.float32
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
        "win_start_sec": win_start_sec,
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
    signal_dtype: torch.dtype = torch.float16,
    fold_idx: int | None = None,
    n_folds: int = 1,
    chunk_idx: int | None = None,
    task_mode: str = "prediction",
) -> Path:
    """Single split chunk packed dict → 별도 .pt (OOM 회피 Stage 5)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_samples = int(split_dict.get("labels", torch.tensor([])).numel())

    save_dict = {
        "split": split_name,
        "data": split_dict,
        "metadata": {
            "task": f"intracranial_hypertension_{task_mode}",
            "source": "MIMIC-III Waveform",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "icp_threshold": ICP_THRESHOLD,
            "sustained_sec": SUSTAINED_SEC,
            "split": split_name,
            "n_samples": n_samples,
            "fold_idx": fold_idx,
            "n_folds": n_folds,
            "chunk_idx": chunk_idx,
            "signal_dtype": str(signal_dtype).replace("torch.", ""),
            "valid_ratio_threshold": DEFAULT_VALID_RATIO_THRESHOLD,
            "pos_mode": pos_mode,
            "pos_stride_sec": pos_stride_sec,
            "neg_stride_sec": neg_stride_sec,
        },
    }

    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    win_int = int(window_sec)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    chunk_suffix = f"_chunk{chunk_idx}" if chunk_idx is not None else ""
    # task_mode 를 파일명에 넣어 detection 산출물이 prediction 산출물과 섞이지 않게 한다
    # (같은 OUT_DIR 공유 시 이름 충돌 방지). 예:
    #   intracranial_hypertension_detection_abp_ecg_w10s_h0min_fold0_train_chunk0.pt
    #   intracranial_hypertension_prediction_abp_icp_ecg_w1200s_h30min_fold0_train.pt
    filename = (
        f"intracranial_hypertension_{task_mode}_{mode_str}_w{win_int}s_h{horizon_min}min"
        f"{fold_suffix}_{split_name}{chunk_suffix}.pt"
    )
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ── 통계 출력 ────────────────────────────────────────────────


def print_stats(name: str, samples: list[ICHSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return

    n_total = len(samples)
    n_ich = sum(1 for s in samples if s.label == 1)
    n_normal = n_total - n_ich
    icps = [s.label_value for s in samples]

    print(f"  {name}: {n_total} samples")
    print(f"    Normal: {n_normal} ({n_normal / n_total * 100:.1f}%)")
    print(f"    ICH:    {n_ich} ({n_ich / n_total * 100:.1f}%)")
    print(
        f"    Future ICP max: [{min(icps):.1f}, {max(icps):.1f}] mmHg, "
        f"mean={np.mean(icps):.1f} +/- {np.std(icps):.1f}"
    )


# ── Sweep ────────────────────────────────────────────────────


def _patient_level_ich_labels(
    cases: list[dict],
    icp_threshold: float = ICP_THRESHOLD,
) -> dict[str, list[int]]:
    """환자별 case 레벨 ICH label (stratification 용).

    각 case 의 ICP 시계열에 threshold 초과 샘플 존재 여부를 binary 로 기록.
    """
    patient_to_labels: dict[str, list[int]] = {}
    for case in cases:
        pid = case["patient_id"]
        icp = case.get("signals", {}).get("icp")
        if icp is None or len(icp) == 0:
            patient_to_labels.setdefault(pid, []).append(0)
            continue
        valid = icp[~np.isnan(icp)] if hasattr(icp, "__len__") else icp
        has_high = bool(np.any(valid > icp_threshold)) if len(valid) else False
        patient_to_labels.setdefault(pid, []).append(1 if has_high else 0)
    return patient_to_labels


def _patient_level_onset_labels(
    cases: list[dict],
    icp_threshold: float = ICP_THRESHOLD,
    event_consec: int = EVENT_CONSEC,
) -> dict[str, list[int]]:
    """환자별 case 레벨 crisis-onset 유무 (anchored stratification 용).

    각 세그먼트 case 의 1분 median 블록에 확정 onset(5×1min>threshold)이 하나라도
    있으면 1. anchored 양성 정의와 일치하는 stratification.
    """
    patient_to_labels: dict[str, list[int]] = {}
    for case in cases:
        pid = case["patient_id"]
        icp = case.get("signals", {}).get("icp")
        has_onset = 0
        if icp is not None and len(icp):
            blocks = _icp_median_blocks(icp)
            if _find_onsets(blocks, icp_threshold, event_consec):
                has_onset = 1
        patient_to_labels.setdefault(pid, []).append(has_onset)
    return patient_to_labels


def prepare_ich_sweep(
    waveform_dir: str,
    input_signals: list[str],
    window_secs: list[float],
    horizon_mins: list[float],
    stride_sec: float = 30.0,
    n_folds: int = 5,
    out_dir: str = "outputs/downstream/intracranial_hypertension",
    split_mode: str = "kfold",
    signal_dtype: torch.dtype = torch.float16,
    seed: int = 42,
    required_signals: list[str] | None = None,
    icp_threshold: float = ICP_THRESHOLD,
    min_sample_gap_sec: float = 1200.0,
    sampling_mode: str = "unbiased",
    sustained_sec: float = SUSTAINED_SEC,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    task_mode: str = "prediction",
    scan_dir: bool = True,
    neg_stride_sec: float = 600.0,
    event_consec: int = EVENT_CONSEC,
    pos_mode: str = "dense",
    pos_stride_sec: float = 60.0,
) -> list[Path]:
    """(window, horizon) 조합 × stratified K-fold CV 데이터셋 생성.

    split_mode:
        "kfold"        : Stratified patient-level K-fold CV (default, n_folds=5).
        "loso_export"  : 모든 샘플을 train에 저장(test 비움) — subject_ids로
                         외부에서 LeaveOneSubjectOut CV 수행용.

    sampling_mode == "anchored": Güiza onset-anchor 양성 + crisis-free 균일 stride
        음성. 정렬(aligned) 세그먼트 로더(load_aligned_cases) 사용, ABP·ICP 시간축
        일치 보장. (unbiased/biased 는 구 10s-평균 dense 모드.)
    """
    is_anchored = sampling_mode == "anchored"
    # detection(aICP식 동시 탐지): horizon 무의미 → 단일 0 으로 고정(파일명 h0min).
    if task_mode == "detection":
        horizon_mins = [0.0]

    mode_str = " + ".join(s.upper() for s in input_signals)
    print(f"\n{'=' * 60}")
    print(f"  Intracranial Hypertension {task_mode.capitalize()} - MIMIC-III")
    print(f"  Waveform: {waveform_dir}")
    print(f"  Input:    {mode_str}  (ICP=label only)")
    print(f"  Windows:  {window_secs}")
    print(f"  Horizons: {horizon_mins}" + ("  (detection: 동시 라벨, horizon 무시)" if task_mode == "detection" else ""))
    print(f"  ICP threshold: {icp_threshold} mmHg, sustained: {sustained_sec}s, "
          f"valid_ratio: {valid_ratio_threshold}")
    print(f"{'=' * 60}")

    # 1. 데이터 로딩
    print("\n[1/3] Loading ICP waveform records...")
    if is_anchored:
        # 정렬 세그먼트 로더 (ABP·ICP 시간축 일치). min_seg = window + 최소 horizon.
        min_seg_sec = min(window_secs) + min(h * 60.0 for h in horizon_mins)
        cases = load_aligned_cases(
            Path(waveform_dir), input_signals,
            min_seg_sec=min_seg_sec, scan_dir=scan_dir,
        )
    else:
        # default: input_signals 전체를 강제 (모든 sample 이 동일 채널)
        effective_required = (
            required_signals if required_signals is not None else input_signals
        )
        cases = load_patient_signals(
            Path(waveform_dir), required_signals=effective_required,
        )
    if not cases:
        print("ERROR: No ICP records found.", file=sys.stderr)
        sys.exit(1)

    # 2. Patient-level split
    patient_ids = sorted({c["patient_id"] for c in cases})
    if split_mode == "loso_export":
        print(f"\n[2/3] LOSO export mode: all {len(patient_ids)} patients in train "
              "(test empty; subject_ids for external LOSO CV).")
        splits = [(set(patient_ids), set(), set())]
    else:
        print(f"\n[2/3] Stratified {n_folds}-fold patient-level CV...")
        if is_anchored:
            patient_to_labels = _patient_level_onset_labels(
                cases, icp_threshold=icp_threshold, event_consec=event_consec,
            )
        else:
            patient_to_labels = _patient_level_ich_labels(
                cases, icp_threshold=icp_threshold,
            )
        pos_pts = sum(1 for pid in patient_ids if any(patient_to_labels.get(pid, [])))
        crit = ("any crisis onset" if is_anchored else f"any ICP>{icp_threshold}")
        print(
            f"  Patient-level positive ({crit}): "
            f"{pos_pts}/{len(patient_ids)} "
            f"({100.0 * pos_pts / max(1, len(patient_ids)):.1f}%)"
        )
        splits = stratified_kfold_patient_splits(
            patient_ids, patient_to_labels, n_folds=n_folds, seed=seed,
        )
        print(summarize_splits(splits, patient_to_labels))

    # 3. 조합 × fold 윈도우 추출 + 저장
    combos = [(w, h) for w in window_secs for h in horizon_mins]
    total_runs = len(combos) * len(splits)
    print(f"\n[3/3] Generating {len(combos)} combo × {len(splits)} fold = {total_runs} datasets...")

    saved_paths: list[Path] = []
    run_idx = 0
    # signal_dtype (torch) → str for apply_gap_mask_multichannel
    sample_dtype_str = str(signal_dtype).replace("torch.", "")
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
            # OOM 회피 (Stage 5, 2026-07 재강화) — byte-예산 chunked save.
            #   기존 고정 CASES_PER_CHUNK=200 은 window 크기에 무관해, window=1200s
            #   처럼 sample 1개가 window_sec×100×2byte×채널로 커지는 경우 200 case
            #   누적 buffer 가 수백 GB → OOM(Killed). 이제 case 를 하나씩 흘리며
            #   누적 sample byte 가 예산을 넘으면 flush → window 크기와 무관하게
            #   buffer peak 를 고정한다. (case 단위 호출은 extract 가 case 별 독립
            #   처리라 결과·gap_stats byte-identical.) chunk 수가 늘어도 run.py 의
            #   load_prepared_split_chunked 가 glob 로 모으므로 안전.
            cur_fold_idx = fold_idx if len(splits) > 1 else None
            CHUNK_BYTES_BUDGET = 4 * 1024 ** 3  # 누적 4GB 초과 시 flush

            for split_name, split_cases in (
                ("train", train_cases),
                ("val", val_cases),
                ("test", test_cases),
            ):
                if not split_cases:
                    continue
                gap_stats = GapStats()
                chunk_idx = 0
                total = 0
                buffer: list[ICHSample] = []
                buf_bytes = 0

                def _flush(buf: list[ICHSample], cidx: int, split_name: str) -> int:
                    packed = _consume_to_tensors_ich(buf, input_signals, signal_dtype)
                    buf.clear()
                    gc.collect()
                    n_in_chunk = int(packed.get("labels", torch.tensor([])).numel())
                    save_path = save_split_dataset(
                        packed, split_name, input_signals,
                        horizon_sec, window_sec, out_dir,
                        signal_dtype=signal_dtype,
                        fold_idx=cur_fold_idx,
                        n_folds=len(splits),
                        chunk_idx=cidx,
                        task_mode=task_mode,
                    )
                    saved_paths.append(save_path)
                    del packed
                    gc.collect()
                    return n_in_chunk

                for case in split_cases:
                    if is_anchored:
                        samples = extract_anchored_samples(
                            [case], input_signals,
                            window_sec=window_sec, horizon_sec=horizon_sec,
                            neg_stride_sec=neg_stride_sec,
                            icp_threshold=icp_threshold,
                            event_consec=event_consec,
                            valid_ratio_threshold=valid_ratio_threshold,
                            gap_stats=gap_stats,
                            sample_dtype=sample_dtype_str,
                            pos_mode=pos_mode,
                            pos_stride_sec=pos_stride_sec,
                        )
                    else:
                        samples = extract_forecast_samples(
                            [case], input_signals, window_sec, stride_sec, horizon_sec,
                            gap_stats=gap_stats,
                            sample_dtype=sample_dtype_str,
                            icp_threshold=icp_threshold,
                            min_sample_gap_sec=min_sample_gap_sec,
                            sampling_mode=sampling_mode,
                            sustained_sec=sustained_sec,
                            valid_ratio_threshold=valid_ratio_threshold,
                            task_mode=task_mode,
                        )
                    for s in samples:
                        buffer.append(s)
                        buf_bytes += sum(a.nbytes for a in s.input_signals.values())
                        buf_bytes += sum(m.nbytes for m in s.input_gap_masks.values())
                    samples.clear()
                    del samples
                    if buf_bytes >= CHUNK_BYTES_BUDGET and buffer:
                        total += _flush(buffer, chunk_idx, split_name)
                        chunk_idx += 1
                        buf_bytes = 0
                        gc.collect()

                if buffer:  # 잔여 flush
                    total += _flush(buffer, chunk_idx, split_name)
                    chunk_idx += 1
                    buf_bytes = 0

                print(f"    {split_name.capitalize()}: {chunk_idx} chunk(s), {total} samples")
                print(gap_stats.summary())

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(saved_paths)} files saved across {len(combos)} combo × {len(splits)} fold")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")
    return saved_paths


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intracranial Hypertension Detection - Data Preparation (MIMIC-III)",
    )
    parser.add_argument(
        "--waveform-dir", type=str,
        default="datasets/raw/mimic3-waveform-ich",
        help="MIMIC-III waveform directory (ICP 포함 레코드)",
    )
    parser.add_argument(
        "--input-signals", nargs="+",
        default=["abp", "icp"],
        choices=["icp", "ecg", "abp", "ppg", "co2"],
        help="Input signal types. anchored 기본: ABP + ICP (2ch, prediction 이라 "
             "현재 ICP 를 입력에 포함). label 은 항상 ICP.",
    )
    parser.add_argument(
        "--horizon-mins", nargs="+", type=float, default=[5.0, 10.0, 15.0],
    )
    parser.add_argument(
        "--window-secs", nargs="+", type=float, default=[1200.0],
    )
    parser.add_argument("--stride-sec", type=float, default=30.0,
                        help="구 dense 모드 sliding stride(초). anchored 는 미사용.")
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Stratified patient-level K-fold CV (default 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for fold shuffling.",
    )
    parser.add_argument(
        "--split-mode", type=str, default="kfold",
        choices=["kfold", "loso_export"],
        help="'kfold': stratified patient-level K-fold CV. "
             "'loso_export': all in train, external LOSO CV via subject_ids.",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="outputs/downstream/intracranial_hypertension",
    )
    parser.add_argument(
        "--signal-dtype", type=str, default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for waveform tensors. fp16 halves disk/RAM peak; "
             "run.py auto-casts back to fp32 at load time.",
    )
    parser.add_argument(
        "--required-signals", nargs="+", default=None,
        choices=["icp", "ecg", "abp", "ppg", "co2"],
        help="모든 record 가 반드시 가져야 하는 channel set (intersection cohort). "
             "지정하지 않으면 --input-signals 가 그대로 강제됨. "
             "ICP 는 라벨 산출용으로 항상 자동 포함.",
    )
    parser.add_argument(
        "--icp-threshold", type=float, default=ICP_THRESHOLD,
        help=f"intracranial hypertension 임계 ICP(mmHg). default {ICP_THRESHOLD:.0f} "
             "(Czosnyka 2004 IH 정의). BTF 2016 치료 임계는 22.",
    )
    parser.add_argument(
        "--min-sample-gap-sec", type=float, default=1200.0,
        help="biased 모드 sparse 표본 간격(초). 기본 1200(20분).",
    )
    parser.add_argument(
        "--sustained-sec", type=float, default=SUSTAINED_SEC,
        help=f"positive 판정 지속 요건(초): 미래 horizon 내 ICP>임계가 이 시간 이상 "
             f"연속 지속되어야 ICH. default {SUSTAINED_SEC:.0f}(1분). 낮추면(예: 30) "
             "짧은 에피소드도 양성으로 포함돼 희소 양성 표본 확보(재파싱 필요).",
    )
    parser.add_argument(
        "--valid-ratio", type=float, default=DEFAULT_VALID_RATIO_THRESHOLD,
        help=f"input window 유효 샘플 비율 임계. default {DEFAULT_VALID_RATIO_THRESHOLD}. "
             "낮추면(예: 0.7) gappy window 를 더 포함해 표본 확보(재파싱 필요).",
    )
    parser.add_argument(
        "--sampling-mode", type=str, default="anchored",
        choices=["anchored", "unbiased", "biased"],
        help="anchored(기본: Güiza onset-anchor 양성 + crisis-free 균일 stride 음성, "
             "정렬 세그먼트) | unbiased/biased(구 10s-평균 dense 모드).",
    )
    parser.add_argument(
        "--neg-stride-sec", type=float, default=600.0,
        help="음성 균일 stride(초). prevalence 다이얼(dense 양성 기준): 600(10min)"
             "≈IOH 스케일 3.9/7/9.7%, 300≈2/3.6/5.1%, 1800≈10/18/24%.",
    )
    parser.add_argument(
        "--pos-mode", type=str, default="dense", choices=["dense", "anchored"],
        help="양성 sampling. dense(기본): 예측구간 안 onset=양성 dense sliding → "
             "horizon 따라 prevalence↑(IOH 정합). anchored(레거시): onset 당 1개 → flat.",
    )
    parser.add_argument(
        "--pos-stride-sec", type=float, default=60.0,
        help="dense 양성 sliding stride(초). 기본 60(=1분 median 블록).",
    )
    parser.add_argument(
        "--event-consec", type=int, default=EVENT_CONSEC,
        help=f"crisis 확정 연속 1분블록 수. 기본 {EVENT_CONSEC}(=5min, Güiza).",
    )
    parser.add_argument(
        "--scan-dir", action=argparse.BooleanOptionalAction, default=True,
        help="anchored 로더가 waveform-dir 최상위를 scandir 로 마스터 직접 열거"
             "(flat numeric 대응, 기본 True). --no-scan-dir 이면 rglob(matched).",
    )
    parser.add_argument(
        "--task-mode", type=str, default="prediction",
        choices=["prediction", "detection"],
        help="prediction(미래 horizon IH 예측, 현재-ICH 제외) | "
             "detection(동시 window IH 탐지, aICP식: 입력 ABP+ECG, ICP=라벨전용, "
             "현재-ICH 제외 없음, horizon 무시).",
    )
    args = parser.parse_args()

    # detection 은 ICP 를 라벨 전용으로만 쓴다(circularity 차단). 입력에 icp 가 들어오면
    # trivial + circular 가 되므로 거부한다.
    if args.task_mode == "detection" and "icp" in args.input_signals:
        parser.error(
            "detection 모드에서는 ICP 를 입력(--input-signals)에 넣을 수 없습니다 "
            "(ICP=라벨 전용, circularity 차단). --input-signals 에서 icp 를 빼세요.",
        )

    dtype_map = {"float16": torch.float16, "float32": torch.float32}

    prepare_ich_sweep(
        waveform_dir=args.waveform_dir,
        input_signals=args.input_signals,
        window_secs=args.window_secs,
        horizon_mins=args.horizon_mins,
        stride_sec=args.stride_sec,
        n_folds=args.n_folds,
        seed=args.seed,
        out_dir=args.out_dir,
        split_mode=args.split_mode,
        signal_dtype=dtype_map[args.signal_dtype],
        required_signals=args.required_signals,
        icp_threshold=args.icp_threshold,
        min_sample_gap_sec=args.min_sample_gap_sec,
        sampling_mode=args.sampling_mode,
        sustained_sec=args.sustained_sec,
        valid_ratio_threshold=args.valid_ratio,
        task_mode=args.task_mode,
        scan_dir=args.scan_dir,
        neg_stride_sec=args.neg_stride_sec,
        event_consec=args.event_consec,
        pos_mode=args.pos_mode,
        pos_stride_sec=args.pos_stride_sec,
    )


if __name__ == "__main__":
    main()
