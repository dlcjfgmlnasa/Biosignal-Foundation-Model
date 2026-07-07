# -*- coding:utf-8 -*-
"""Intraoperative Desaturation Prediction — 데이터 준비 (SNUH ventilation_eventcases, self-contained).

미래 horizon 구간 SpO2 < 95% "at any point" (아티팩트 게이트 후) 예측을 위한
(input_window, future_label) 쌍 생성.

Task 성격: Acute Event — respiratory desaturation *prediction* (detection 아님).
  - 입력 = desat *이전* 정상 파형 (CO2/AWP/RESP_Flow/PPG — ECG 제외). PPG 포함=competitive(문헌 관행)
  - 라벨 = 이후 horizon 구간에 (게이트 통과한) SpO2<95% 가 **한 번이라도** 발생하면 1

라벨 정의 = "any point" + 아티팩트 게이트 — 소아논문(PLOS One 2023) 프로토콜 재현:
  any-point 를 쓰려면 반드시 품질 게이트가 세트여야 함 (motion/dropout label noise 차단).
  게이트 = ① SpO2∈[50,100] ② |pulseHR-ecgHR|/ecgHR ≤ 0.20 ③ PI ≥ 0.3.
  threshold 95% = SNUH 소아(PLOS One 2023) 관행. 92=Prescience/WHO, 90=conservative.
  dwell(≥10/30s)은 `--sustained-sec` sensitivity variant.

데이터 소스: SNUH 2024 마취 이상사례(ventilation) 코호트 — raw `.vital` 188개.
  - hypoxemia(#6, VitalDB) 와 라벨 규칙 동일, 소스만 SNUH.
  - SNUH 고유: RESP_Flow(FLOW_WAV) 91% 존재 → 입력에 flow 추가 가능.
  - category `desaturation`(Claude 텍스트) 는 사용하지 않음 — 라벨은 전적으로 SpO2 파형에서 재생성.

트랙 커버리지(188 실측): CO2_WAV 99% · PLETH 99% · ECG_II_WAV 96% · AWP_WAV 95% ·
  FLOW_WAV 91% · ABP 90% · PLETH_SAT_O2 99%. Intellivue 파형 SR 125~250Hz → 100Hz 다운샘플.

사용법:
    python -m downstream.acute_event.desaturation.prepare_data \\
        --vital-dir C:/Projects/ventilation_eventcases/data/vital_files \\
        --meta-xlsx C:/Projects/ventilation_eventcases/ventilation_cases.xlsx \\
        --input-signals co2 awp resp_flow ppg \\
        --required-signals co2 awp \\
        --window-secs 300 --horizon-mins 1 3 5 --n-folds 5 \\
        --out-dir outputs/downstream/desaturation

외부검증(external-test) 모드: --external-test → fold 분할 없이 188 전부 test-only 저장.
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.parser._common import resample_to_target
from downstream._gap_mask import (
    DEFAULT_VALID_RATIO_THRESHOLD,
    GapStats,
    apply_gap_mask_multichannel,
    compute_valid_ratio,
)
from downstream._kfold_utils import stratified_kfold_patient_splits, summarize_splits

TARGET_SR: float = 100.0  # pretraining 과 동일 (100Hz)

# SNUH 트랙 → CARMEN signal_type. 앞이 우선(Intellivue 고해상 파형), 뒤는 폴백(Datex 25Hz).
INPUT_TRACK_CANDIDATES: dict[str, list[str]] = {
    "ecg":       ["Intellivue/ECG_II_WAV"],
    "ppg":       ["Intellivue/PLETH", "ROOT/PLETH"],
    "co2":       ["Intellivue/CO2_WAV", "Datex-Ohmeda/CO2"],
    "awp":       ["Intellivue/AWP_WAV", "Datex-Ohmeda/AWP"],
    "resp_flow": ["Intellivue/FLOW_WAV", "Datex-Ohmeda/Flow"],
    "abp":       ["Intellivue/ABP"],
}
# 라벨 전용 (입력 아님). hypoxemia SPO2_TRACKS 와 동일 항목 포함.
SPO2_TRACKS: list[str] = ["Intellivue/PLETH_SAT_O2", "ROOT/SPO2"]
# 아티팩트 게이트용 (1Hz numeric). 소아 논문(PLOS One 2023) 프로토콜 재현:
#   pulse-ox HR vs ECG HR 20% 초과 괴리 → motion/dropout 아티팩트로 그 SpO2 샘플 제외.
PULSE_HR_TRACKS: list[str] = ["Intellivue/PLETH_HR", "ROOT/BPM"]
ECG_HR_TRACKS: list[str] = ["Intellivue/ECG_HR", "Intellivue/ABP_HR"]
PI_TRACKS: list[str] = ["ROOT/PI", "Intellivue/PLETH_PERF_REL"]  # 저관류 dropout 제외


# ---- 데이터 구조 ----


@dataclass
class ForecastSample:
    """미래 desaturation 예측 샘플."""

    input_signals: dict[str, np.ndarray]    # NaN → 0 fill (float16)
    input_gap_masks: dict[str, np.ndarray]  # bool, True=원본이 NaN (gap)
    label: int         # 0=normal, 1=desaturation in future
    label_value: float  # future SpO2 최솟값 (%)
    case_id: str
    win_start_sec: float
    horizon_sec: float


# ---- raw .vital 로더 (self-contained, gap 보존 100Hz 리샘플) ----


def _to_100hz_gap_preserving(native: np.ndarray, native_sr: float) -> np.ndarray:
    """native(원본 SR, NaN=gap) → 100Hz. 짧은 결측은 보간, 원래 gap 은 NaN 유지.

    scope prepare_data 의 _process_band 와 동일 접근.
    """
    native = native.astype(np.float64, copy=False)
    if native_sr == TARGET_SR:
        return native
    nan_mask = ~np.isfinite(native)
    if nan_mask.all():
        n_out = int(round(len(native) * TARGET_SR / native_sr))
        return np.full(n_out, np.nan)
    filled = native.copy()
    idx = np.arange(len(filled))
    good = ~nan_mask
    # gap 을 선형보간으로 임시 채워 anti-alias 리샘플이 깨지지 않게
    filled[nan_mask] = np.interp(idx[nan_mask], idx[good], filled[good])
    out = resample_to_target(filled, orig_sr=native_sr, target_sr=TARGET_SR)
    # 원래 gap 구간을 다시 NaN 으로 마스킹 (nearest 매핑)
    n_out = len(out)
    src_idx = np.clip(
        np.round(np.arange(n_out) * native_sr / TARGET_SR).astype(int),
        0, len(native) - 1,
    )
    out[nan_mask[src_idx]] = np.nan
    return out


def _pick_track(available: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in available:
            return c
    return None


def _load_1hz(vf, available: set[str], candidates: list[str], n: int) -> np.ndarray | None:
    """후보 중 첫 존재 트랙을 1Hz numeric 으로 로드, 길이 n 정렬 (없으면 None)."""
    trk = _pick_track(available, candidates)
    if trk is None:
        return None
    a = vf.to_numpy([trk], interval=1.0).flatten().astype(np.float64)
    if len(a) >= n:
        return a[:n]
    out = np.full(n, np.nan)
    out[:len(a)] = a
    return out


def gate_spo2_artifacts(
    spo2: np.ndarray,
    pulse_hr: np.ndarray | None,
    ecg_hr: np.ndarray | None,
    pi: np.ndarray | None,
    hr_tol: float = 0.20,
    pi_min: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """SpO2 아티팩트 게이트 (소아 논문 PLOS One 2023 프로토콜 재현).

    NaN 처리(=라벨 계산에서 제외)되는 조건:
      ① range: SpO2 ∉ [50, 100]
      ② HR concordance: |pulse_hr - ecg_hr| / ecg_hr > hr_tol (motion/dropout)
      ③ perfusion: pi < pi_min (저관류 → 부정확)
    HR/PI 트랙이 없으면 해당 게이트는 skip (판정 불가 샘플은 보존).
    Returns (gated_spo2, stats).
    """
    s = spo2.astype(np.float64).copy()
    n = len(s)
    invalid = ~np.isfinite(s) | (s < 50.0) | (s > 100.0)
    n_range = int(invalid.sum())

    n_hr = 0
    if pulse_hr is not None and ecg_hr is not None:
        m = min(n, len(pulse_hr), len(ecg_hr))
        ph, eh = pulse_hr[:m], ecg_hr[:m]
        ok = np.isfinite(ph) & np.isfinite(eh) & (eh > 20.0)  # eh 유효할 때만 판정
        discord = np.zeros(n, dtype=bool)
        discord[:m] = ok & (np.abs(ph - eh) / eh > hr_tol)
        n_hr = int((discord & ~invalid).sum())
        invalid |= discord

    n_pi = 0
    if pi is not None:
        m = min(n, len(pi))
        low = np.zeros(n, dtype=bool)
        p = pi[:m]
        low[:m] = np.isfinite(p) & (p < pi_min)
        n_pi = int((low & ~invalid).sum())
        invalid |= low

    s[invalid] = np.nan
    stats = {"n_total": n, "n_range": n_range, "n_hr_discord": n_hr,
             "n_low_pi": n_pi, "n_gated": int(invalid.sum())}
    return s, stats


def load_case_from_vital(
    vital_path: Path,
    patient_id: str,
    input_signals: list[str],
    required_signals: list[str],
    artifact_gate: bool = True,
    hr_tol: float = 0.20,
    pi_min: float = 0.3,
) -> dict | None:
    """단일 .vital → {case_id, patient_id, signals@100Hz(NaN=gap), spo2@1Hz}.

    required_signals 중 하나라도 없으면 None (case drop).
    """
    import vitaldb

    case_id = vital_path.stem
    try:
        available = set(vitaldb.vital_trks(str(vital_path)))
    except Exception as e:  # noqa: BLE001
        print(f"    [skip] {case_id}: vital_trks 실패 ({e})", file=sys.stderr)
        return None

    # 입력 트랙 해석
    chosen: dict[str, str] = {}
    for stype in input_signals:
        trk = _pick_track(available, INPUT_TRACK_CANDIDATES.get(stype, []))
        if trk is not None:
            chosen[stype] = trk
    # required 검증
    missing = [s for s in required_signals if s not in chosen]
    if missing:
        print(f"    [skip] {case_id}: required 없음 {missing}", file=sys.stderr)
        return None

    # SpO2 라벨 트랙
    spo2_trk = _pick_track(available, SPO2_TRACKS)
    if spo2_trk is None:
        print(f"    [skip] {case_id}: SpO2 트랙 없음", file=sys.stderr)
        return None

    # 게이트 트랙 해석 (있으면 로드)
    gate_trks = []
    if artifact_gate:
        for cands in (PULSE_HR_TRACKS, ECG_HR_TRACKS, PI_TRACKS):
            t = _pick_track(available, cands)
            if t is not None:
                gate_trks.append(t)

    # 로드 (필요한 트랙만)
    load_names = list(chosen.values()) + [spo2_trk] + gate_trks
    try:
        vf = vitaldb.VitalFile(str(vital_path), track_names=load_names)
    except Exception as e:  # noqa: BLE001
        print(f"    [skip] {case_id}: VitalFile 로드 실패 ({e})", file=sys.stderr)
        return None

    signals: dict[str, np.ndarray] = {}
    for stype, trk in chosen.items():
        native_sr = float(getattr(vf.trks.get(trk), "srate", 0.0) or 0.0)
        if native_sr <= 0:
            continue
        native = vf.to_numpy([trk], interval=1.0 / native_sr).flatten()
        signals[stype] = _to_100hz_gap_preserving(native, native_sr)

    if not all(s in signals for s in required_signals):
        return None

    # 공통 길이로 트림
    n_common = min(len(v) for v in signals.values())
    signals = {k: v[:n_common] for k, v in signals.items()}

    # SpO2 → 1Hz numeric
    spo2 = vf.to_numpy([spo2_trk], interval=1.0).flatten().astype(np.float64)
    n_spo2 = len(spo2)

    gate_stats = None
    if artifact_gate:
        pulse_hr = _load_1hz(vf, available, PULSE_HR_TRACKS, n_spo2)
        ecg_hr = _load_1hz(vf, available, ECG_HR_TRACKS, n_spo2)
        pi = _load_1hz(vf, available, PI_TRACKS, n_spo2)
        spo2, gate_stats = gate_spo2_artifacts(
            spo2, pulse_hr, ecg_hr, pi, hr_tol=hr_tol, pi_min=pi_min)

    return {
        "case_id": case_id,
        "patient_id": patient_id,
        "signals": signals,
        "spo2": spo2,          # 게이트 적용된 SpO2 (아티팩트 → NaN)
        "gate_stats": gate_stats,
    }


# ---- 라벨 + window 추출 (hypoxemia 와 동일 규칙) ----


def _has_sustained_desat(future_spo2: list[float], threshold: float, min_consecutive: int) -> bool:
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
    input_signals: list[str],
    window_sec: float,
    stride_sec: float,
    horizon_sec: float,
    spo2_threshold: float = 90.0,
    sustained_sec: float = 60.0,
    valid_ratio_threshold: float = DEFAULT_VALID_RATIO_THRESHOLD,
    exclude_already_low: bool = True,
    task_mode: str = "prediction",
    gap_stats: GapStats | None = None,
    sample_dtype: str = "float16",
) -> list[ForecastSample]:
    """SpO2 trend + waveform window 정렬해서 (input, label) 쌍 추출.

    wave(100Hz) index/100 == spo2(1Hz) index(초). 둘 다 같은 .vital dtstart 기준이라 정렬됨.
    task_mode:
      - "prediction": 라벨 = 미래 [win_end, win_end+horizon] 의 desat (조기예측).
        exclude_already_low 로 window 끝에 이미 desat 이면 drop(leakage 방지).
      - "detection": 라벨 = 입력 window 자체 [win_start, win_end] 의 concurrent desat.
        exclude_already_low 무시(진행중 desat 이 곧 positive).
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    horizon_samples = int(horizon_sec * TARGET_SR)
    total_needed = win_samples + horizon_samples
    min_consecutive = max(1, int(sustained_sec))  # 1Hz

    samples: list[ForecastSample] = []
    for case in cases:
        signals = case["signals"]
        spo2 = case["spo2"]
        if not signals or spo2 is None or len(spo2) == 0:
            continue
        n_total = min(len(s) for s in signals.values())
        if n_total < total_needed:
            continue

        for start in range(0, n_total - total_needed + 1, stride_samples):
            input_dict = {}
            for stype in input_signals:
                if stype in signals:
                    input_dict[stype] = signals[stype][start:start + win_samples]
            if not input_dict:
                continue

            # Step 1: gap-policy window drop
            if compute_valid_ratio(list(input_dict.values())) < valid_ratio_threshold:
                if gap_stats is not None:
                    gap_stats.add_drop()
                continue

            win_end_sec = (start + win_samples) / TARGET_SR
            if task_mode == "detection":
                # 라벨 window = 입력 window 자체 (concurrent). exclude_already_low 무시.
                l_start = int(start / TARGET_SR)
                l_end = int(win_end_sec)
            else:  # prediction: 미래 horizon
                l_start = int(win_end_sec)
                l_end = int(win_end_sec + horizon_sec)
                if l_end > len(spo2):
                    if gap_stats is not None:
                        gap_stats.add_drop()
                    continue
                # 이미 desat 중인 window 제외 (window 끝 직전 5초 SpO2 기준)
                if exclude_already_low:
                    pre = spo2[max(0, l_start - 5):l_start]
                    pre = pre[(pre >= 50.0) & (pre <= 100.0)]
                    if len(pre) and float(np.min(pre)) < spo2_threshold:
                        if gap_stats is not None:
                            gap_stats.add_drop()
                        continue

            lbl_spo2 = spo2[l_start:l_end]
            lbl_spo2 = lbl_spo2[np.isfinite(lbl_spo2)]
            lbl_spo2 = lbl_spo2[(lbl_spo2 >= 50.0) & (lbl_spo2 <= 100.0)]
            if len(lbl_spo2) < max(1, min_consecutive // 2):
                if gap_stats is not None:
                    gap_stats.add_drop()
                continue

            label = 1 if _has_sustained_desat(
                lbl_spo2.tolist(), spo2_threshold, min_consecutive) else 0
            min_future_spo2 = float(lbl_spo2.min())

            filled_dict, gap_mask_dict = apply_gap_mask_multichannel(
                input_dict, output_dtype=sample_dtype)
            if gap_stats is not None:
                n_s = sum(a.size for a in filled_dict.values())
                n_g = sum(int(m.sum()) for m in gap_mask_dict.values())
                gap_stats.add_window(n_s, n_g)

            # 부재 input 채널을 all-gap 으로 패딩 → 샘플 간 채널 key/순서 일관 (packing 안전).
            # 없는 modality(예: FLOW_WAV 무 → resp_flow)는 전체 gap-mask 되어 모델에서 마스킹됨.
            ordered_filled: dict[str, np.ndarray] = {}
            ordered_gap: dict[str, np.ndarray] = {}
            for st in input_signals:
                if st in filled_dict:
                    ordered_filled[st] = filled_dict[st]
                    ordered_gap[st] = gap_mask_dict[st]
                else:
                    ordered_filled[st] = np.zeros(win_samples, dtype=sample_dtype)
                    ordered_gap[st] = np.ones(win_samples, dtype=bool)

            samples.append(ForecastSample(
                input_signals=ordered_filled,
                input_gap_masks=ordered_gap,
                label=label,
                label_value=min_future_spo2,
                case_id=case["case_id"],
                win_start_sec=start / TARGET_SR,
                horizon_sec=horizon_sec,
            ))
    return samples


# ---- 저장 (hypoxemia 포맷과 호환, task/source 만 다름) ----


def pack_samples_to_dict(samples: list[ForecastSample]) -> dict:
    # ⚠ 키 이름은 run.py(hypoxemia 계열)가 기대하는 "signals"/"gap_masks" 여야 함.
    #   run.py --data-path 분기가 train_data["signals"] 로 읽으므로 다르면 KeyError.
    if not samples:
        return {"signals": {}, "gap_masks": {},
                "labels": torch.tensor([]), "label_values": torch.tensor([]),
                "case_ids": [], "win_starts": []}
    stypes = list(samples[0].input_signals.keys())
    # bfloat16 저장 (모델 학습 dtype 과 정합). numpy 는 bf16 미지원이라 torch 로 캐스팅.
    #   → run.py 로드시 .float().numpy() 로 되돌림 (numpy bf16 불가라 .numpy() 직접 호출 금지).
    signals = {
        st: torch.from_numpy(np.stack([s.input_signals[st] for s in samples])).to(torch.bfloat16)
        for st in stypes
    }
    gap_masks = {
        st: torch.from_numpy(np.stack([s.input_gap_masks[st] for s in samples]))
        for st in stypes
    }
    return {
        "signals": signals,
        "gap_masks": gap_masks,
        "labels": torch.tensor([s.label for s in samples], dtype=torch.long),
        "label_values": torch.tensor([s.label_value for s in samples], dtype=torch.float32),
        "case_ids": [s.case_id for s in samples],
        "win_starts": [s.win_start_sec for s in samples],
    }


def save_split_dataset(
    split_dict: dict,
    split_name: str,
    input_signals: list[str],
    horizon_sec: float,
    window_sec: float,
    out_dir: str,
    spo2_threshold: float,
    sustained_sec: float,
    signal_dtype: str = "bfloat16",
    fold_idx: int | None = None,
    n_folds: int = 1,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_samples = int(split_dict.get("labels", torch.tensor([])).numel())
    save_dict = {
        "split": split_name,
        "data": split_dict,
        "metadata": {
            "task": "desaturation_forecast",
            "source": "snuh_ventilation_eventcases",
            "input_signals": input_signals,
            "horizon_sec": horizon_sec,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "spo2_threshold": spo2_threshold,
            "sustained_sec": sustained_sec,
            "gap_policy": "drop+mask",
            "valid_ratio_threshold": DEFAULT_VALID_RATIO_THRESHOLD,
            "split": split_name,
            "n_samples": n_samples,
            "signal_dtype": signal_dtype,
            "fold_idx": fold_idx,
            "n_folds": n_folds,
        },
    }
    mode_str = "_".join(input_signals)
    horizon_min = int(horizon_sec / 60)
    win_int = int(window_sec)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    filename = (f"desaturation_{mode_str}_w{win_int}s_h{horizon_min}min"
                f"{fold_suffix}_{split_name}.pt")
    save_path = out_path / filename
    torch.save(save_dict, save_path)
    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {save_path.name} ({size_mb:.1f} MB, n={n_samples})")
    return save_path


def print_stats(name: str, samples: list[ForecastSample]) -> None:
    n = len(samples)
    n_pos = sum(1 for s in samples if s.label == 1)
    vals = [s.label_value for s in samples]
    print(f"  [{name}] n={n}  pos={n_pos} ({100*n_pos/max(1,n):.1f}%)  "
          f"future SpO2 min: median={np.median(vals) if vals else float('nan'):.1f}")


# ---- patient 매핑 (case_id → 환자 ID) ----


def load_patient_map(meta_xlsx: Path | None) -> dict[str, str]:
    """xlsx events 시트에서 case_id → 환자 ID 매핑. 없으면 빈 dict (파일명 prefix 폴백)."""
    if meta_xlsx is None or not meta_xlsx.exists():
        return {}
    import pandas as pd
    ev = pd.ExcelFile(meta_xlsx).parse("events")
    cid_col = ev.columns[0]      # case_id
    pid_col = ev.columns[1]      # 환자 ID (mojibake 방지 위해 위치로 접근)
    m: dict[str, str] = {}
    for _, r in ev.iterrows():
        cid = str(r[cid_col]); pid = str(r[pid_col])
        if cid and cid != "nan":
            m[cid] = pid
    return m


# ---- main ----


def main() -> None:
    parser = argparse.ArgumentParser(description="SNUH desaturation prediction 데이터 준비")
    parser.add_argument("--vital-dir", required=True, help="SNUH .vital 디렉토리 (188개)")
    parser.add_argument("--meta-xlsx", default=None, help="ventilation_cases.xlsx (case_id→환자ID)")
    # CO2·AWP·RESP_Flow·PPG (호흡 + PPG). PPG 는 SpO2 와 같은 센서라 예측력↑(문헌 관행,
    # PLOS One 2023 은 SpO2 자체를 입력) — competitive 버전. ECG 제외. respiratory-only
    # (co2 awp resp_flow) 로 leakage-clean ablation 하려면 --input-signals override.
    parser.add_argument("--input-signals", nargs="+",
                        default=["co2", "awp", "resp_flow", "ppg"])
    # 기본 None → input-signals 전부 필수(신호 고정: 모든 케이스가 입력 채널을 실제로 다 보유,
    # 패딩 없음). 일부만 필수로 완화하려면 명시적으로 지정(없는 채널은 all-gap 패딩).
    parser.add_argument("--required-signals", nargs="+", default=None)
    parser.add_argument("--window-secs", nargs="+", type=float, default=[300.0])
    # horizon 1/3/5min = 임상·선행연구(IOH/hypoxemia 예측) 정렬. 15min = stress test
    # (예측 지평선 한계 측정 — AUPRC 붕괴는 실패가 아니라 "호흡역학→desat 예측의
    #  생리학적 지평선"을 실증하는 곡선으로 해석).
    parser.add_argument("--horizon-mins", nargs="+", type=float, default=[1.0, 3.0, 5.0])
    parser.add_argument("--stride-sec", type=float, default=30.0)
    parser.add_argument("--spo2-threshold", type=float, default=95.0,
                        help="SpO2 임계 (%%). 95=SNUH 소아(PLOS One 2023) 관행(기본), "
                             "92=Prescience/WHO, 90=conservative.")
    parser.add_argument("--sustained-sec", type=float, default=1.0,
                        help="임계 아래 최소 연속 초. 1='any point'(소아논문+게이트 관행, 기본). "
                             "10/30 은 dwell sensitivity variant.")
    parser.add_argument("--no-artifact-gate", action="store_true",
                        help="SpO2 아티팩트 게이트(HR concordance+저관류) 비활성화")
    parser.add_argument("--hr-tol", type=float, default=0.20,
                        help="pulse-HR vs ECG-HR 괴리 허용비 (기본 0.20 = 20%%, 소아논문)")
    parser.add_argument("--pi-min", type=float, default=0.3,
                        help="perfusion index 하한 (미만 → 저관류 제외)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=None, help="디버그: 앞 N개만")
    parser.add_argument("--external-test", action="store_true",
                        help="fold 분할 없이 전체를 test-only 로 저장 (외부검증)")
    parser.add_argument("--task-mode", choices=["prediction", "detection"], default="prediction",
                        help="prediction=미래 horizon desat 예측(기본). "
                             "detection=입력 window 내 concurrent desat 분류(horizon 무시, h0).")
    parser.add_argument("--out-dir", default="outputs/downstream/desaturation")
    args = parser.parse_args()

    # detection 은 horizon 개념이 없음 → 단일 pass(h0)로 강제. 파일명 _h0min 로 prediction 과 구분.
    if args.task_mode == "detection":
        args.horizon_mins = [0.0]

    # 신호 고정: required 미지정 시 input 전부 필수 → 모든 케이스가 입력 채널을 실제 보유(패딩 없음).
    if args.required_signals is None:
        args.required_signals = list(args.input_signals)

    vital_dir = Path(args.vital_dir)
    files = sorted(vital_dir.glob("*.vital"))
    if args.max_cases:
        files = files[:args.max_cases]
    if not files:
        print(f"ERROR: {vital_dir} 에 .vital 없음", file=sys.stderr); sys.exit(1)

    patient_map = load_patient_map(Path(args.meta_xlsx) if args.meta_xlsx else None)
    print(f"[1/4] {len(files)} .vital 로딩 (patient_map={len(patient_map)}건)...")

    gate_on = not args.no_artifact_gate
    print(f"  아티팩트 게이트: {'ON' if gate_on else 'OFF'}"
          + (f" (HR 괴리>{args.hr_tol:.0%}, PI<{args.pi_min})" if gate_on else ""))
    cases: list[dict] = []
    gate_tot = {"n_total": 0, "n_range": 0, "n_hr_discord": 0, "n_low_pi": 0, "n_gated": 0}
    for i, f in enumerate(files):
        cid = f.stem
        pid = patient_map.get(cid, cid.split("_")[0])  # 폴백: 파일명 prefix
        case = load_case_from_vital(
            f, pid, args.input_signals, args.required_signals,
            artifact_gate=gate_on, hr_tol=args.hr_tol, pi_min=args.pi_min)
        if case is not None:
            cases.append(case)
            if case.get("gate_stats"):
                for k in gate_tot:
                    gate_tot[k] += case["gate_stats"].get(k, 0)
        if (i + 1) % 20 == 0:
            print(f"    ...{i+1}/{len(files)} (loaded {len(cases)})")
    print(f"  → {len(cases)}/{len(files)} case 로드 성공")
    if gate_on and gate_tot["n_total"]:
        t = gate_tot
        print(f"  게이트 제외 SpO2 샘플: {t['n_gated']}/{t['n_total']} "
              f"({100*t['n_gated']/t['n_total']:.1f}%) "
              f"[range {t['n_range']} · HR괴리 {t['n_hr_discord']} · 저관류 {t['n_low_pi']}]")
    if not cases:
        print("ERROR: 로드된 case 없음", file=sys.stderr); sys.exit(1)

    # patient-level desat 유무 (stratify 용): 녹화 내 SpO2<thr ≥60s 존재?
    patient_to_labels: dict[str, list[int]] = {}
    for c in cases:
        s = c["spo2"]; s = s[(s >= 50) & (s <= 100)]
        has = int(_has_sustained_desat(s.tolist(), args.spo2_threshold, int(args.sustained_sec)))
        patient_to_labels.setdefault(c["patient_id"], []).append(has)
    patient_ids = sorted(patient_to_labels.keys())
    pos_pts = sum(1 for p in patient_ids if any(patient_to_labels[p]))
    print(f"  환자 {len(patient_ids)}명 중 desat 보유 {pos_pts}명 ({100*pos_pts/len(patient_ids):.0f}%)")

    combos = [(w, h * 60.0) for w in args.window_secs for h in args.horizon_mins]

    for window_sec, horizon_sec in combos:
        _lbl = ("concurrent(입력 window 내)" if args.task_mode == "detection"
                else f"미래 {horizon_sec/60:.0f}min 내")
        print(f"\n[2/4] window={window_sec:.0f}s [{args.task_mode}] "
              f"| label: {_lbl} SpO2<{args.spo2_threshold:.0f}%")

        if args.external_test:
            gap = GapStats()
            samples = extract_forecast_samples(
                cases, args.input_signals, window_sec, args.stride_sec, horizon_sec,
                args.spo2_threshold, args.sustained_sec, task_mode=args.task_mode, gap_stats=gap)
            print_stats("external_test", samples)
            save_split_dataset(
                pack_samples_to_dict(samples), "test", args.input_signals,
                horizon_sec, window_sec, args.out_dir,
                args.spo2_threshold, args.sustained_sec, n_folds=1)
            del samples; gc.collect()
            continue

        splits = stratified_kfold_patient_splits(
            patient_ids, patient_to_labels, n_folds=args.n_folds, seed=args.seed)
        print(summarize_splits(splits, patient_to_labels))

        for fold_idx, (train_pids, val_pids, test_pids) in enumerate(splits):
            print(f"  --- fold {fold_idx} ---")
            for split_name, pids in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
                pset = set(pids)
                sub = [c for c in cases if c["patient_id"] in pset]
                gap = GapStats()
                samples = extract_forecast_samples(
                    sub, args.input_signals, window_sec, args.stride_sec, horizon_sec,
                    args.spo2_threshold, args.sustained_sec, task_mode=args.task_mode, gap_stats=gap)
                print_stats(split_name, samples)
                save_split_dataset(
                    pack_samples_to_dict(samples), split_name, args.input_signals,
                    horizon_sec, window_sec, args.out_dir,
                    args.spo2_threshold, args.sustained_sec,
                    fold_idx=fold_idx, n_folds=args.n_folds)
                del samples; gc.collect()

    print(f"\n[4/4] 완료 → {args.out_dir}")


if __name__ == "__main__":
    main()
