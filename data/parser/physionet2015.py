# -*- coding:utf-8 -*-
"""PhysioNet/CinC Challenge 2015 — ICU False Alarm Reduction 파서.

250Hz WFDB 레코드 (ECG 2ch + ABP/PPG) → 100Hz .pt 변환.
알람 전 5분(short) 또는 5분30초(long) 구간을 파싱하여 downstream task에 사용한다.

데이터 구조:
    raw/physionet-challenge-2015/training/
    ├── ALARMS              ← 라벨 (record,alarm_type,0/1)
    ├── RECORDS             ← 레코드 목록
    ├── a100l.hea + .mat    (Asystole, long)
    ├── a100s.hea + .mat    (Asystole, short)
    ├── v200l.hea + .mat    (V-Tach, long)
    └── ...

레코드 네이밍:
    첫 글자: 알람 타입 (a/b/t/v/f)
    숫자: 레코드 ID
    마지막 글자: l=long(5min+30s post-alarm), s=short(5min pre-alarm only)

ALARMS 파일 포맷:
    record_name,Alarm_Type,label(0=false,1=true)
    예: v100s,Ventricular_Tachycardia,0

사용법:
    python -m data.parser.physionet2015 \
        --raw-dir datasets/raw/physionet-challenge-2015 \
        --out-dir datasets/processed/physionet2015 \
        --max-records 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data.parser._common import resample_to_target

TARGET_SR: float = 100.0
NATIVE_SR: float = 250.0


# ── 신호별 전처리 설정 (VitalDB 파서와 동일 파이프라인) ────────────


def _apply_bandpass(data: np.ndarray, lo: float, hi: float, sr: float) -> np.ndarray:
    """Butterworth 대역통과 필터 (1D)."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if hi <= lo:
        return data
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data).astype(data.dtype)


def _apply_lowpass(data: np.ndarray, hi: float, sr: float) -> np.ndarray:
    """Butterworth 저역통과 필터 (1D). DC 성분(절대값)을 보존한다."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if hi <= 0:
        return data
    sos = butter(4, hi / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, data).astype(data.dtype)


def _apply_notch_filter(
    data: np.ndarray, freq: float, sr: float, Q: float = 30.0
) -> np.ndarray:
    """전원 간섭(50/60Hz) 제거를 위한 notch filter."""
    from scipy.signal import filtfilt, iirnotch

    nyq = sr / 2.0
    if freq >= nyq:
        return data
    b, a = iirnotch(freq / nyq, Q)
    return filtfilt(b, a, data).astype(data.dtype)


def _apply_median_filter(data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """임펄스 노이즈 제거를 위한 median filter."""
    from scipy.signal import medfilt

    if kernel_size < 3:
        return data
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(data, kernel_size=kernel_size).astype(data.dtype)


def _detect_spikes(
    data: np.ndarray, sr: float, threshold_std: float = 10.0, blank_ms: float = 100.0
) -> tuple[np.ndarray, int]:
    """급격한 진폭 변화(spike/artifact) 구간을 NaN으로 마킹한다."""
    out = data.copy()
    diff = np.abs(np.diff(out, prepend=out[0]))
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) * 1.4826
    if mad < 1e-10:
        return out, 0
    spike_mask = diff > (med + threshold_std * mad)
    if not spike_mask.any():
        return out, 0
    blank_samples = int(blank_ms / 1000.0 * sr)
    for idx in np.where(spike_mask)[0]:
        start = max(0, idx - blank_samples)
        end = min(len(out), idx + blank_samples + 1)
        out[start:end] = np.nan
    n_blanked = int(np.isnan(out).sum() - np.isnan(data).sum())
    return out, n_blanked


# 신호별 필터링 설정 (VitalDB 파서와 동일한 필터 파라미터)
# 주의: downstream 데이터이므로 품질 체크로 채널을 폐기하지 않음.
#       필터링만 적용하여 pretraining과 동일한 주파수 특성을 맞춘다.
FILTER_CONFIGS: dict[str, dict] = {
    "ecg": {
        "filter": "bandpass",
        "freq": (0.5, 40.0),
        "notch": 60.0,
    },
    "abp": {
        "filter": "lowpass",
        "freq": (0.0, 15.0),
        "median_kernel": 5,
    },
    "ppg": {
        "filter": "lowpass",
        "freq": (0.0, 8.0),
        "notch": 60.0,
        "median_kernel": 5,
    },
}


def preprocess_signal(signal: np.ndarray, sig_type: str, sr: float) -> np.ndarray:
    """Pretraining과 동일한 필터링 파이프라인을 적용한다.

    Downstream 데이터이므로 품질 체크/채널 폐기는 하지 않고,
    주파수 특성만 pretraining과 일치시킨다.

    1. NaN 보간
    2. Median filter (ABP, PPG — 임펄스 노이즈 제거)
    3. Notch filter (ECG, PPG — 전원 간섭 제거)
    4. Bandpass / Lowpass filter
    """
    cfg = FILTER_CONFIGS.get(sig_type)
    if cfg is None:
        return signal

    # 1. NaN 보간 (필터 적용 전에 NaN 제거 필요)
    nan_ratio = float(np.isnan(signal).mean())
    if nan_ratio > 0 and nan_ratio < 1.0:
        nans = np.isnan(signal)
        x = np.arange(len(signal))
        signal = signal.copy()
        signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])

    # 2. Median filter
    kernel = cfg.get("median_kernel", 0)
    if kernel >= 3:
        signal = _apply_median_filter(signal, kernel)

    # 3. Notch filter
    notch = cfg.get("notch")
    if notch is not None:
        signal = _apply_notch_filter(signal, notch, sr)

    # 4. Bandpass / Lowpass
    ftype = cfg.get("filter", "none")
    freq = cfg.get("freq")
    if ftype == "bandpass" and freq is not None:
        signal = _apply_bandpass(signal, freq[0], freq[1], sr)
    elif ftype == "lowpass" and freq is not None:
        signal = _apply_lowpass(signal, freq[1], sr)

    return signal

# ALARMS 파일의 alarm 이름 → 통일 이름
ALARM_NAME_NORMALIZE: dict[str, str] = {
    "Asystole": "asystole",
    "Bradycardia": "extreme_bradycardia",
    "Tachycardia": "extreme_tachycardia",
    "Ventricular_Tachycardia": "ventricular_tachycardia",
    "Ventricular_Flutter_Fib": "ventricular_flutter_fib",
}

# 레코드 첫 글자 → alarm type (폴백)
ALARM_CODE_MAP: dict[str, str] = {
    "a": "asystole",
    "b": "extreme_bradycardia",
    "t": "extreme_tachycardia",
    "v": "ventricular_tachycardia",
    "f": "ventricular_flutter_fib",
}

# 채널명 → signal_type 매핑
SIGNAL_NAME_MAP: dict[str, str] = {
    # ECG leads
    "I": "ecg", "II": "ecg", "III": "ecg",
    "V": "ecg", "V1": "ecg", "V2": "ecg", "V5": "ecg",
    "aVR": "ecg", "aVL": "ecg", "aVF": "ecg",
    "MCL": "ecg", "MCL1": "ecg",
    # ABP
    "ABP": "abp", "ART": "abp", "AOBP": "abp",
    # PPG
    "PLETH": "ppg",
}

# signal_type → 정수 코드 (spatial_map.py 기준)
SIGNAL_TYPE_INT: dict[str, int] = {
    "ecg": 0,
    "abp": 1,
    "ppg": 2,
}


def load_alarms(raw_dir: Path) -> dict[str, dict]:
    """ALARMS 파일에서 레코드별 (alarm_type, label) 로드.

    ALARMS 형식: record_name,Alarm_Type,0_or_1 (헤더 없음)

    Returns
    -------
    {record_name: {"alarm_type": str, "label": bool}}
    """
    alarms: dict[str, dict] = {}

    for alarms_file in raw_dir.rglob("ALARMS"):
        with open(alarms_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    name = parts[0].strip()
                    alarm_raw = parts[1].strip()
                    label_int = int(parts[2].strip())
                    alarms[name] = {
                        "alarm_type": ALARM_NAME_NORMALIZE.get(alarm_raw, alarm_raw),
                        "label": label_int == 1,
                    }

    return alarms


def parse_record(
    record_path: Path,
    out_dir: Path,
    alarm_type: str,
    label: bool,
) -> dict | None:
    """단일 WFDB 레코드(.hea + .mat)를 파싱하여 채널별 .pt로 저장한다.

    Parameters
    ----------
    record_path: .hea 파일 경로 (확장자 제외한 stem).
    out_dir: 출력 디렉토리.
    alarm_type: 알람 타입 문자열.
    label: True=real alarm, False=false alarm.

    Returns
    -------
    저장된 레코드 정보 딕셔너리, 실패 시 None.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    try:
        rec = wfdb.rdrecord(str(record_path))
    except Exception as e:
        print(f"  SKIP {record_path.name}: {e}")
        return None

    if rec.p_signal is None or rec.sig_len == 0:
        return None

    record_name = record_path.name
    signals_saved: list[dict] = []
    signal_types_found: list[str] = []

    fs = float(rec.fs)

    for ch_idx, sig_name in enumerate(rec.sig_name):
        sig_type = SIGNAL_NAME_MAP.get(sig_name)
        if sig_type is None:
            continue

        signal = rec.p_signal[:, ch_idx].astype(np.float64)

        # 필터링 (NaN 보간 → median → notch → bandpass/lowpass)
        # native SR(250Hz)에서 수행 후 리샘플링
        signal = preprocess_signal(signal, sig_type, fs)

        # 리샘플링 250Hz → 100Hz
        if fs != TARGET_SR:
            signal = resample_to_target(signal, fs, TARGET_SR)

        tensor = torch.from_numpy(signal).float().unsqueeze(0)  # (1, T)
        stype_int = SIGNAL_TYPE_INT.get(sig_type, -1)

        # 채널별 고유 파일명 (ECG가 2ch일 수 있음)
        ch_suffix = f"{sig_type}_{sig_name.replace('/', '_')}"
        out_name = f"{record_name}_{ch_suffix}.pt"
        torch.save(tensor, out_dir / out_name)

        signals_saved.append({
            "file": out_name,
            "signal_type": sig_type,
            "signal_type_int": stype_int,
            "signal_name": sig_name,
            "n_samples": tensor.shape[-1],
            "duration_sec": round(tensor.shape[-1] / TARGET_SR, 2),
        })
        if sig_type not in signal_types_found:
            signal_types_found.append(sig_type)

    if not signals_saved:
        return None

    return {
        "record": record_name,
        "alarm_type": alarm_type,
        "label": label,
        "signals": signals_saved,
        "signal_types": signal_types_found,
        "sampling_rate": TARGET_SR,
        "native_sr": fs,
        "n_channels": len(rec.sig_name),
        "all_channel_names": rec.sig_name,
        "duration_sec": round(rec.sig_len / rec.fs, 2),
        "record_type": "long" if record_name.endswith("l") else "short",
    }


def parse_dataset(
    raw_dir: str,
    out_dir: str,
    max_records: int | None = None,
) -> None:
    """PhysioNet 2015 Challenge 전체 파싱.

    Parameters
    ----------
    raw_dir: raw 데이터 루트 (training/ 디렉토리가 있는 경로).
    out_dir: 출력 디렉토리.
    max_records: 최대 레코드 수 (디버깅용).
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ALARMS 파일 로드
    alarms = load_alarms(raw_path)
    if alarms:
        print(f"Loaded {len(alarms)} alarm labels from ALARMS file(s)")
    else:
        print("  WARNING: No ALARMS file found — using record name prefix as fallback")

    # .hea 파일 탐색
    hea_files = sorted(raw_path.rglob("*.hea"))
    print(f"Found {len(hea_files)} records")

    if max_records is not None:
        hea_files = hea_files[:max_records]
        print(f"  Processing first {max_records} records")

    manifest: list[dict] = []
    alarm_counts: dict[str, dict[str, int]] = {}
    signal_type_counts: dict[str, int] = {}
    channel_name_set: set[str] = set()
    n_success = 0
    n_skip = 0

    for i, hea_path in enumerate(hea_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{len(hea_files)}] {hea_path.stem}...")

        record_stem = str(hea_path.with_suffix(""))
        record_name = hea_path.stem

        # 알람 타입 및 라벨 결정
        if record_name in alarms:
            alarm_type = alarms[record_name]["alarm_type"]
            label = alarms[record_name]["label"]
        else:
            alarm_type = ALARM_CODE_MAP.get(record_name[0].lower(), "unknown")
            label = False  # 보수적 폴백

        # 이미 파싱된 경우 스킵
        existing_pts = list(out_path.glob(f"{record_name}_*.pt"))
        if existing_pts:
            n_skip += 1
            manifest.append({
                "record": record_name,
                "alarm_type": alarm_type,
                "label": label,
                "signals": [{"file": p.name} for p in existing_pts],
                "sampling_rate": TARGET_SR,
            })
            if alarm_type not in alarm_counts:
                alarm_counts[alarm_type] = {"true": 0, "false": 0}
            alarm_counts[alarm_type]["true" if label else "false"] += 1
            continue

        result = parse_record(
            Path(record_stem), out_path, alarm_type, label,
        )

        if result is None:
            n_skip += 1
            continue

        manifest.append(result)
        n_success += 1

        # 통계 수집
        if alarm_type not in alarm_counts:
            alarm_counts[alarm_type] = {"true": 0, "false": 0}
        alarm_counts[alarm_type]["true" if label else "false"] += 1

        for sig_info in result.get("signals", []):
            st = sig_info.get("signal_type", "unknown")
            signal_type_counts[st] = signal_type_counts.get(st, 0) + 1
        for ch_name in result.get("all_channel_names", []):
            channel_name_set.add(ch_name)

    # manifest 저장
    total_true = sum(c["true"] for c in alarm_counts.values())
    total_false = sum(c["false"] for c in alarm_counts.values())

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "dataset": "PhysioNet-Challenge-2015",
            "task": "ICU False Alarm Reduction",
            "sampling_rate": TARGET_SR,
            "native_sr": NATIVE_SR,
            "n_records": len(manifest),
            "n_true_alarms": total_true,
            "n_false_alarms": total_false,
            "alarm_distribution": alarm_counts,
            "signal_type_counts": signal_type_counts,
            "unique_channel_names": sorted(channel_name_set),
            "records": manifest,
        }, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  PhysioNet 2015 Challenge Parsing Complete")
    print(f"  New: {n_success}, Skipped (existing): {n_skip}")
    print(f"  Total records: {len(manifest)}")
    print(f"  True alarms: {total_true}, False alarms: {total_false}")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  Alarm distribution:")
    for atype in sorted(alarm_counts.keys()):
        counts = alarm_counts[atype]
        total = counts["true"] + counts["false"]
        false_pct = counts["false"] / max(total, 1) * 100
        print(f"    {atype:30s}: True={counts['true']:4d}  False={counts['false']:4d}  "
              f"(False {false_pct:.0f}%)")
    print(f"\n  Signal types: {signal_type_counts}")
    print(f"  Channel names found: {sorted(channel_name_set)}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhysioNet 2015 Challenge — ICU False Alarm Reduction Parser",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Raw data directory (contains training/ with .hea/.mat files)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="datasets/processed/anomaly_detection",
        help="Output directory for processed .pt files",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Max records to process (None=all)",
    )
    args = parser.parse_args()

    parse_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()
