# -*- coding:utf-8 -*-
"""VitalDB (.vital) → datasets/processed/ 변환 스크립트.

VitalDB(https://vitaldb.net)의 수술 중 모니터링 데이터를 파싱하여
zarr 압축 포맷으로 저장한다. 신호별 physiological range check와
bandpass filtering을 적용하고, 모든 유효 세그먼트를 개별 zarr로 저장한다.

신호 타입 매핑:
  ECG(0), ABP(1), EEG(2), PPG(3), CVP(4), CO2(5), AWP(6)

사용법:
  # 트랙 탐색
  python -m data.parser.vitaldb --raw datasets/raw/vitaldb --discover --max-files 3

  # 단일 파일 테스트
  python -m data.parser.vitaldb --raw datasets/raw/vitaldb --out datasets/processed --max-files 1

  # 전체 파싱
  python -m data.parser.vitaldb --raw datasets/raw/vitaldb --out datasets/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.parser._common import resample_to_target, save_recording_zarr

# 목표 sampling rate (Hz)
TARGET_SR: float = 100.0


# ── 신호별 전처리 설정 ──────────────────────────────────────────


@dataclass
class SignalConfig:
    """신호 타입별 전처리 파라미터."""
    valid_range: tuple[float, float] | None  # (min, max) — None이면 range check 안 함
    bandpass: tuple[float, float] | None     # (lo, hi) Hz — None이면 필터 안 함


SIGNAL_CONFIGS: dict[str, SignalConfig] = {
    "ecg": SignalConfig(valid_range=(-5.0, 5.0),       bandpass=(0.5, 40.0)),
    "abp": SignalConfig(valid_range=(0.0, 300.0),      bandpass=(0.5, 40.0)),
    "eeg": SignalConfig(valid_range=(-500.0, 500.0),   bandpass=(0.5, 45.0)),
    "ppg": SignalConfig(valid_range=None,               bandpass=(0.5, 8.0)),
    "cvp": SignalConfig(valid_range=(-5.0, 40.0),      bandpass=(0.5, 20.0)),
    "co2": SignalConfig(valid_range=(0.0, 100.0),      bandpass=None),
    "awp": SignalConfig(valid_range=(-10.0, 80.0),     bandpass=None),
}


# ── VitalDB 트랙 매핑 ──────────────────────────────────────────


# VitalDB 트랙명 → (signal_type_key, local_spatial_id)
TRACK_MAP: dict[str, tuple[str, int]] = {
    # ECG (0)
    "SNUADC/ECG_II": ("ecg", 2),
    "SNUADC/ECG_I": ("ecg", 1),
    "SNUADC/ECG_III": ("ecg", 3),
    "SNUADC/ECG_V5": ("ecg", 11),
    "Solar8000/ECG_II": ("ecg", 2),
    # ABP (1)
    "SNUADC/ART": ("abp", 1),
    "SNUADC/FEM": ("abp", 2),
    # EEG (2)
    "SNUADC/EEG_BIS": ("eeg", 0),
    "SNUADC/EEG1": ("eeg", 0),
    "SNUADC/EEG2": ("eeg", 0),
    # PPG (3)
    "Solar8000/PLETH": ("ppg", 1),
    "SNUADC/PLETH": ("ppg", 1),
    # CVP (4)
    "SNUADC/CVP": ("cvp", 0),
    # CO2 (5)
    "Solar8000/CO2": ("co2", 0),
    "Primus/CO2": ("co2", 0),
    # AWP (6)
    "Solar8000/AWP": ("awp", 0),
    "Primus/AWP": ("awp", 0),
}

SIGNAL_TYPES: dict[str, int] = {
    "ecg": 0, "abp": 1, "eeg": 2, "ppg": 3, "cvp": 4, "co2": 5, "awp": 6,
}


# ── 전처리 함수 ────────────────────────────────────────────────


def _apply_range_check(data: np.ndarray, valid_range: tuple[float, float]) -> tuple[np.ndarray, int]:
    """범위 밖 값을 NaN으로 마킹한다. (1D inplace-safe copy)"""
    lo, hi = valid_range
    out = data.copy()
    mask = (out < lo) | (out > hi)
    n_bad = int(mask.sum())
    if n_bad > 0:
        out[mask] = np.nan
    return out, n_bad


def _apply_bandpass(data: np.ndarray, lo: float, hi: float, sr: float) -> np.ndarray:
    """Butterworth bandpass filter. (1D)"""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if hi <= lo:
        return data

    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data).astype(data.dtype)


def _detect_electrocautery(data: np.ndarray, sr: float, threshold_std: float = 10.0,
                           blank_ms: float = 100.0) -> tuple[np.ndarray, int]:
    """전기소작기 아티팩트 구간을 NaN으로 마킹한다. (ECG/EEG용)

    급격한 진폭 변화(미분의 절대값)가 threshold_std배 이상인 구간을
    전후 blank_ms만큼 확장하여 NaN 처리한다.
    """
    out = data.copy()
    diff = np.abs(np.diff(out, prepend=out[0]))
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) * 1.4826  # MAD → std 추정
    if mad < 1e-10:
        return out, 0

    spike_mask = diff > (med + threshold_std * mad)
    if not spike_mask.any():
        return out, 0

    # blank_ms만큼 전후 확장
    blank_samples = int(blank_ms / 1000.0 * sr)
    spike_idx = np.where(spike_mask)[0]
    for idx in spike_idx:
        start = max(0, idx - blank_samples)
        end = min(len(out), idx + blank_samples + 1)
        out[start:end] = np.nan

    n_blanked = np.isnan(out).sum() - np.isnan(data).sum()
    return out, int(n_blanked)


def _extract_nan_free_segments(
    data: np.ndarray,  # (T,) float
    min_samples: int,
) -> list[np.ndarray]:
    """NaN 구간을 제거하고 연속 valid 세그먼트를 반환한다."""
    valid = ~np.isnan(data)
    segments: list[np.ndarray] = []

    diff = np.diff(valid.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        if e - s >= min_samples:
            segments.append(data[s:e])

    return segments


# ── 탐색 / 파싱 ────────────────────────────────────────────────


def discover_tracks(vital_path: Path) -> list[str]:
    """파일 내 트랙명을 출력한다 (로컬 탐색용)."""
    import vitaldb

    vf = vitaldb.VitalFile(str(vital_path))
    tracks = vf.get_track_names()
    return tracks


def _parse_subject_id(vital_path: Path) -> tuple[str, str]:
    """파일명에서 (subject_id, session_id)를 추출한다."""
    stem = vital_path.stem
    digits = "".join(c for c in stem if c.isdigit())
    if not digits:
        digits = stem
    subject_id = f"VDB_{int(digits):04d}"
    session_id = f"{subject_id}_S0"
    return subject_id, session_id


def process_vital(
    vital_path: Path,
    out_dir: Path,
    min_duration_s: float = 60.0,
) -> tuple[str, str, list[dict]]:
    """단일 .vital 파일을 처리하여 zarr 파일들을 저장한다.

    모든 유효 세그먼트를 개별 zarr로 저장한다.

    Returns
    -------
    (subject_id, session_id, recordings)
    """
    import vitaldb

    subject_id, session_id = _parse_subject_id(vital_path)
    subj_out = out_dir / subject_id
    subj_out.mkdir(parents=True, exist_ok=True)

    vf = vitaldb.VitalFile(str(vital_path))
    available_tracks = vf.get_track_names()

    recordings: list[dict] = []
    processed_keys: set[tuple[int, int]] = set()  # (signal_type, spatial_id) 중복 방지

    for track_name in available_tracks:
        if track_name not in TRACK_MAP:
            continue

        stype_key, spatial_id = TRACK_MAP[track_name]
        signal_type = SIGNAL_TYPES[stype_key]
        cfg = SIGNAL_CONFIGS[stype_key]

        # 동일 (signal_type, spatial_id) 중복 시 첫 번째만 처리
        key = (signal_type, spatial_id)
        if key in processed_keys:
            continue

        # Native sampling rate로 데이터 추출
        try:
            trk = vf.find_track(track_name)
            native_sr = trk.srate if trk is not None and trk.srate > 0 else 0
            if native_sr <= 0:
                native_sr = 500.0
            data = vf.to_numpy(track_name, interval=1.0 / native_sr)
        except Exception as exc:
            print(f"    [WARN] {track_name} 로드 실패: {exc}", file=sys.stderr)
            continue

        if data is None or len(data) == 0:
            continue

        data = data.flatten()

        # ── Step 1: Physiological range check ──
        if cfg.valid_range is not None:
            data, n_bad = _apply_range_check(data, cfg.valid_range)
            if n_bad > 0:
                pct = n_bad / len(data) * 100
                print(f"    [RANGE] {track_name}: {n_bad} samples ({pct:.1f}%) out of range", file=sys.stderr)

        # ── Step 2: 전기소작기 아티팩트 제거 (ECG/EEG만) ──
        if stype_key in ("ecg", "eeg"):
            data, n_blanked = _detect_electrocautery(data, native_sr)
            if n_blanked > 0:
                pct = n_blanked / len(data) * 100
                print(f"    [CAUTERY] {track_name}: {n_blanked} samples ({pct:.1f}%) blanked", file=sys.stderr)

        # ── Step 3: NaN-free 세그먼트 추출 ──
        min_samples = int(min_duration_s * native_sr)
        segments = _extract_nan_free_segments(data, min_samples)
        if not segments:
            print(f"    [SKIP] {track_name}: 유효 세그먼트 없음 (min={min_duration_s}s)", file=sys.stderr)
            continue

        # ── Step 4: 각 세그먼트별 처리 & 저장 ──
        seg_count = 0
        for seg_idx, segment in enumerate(segments):
            # Flatline 검사
            if segment.std() < 1e-6:
                print(f"    [SKIP] {track_name} seg{seg_idx}: 플랫라인", file=sys.stderr)
                continue

            # Bandpass filtering (range check/cautery 이후 clean 세그먼트에 적용)
            if cfg.bandpass is not None:
                segment = _apply_bandpass(segment, cfg.bandpass[0], cfg.bandpass[1], native_sr)

            # Resampling → TARGET_SR (100Hz)
            if native_sr != TARGET_SR:
                segment = resample_to_target(segment, orig_sr=native_sr, target_sr=TARGET_SR)

            # (1, T) 형태로 저장
            channel_data = segment.reshape(1, -1).astype(np.float32)

            # 파일명: seg 인덱스 포함
            fname = f"{session_id}_{stype_key}_{spatial_id}_seg{seg_idx}.zarr"
            save_recording_zarr(channel_data, subj_out / fname)

            recordings.append({
                "signal_type": signal_type,
                "file": fname,
                "n_channels": 1,
                "sampling_rate": TARGET_SR,
                "n_timesteps": channel_data.shape[1],
                "spatial_ids": [spatial_id],
            })
            seg_count += 1
            duration_s = channel_data.shape[1] / TARGET_SR
            print(f"    saved {fname}  shape={tuple(channel_data.shape)}  {duration_s:.0f}s")

        if seg_count > 0:
            processed_keys.add(key)
            if seg_count > 1:
                print(f"    [{track_name}] {seg_count}개 세그먼트 저장")

    return subject_id, session_id, recordings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VitalDB (.vital) → datasets/processed/ zarr 변환"
    )
    parser.add_argument(
        "--raw", required=True,
        help="VitalDB .vital 파일이 있는 디렉토리",
    )
    parser.add_argument(
        "--out", default=None,
        help="처리 결과를 저장할 루트 디렉토리 (--discover 시 불필요)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=60.0,
        help="최소 유효 신호 길이 (초, 기본 60)",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="트랙 탐색 모드: 파일 내 트랙명만 출력",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="처리할 최대 파일 수 (테스트용)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    vital_files = sorted(raw_dir.glob("*.vital"))
    if not vital_files:
        print(f"ERROR: {raw_dir}에 .vital 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    if args.max_files is not None:
        vital_files = vital_files[: args.max_files]

    print(f".vital 파일 {len(vital_files)}개 발견\n")

    # ── Discover 모드 ──
    if args.discover:
        for vf_path in vital_files:
            print(f"[{vf_path.name}]")
            try:
                tracks = discover_tracks(vf_path)
                for t in tracks:
                    mapped = TRACK_MAP.get(t)
                    tag = f" → {mapped[0]}({mapped[1]})" if mapped else ""
                    print(f"    {t}{tag}")
            except Exception as exc:
                print(f"    [ERROR] {exc}", file=sys.stderr)
            print()
        return

    # ── 파싱 모드 ──
    if args.out is None:
        print("ERROR: --out 경로를 지정하세요.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_sessions: dict[str, list[dict]] = {}

    for vf_path in vital_files:
        print(f"[{vf_path.name}]")
        try:
            subject_id, session_id, recordings = process_vital(
                vf_path, out_dir, min_duration_s=args.min_duration,
            )
        except Exception as exc:
            print(f"    [ERROR] {exc}", file=sys.stderr)
            continue

        if not recordings:
            print(f"    [SKIP] 유효 레코딩 없음")
            continue

        subject_sessions.setdefault(subject_id, []).append(
            {"session_id": session_id, "recordings": recordings}
        )

    # subject별 manifest.json 저장
    global_index: list[dict] = []
    for subject_id, sessions in sorted(subject_sessions.items()):
        manifest = {
            "subject_id": subject_id,
            "source": "vitaldb",
            "sessions": sessions,
        }
        manifest_path = out_dir / subject_id / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        global_index.append(
            {"subject_id": subject_id, "manifest": f"{subject_id}/manifest.json"}
        )

    # manifest.jsonl에 추가 (기존 항목 유지)
    jsonl_path = out_dir / "manifest.jsonl"
    existing: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    existing.add(entry["subject_id"])

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for entry in global_index:
            if entry["subject_id"] not in existing:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"\n완료: {len(subject_sessions)}명 처리 → {out_dir}"
        f"\n인덱스: {jsonl_path}"
    )


if __name__ == "__main__":
    main()
