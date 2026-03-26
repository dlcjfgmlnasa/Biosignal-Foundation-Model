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

from data.parser._common import resample_to_target, save_recording_zarr, segment_quality_score

# 목표 sampling rate (Hz)
TARGET_SR: float = 100.0


# ── 신호별 전처리 설정 ──────────────────────────────────────────


@dataclass
class SignalConfig:
    """신호 타입별 전처리 파라미터.

    품질 기준 (segment_quality_score 연동):
        max_flatline_ratio: 연속 동일 값 비율 상한 (초과 시 불량)
        max_clip_ratio: min/max 고정 비율 상한
        max_high_freq_ratio: 고주파 에너지 비율 상한 (신호별 특성에 맞게 설정)

    high_freq_ratio 근거 (실측 + 시각 검증, 2026-03-26):
        ECG: QRS spike → 정상도 ~0.4, P99=1.97 → 3.0
        ABP: 매우 부드러운 파형, P99=0.03 → 0.5
        EEG: alpha/beta 고주파 → 합성 clean=0.38 → 2.0
        PPG: 부드러운 파형, hf>0.05부터 노이즈 → 0.05 (시각 검증)
        CVP: 저주파, 합성 clean=0.0004 → 0.5
        CO2: 느린 capnogram, flatline 구간 → hf=1.0, flatline=0.3
        AWP: P95=0.54, 1.0 이상 스파이크 → 1.0
    """
    valid_range: tuple[float, float] | None  # (min, max) — None이면 range check 안 함
    bandpass: tuple[float, float] | None     # (lo, hi) Hz — None이면 필터 안 함
    max_flatline_ratio: float = 0.5          # 50% 이상 flat이면 불량
    max_clip_ratio: float = 0.1              # 10% 이상 clipping이면 불량
    max_high_freq_ratio: float = 2.0         # 기본값; 신호별로 아래에서 재정의
    min_amplitude: float = 0.0               # 최소 peak-to-peak 진폭 (0=비활성)
    min_high_freq_ratio: float = 0.0         # 최소 hf ratio (0=비활성, ECG용: QRS 없으면 불량)


SIGNAL_CONFIGS: dict[str, SignalConfig] = {
    "ecg": SignalConfig(valid_range=(-5.0, 5.0),       bandpass=(0.5, 40.0),  max_high_freq_ratio=1.0, min_amplitude=0.3, min_high_freq_ratio=0.05),
    "abp": SignalConfig(valid_range=(0.0, 300.0),      bandpass=(0.5, 40.0),  max_high_freq_ratio=0.5),
    "eeg": SignalConfig(valid_range=(-500.0, 500.0),   bandpass=(0.5, 45.0),  max_high_freq_ratio=2.0),
    "ppg": SignalConfig(valid_range=None,               bandpass=(0.5, 8.0),   max_high_freq_ratio=0.05, min_amplitude=5.0),
    "cvp": SignalConfig(valid_range=(-5.0, 40.0),      bandpass=(0.5, 20.0),  max_high_freq_ratio=0.5),
    "co2": SignalConfig(valid_range=(0.0, 100.0),      bandpass=None,          max_high_freq_ratio=1.0, max_flatline_ratio=0.3, min_amplitude=5.0),
    "awp": SignalConfig(valid_range=(-10.0, 80.0),     bandpass=None,          max_high_freq_ratio=1.0, min_amplitude=2.0),
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
    "BIS/EEG1_WAV": ("eeg", 0),
    "BIS/EEG2_WAV": ("eeg", 0),
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
    """범위 밖 값을 NaN으로 마킹한다 (1D, 원본 비파괴 복사)."""
    lo, hi = valid_range
    out = data.copy()
    mask = (out < lo) | (out > hi)
    n_bad = int(mask.sum())
    if n_bad > 0:
        out[mask] = np.nan
    return out, n_bad


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
    """NaN 구간을 제거하고 연속 유효 세그먼트를 반환한다."""
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


def _save_subject_manifest(
    subj_dir: Path,
    subject_id: str,
    session_id: str,
    recordings: list[dict],
) -> None:
    """subject의 manifest.json을 즉시 갱신한다.

    기존 manifest가 있으면 세션/레코딩을 병합하고,
    없으면 새로 생성한다. zarr 저장 직후 호출하여
    중단 시에도 manifest 유실을 방지한다.
    """
    manifest_path = subj_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        # 같은 session_id가 있으면 recordings 병합, 없으면 세션 추가
        existing_session = None
        for s in manifest["sessions"]:
            if s["session_id"] == session_id:
                existing_session = s
                break
        if existing_session is not None:
            existing_files = {r["file"] for r in existing_session["recordings"]}
            for rec in recordings:
                if rec["file"] not in existing_files:
                    existing_session["recordings"].append(rec)
        else:
            manifest["sessions"].append(
                {"session_id": session_id, "recordings": recordings}
            )
    else:
        manifest = {
            "subject_id": subject_id,
            "source": "vitaldb",
            "sessions": [{"session_id": session_id, "recordings": recordings}],
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def process_vital(
    vital_path: Path,
    out_dir: Path,
    min_duration_s: float = 60.0,
    signal_types: set[int] | None = None,
) -> tuple[str, str, list[dict]]:
    """단일 .vital 파일을 처리하여 zarr 파일들을 저장한다.

    각 트랙의 zarr 저장 직후 manifest.json을 갱신하여,
    중간 중단 시에도 이미 저장된 데이터의 manifest가 유지된다.

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

        # signal_types 필터: 지정된 타입만 파싱
        if signal_types is not None and signal_type not in signal_types:
            continue

        # 동일 (signal_type, spatial_id) 중복 시 첫 번째만 처리
        key = (signal_type, spatial_id)
        if key in processed_keys:
            continue

        # 원본 sampling rate로 데이터 추출
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
        track_recordings: list[dict] = []
        for seg_idx, segment in enumerate(segments):
            # 품질 검사 (flatline, clipping, noise, amplitude) — 신호별 threshold 적용
            qscore = segment_quality_score(
                segment,
                max_flatline_ratio=cfg.max_flatline_ratio,
                max_clip_ratio=cfg.max_clip_ratio,
                max_high_freq_ratio=cfg.max_high_freq_ratio,
                min_amplitude=cfg.min_amplitude,
                min_high_freq_ratio=cfg.min_high_freq_ratio,
            )
            if not qscore["pass"]:
                reasons = []
                if qscore["flatline_ratio"] >= cfg.max_flatline_ratio:
                    reasons.append(f"flat={qscore['flatline_ratio']:.2f}")
                if qscore["clip_ratio"] >= cfg.max_clip_ratio:
                    reasons.append(f"clip={qscore['clip_ratio']:.2f}")
                if qscore["high_freq_ratio"] >= cfg.max_high_freq_ratio:
                    reasons.append(f"hf={qscore['high_freq_ratio']:.2f}")
                if qscore["amplitude"] < cfg.min_amplitude:
                    reasons.append(f"amp={qscore['amplitude']:.2f}<{cfg.min_amplitude}")
                if qscore["high_freq_ratio"] < cfg.min_high_freq_ratio:
                    reasons.append(f"hf_low={qscore['high_freq_ratio']:.4f}<{cfg.min_high_freq_ratio}")
                print(f"    [SKIP] {track_name} seg{seg_idx}: 품질 불량 ({', '.join(reasons)})", file=sys.stderr)
                continue

            # 대역통과 필터링 (range check/cautery 이후 정상 세그먼트에 적용)
            if cfg.bandpass is not None:
                segment = _apply_bandpass(segment, cfg.bandpass[0], cfg.bandpass[1], native_sr)

            # 리샘플링 → TARGET_SR (100Hz)
            if native_sr != TARGET_SR:
                segment = resample_to_target(segment, orig_sr=native_sr, target_sr=TARGET_SR)

            # (1, T) 형태로 저장
            channel_data = segment.reshape(1, -1).astype(np.float32)

            # 파일명: seg 인덱스 포함
            fname = f"{session_id}_{stype_key}_{spatial_id}_seg{seg_idx}.zarr"
            save_recording_zarr(channel_data, subj_out / fname)

            rec = {
                "signal_type": signal_type,
                "file": fname,
                "n_channels": 1,
                "sampling_rate": TARGET_SR,
                "n_timesteps": channel_data.shape[1],
                "spatial_ids": [spatial_id],
            }
            track_recordings.append(rec)
            recordings.append(rec)
            seg_count += 1
            duration_s = channel_data.shape[1] / TARGET_SR
            print(f"    saved {fname}  shape={tuple(channel_data.shape)}  {duration_s:.0f}s")

        if seg_count > 0:
            processed_keys.add(key)
            # 트랙 처리 완료 즉시 manifest 갱신
            _save_subject_manifest(subj_out, subject_id, session_id, track_recordings)
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
    parser.add_argument(
        "--signal-types", type=int, nargs="+", default=None,
        help="파싱할 signal type IDs (0=ECG,1=ABP,2=EEG,3=PPG,4=CVP,5=CO2,6=AWP). 미지정 시 전부.",
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

    # manifest.jsonl 기존 항목 로드
    jsonl_path = out_dir / "manifest.jsonl"
    existing_subjects: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_subjects.add(json.loads(line)["subject_id"])

    n_processed = 0

    for vf_path in vital_files:
        print(f"[{vf_path.name}]")
        try:
            sig_filter = set(args.signal_types) if args.signal_types else None
            subject_id, session_id, recordings = process_vital(
                vf_path, out_dir, min_duration_s=args.min_duration,
                signal_types=sig_filter,
            )
        except Exception as exc:
            print(f"    [ERROR] {exc}", file=sys.stderr)
            continue

        if not recordings:
            print(f"    [SKIP] 유효 레코딩 없음")
            continue

        # manifest.json은 process_vital() 내부에서 트랙마다 즉시 저장됨
        # 여기서는 manifest.jsonl 인덱스만 관리
        if subject_id not in existing_subjects:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"subject_id": subject_id, "manifest": f"{subject_id}/manifest.json"},
                    ensure_ascii=False,
                ) + "\n")
            existing_subjects.add(subject_id)

        n_processed += 1

    print(
        f"\n완료: {n_processed}명 처리 → {out_dir}"
        f"\n인덱스: {jsonl_path}"
    )


if __name__ == "__main__":
    main()
