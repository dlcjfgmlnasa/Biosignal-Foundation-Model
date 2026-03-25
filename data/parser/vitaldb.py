# -*- coding:utf-8 -*-
"""VitalDB (.vital) → datasets/processed/ 변환 스크립트.

VitalDB(https://vitaldb.net)의 수술 중 모니터링 데이터를 파싱하여
zarr 압축 포맷으로 저장한다.

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
from pathlib import Path

import numpy as np

from data.parser._common import resample_to_target, save_recording_zarr

# 목표 sampling rate (Hz)
TARGET_SR: float = 100.0

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


def discover_tracks(vital_path: Path) -> list[str]:
    """파일 내 트랙명을 출력한다 (로컬 탐색용)."""
    import vitaldb

    vf = vitaldb.VitalFile(str(vital_path))
    tracks = vf.get_track_names()
    return tracks


def _parse_subject_id(vital_path: Path) -> tuple[str, str]:
    """파일명에서 (subject_id, session_id)를 추출한다.

    예: 00042.vital → ("VDB_0042", "VDB_0042_S0")
    """
    stem = vital_path.stem
    # 숫자만 추출하여 zero-pad
    digits = "".join(c for c in stem if c.isdigit())
    if not digits:
        digits = stem
    subject_id = f"VDB_{int(digits):04d}"
    session_id = f"{subject_id}_S0"
    return subject_id, session_id


def _extract_nan_free_segments(
    data: np.ndarray,  # (T,) float
    min_samples: int,
) -> list[np.ndarray]:
    """NaN 구간을 제거하고 연속 valid 세그먼트를 반환한다.

    Parameters
    ----------
    data:
        1D 신호 배열. NaN이 포함될 수 있음.
    min_samples:
        최소 세그먼트 길이 (샘플 수).

    Returns
    -------
    min_samples 이상인 연속 valid 세그먼트 리스트.
    """
    valid = ~np.isnan(data)
    segments: list[np.ndarray] = []

    # valid 구간의 시작/끝 찾기
    diff = np.diff(valid.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        if e - s >= min_samples:
            segments.append(data[s:e])

    return segments


def _normalize(data: np.ndarray) -> np.ndarray:
    """채널별 z-score 정규화. (C, T) → (C, T) float32."""
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((data - mean) / std).astype(np.float32)


def process_vital(
    vital_path: Path,
    out_dir: Path,
    min_duration_s: float = 60.0,
) -> tuple[str, str, list[dict]]:
    """단일 .vital 파일을 처리하여 zarr 파일들을 저장한다.

    Parameters
    ----------
    vital_path:
        .vital 파일 경로.
    out_dir:
        datasets/processed/ 루트 경로.
    min_duration_s:
        최소 유효 신호 길이 (초).

    Returns
    -------
    (subject_id, session_id, recordings)
    recordings는 manifest.json sessions[].recordings 항목 리스트.
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

        # 동일 (signal_type, spatial_id) 중복 시 첫 번째만 처리
        key = (signal_type, spatial_id)
        if key in processed_keys:
            continue

        # Native sampling rate로 데이터 추출
        try:
            native_sr = vf.get_track_info(track_name).get("srate", 0)
            if native_sr <= 0:
                # fallback: 일반적인 SR 추정
                native_sr = 500.0
            data = vf.to_numpy(track_name, interval=1.0 / native_sr)
        except Exception as exc:
            print(
                f"    [WARN] {track_name} 로드 실패: {exc}",
                file=sys.stderr,
            )
            continue

        if data is None or len(data) == 0:
            continue

        data = data.flatten()

        # NaN-free 세그먼트 추출
        min_samples = int(min_duration_s * native_sr)
        segments = _extract_nan_free_segments(data, min_samples)
        if not segments:
            print(
                f"    [SKIP] {track_name}: 유효 세그먼트 없음 (min={min_duration_s}s)",
                file=sys.stderr,
            )
            continue

        # 가장 긴 세그먼트 선택
        segment = max(segments, key=len)

        # Flatline 검사
        if segment.std() < 1e-6:
            print(
                f"    [SKIP] {track_name}: 플랫라인",
                file=sys.stderr,
            )
            continue

        # Resampling → TARGET_SR (100Hz)
        if native_sr != TARGET_SR:
            segment = resample_to_target(segment, orig_sr=native_sr, target_sr=TARGET_SR)
            print(
                f"    [RESAMPLE] {track_name}: {native_sr:.0f}Hz → {TARGET_SR:.0f}Hz "
                f"({len(segment)} samples)",
                file=sys.stderr,
            )

        # (1, T) 형태로 정규화
        channel_data = segment.reshape(1, -1)
        channel_data = _normalize(channel_data)

        # zarr로 저장
        fname = f"{session_id}_{stype_key}_{spatial_id}.zarr"
        save_recording_zarr(channel_data, subj_out / fname)

        recordings.append(
            {
                "signal_type": signal_type,
                "file": fname,
                "n_channels": 1,
                "sampling_rate": TARGET_SR,
                "n_timesteps": channel_data.shape[1],
                "spatial_ids": [spatial_id],
            }
        )
        processed_keys.add(key)
        print(f"    saved {fname}  shape={tuple(channel_data.shape)}  fs={TARGET_SR:.0f}Hz")

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
