"""Sleep-EDF Database → datasets/processed/ 변환 스크립트.

파일명 규칙:
  SC4XYZ...-PSG.edf  → subject_id="SC_XY", session_id="SC4XYZ..."
  ST7XYZ...-PSG.edf  → subject_id="ST_XY", session_id="ST7XYZ..."

신호 타입 매핑 (signal_type):
  EEG Fpz-Cz, EEG Pz-Oz  → eeg (2)  → {session}_eeg.pt

사용법:
  python -m data.parser.sleep_edf \\
      --raw  datasets/raw/sleep-edf-database-expanded-1.0.0 \\
      --out  datasets/processed \\
      [--bandpass] [--notch 50]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from data.parser._common import resample_to_target

# 모든 출력 신호의 목표 sampling rate (Hz)
TARGET_SR: float = 100.0

SIGNAL_TYPES: dict[str, int] = {
    "ecg": 0,
    "abp": 1,
    "eeg": 2,
    "ppg": 3,
    "cvp": 4,
    "co2": 5,
    "awp": 6,
}

# Sleep-EDF EDF 채널명 → signal_type 키 매핑
# EOG, Temp rectal, Event marker/Marker는 의도적으로 제외
CHANNEL_MAP: dict[str, str] = {
    "EEG Fpz-Cz": "eeg",
    "EEG Pz-Oz": "eeg",
}
# Sleep-EDF 채널명 → 로컬 spatial_id 매핑
CHANNEL_SPATIAL: dict[str, int] = {
    "EEG Fpz-Cz": 22,   # EEG Fpz-Cz bipolar montage
    "EEG Pz-Oz": 23,    # EEG Pz-Oz bipolar montage
}


def _parse_subject(stem: str) -> tuple[str, str]:
    """PSG 파일 스템에서 (subject_id, session_id)를 추출한다.

    Examples
    --------
    >>> _parse_subject("SC4001E0-PSG")
    ('SC_00', 'SC4001E0')
    >>> _parse_subject("ST7011J0-PSG")
    ('ST_01', 'ST7011J0')
    """
    m = re.match(r"(SC|ST)\d(\d{2})\d.*-PSG", stem)
    if m is None:
        raise ValueError(f"파일명 파싱 실패: {stem!r}")
    study, subject_num = m.group(1), m.group(2)
    session_id = stem.split("-PSG")[0]
    subject_id = f"{study}_{subject_num}"
    return subject_id, session_id


def _quality_gate(
    data: np.ndarray,
    ch_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """NaN 또는 플랫라인인 채널을 제거한다.

    Parameters
    ----------
    data:
        (C, T) float64 배열.
    ch_names:
        채널 이름 목록. data와 길이가 같아야 한다.

    Returns
    -------
    통과한 채널만 담은 (C', T) 배열과 해당 채널명 목록.
    """
    keep: list[bool] = []
    for i, ch in enumerate(ch_names):
        ch_data = data[i]
        if np.any(np.isnan(ch_data)):
            print(f"    [QC] NaN 검출 → 제외: {ch}", file=sys.stderr)
            keep.append(False)
        elif ch_data.std() < 1e-6:
            print(f"    [QC] 플랫라인 검출 → 제외: {ch}", file=sys.stderr)
            keep.append(False)
        else:
            keep.append(True)
    mask = np.array(keep)
    return data[mask], [ch for ch, k in zip(ch_names, keep) if k]


def _normalize(data: np.ndarray) -> np.ndarray:
    """채널별 독립 z-score 정규화. (C, T) → (C, T) float32."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((data - mean) / std).astype(np.float32)


def process_psg(
    psg_path: Path,
    out_dir: Path,
    bandpass: bool = False,
    notch_freq: float | None = None,
) -> tuple[str, str, list[dict]]:
    """PSG EDF 파일 하나를 처리해 .pt 파일들을 저장한다.

    Parameters
    ----------
    psg_path:
        *-PSG.edf 파일 경로.
    out_dir:
        datasets/processed/ 루트 경로.
    bandpass:
        True이면 각 신호 타입별 기본 밴드패스 필터를 적용한다.
        EEG/EMG/Resp 모두 0.5–50 Hz (원본 fs 100Hz 기준).
    notch_freq:
        노치 필터 주파수(Hz). None이면 미적용.

    Returns
    -------
    (subject_id, session_id, recordings)
    recordings는 manifest.json sessions[].recordings 항목 리스트.
    """
    import mne

    mne.set_log_level("ERROR")

    subject_id, session_id = _parse_subject(psg_path.stem)
    subj_out = out_dir / subject_id
    subj_out.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_edf(str(psg_path), preload=False)

    # EDF 헤더에서 채널별 native sampling rate 추출
    # n_samps[i] = i번째 채널의 data record당 샘플 수
    # record_length = data record 길이(초)
    ch_native_fs: dict[str, float] = {}
    try:
        edf_info = raw._raw_extras[0]
        record_length: float = edf_info["record_length"]
        n_samps = edf_info["n_samps"]  # (n_channels,)
        ch_native_fs = {
            ch: float(n_samps[i] / record_length)
            for i, ch in enumerate(raw.ch_names)
        }
    except (AttributeError, IndexError, KeyError, TypeError):
        # MNE 버전에 따라 _raw_extras 구조가 다를 수 있음 — fallback
        fallback_fs = float(raw.info["sfreq"])
        print(
            f"    [WARN] _raw_extras 접근 실패, fallback fs={fallback_fs}Hz 사용",
            file=sys.stderr,
        )
        ch_native_fs = {ch: fallback_fs for ch in raw.ch_names}

    # 채널별 native sampling rate 검증 로그
    unique_fs = sorted(set(ch_native_fs.values()))
    if len(unique_fs) > 1:
        print(f"    [INFO] 채널별 sampling rate 상이: {ch_native_fs}", file=sys.stderr)

    # 채널 → signal_type 그룹핑
    groups: dict[str, list[str]] = {}
    for ch in raw.ch_names:
        stype = CHANNEL_MAP.get(ch)
        if stype is not None:
            groups.setdefault(stype, []).append(ch)

    recordings: list[dict] = []
    for stype_key, ch_names in groups.items():
        # 그룹 내 native sampling rate (동일 타입 채널은 같은 fs라고 가정)
        group_fs = ch_native_fs[ch_names[0]]

        # 해당 채널만 pick하여 native rate로 로드
        group_raw = raw.copy().pick(ch_names)
        group_raw.load_data()
        if group_raw.info["sfreq"] != group_fs:
            group_raw.resample(group_fs)

        if bandpass:
            lo = 0.5
            hi = min(50.0, group_fs / 2.0 - 1.0)  # Nyquist 이하로 제한
            if hi <= lo:
                print(
                    f"    [WARN] {stype_key} fs={group_fs}Hz → Nyquist 제한으로 "
                    f"bandpass 불가 (hi={hi:.1f} <= lo={lo}), 필터 스킵",
                    file=sys.stderr,
                )
            else:
                group_raw.filter(l_freq=lo, h_freq=hi, fir_design="firwin")
        if notch_freq is not None and notch_freq < group_fs / 2.0:
            group_raw.notch_filter(freqs=notch_freq)

        data = group_raw.get_data()  # (C, T) float64

        data, ch_names = _quality_gate(data, ch_names)
        if data.shape[0] == 0:
            print(
                f"    [SKIP] {session_id}/{stype_key}: 모든 채널 제외됨",
                file=sys.stderr,
            )
            continue

        # Resampling → TARGET_SR (100 Hz)
        if group_fs != TARGET_SR:
            data = resample_to_target(data, orig_sr=group_fs, target_sr=TARGET_SR)
            print(
                f"    [RESAMPLE] {stype_key}: {group_fs:.0f}Hz → {TARGET_SR:.0f}Hz "
                f"({data.shape[1]} samples)",
                file=sys.stderr,
            )

        tensor = torch.from_numpy(_normalize(data))  # (C, T) float32
        fname = f"{session_id}_{stype_key}.pt"
        torch.save(tensor, subj_out / fname)

        recordings.append(
            {
                "signal_type": SIGNAL_TYPES[stype_key],
                "file": fname,
                "n_channels": tensor.shape[0],
                "sampling_rate": TARGET_SR,
                "n_timesteps": tensor.shape[1],
                "spatial_ids": [
                    CHANNEL_SPATIAL.get(ch, 0) for ch in ch_names
                ],
            }
        )
        print(f"    saved {fname}  shape={tuple(tensor.shape)}  fs={TARGET_SR:.0f}Hz")

    return subject_id, session_id, recordings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sleep-EDF Database를 datasets/processed/ 구조로 변환한다."
    )
    parser.add_argument(
        "--raw",
        default="../../datasets/raw/sleep-edf-database-expanded-1.0.0/sleep-cassette",
        help="sleep-edf 원본 디렉토리",
    )
    parser.add_argument(
        "--out",
        default="../../datasets/processed",
        help="처리 결과를 저장할 루트 디렉토리",
    )
    parser.add_argument(
        "--bandpass",
        action="store_true",
        help="0.5-50Hz 밴드패스 필터 적용",
    )
    parser.add_argument(
        "--notch",
        type=float,
        default=None,
        metavar="FREQ",
        help="노치 필터 주파수 (예: 50 또는 60)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    psg_files = sorted(raw_dir.rglob("*-PSG.edf"))
    print(f"PSG 파일 {len(psg_files)}개 발견\n")

    # subject_id → sessions 누적
    subject_sessions: dict[str, list[dict]] = {}

    for psg_path in psg_files:
        print(f"[{psg_path.name}]")
        try:
            subject_id, session_id, recordings = process_psg(
                psg_path,
                out_dir,
                bandpass=args.bandpass,
                notch_freq=args.notch,
            )
        except Exception as exc:
            print(f"    [ERROR] {exc}", file=sys.stderr)
            continue

        if not recordings:
            continue

        subject_sessions.setdefault(subject_id, []).append(
            {"session_id": session_id, "recordings": recordings}
        )

    # subject별 manifest.json 저장
    global_index: list[dict] = []
    for subject_id, sessions in sorted(subject_sessions.items()):
        manifest = {
            "subject_id": subject_id,
            "source": "sleep-edf",
            "sessions": sessions,
        }
        manifest_path = out_dir / subject_id / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        global_index.append(
            {"subject_id": subject_id, "manifest": f"{subject_id}/manifest.json"}
        )

    # 전체 인덱스 manifest.jsonl 저장
    jsonl_path = out_dir / "manifest.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in global_index:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"\n완료: {len(subject_sessions)}명 처리 → {out_dir}"
        f"\n인덱스: {jsonl_path}"
    )


if __name__ == "__main__":
    main()
