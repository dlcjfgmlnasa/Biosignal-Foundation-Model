# -*- coding: utf-8 -*-
"""MIMIC-III WFDB layout/sub-record .hea 헤더를 스캔해서 채널 가용성 통계.

CO2, AWP, RESP, ICP 같은 sparse modality 가 어느 환자에 있는지 사전 파악용.
.dat 다운로드 전에 *.hea 만 받은 상태에서 호출 가능.

채널 추출 방식:
  WFDB .hea 라인 포맷: <filename> <fmt> <gain> <bits> <baseline> ... <channel_name>
  channel_name 은 마지막 토큰 (단순화). layout / sub-record .hea 둘 다 같은 형식.

사용법:
  python -m scripts.scan_mimic_channels \\
      --root /path/to/mimic3-waveform-vent_need \\
      --channels CO2 RESP ABP ECG_II PLETH AWP
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


# layout .hea 의 신호 라인 매칭: 첫 글자 ~ 으로 시작 (gain 없는 layout)
# 또는 일반 sub-record (filename.dat 으로 시작)
LAYOUT_LINE_RE = re.compile(r"^\s*~?\S+\s+\d+")


def extract_channels_from_hea(path: Path) -> list[str]:
    """단일 .hea 파일에서 채널 이름 추출 (마지막 토큰)."""
    channels: list[str] = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return []
    if not lines:
        return []
    # 첫 줄: 헤더 — record_name n_signals fs n_samples ...
    header = lines[0].split()
    if len(header) < 2 or not header[1].isdigit():
        return []
    n_sig = int(header[1])
    for line in lines[1 : 1 + n_sig]:
        toks = line.strip().split()
        if not toks:
            continue
        # 마지막 토큰이 채널 이름 (보통)
        chan = toks[-1]
        if chan.startswith("#"):
            continue
        channels.append(chan)
    return channels


def is_layout_hea(path: Path) -> bool:
    """파일명이 *_layout.hea 인지."""
    return path.stem.endswith("_layout")


def is_master_hea(path: Path) -> bool:
    """master record .hea (sub-record 들을 참조) 인지 — 첫 신호 라인이 sub-record 이름."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return False
    if len(lines) < 2:
        return False
    # master 의 sub-record 라인은 단순히 <name> <offset_sec> 형식, 채널 정보 없음
    second_line_toks = lines[1].strip().split()
    return len(second_line_toks) == 2 and second_line_toks[1].lstrip("-").isdigit()


def scan_subject_dir(subj_dir: Path) -> set[str]:
    """한 환자 폴더의 모든 .hea 에서 추출된 채널 union."""
    channels: set[str] = set()
    for hea in subj_dir.glob("*.hea"):
        # master .hea 는 스킵 (채널 정보 없음)
        if is_master_hea(hea):
            continue
        # numerics (.hea ends with 'n') 도 스킵
        if hea.stem.endswith("n"):
            continue
        for ch in extract_channels_from_hea(hea):
            channels.add(ch)
    return channels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan MIMIC-III WFDB headers for channel availability"
    )
    parser.add_argument(
        "--root", required=True, type=str,
        help="MIMIC-III WFDB cohort root (e.g., mimic3-waveform-vent_need)",
    )
    parser.add_argument(
        "--channels", nargs="+",
        default=["CO2", "RESP", "ABP", "ART", "II", "PLETH", "AWP", "ICP", "CVP"],
        help="확인할 채널 목록",
    )
    parser.add_argument(
        "--show-top-channels", type=int, default=20,
        help="가장 많이 등장한 채널 N 개 표시",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root not found: {root}")
        return

    # subject_dir 들: p00/p000020 형식
    subj_dirs = []
    for parent in sorted(root.iterdir()):
        if parent.is_dir() and parent.name.startswith("p"):
            for sd in sorted(parent.iterdir()):
                if sd.is_dir() and sd.name.startswith("p"):
                    subj_dirs.append(sd)

    print(f"Found {len(subj_dirs)} subject dirs in {root}")
    if not subj_dirs:
        print("(No subjects. Check that --root is correct.)")
        return

    # 채널별 환자 count + 전체 채널 빈도
    channel_to_subjects: dict[str, set[str]] = defaultdict(set)
    all_channels_count: Counter = Counter()
    n_with_any_layout = 0

    for sd in subj_dirs:
        chans = scan_subject_dir(sd)
        if chans:
            n_with_any_layout += 1
        for ch in chans:
            channel_to_subjects[ch].add(sd.name)
            all_channels_count[ch] += 1

    print(f"\nSubjects with extractable layout: {n_with_any_layout}/{len(subj_dirs)} "
          f"({100 * n_with_any_layout / max(1, len(subj_dirs)):.1f}%)")
    if n_with_any_layout == 0:
        print("\n[WARN] No layout headers found. You likely only downloaded master .hea.")
        print("       Run *_headers.txt wget to fetch all .hea (sub-records included).")
        return

    print("\n=== Channel availability per subject ===")
    print(f"{'Channel':<12}  {'n_subjects':>10}  {'% of total':>10}  {'% of with-layout':>15}")
    for ch in args.channels:
        n_sub = len(channel_to_subjects.get(ch, set()))
        pct_total = 100 * n_sub / max(1, len(subj_dirs))
        pct_layout = 100 * n_sub / max(1, n_with_any_layout)
        print(f"{ch:<12}  {n_sub:>10}  {pct_total:>9.1f}%  {pct_layout:>14.1f}%")

    print(f"\n=== Top {args.show_top_channels} channels in cohort ===")
    for ch, cnt in all_channels_count.most_common(args.show_top_channels):
        print(f"  {ch:<20}  {cnt}")


if __name__ == "__main__":
    main()
