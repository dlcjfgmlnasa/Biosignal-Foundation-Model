"""raw .vital 파일의 NaN 패턴 진단.

- track 별 native_sr (vitaldb 보고 vs 실측)
- NaN gap 의 시간 간격 분포 — 일정 cycle (예: 30s) 마다 발생하는지
- valid → NaN 전이 위치 시각화

사용법:
    python -m scripts.diagnose_nan_pattern \
        --vital /home/coder/workspace/datasets/.../some.vital \
        --tracks Intellivue/ECG_II Intellivue/PLETH

    # 또는 list 에서 random 1개 자동
    python -m scripts.diagnose_nan_pattern \
        --list /home/coder/workspace/updown/kmimic_vital_remaining.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

try:
    import vitaldb
except ImportError:
    raise SystemExit("vitaldb 패키지 필요")


def analyze_track(vf, track_name: str) -> None:
    print(f"\n--- {track_name} ---")
    trk = vf.find_track(track_name)
    if trk is None:
        print("  (track not found)")
        return

    reported_sr = float(trk.srate) if trk.srate else 0.0
    print(f"  reported srate (vitaldb): {reported_sr}")

    # to_numpy 로 데이터 추출 (parser 와 동일 인터벌)
    used_sr = reported_sr if reported_sr > 0 else 500.0
    data = vf.to_numpy(track_name, interval=1.0 / used_sr)
    if data is None or data.size == 0:
        print("  (empty)")
        return
    data = data.flatten()

    n = data.size
    nan_mask = np.isnan(data)
    nan_count = int(nan_mask.sum())
    nan_pct = 100 * nan_count / n
    print(f"  data length:  {n} samples (= {n/used_sr:.1f}s @ {used_sr}Hz)")
    print(f"  NaN samples:  {nan_count} ({nan_pct:.2f}%)")

    if nan_count == 0:
        print("  (no NaN — 깨끗한 데이터)")
        return
    if nan_count == n:
        print("  (all NaN)")
        return

    # 연속 valid 구간 길이 분포
    diff = np.diff(nan_mask.astype(np.int8), prepend=1, append=1)
    valid_starts = np.where(diff == -1)[0]
    valid_ends = np.where(diff == 1)[0]
    valid_lens = valid_ends - valid_starts
    valid_secs = valid_lens / used_sr

    print(f"  연속 valid 구간 수: {len(valid_lens)}")
    if len(valid_lens) > 0:
        print(f"    길이 분포 (초):")
        print(f"      min={valid_secs.min():.2f}  p25={np.percentile(valid_secs, 25):.2f}  "
              f"p50={np.median(valid_secs):.2f}  p75={np.percentile(valid_secs, 75):.2f}  "
              f"max={valid_secs.max():.2f}")
        # 30초 cycle 검증
        n_30 = int(((valid_secs > 28) & (valid_secs < 32)).sum())
        n_60 = int(((valid_secs > 58) & (valid_secs < 62)).sum())
        print(f"      28-32s 구간: {n_30} ({100*n_30/len(valid_secs):.1f}%)")
        print(f"      58-62s 구간: {n_60} ({100*n_60/len(valid_secs):.1f}%)")

    # NaN gap 길이 분포
    nan_starts = np.where(diff == 1)[0][:-1]   # 마지막은 sentinel
    nan_ends = np.where(diff == -1)[0]
    if len(nan_starts) > 0 and len(nan_ends) > 0:
        # Align: gap = (start_i, end_i) where start < end
        gap_lens = []
        for s in nan_starts:
            ends_after = nan_ends[nan_ends > s]
            if len(ends_after) > 0:
                gap_lens.append(int(ends_after[0] - s))
        gap_lens = np.array(gap_lens) / used_sr
        if len(gap_lens) > 0:
            print(f"  NaN gap 길이 분포 (초):")
            print(f"    n={len(gap_lens)}, min={gap_lens.min():.3f}, "
                  f"p50={np.median(gap_lens):.3f}, max={gap_lens.max():.3f}")

    # NaN 전이 시점 분포 — 30초 cycle 검증
    transitions = valid_starts / used_sr  # valid 가 시작되는 시각 (초)
    if len(transitions) > 1:
        intervals = np.diff(transitions)
        print(f"  연속 valid block 시작 간격 (초):")
        print(f"    n={len(intervals)}, min={intervals.min():.2f}, "
              f"p25={np.percentile(intervals, 25):.2f}, "
              f"p50={np.median(intervals):.2f}, p75={np.percentile(intervals, 75):.2f}, "
              f"max={intervals.max():.2f}")
        # 30s 또는 60s 주기성 검증
        n_30 = int(((intervals > 28) & (intervals < 32)).sum())
        n_60 = int(((intervals > 58) & (intervals < 62)).sum())
        print(f"    28-32s 간격: {n_30} ({100*n_30/len(intervals):.1f}%)")
        print(f"    58-62s 간격: {n_60} ({100*n_60/len(intervals):.1f}%)")

    # 첫 10개 valid 구간 상세
    print(f"  첫 10개 valid 구간 [start, end, len(s)]:")
    for i, (s, e, l) in enumerate(zip(valid_starts[:10], valid_ends[:10], valid_secs[:10])):
        print(f"    {i}: start={s/used_sr:7.2f}s  end={e/used_sr:7.2f}s  len={l:6.2f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vital", type=Path, default=None,
                   help="단일 .vital 파일 경로")
    p.add_argument("--list", type=Path, default=None,
                   help="path list 에서 random 1개 자동")
    p.add_argument("--tracks", nargs="+", default=None,
                   help="특정 track 만 (기본: ECG_II, PLETH, ABP, RESP 자동 매칭)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.vital is None:
        if args.list is None:
            raise SystemExit("--vital 또는 --list 중 하나 지정")
        with open(args.list) as f:
            paths = [line.strip() for line in f if line.strip()]
        rng = random.Random(args.seed)
        args.vital = Path(rng.choice(paths))

    print(f"File: {args.vital}")
    vf = vitaldb.VitalFile(str(args.vital))
    available = vf.get_track_names()
    print(f"Available tracks ({len(available)}): {available[:15]}{' ...' if len(available)>15 else ''}")

    if args.tracks:
        targets = args.tracks
    else:
        # ECG, PPG, ABP, RESP 패턴으로 자동 매칭
        candidates_keys = ["ECG_II", "ECG_I", "PLETH", "ABP", "ART", "RESP", "AWP_WAV", "FLOW_WAV"]
        targets = []
        for tn in available:
            for k in candidates_keys:
                if tn.endswith("/" + k) or tn.endswith(k):
                    targets.append(tn)
                    break
            if len(targets) >= 5:
                break

    for tn in targets:
        analyze_track(vf, tn)


if __name__ == "__main__":
    main()
