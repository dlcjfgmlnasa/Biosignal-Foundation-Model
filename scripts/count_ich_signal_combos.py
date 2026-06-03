# -*- coding: utf-8 -*-
"""ICH cohort signal combo diagnostic.

`mimic3-waveform-ich/` 의 각 master record header(.hea + _layout.hea) 만 읽어서
어떤 channel 조합을 갖고 있는지 집계한다. 실제 signal load 없이 빠르게 수행.

사용법:
    python -m scripts.count_ich_signal_combos \
        --waveform-dir /home/coder/workspace/updown/raw/mimic3-waveform-ich

출력:
  - record-level cohort size: 5ch / 4ch / 3ch / 2ch / ICP-only
  - patient-level cohort size (dedup by p-ID prefix)
  - drop-rate trade-off table
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

# MIMIC-III 채널명 → signal_type (ICH prepare_data.py 와 동일)
CHANNEL_MAP: dict[str, str] = {
    "II": "ecg", "I": "ecg", "III": "ecg", "V": "ecg", "V5": "ecg",
    "MCL1": "ecg", "MCL": "ecg",
    "ABP": "abp", "ART": "abp",
    "PLETH": "ppg",
    "CO2": "co2", "RESP_CO2": "co2",
    "ICP": "icp",
}


def _scan_record_channels(master_hea: Path) -> set[str]:
    """master .hea 의 segment 목록 → 각 segment .hea 읽어 channel union."""
    channels: set[str] = set()
    try:
        with open(master_hea, encoding="latin-1") as f:
            lines = f.readlines()
    except Exception:
        return channels

    record_dir = master_hea.parent
    # line 1+: segment names (or sig channels for non-multi-segment)
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        # multi-segment record entry: "<seg_name> <samples>"
        # ↑ first 가 seg 이름이면 그 seg .hea 도 읽어야 함
        seg_hea = record_dir / f"{first}.hea"
        if seg_hea.exists() and seg_hea != master_hea:
            try:
                with open(seg_hea, encoding="latin-1") as g:
                    seg_lines = g.readlines()
            except Exception:
                continue
            for sl in seg_lines[1:]:
                sl = sl.strip()
                if not sl or sl.startswith("#"):
                    continue
                tokens = sl.split()
                # signal entry: 마지막 토큰이 채널명일 가능성
                # WFDB header 의 signal line: "<file> <format> <gain> <res> <zero> <init> <chksum> <bsize> <desc>"
                if len(tokens) >= 9:
                    ch_name = tokens[-1]
                    stype = CHANNEL_MAP.get(ch_name)
                    if stype:
                        channels.add(stype)
        else:
            # 단일-segment record: 이 line 자체가 signal entry
            tokens = line.split()
            if len(tokens) >= 9:
                ch_name = tokens[-1]
                stype = CHANNEL_MAP.get(ch_name)
                if stype:
                    channels.add(stype)
    return channels


def main() -> None:
    parser = argparse.ArgumentParser(description="ICH cohort signal combo counter")
    parser.add_argument("--waveform-dir", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.waveform_dir)
    if not root.is_dir():
        raise SystemExit(f"ERROR: not a directory: {root}")

    # master .hea 만 (segment .hea 는 제외: layout / numeric / 숫자 segment)
    master_files = sorted(
        p for p in root.rglob("p[0-9]*-*.hea")
        if not p.stem.endswith("n")
        and "_layout" not in p.stem
    )
    print(f"Found {len(master_files)} master .hea files")

    record_combos: list[frozenset[str]] = []
    patient_combos: dict[str, set[str]] = defaultdict(set)

    for i, hea in enumerate(master_files, 1):
        chans = _scan_record_channels(hea)
        record_combos.append(frozenset(chans))
        # patient_id = pXXXXXX (master stem 의 prefix)
        pid = hea.stem.split("-", 1)[0]
        patient_combos[pid] |= chans
        if i % 50 == 0:
            print(f"  scanned {i}/{len(master_files)}")

    print(f"\nUnique patients: {len(patient_combos)}")

    # ── 조합별 record/patient 카운트 ─────────────────────────
    def has_all(chans: set[str], req: list[str]) -> bool:
        return all(s in chans for s in req)

    requirements = [
        ("ICP only",                          ["icp"]),
        ("ABP+ICP (2ch)",                     ["abp", "icp"]),
        ("ABP+ECG+ICP (3ch)",                 ["abp", "ecg", "icp"]),
        ("ABP+ECG+PPG+ICP (4ch)",             ["abp", "ecg", "ppg", "icp"]),
        ("ABP+ECG+PPG+CO2+ICP (5ch)",         ["abp", "ecg", "ppg", "co2", "icp"]),
    ]

    print("\n" + "=" * 78)
    print(f"{'Required signals':<35} {'records':>10} {'patients':>10} {'rec_rate':>10}")
    print("-" * 78)
    n_rec = len(record_combos)
    n_pt = len(patient_combos)
    for label, req in requirements:
        rec_hit = sum(1 for c in record_combos if has_all(set(c), req))
        pt_hit = sum(1 for chans in patient_combos.values() if has_all(chans, req))
        rec_rate = 100.0 * rec_hit / max(1, n_rec)
        print(f"{label:<35} {rec_hit:>10} {pt_hit:>10} {rec_rate:>9.1f}%")
    print("=" * 78)

    # ── Combo 분포 ──────────────────────────────────────────
    print("\nTop combo distribution (record-level):")
    combo_counter = Counter(record_combos)
    for combo, cnt in combo_counter.most_common(15):
        sorted_combo = sorted(combo) if combo else ["<empty>"]
        print(f"  {cnt:>5}  {{{', '.join(sorted_combo)}}}")


if __name__ == "__main__":
    main()
