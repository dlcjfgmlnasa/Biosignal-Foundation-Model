# -*- coding: utf-8 -*-
"""ICH cohort signal combo diagnostic — fast variant.

MIMIC-III multi-segment record 의 `_layout` segment 1개만 읽어 channel 집합을
파악한다. 신호 load 없음 + ThreadPool 병렬 → 네트워크 마운트에서도 빠름.

사용법:
    python -m scripts.count_ich_signal_combos \
        --waveform-dir /home/coder/workspace/updown/raw/mimic3-waveform-ich \
        --workers 16
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# MIMIC-III 채널명 → signal_type (prepare_data.py 와 동일)
CHANNEL_MAP: dict[str, str] = {
    "II": "ecg", "I": "ecg", "III": "ecg", "V": "ecg", "V5": "ecg",
    "MCL1": "ecg", "MCL": "ecg",
    "ABP": "abp", "ART": "abp",
    "PLETH": "ppg",
    "CO2": "co2", "RESP_CO2": "co2",
    "ICP": "icp",
}


def _channels_from_hea(hea_path: Path) -> set[str]:
    """단일 .hea 파일에서 signal channel 이름 집합 추출."""
    out: set[str] = set()
    try:
        with open(hea_path, encoding="latin-1") as f:
            lines = f.readlines()
    except Exception:
        return out
    # line 0 은 record header line. line 1+ 가 signal entry.
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) >= 9:
            ch_name = tokens[-1]
            stype = CHANNEL_MAP.get(ch_name)
            if stype:
                out.add(stype)
    return out


def _scan_record(master_hea: Path) -> tuple[Path, set[str]]:
    """master .hea → channel 집합. layout segment 1개만 읽음.

    Multi-segment record format:
        line 0: header (record_name/N_seg ...)
        line 1+: segment entries — 그 중 '_layout' 으로 끝나는 segment 가
        전체 channel 정의를 가짐.

    fallback: layout 못 찾으면 master 자체에서 추출.
    """
    record_dir = master_hea.parent
    layout_path: Path | None = None
    try:
        with open(master_hea, encoding="latin-1") as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                first = line.split()[0]
                if first.endswith("_layout"):
                    candidate = record_dir / f"{first}.hea"
                    if candidate.exists():
                        layout_path = candidate
                        break
    except Exception:
        pass

    if layout_path is not None:
        return master_hea, _channels_from_hea(layout_path)
    # Fallback: master 자체가 single-segment record
    return master_hea, _channels_from_hea(master_hea)


def main() -> None:
    parser = argparse.ArgumentParser(description="ICH cohort signal combo counter (fast)")
    parser.add_argument("--waveform-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.waveform_dir)
    if not root.is_dir():
        raise SystemExit(f"ERROR: not a directory: {root}")

    # master .hea: pXX/pXXXXXX/pXXXXXX-YYYY-MM-DD-HH-MM.hea (2-level)
    # rglob 는 네트워크 마운트에서 수만 개 .hea 를 다 훑어 느림 — 구조 명시 glob.
    print(f"Enumerating master .hea under {root} ...", flush=True)
    top_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    master_files: list[Path] = []
    for top in tqdm(top_dirs, desc="walk pXX", unit="dir", dynamic_ncols=True):
        for patient_dir in top.iterdir():
            if not patient_dir.is_dir():
                continue
            for hea in patient_dir.glob("p[0-9]*-*.hea"):
                if hea.stem.endswith("n"):
                    continue
                if "_layout" in hea.stem:
                    continue
                master_files.append(hea)
    master_files.sort()
    print(f"Found {len(master_files)} master .hea files; workers={args.workers}")

    record_combos: list[frozenset[str]] = []
    patient_combos: dict[str, set[str]] = defaultdict(set)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_scan_record, m): m for m in master_files}
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="scan",
            unit="rec",
            dynamic_ncols=True,
        )
        for fut in pbar:
            master, chans = fut.result()
            record_combos.append(frozenset(chans))
            pid = master.stem.split("-", 1)[0]
            patient_combos[pid] |= chans
            pbar.set_postfix(patients=len(patient_combos))
        pbar.close()

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

    n_rec = len(record_combos)
    print("\n" + "=" * 78)
    print(f"{'Required signals':<35} {'records':>10} {'patients':>10} {'rec_rate':>10}")
    print("-" * 78)
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
