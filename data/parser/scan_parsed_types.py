# -*- coding:utf-8 -*-
"""파싱된 트리의 signal_type 정수 ↔ 파일명 키를 교차 대조한다.

"CO2/AWP 가 없다" 가 정말 **없는 것**인지, 아니면 **번호(ID) 매핑이 어긋나** 다른
signal_type 으로 읽히는 것인지 가른다.

두 개의 독립된 증거를 쓴다:
  1. manifest.json 의 ``signal_type`` (정수)
  2. `.pt` 파일명에 박힌 ``stype_key`` (문자열) — parser 가
     ``f"{session_id}_{stype_key}_{spatial_id}_seg{i}_{g}.pt"`` 로 저장한다.

정수↔문자열 교차표가 대각이 아니면 매핑 드리프트다. 둘 다 co2/awp 가 0 이면
애초에 파싱되지 않은 것이다.

부수적으로 잡는 것:
  - manifest 없는 top-level 디렉토리 (subject_id 형식 불일치·다른 split 트리)
  - 디스크에는 있는데 manifest 에 없는 `.pt` (orphan)
  - manifest 엔 있는데 파일이 없는 recording (ghost)

사용법:
    python -m data.parser.scan_parsed_types --data-dir <parsed_root>
    python -m data.parser.scan_parsed_types --data-dir <parsed_root> --max-subjects 500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# parser 가 디스크에 쓰는 키 (data/parser/vitaldb.py SIGNAL_TYPES 의 key)
DISK_KEYS = ("ecg", "abp", "ppg", "cvp", "co2", "awp", "pap", "icp", "resp")
# 디스크 spec (파싱 당시 정수). runtime v2 와 6/7/8 이 다르므로 여기서는 디스크 기준.
DISK_SPEC = {"ecg": 0, "abp": 1, "ppg": 2, "cvp": 3, "co2": 4,
             "awp": 5, "pap": 6, "icp": 7, "resp": 8}

_FNAME_RE = re.compile(
    r"_(?P<key>" + "|".join(sorted(DISK_KEYS, key=len, reverse=True)) + r")"
    r"_(?P<spatial>\d+)_seg\d+_\d+\.pt$"
)


def _key_from_filename(name: str) -> str | None:
    m = _FNAME_RE.search(name)
    return m.group("key") if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description="parsed 트리 signal_type 교차 대조")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        print(f"ERROR: 디렉토리 없음: {root}", file=sys.stderr)
        sys.exit(1)

    subj_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if args.max_subjects:
        subj_dirs = subj_dirs[: args.max_subjects]
    print(f"top-level 디렉토리: {len(subj_dirs)}  ({root})\n")

    int_hist: Counter[int] = Counter()          # manifest 의 signal_type 정수
    key_hist_manifest: Counter[str] = Counter() # manifest rec 의 파일명 → 키
    key_hist_disk: Counter[str] = Counter()     # 디스크 .pt 파일명 → 키
    cross: dict[int, Counter[str]] = defaultdict(Counter)  # 정수 → 파일명 키
    subj_with_key: dict[str, set[str]] = defaultdict(set)

    no_manifest: list[str] = []
    orphan = Counter()   # 디스크엔 있는데 manifest 에 없음
    ghost = Counter()    # manifest 엔 있는데 파일 없음
    name_shapes: Counter[str] = Counter()
    # 키별 subject **위치**(정렬 인덱스) — 특정 구간에만 몰려 있으면 batch 효과다.
    key_positions: dict[str, list[int]] = defaultdict(list)

    for idx, sd in enumerate(subj_dirs):
        name_shapes[re.sub(r"\d+", "#", sd.name)] += 1
        mp = sd / "manifest.json"
        disk_files = {p.name for p in sd.glob("*.pt")}
        for f in disk_files:
            k = _key_from_filename(f)
            key_hist_disk[k or "UNPARSED_NAME"] += 1

        if not mp.is_file():
            no_manifest.append(sd.name)
            for f in disk_files:
                orphan[_key_from_filename(f) or "?"] += 1
            continue
        try:
            with open(mp, encoding="utf-8") as fh:
                m = json.load(fh)
        except (json.JSONDecodeError, OSError):
            no_manifest.append(sd.name + " (손상)")
            continue

        listed: set[str] = set()
        for sess in m.get("sessions", []):
            for rec in sess.get("recordings", []):
                st = rec.get("signal_type")
                fn = rec.get("file", "")
                if st is None:
                    continue
                st = int(st)
                k = _key_from_filename(fn)
                int_hist[st] += 1
                if k:
                    key_hist_manifest[k] += 1
                    cross[st][k] += 1
                    if sd.name not in subj_with_key[k]:
                        key_positions[k].append(idx)
                    subj_with_key[k].add(sd.name)
                else:
                    cross[st]["UNPARSED_NAME"] += 1
                listed.add(fn)
                if fn not in disk_files:
                    ghost[k or "?"] += 1
        for f in disk_files - listed:
            orphan[_key_from_filename(f) or "?"] += 1

    n = len(subj_dirs)
    print("── subject 디렉토리 이름 패턴 (숫자→#) ──")
    for shape, c in name_shapes.most_common(5):
        print(f"  {shape:24s} {c}")
    if no_manifest:
        print(f"\n  ⚠ manifest 없는 디렉토리 {len(no_manifest)}개: {no_manifest[:5]}")

    print("\n── manifest signal_type 정수 히스토그램 ──")
    if not int_hist:
        print("  (없음)")
    for st in sorted(int_hist):
        expect = next((k for k, v in DISK_SPEC.items() if v == st), "?")
        print(f"  {st}  n={int_hist[st]:7d}   (disk spec: {expect})")

    print("\n── 정수 ↔ 파일명 키 교차표 (대각이 아니면 매핑 드리프트) ──")
    for st in sorted(cross):
        expect = next((k for k, v in DISK_SPEC.items() if v == st), "?")
        pairs = ", ".join(f"{k}×{c}" for k, c in cross[st].most_common())
        flag = "" if list(cross[st]) == [expect] else "   ← 불일치!"
        print(f"  signal_type={st:2d} (기대 {expect:5s}) : {pairs}{flag}")

    print("\n── 파일명 키 기준 보유 subject 수 (정수 매핑과 무관) ──")
    for k in DISK_KEYS:
        s = len(subj_with_key.get(k, ()))
        print(f"  {k:5s} subjects={s:5d} ({100.0*s/max(1,n):5.1f}%)  files={key_hist_disk.get(k,0)}")

    # 위치 분포: subject 정렬 인덱스를 10 구간으로 나눠 보유율을 본다.
    # 특정 구간에만 몰려 있으면 "여러 번에 나눠 파싱했고 옵션이 달랐다" 는 뜻이다.
    print("\n── 키별 subject 위치 분포 (정렬 인덱스 10분위, 각 칸 = 해당 구간 보유율) ──")
    B = 10
    bounds = [(i * n // B, (i + 1) * n // B) for i in range(B)]
    print("  " + "key".ljust(6) + "".join(f"{i*10:>3d}%" + " " for i in range(B)) + "  첫/끝 subject")
    for k in DISK_KEYS:
        pos = key_positions.get(k, [])
        if not pos:
            continue
        row = "  " + k.ljust(6)
        pset = set(pos)
        for lo, hi in bounds:
            width = max(1, hi - lo)
            got = sum(1 for i in pset if lo <= i < hi)
            row += f"{int(100.0*got/width):>3d}% "
        first = subj_dirs[min(pos)].name
        last = subj_dirs[max(pos)].name
        print(f"{row}  {first} … {last}")
    print("  (co2/awp 만 특정 구간에 몰려 있으면 → 파싱을 나눠 했고 옵션이 달랐다)")

    if orphan:
        print("\n  ⚠ orphan .pt (디스크에 있으나 manifest 에 없음): " +
              ", ".join(f"{k}={v}" for k, v in orphan.most_common()))
    if ghost:
        print("  ⚠ ghost recording (manifest 에 있으나 파일 없음): " +
              ", ".join(f"{k}={v}" for k, v in ghost.most_common()))

    print("\n해석:")
    print("  교차표 대각 + co2/awp 0  → 애초에 파싱 안 됨 (--signal-types 제외)")
    print("  교차표 불일치            → 정수 매핑 드리프트. 파일명 키가 정답.")
    print("  orphan co2/awp 다수      → 파일은 있는데 manifest 누락 → 재-manifest 필요")


if __name__ == "__main__":
    main()
