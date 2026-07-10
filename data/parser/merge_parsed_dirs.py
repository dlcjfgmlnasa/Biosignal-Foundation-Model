# -*- coding:utf-8 -*-
"""파싱된 두 트리를 병합한다 — 누락 signal_type 을 사후에 채워 넣기 위한 도구.

배경: `parse_vital_file` 의 resume 은 "세션의 recording 파일이 모두 존재하면
subject 를 통째로 SKIP" 한다 (vitaldb.py 의 Skip 체크). 따라서 이미 파싱된 트리에
`--signal-types 4 5` 로 다시 돌려도 **아무것도 추가되지 않는다**.

해결: 누락 signal 만 **별도 out-dir** 로 파싱한 뒤 이 스크립트로 병합한다.

    # 1) CO2(4)·AWP(5) 만 aux 트리로 파싱
    python -m data.parser.vitaldb --raw <raw_vital_dir> \
        --out <parsed_aux> --signal-types 4 5 --workers 16

    # 2) dry-run 으로 확인 → 실제 병합
    python -m data.parser.merge_parsed_dirs --src <parsed_aux> --dst <parsed_main>
    python -m data.parser.merge_parsed_dirs --src <parsed_aux> --dst <parsed_main> --apply --mode move

    # 3) 검증
    python -m downstream.acute_event.hypoxemia.inspect_cohort --data-dir <parsed_main>

안전장치:
  - 기본 dry-run. 실제 쓰기는 ``--apply`` 필요.
  - dst manifest 는 덮어쓰기 전에 ``manifest.json.bak`` 으로 백업.
  - recording 은 파일명 기준 dedup (parser 의 병합 규칙과 동일).
  - dst 에 같은 파일명이 이미 있으면 건너뛴다 (덮어쓰지 않음).

전제: src·dst 가 **같은 파서·같은 raw 파일**에서 나왔어야 한다. `start_sample` 은
dtstart 원점의 100Hz 절대 인덱스이므로 트리가 달라도 시간축이 일치한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from data.spatial_map import SIGNAL_TYPE_TO_KEY


def _load_manifest(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] manifest 손상: {p} ({exc})", file=sys.stderr)
        return None


def _sessions_by_id(manifest: dict) -> dict[str, dict]:
    return {s.get("session_id"): s for s in manifest.get("sessions", [])}


def merge_subject(
    src_dir: Path,
    dst_dir: Path,
    mode: str,
    apply: bool,
    stats: dict[str, int],
    per_signal: dict[str, int],
) -> None:
    src_m = _load_manifest(src_dir / "manifest.json")
    if src_m is None:
        return

    dst_dir_exists = dst_dir.is_dir()
    dst_m = _load_manifest(dst_dir / "manifest.json")
    if dst_m is None:
        if not dst_dir_exists:
            stats["new_subject"] += 1
        dst_m = {
            "subject_id": src_m.get("subject_id", src_dir.name),
            "source": src_m.get("source", "vitaldb"),
            "sessions": [],
        }

    dst_sessions = _sessions_by_id(dst_m)
    added_here = 0

    for src_sess in src_m.get("sessions", []):
        sid = src_sess.get("session_id")
        dst_sess = dst_sessions.get(sid)
        if dst_sess is None:
            dst_sess = {"session_id": sid, "recordings": []}
            dst_m.setdefault("sessions", []).append(dst_sess)
            dst_sessions[sid] = dst_sess

        existing_files = {r["file"] for r in dst_sess.get("recordings", [])}

        for rec in src_sess.get("recordings", []):
            fname = rec.get("file")
            if not fname:
                continue
            src_pt = src_dir / fname
            if not src_pt.is_file():
                stats["missing_pt"] += 1
                continue
            if fname in existing_files:
                stats["dup_skip"] += 1
                continue
            dst_pt = dst_dir / fname
            if dst_pt.exists():
                # manifest 엔 없는데 파일만 있는 이상 상태 → 덮어쓰지 않는다
                stats["file_exists_skip"] += 1
                continue

            if apply:
                dst_dir.mkdir(parents=True, exist_ok=True)
                if mode == "move":
                    shutil.move(str(src_pt), str(dst_pt))
                elif mode == "copy":
                    shutil.copy2(src_pt, dst_pt)
                else:  # symlink
                    dst_pt.symlink_to(src_pt.resolve())

            dst_sess.setdefault("recordings", []).append(rec)
            existing_files.add(fname)
            added_here += 1
            key = SIGNAL_TYPE_TO_KEY.get(int(rec.get("signal_type", -1)), "?")
            per_signal[key] += 1

    if added_here == 0:
        return
    stats["recordings_added"] += added_here
    stats["subjects_touched"] += 1

    if apply:
        dst_dir.mkdir(parents=True, exist_ok=True)
        mp = dst_dir / "manifest.json"
        if mp.is_file():
            shutil.copy2(mp, dst_dir / "manifest.json.bak")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(dst_m, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="parsed 트리 병합 (누락 signal 채우기)")
    ap.add_argument("--src", required=True, help="추가로 파싱한 트리 (예: co2/awp 전용)")
    ap.add_argument("--dst", required=True, help="기존 트리 (여기에 채워 넣는다)")
    ap.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    ap.add_argument("--apply", action="store_true", help="실제로 쓴다 (기본 dry-run)")
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    if not src_root.is_dir():
        print(f"ERROR: src 없음: {src_root}", file=sys.stderr)
        sys.exit(1)
    if not dst_root.is_dir():
        print(f"ERROR: dst 없음: {dst_root}", file=sys.stderr)
        sys.exit(1)

    subs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    print(f"{'APPLY' if args.apply else 'DRY-RUN'}  mode={args.mode}")
    print(f"  src: {src_root}  ({len(subs)} subjects)")
    print(f"  dst: {dst_root}\n")

    stats: dict[str, int] = defaultdict(int)
    per_signal: dict[str, int] = defaultdict(int)

    try:
        from tqdm import tqdm
        it = tqdm(subs, desc="merging", unit="subj", dynamic_ncols=True)
    except ImportError:
        it = subs

    for sd in it:
        merge_subject(sd, dst_root / sd.name, args.mode, args.apply, stats, per_signal)

    print("\n── 결과 ──")
    print(f"  subjects touched : {stats['subjects_touched']}")
    print(f"  new subjects     : {stats['new_subject']}")
    print(f"  recordings added : {stats['recordings_added']}")
    print(f"  dup (manifest)   : {stats['dup_skip']}")
    print(f"  file exists skip : {stats['file_exists_skip']}")
    print(f"  missing .pt      : {stats['missing_pt']}")
    if per_signal:
        print("  signal 별 추가   : " + ", ".join(
            f"{k}={v}" for k, v in sorted(per_signal.items())))
    if not args.apply:
        print("\n  (dry-run 이었습니다. 실제 병합하려면 --apply)")


if __name__ == "__main__":
    main()
