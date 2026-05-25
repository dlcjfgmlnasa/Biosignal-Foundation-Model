# -*- coding:utf-8 -*-
"""K-MIMIC 실제 환자 수 복원 (재파싱 없이).

파싱 시 ``--subject-from-parent`` 로 subject_id 를 **bucket 레벨**로 잡아
(예: ``VITALDB/398/3986/stay/file.vital`` → ``VDB_0398``), manifest 의 고유
subject_id 는 환자가 아니라 bucket 수(예: 725)가 된다. 환자(3986) 정보는
원본 경로에만 있고 processed manifest 에는 없다.

이 스크립트는 **재파싱 없이** 원본 ``.vital`` 경로 리스트와 manifest_full 을
대조해 환자 수를 복원한다.

  - **Source 환자** = 리스트의 고유 환자 디렉토리 수 (= 파싱에 *투입*한 환자).
  - **Used 환자**   = manifest 의 각 session_id 를 원본 경로의 session_id 로
    매핑해, quality filter 를 통과해 *실제 학습에 들어간* 고유 환자 수.

session_id 매칭은 ``data.parser.vitaldb._parse_subject_id`` 의 로직을 그대로
복제(아래 ``_derive_ids``)해, 파서가 만든 session_id 와 정확히 일치시킨다.
한 session_id 가 서로 다른 환자로 매핑되면 collision 으로 카운트한다.

속도: 텍스트 파일 2개만 읽고 in-memory dict 연산 — per-file stat 없음. ~수 초.

사용법:
    python -m scripts.recover_patient_count \\
        --vital-list /home/coder/workspace/updown/kmimic_vital_all.txt \\
        --manifest-full /home/coder/workspace/updown/bio_fm/data/train/k_mimic_full_v2/manifest_full.jsonl

경로 구조가 다르면 --subject-from-parent / --patient-from-parent 로 맞춘다
(둘 다 1-based: N → ``path.parents[N-1]``). 기본은 bucket=3, patient=2
즉 ``VITALDB/<bucket>/<patient>/<stay>/<file>.vital``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it=None, **_kwargs):  # type: ignore[no-redef]
        return it if it is not None else []


def _derive_ids(vital_path: Path, subject_from_parent: int) -> tuple[str, str]:
    """data.parser.vitaldb._parse_subject_id (subject_from_parent>0) 미러.

    파서와 동일한 (subject_id, session_id) 를 만들어 manifest 의 session_id 와
    정확히 join 되게 한다. 로직이 파서에서 바뀌면 여기도 갱신할 것.
    """
    parent_name = vital_path.parents[subject_from_parent - 1].name
    digits = "".join(c for c in parent_name if c.isdigit()) or parent_name
    try:
        subject_id = f"VDB_{int(digits):04d}"
    except ValueError:
        subject_id = f"VDB_{digits}"
    stem_digits = "".join(c for c in vital_path.stem if c.isdigit())
    session_tag = stem_digits if stem_digits else vital_path.stem
    return subject_id, f"{subject_id}_S_{session_tag}"


def _iter_session_ids(manifest_full: Path):
    """manifest_full.jsonl → 모든 session_id yield (중복 포함)."""
    with open(manifest_full, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sm = json.loads(line)
            except json.JSONDecodeError:
                continue
            for sess in sm.get("sessions", []):
                sid = sess.get("session_id")
                if sid:
                    yield sid


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vital-list", required=True, type=Path,
                   help="전체 .vital 경로 리스트 (한 줄당 1개)")
    p.add_argument("--manifest-full", required=True, type=Path,
                   help="processed manifest_full.jsonl")
    p.add_argument("--subject-from-parent", type=int, default=3,
                   help="파싱 시 사용한 값 = bucket 깊이 (1-based). default 3")
    p.add_argument("--patient-from-parent", type=int, default=2,
                   help="진짜 환자 디렉토리 깊이 (1-based). default 2 → parents[1]")
    p.add_argument("--show-examples", type=int, default=5,
                   help="unmatched/collision 예시 출력 개수")
    args = p.parse_args()

    sfp = args.subject_from_parent
    pfp = args.patient_from_parent

    if not args.vital_list.exists():
        print(f"[ERROR] {args.vital_list} 없음", file=sys.stderr)
        sys.exit(1)
    if not args.manifest_full.exists():
        print(f"[ERROR] {args.manifest_full} 없음", file=sys.stderr)
        sys.exit(1)

    # ── (1) Source: 원본 경로 리스트 ──
    print("=" * 72)
    print("  Source (.vital 경로 리스트)")
    print("=" * 72)
    session_to_patient: dict[str, tuple] = {}
    collisions: set[str] = set()
    collision_examples: list[str] = []
    source_patients: set[tuple] = set()
    n_paths = 0
    n_bad_path = 0

    with open(args.vital_list, encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading vital list", unit="path", mininterval=0.5):
            line = line.strip()
            if not line:
                continue
            n_paths += 1
            vp = Path(line)
            try:
                sid, sess = _derive_ids(vp, sfp)
                patient_name = vp.parents[pfp - 1].name
            except IndexError:
                n_bad_path += 1
                continue
            pk = (sid, patient_name)  # bucket + 환자디렉토리 (안전한 고유키)
            source_patients.add(pk)
            prev = session_to_patient.get(sess)
            if prev is None:
                session_to_patient[sess] = pk
            elif prev != pk and sess not in collisions:
                collisions.add(sess)
                if len(collision_examples) < args.show_examples:
                    collision_examples.append(f"{sess}: {prev} vs {pk}")

    print(f"  .vital paths              : {n_paths:>12,}")
    if n_bad_path:
        print(f"  깊이 부족 path (skip)     : {n_bad_path:>12,}")
    print(f"  unique sessions (source)  : {len(session_to_patient):>12,}")
    print(f"  collision sessions        : {len(collisions):>12,}  "
          f"(한 session_id 가 2+ 환자 — 매핑 모호)")
    print(f"  >> SOURCE 환자 수         : {len(source_patients):>12,}  "
          f"(파싱에 투입된 고유 환자)")

    # ── (2) Used: manifest_full 의 session 을 환자로 환원 ──
    print("\n" + "=" * 72)
    print("  Used (manifest_full → 환자 환원)")
    print("=" * 72)
    used_sessions: set[str] = set()
    for sid in tqdm(
        _iter_session_ids(args.manifest_full),
        desc="Reading manifest sessions", unit="sess", mininterval=0.5,
    ):
        used_sessions.add(sid)

    used_patients: set[tuple] = set()
    n_unmatched = 0
    n_ambiguous = 0
    unmatched_examples: list[str] = []
    for sess in used_sessions:
        if sess in collisions:
            n_ambiguous += 1
        pk = session_to_patient.get(sess)
        if pk is None:
            n_unmatched += 1
            if len(unmatched_examples) < args.show_examples:
                unmatched_examples.append(sess)
        else:
            used_patients.add(pk)

    print(f"  unique sessions (manifest): {len(used_sessions):>12,}")
    print(f"  unmatched sessions        : {n_unmatched:>12,}  "
          f"(source 리스트에 대응 경로 없음)")
    print(f"  ambiguous sessions        : {n_ambiguous:>12,}  "
          f"(collision 으로 환자 불확정)")
    print(f"  >> USED 환자 수           : {len(used_patients):>12,}  "
          f"(quality 통과 후 실제 학습 환자)")

    # ── 진단 예시 ──
    if collision_examples:
        print(f"\n  [collision 예시 {len(collision_examples)}]")
        for e in collision_examples:
            print(f"    {e}")
    if unmatched_examples:
        print(f"\n  [unmatched 예시 {len(unmatched_examples)}]")
        for e in unmatched_examples:
            print(f"    {e}")

    print("\n" + "=" * 72)
    if n_unmatched or len(collisions):
        print(f"  [참고] unmatched {n_unmatched} / collision {len(collisions)} 존재 — "
              f"USED 환자 수는 ±오차 가능. 경로 깊이(--*-from-parent) 확인 권장.")
    print(f"  SOURCE 환자 {len(source_patients):,} → USED 환자 {len(used_patients):,} "
          f"(차이 = quality filter 로 통째 탈락한 환자)")


if __name__ == "__main__":
    main()
