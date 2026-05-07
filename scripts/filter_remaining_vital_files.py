"""파싱 완료된 .vital 파일을 제외한 remaining list 생성.

`parse_to_shard.py` 의 [SKIP] 체크는 파일당 manifest.json + recording exist 검증을
수행하기 때문에, 이미 대부분 완료된 상황에서 재실행 시 NAS I/O 누적으로 느려진다.
이 스크립트는 처음부터 완료된 (subject_id, session_id) 를 제외한 list 를 만들어
`--from-list` 에 넘기게 한다.

사용법:
    python -m scripts.filter_remaining_vital_files \
        --processed /home/coder/workspace/updown/bio_fm/data/train/k_mimic_full \
        --in-list /home/coder/workspace/updown/kmimic_vital_all.txt \
        --out-list /home/coder/workspace/updown/kmimic_vital_remaining.txt \
        --subject-from-parent 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def derive_session_id(vital_path: Path, subject_from_parent: int) -> tuple[str, str]:
    """`data/parser/vitaldb.py:_parse_subject_id` 와 동일 규칙."""
    if subject_from_parent > 0:
        parent_name = vital_path.parents[subject_from_parent - 1].name
        digits = "".join(c for c in parent_name if c.isdigit())
        if not digits:
            digits = parent_name
        subject_id = f"VDB_{int(digits):04d}"
        stem_digits = "".join(c for c in vital_path.stem if c.isdigit())
        session_tag = stem_digits if stem_digits else vital_path.stem
        session_id = f"{subject_id}_S_{session_tag}"
    else:
        stem = vital_path.stem
        digits = "".join(c for c in stem if c.isdigit())
        if not digits:
            digits = stem
        subject_id = f"VDB_{int(digits):04d}"
        session_id = f"{subject_id}_S0"
    return subject_id, session_id


def collect_completed(processed: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    n_manifest = 0
    n_corrupt = 0
    for manifest_path in processed.rglob("manifest.json"):
        n_manifest += 1
        try:
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            n_corrupt += 1
            continue

        subj_dir = manifest_path.parent
        subject_id = data.get("subject_id") or subj_dir.name

        for sess in data.get("sessions", []):
            session_id = sess.get("session_id")
            recs = sess.get("recordings", [])
            if not session_id or not recs:
                continue
            if all((subj_dir / r["file"]).exists() for r in recs):
                completed.add((subject_id, session_id))
    print(f"  manifest.json found: {n_manifest} (corrupt: {n_corrupt})")
    print(f"  completed sessions:  {len(completed)}")
    return completed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, required=True,
                   help="parse_to_shard 의 --processed-tmp 와 동일")
    p.add_argument("--in-list", type=Path, required=True,
                   help="원본 .vital 파일 경로 list (예: kmimic_vital_all.txt)")
    p.add_argument("--out-list", type=Path, required=True,
                   help="remaining list 출력 경로")
    p.add_argument("--subject-from-parent", type=int, default=2)
    args = p.parse_args()

    print(f"Scanning completed sessions in {args.processed} ...")
    completed = collect_completed(args.processed)

    print(f"Filtering {args.in_list} ...")
    n_total = 0
    n_remaining = 0
    n_unparseable = 0
    with open(args.in_list, encoding="utf-8") as fin, \
            open(args.out_list, "w", encoding="utf-8") as fout:
        for line in fin:
            path = line.strip()
            if not path:
                continue
            n_total += 1
            try:
                subj, sess = derive_session_id(Path(path), args.subject_from_parent)
            except Exception:
                n_unparseable += 1
                fout.write(path + "\n")
                n_remaining += 1
                continue
            if (subj, sess) in completed:
                continue
            fout.write(path + "\n")
            n_remaining += 1

    print(f"Total:       {n_total}")
    print(f"Remaining:   {n_remaining}")
    print(f"Skipped:     {n_total - n_remaining}")
    print(f"Unparseable: {n_unparseable}")
    print(f"Wrote:       {args.out_list}")


if __name__ == "__main__":
    main()
