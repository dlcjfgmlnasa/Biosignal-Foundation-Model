"""손상 subject 리스트에 해당하는 .vital 파일만 골라 재파싱 리스트를 만든다.

`find_corrupted_subjects.py` 의 출력은 결과물 디렉토리 경로 (subject_id) 만 담고
있어서, 그대로 `parse_to_shard.py --from-list` 에 넘길 수 없다. 이 스크립트는
손상 subject_id 와 원본 .vital 전체 리스트를 cross-reference 하여 재파싱 대상
.vital 파일 경로만 추출한다 (`filter_remaining_vital_files.py` 의 반대 방향).

사용법:
    python -m scripts.select_corrupt_vital_files \\
        --corrupted corrupted_subjects.txt \\
        --in-list /home/coder/workspace/updown/kmimic_vital_all.txt \\
        --out-list kmimic_vital_remaining_corrupt.txt \\
        --subject-from-parent 2

입력 형식:
    --corrupted: `find_corrupted_subjects.py --out` 의 출력.
                 `<path>\\t<reason>` (default) 또는 `<subject_id>` (--name-only) 모두 지원.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm


def derive_subject_id(vital_path: Path, subject_from_parent: int) -> str:
    """`data/parser/vitaldb.py:_parse_subject_id` 와 동일 규칙으로 subject_id 만 반환."""
    if subject_from_parent > 0:
        parent_name = vital_path.parents[subject_from_parent - 1].name
        digits = "".join(c for c in parent_name if c.isdigit())
        if not digits:
            digits = parent_name
        return f"VDB_{int(digits):04d}"
    stem = vital_path.stem
    digits = "".join(c for c in stem if c.isdigit())
    if not digits:
        digits = stem
    return f"VDB_{int(digits):04d}"


def load_corrupted_subject_ids(corrupted_path: Path) -> set[str]:
    """corrupted_subjects.txt 에서 subject_id set 추출.

    각 라인은 `<path>\\t<reason>` 또는 단독 `<subject_id>` 형식.
    path 가 들어있으면 마지막 컴포넌트(=디렉토리명=subject_id) 를 사용.
    """
    ids: set[str] = set()
    with open(corrupted_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split("\t", 1)[0]
            # 경로면 마지막 컴포넌트, 아니면 그대로
            subject_id = Path(token).name if ("/" in token or "\\" in token) else token
            ids.add(subject_id)
    return ids


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--corrupted", type=Path, required=True,
        help="find_corrupted_subjects.py 의 --out 결과 파일",
    )
    p.add_argument(
        "--in-list", type=Path, required=True,
        help="원본 .vital 파일 경로 전체 리스트 (예: kmimic_vital_all.txt)",
    )
    p.add_argument(
        "--out-list", type=Path, required=True,
        help="재파싱 대상 .vital 파일 리스트 출력 경로",
    )
    p.add_argument(
        "--subject-from-parent", type=int, default=2,
        help="parse_to_shard 실행 시 사용한 값과 동일해야 함 (K-MIMIC 보통 2)",
    )
    args = p.parse_args()

    print(f"Loading corrupted subject_ids from {args.corrupted} ...")
    corrupted_ids = load_corrupted_subject_ids(args.corrupted)
    print(f"  {len(corrupted_ids)} corrupted subject_ids")
    if not corrupted_ids:
        print("  손상 subject 가 없습니다. 종료.")
        args.out_list.write_text("", encoding="utf-8")
        return

    print(f"Counting lines in {args.in_list} ...")
    with open(args.in_list, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"  {total_lines} lines")

    print(f"Filtering {args.in_list} ...")
    n_total = 0
    n_matched = 0
    n_unparseable = 0
    matched_subjects: set[str] = set()
    with open(args.in_list, encoding="utf-8") as fin, \
            open(args.out_list, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_lines, desc="Filtering", unit="path"):
            path = line.strip()
            if not path:
                continue
            n_total += 1
            try:
                subj = derive_subject_id(Path(path), args.subject_from_parent)
            except Exception:
                n_unparseable += 1
                continue
            if subj in corrupted_ids:
                fout.write(path + "\n")
                n_matched += 1
                matched_subjects.add(subj)

    unmatched = corrupted_ids - matched_subjects
    print(f"\nTotal .vital scanned:      {n_total}")
    print(f"Matched to corrupted:      {n_matched} (across {len(matched_subjects)} subjects)")
    print(f"Unparseable .vital paths:  {n_unparseable}")
    print(f"Wrote:                     {args.out_list}")
    if unmatched:
        print(
            f"\n[WARN] {len(unmatched)} corrupted subjects 가 in-list 에서 발견되지 않음 — "
            "in-list 가 최신인지 또는 --subject-from-parent 값이 맞는지 확인 필요"
        )
        for s in sorted(unmatched)[:10]:
            print(f"  {s}")
        if len(unmatched) > 10:
            print(f"  ... ({len(unmatched) - 10} more)")


if __name__ == "__main__":
    main()