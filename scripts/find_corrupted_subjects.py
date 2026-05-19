"""손상된 manifest.json 을 가진 subject 디렉토리를 찾아낸다.

두 가지 모드:

1. ``--scan <processed_dir>`` — 데이터 디렉토리를 직접 스캔.
   각 subject 폴더의 manifest.json 을 json.load 시도하고,
   다음 케이스를 손상으로 판정:
     - JSONDecodeError / OSError 발생
     - manifest.json 누락인데 .pt 파일은 존재
     - sessions 가 비어있는데 .pt 파일은 존재
     - sessions[*].recordings 의 file 이 디스크에 부분만 존재

2. ``--log <stderr_file>`` — 파싱 stderr 로그에서 ``[WARN] ... 손상`` 라인 추출.
   subprocess stdout/stderr 를 파일로 리다이렉트해 둔 경우 사용.

사용 예:
    # 디렉토리 스캔 (권장)
    python -m scripts.find_corrupted_subjects \\
        --scan /home/coder/workspace/updown/bio_fm/data/train/k_mimic_full_v2 \\
        --out corrupted_subjects.txt

    # 과거 stderr 로그 파싱
    python -m scripts.find_corrupted_subjects \\
        --log parse.stderr.log \\
        --out corrupted_subjects.txt

출력 형식 (out 파일):
    한 줄에 하나의 subject 디렉토리 절대 경로.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


CORRUPT_LINE_RE = re.compile(
    r"\[WARN\]\s+(?P<path>\S+/manifest\.json)\s+손상"
)


def _classify_subject(subj_dir: Path) -> tuple[str, str] | None:
    """subject 폴더 하나를 검사. 손상이면 (subject_id, reason) 반환, 정상이면 None."""
    manifest_path = subj_dir / "manifest.json"
    pt_files = list(subj_dir.glob("*.pt"))
    has_pt = len(pt_files) > 0

    if not manifest_path.exists():
        if has_pt:
            return subj_dir.name, f"manifest.json missing ({len(pt_files)} .pt orphans)"
        return None  # 빈 폴더, 무시

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return subj_dir.name, f"json parse error: {exc}"

    sessions = manifest.get("sessions", [])
    if not sessions:
        if has_pt:
            return subj_dir.name, f"empty sessions ({len(pt_files)} .pt orphans)"
        return None

    listed_files: set[str] = set()
    for sess in sessions:
        for rec in sess.get("recordings", []):
            f = rec.get("file")
            if f:
                listed_files.add(f)

    if not listed_files:
        if has_pt:
            return subj_dir.name, f"no recordings listed ({len(pt_files)} .pt orphans)"
        return None

    # manifest 에 적혀있지만 디스크에 없는 .pt 가 있는지
    missing_on_disk = [f for f in listed_files if not (subj_dir / f).exists()]
    if missing_on_disk:
        return (
            subj_dir.name,
            f"{len(missing_on_disk)} recordings missing on disk "
            f"(e.g., {missing_on_disk[0]})",
        )

    # 디스크에 있지만 manifest 에 없는 .pt 가 있는지 (manifest 가 잘려나간 케이스)
    on_disk = {p.name for p in pt_files}
    orphans = on_disk - listed_files
    if orphans:
        return (
            subj_dir.name,
            f"{len(orphans)} .pt orphans not in manifest "
            f"(e.g., {next(iter(orphans))})",
        )

    return None


def scan_processed_dir(processed_dir: Path, workers: int) -> list[tuple[Path, str]]:
    subj_dirs = [p for p in processed_dir.iterdir() if p.is_dir()]
    results: list[tuple[Path, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_classify_subject, d): d for d in subj_dirs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="scanning"):
            d = futures[fut]
            res = fut.result()
            if res is not None:
                _, reason = res
                results.append((d, reason))

    return sorted(results, key=lambda x: x[0].name)


def parse_log_file(log_path: Path) -> list[Path]:
    """stderr 로그에서 [WARN] ... 손상 라인의 manifest 경로를 추출하여
    부모 (subject) 디렉토리를 반환.
    """
    seen: set[Path] = set()
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = CORRUPT_LINE_RE.search(line)
            if m:
                manifest_path = Path(m.group("path"))
                seen.add(manifest_path.parent)
    return sorted(seen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scan", type=Path, help="processed 디렉토리 (subject 폴더들 위)")
    src.add_argument("--log", type=Path, help="파싱 stderr 로그 파일")
    p.add_argument("--out", type=Path, required=True, help="손상 subject 경로 리스트 출력 파일")
    p.add_argument("--workers", type=int, default=16, help="--scan 모드 병렬도")
    p.add_argument(
        "--name-only",
        action="store_true",
        help="절대 경로 대신 subject_id (디렉토리 이름) 만 출력",
    )
    args = p.parse_args()

    if args.scan:
        if not args.scan.is_dir():
            print(f"[ERROR] {args.scan} is not a directory", file=sys.stderr)
            sys.exit(1)
        results = scan_processed_dir(args.scan, args.workers)
        with open(args.out, "w", encoding="utf-8") as f:
            for d, reason in results:
                token = d.name if args.name_only else str(d.resolve())
                f.write(f"{token}\t{reason}\n")
        print(
            f"[OK] {len(results)} corrupted subjects → {args.out}",
            file=sys.stderr,
        )
        for d, reason in results[:10]:
            print(f"  {d.name}: {reason}", file=sys.stderr)
        if len(results) > 10:
            print(f"  ... ({len(results) - 10} more)", file=sys.stderr)
    else:
        if not args.log.is_file():
            print(f"[ERROR] {args.log} is not a file", file=sys.stderr)
            sys.exit(1)
        subj_dirs = parse_log_file(args.log)
        with open(args.out, "w", encoding="utf-8") as f:
            for d in subj_dirs:
                token = d.name if args.name_only else str(d)
                f.write(f"{token}\n")
        print(
            f"[OK] {len(subj_dirs)} subjects flagged in log → {args.out}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
