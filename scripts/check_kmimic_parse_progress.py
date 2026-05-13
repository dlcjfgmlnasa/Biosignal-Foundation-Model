# -*- coding: utf-8 -*-
"""K-MIMIC 파싱 진척률 추적.

각 subject manifest 의 ``sessions[]`` 수 = 처리된 ``.vital`` 파일 수.
이것을 전체 .vital 파일 목록(`--vital-list`) 또는 raw 디렉토리 scan
(`--vital-root`) 와 대조하여 file-level 진척률을 산출한다.

bucket 별 breakdown 도 함께 출력하여 어느 bucket 이 끝났고 어느 bucket 이
미처리인지 한눈에 본다.

사용법
------
    # vital list 파일 사용 (권장 — find 캐시)
    python -m scripts.check_kmimic_parse_progress \
        --processed-dir ../updown/bio_fm/data/train/k_mimic_full_v2/ \
        --vital-list /home/coder/workspace/updown/kmimic_vital_all.txt

    # raw 디렉토리 직접 scan (캐시 없을 때)
    python -m scripts.check_kmimic_parse_progress \
        --processed-dir ../updown/bio_fm/data/train/k_mimic_full_v2/ \
        --vital-root /path/to/K-MIMIC-MORTAL/1.0.0/VITALDB/

    # 일부만 빠르게 확인 (manifest 처음 N개)
    python -m scripts.check_kmimic_parse_progress \
        --processed-dir ../updown/bio_fm/data/train/k_mimic_full_v2/ \
        --vital-list /home/coder/workspace/updown/kmimic_vital_all.txt \
        -n 200

    # CSV (bucket 별 진척) 저장
    python -m scripts.check_kmimic_parse_progress ... --csv progress.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(it, **_kwargs):  # type: ignore[no-redef]
        return it


# ── Vital 파일 목록 로더 ──────────────────────────────────────


_BUCKET_RE = re.compile(r"VITALDB/(\d+)/")


def _extract_bucket(path_str: str) -> str:
    """경로에서 bucket(hadm_id, 3-4자리) 추출. 실패 시 'unknown'."""
    m = _BUCKET_RE.search(path_str)
    if m:
        return m.group(1)
    # fallback: VITALDB/ 토큰 없으면 path 의 상위 2단계 중 숫자 디렉토리
    parts = Path(path_str).parts
    for p in parts:
        if p.isdigit():
            return p
    return "unknown"


def _count_lines(path: Path) -> int:
    """파일 라인 수 (progress bar total 용)."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def load_vital_list(list_path: Path) -> tuple[int, dict[str, int]]:
    """vital list 파일 → (total_count, bucket_count_dict)."""
    bucket_total: dict[str, int] = defaultdict(int)
    n = 0
    total_lines = _count_lines(list_path)
    with open(list_path, encoding="utf-8") as f:
        for line in tqdm(
            f, total=total_lines, desc="vital list", unit="line", mininterval=0.5,
        ):
            line = line.strip()
            if not line:
                continue
            n += 1
            bucket_total[_extract_bucket(line)] += 1
    return n, dict(bucket_total)


def scan_vital_root(root: Path) -> tuple[int, dict[str, int]]:
    """raw .vital 디렉토리 walk → (total_count, bucket_count_dict)."""
    bucket_total: dict[str, int] = defaultdict(int)
    n = 0
    for vp in tqdm(
        root.rglob("*.vital"), desc="rglob *.vital", unit="file", mininterval=0.5,
    ):
        n += 1
        bucket_total[_extract_bucket(str(vp))] += 1
    return n, dict(bucket_total)


# ── 처리된 subject manifest 로더 ────────────────────────────────


def _enumerate_manifest_paths(processed_dir: Path) -> list[Path]:
    """스캔 대상 manifest 경로 목록 (tqdm total 용)."""
    idx = processed_dir / "manifest.jsonl"
    if idx.exists():
        paths: list[Path] = []
        with open(idx, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    meta = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mf_path = processed_dir / meta["manifest"]
                if mf_path.exists():
                    paths.append(mf_path)
        return paths
    return sorted(processed_dir.glob("**/manifest.json"))


def iter_processed_manifests(processed_dir: Path, limit: int | None = None):
    """processed_dir 의 subject manifest 를 yield (tqdm 진척표시 포함)."""
    paths = _enumerate_manifest_paths(processed_dir)
    if limit is not None:
        paths = paths[:limit]
    for mf_path in tqdm(
        paths, desc="manifests", unit="file", mininterval=0.5,
    ):
        try:
            with open(mf_path, encoding="utf-8") as mf:
                yield json.load(mf), mf_path
        except (OSError, json.JSONDecodeError):
            continue


def aggregate_processed(
    processed_dir: Path, limit: int | None = None,
) -> tuple[int, int, dict[str, int]]:
    """
    Returns
    -------
    n_subjects, n_sessions_total, bucket_session_count
        bucket_session_count: bucket(추정) → 처리 session 수
            session_id 는 ``VDB_<subject>_S_<stem>`` 형식 — bucket 정보 없음.
            대신 manifest 파일의 부모 디렉토리에서 추정하지만, 신뢰도 낮음.
            (.vital 의 bucket 정보는 session_id 에 보존되지 않음)
    """
    n_subjects = 0
    n_sessions = 0
    bucket_sess: dict[str, int] = defaultdict(int)

    for sm, _mf_path in iter_processed_manifests(processed_dir, limit=limit):
        n_subjects += 1
        sessions = sm.get("sessions", [])
        if not isinstance(sessions, list):
            continue
        n_sessions += len(sessions)
        # bucket 추정 불가 (session_id 에 bucket 없음) — 일단 unknown
        bucket_sess["unknown"] += len(sessions)

    return n_subjects, n_sessions, dict(bucket_sess)


# ── 출력 ───────────────────────────────────────────────────────


def _fmt_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{num / denom * 100:5.2f}%"


def print_report(
    total_vital: int,
    bucket_total: dict[str, int],
    n_subjects: int,
    n_sessions: int,
    bucket_sess: dict[str, int],
) -> None:
    print()
    print("=" * 78)
    print("  K-MIMIC 파싱 진척률 (file-level)")
    print("=" * 78)
    print(f"  Target .vital files : {total_vital:>12,}")
    print(f"  Processed sessions  : {n_sessions:>12,}  ← processed .vital 수와 동일")
    print(f"  Subject manifests   : {n_subjects:>12,}")
    print()
    print(f"  Progress (sessions/target) : {_fmt_pct(n_sessions, total_vital)}")
    print(f"  Remaining .vital files     : {max(total_vital - n_sessions, 0):>12,}")
    print()

    # Bucket 별 (target side 만 — processed side bucket 추정 불가)
    if bucket_total:
        print("## Target bucket 분포")
        print(f"  {'bucket':<10s} {'count':>10s} {'%':>8s}")
        for bk in sorted(bucket_total.keys(), key=lambda x: (-bucket_total[x], x))[:20]:
            cnt = bucket_total[bk]
            print(f"  {bk:<10s} {cnt:>10,} {cnt / total_vital * 100:>7.2f}%")
        if len(bucket_total) > 20:
            print(f"  ... (+{len(bucket_total) - 20} 개 bucket)")

    print()
    print("## 평균/추정")
    if n_subjects > 0:
        print(
            f"  Sessions / subject (현재까지) : {n_sessions / n_subjects:>10.2f}"
        )
        # 같은 subject 의 .vital 가 미처리로 남아있을 수 있어, 이 비율은 lower bound
    if n_sessions > 0 and total_vital > n_sessions:
        scale = total_vital / n_sessions
        print(
            f"  최종 추정 배율 (target/processed): ×{scale:.2f}"
        )
        print(
            f"    → 같은 subject 의 미처리 .vital 가 추가되면 recording/duration 도"
            f" 약 ×{scale:.2f} 증가 예상 (subject 수는 그보다 적게 증가)"
        )

    print("=" * 78)


def write_csv(
    csv_path: Path,
    bucket_total: dict[str, int],
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "target_vital_count"])
        for bk in sorted(bucket_total.keys(), key=lambda x: (-bucket_total[x], x)):
            w.writerow([bk, bucket_total[bk]])
    print(f"  CSV 저장: {csv_path}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="K-MIMIC 파싱 진척률 추적 (processed sessions vs target .vital 수)"
    )
    ap.add_argument(
        "--processed-dir", "-d", type=str, required=True,
        help="처리된 디렉토리 (subject manifest 가 있는 곳)",
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--vital-list", type=str, default=None,
        help="전체 .vital 경로 list 파일 (한 줄에 하나, 예: kmimic_vital_all.txt)",
    )
    src.add_argument(
        "--vital-root", type=str, default=None,
        help="raw .vital 디렉토리 (rglob 으로 스캔, list 캐시 없을 때)",
    )

    ap.add_argument(
        "--max-subjects", "-n", type=int, default=None,
        help="테스트용: 처음 N개 subject manifest 만 스캔.",
    )
    ap.add_argument(
        "--csv", type=str, default=None,
        help="bucket 분포를 CSV 로 저장.",
    )
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        print(f"ERROR: {processed_dir} 없음", file=sys.stderr)
        sys.exit(1)

    # Target 로드
    if args.vital_list:
        list_path = Path(args.vital_list)
        if not list_path.exists():
            print(f"ERROR: {list_path} 없음", file=sys.stderr)
            sys.exit(1)
        print(f"  Loading vital list: {list_path}", file=sys.stderr)
        total_vital, bucket_total = load_vital_list(list_path)
    else:
        vital_root = Path(args.vital_root)
        if not vital_root.exists():
            print(f"ERROR: {vital_root} 없음", file=sys.stderr)
            sys.exit(1)
        print(f"  Scanning vital root: {vital_root} (rglob *.vital)", file=sys.stderr)
        total_vital, bucket_total = scan_vital_root(vital_root)

    if total_vital == 0:
        print("ERROR: target .vital 파일 0건", file=sys.stderr)
        sys.exit(1)

    # Processed 로드
    print(f"  Scanning processed manifests: {processed_dir}", file=sys.stderr)
    n_subjects, n_sessions, bucket_sess = aggregate_processed(
        processed_dir, limit=args.max_subjects,
    )

    print_report(total_vital, bucket_total, n_subjects, n_sessions, bucket_sess)

    if args.csv:
        write_csv(Path(args.csv), bucket_total)


if __name__ == "__main__":
    main()
