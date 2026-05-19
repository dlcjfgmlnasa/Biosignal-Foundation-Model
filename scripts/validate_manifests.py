"""Sharding 전 manifest 무결성·spec 적합성 검증.

본 sharding (수 시간~수십 시간) 들어가기 전에, processed/ 디렉토리의
모든 subject manifest.json 을 빠르게 스캔하여 다음 항목을 확인한다.

검증 항목:
  1. signal_type ∈ {0..8} (현 spec, ECG~RESP 9종)
  2. (signal_type, spatial_id) 페어가 ``data/spatial_map.SPATIAL_MAP`` 범위 내
  3. ``n_channels == len(spatial_ids)``
  4. ``sampling_rate == 100`` (CLAUDE.md TARGET_SR)
  5. ``n_timesteps > 0``
  6. ``recordings`` 비어있지 않음

리포트:
  - signal_type 분포 (전체 recording / 전체 duration[h] / unique subject 수)
  - (signal_type, spatial_id) 분포
  - violation 카운트 + 첫 N개 예시
  - 예상 전체 토큰 수 (patch_size=200 가정)

사용법:
    python -m scripts.validate_manifests \\
        --processed /home/coder/workspace/updown/bio_fm/data/train/k_mimic_full_v2 \\
        --workers 32 \\
        --max-examples 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# spec import — repo root 가 cwd 라고 가정
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.spatial_map import SPATIAL_MAP  # noqa: E402


EXPECTED_SR = 100
SIGNAL_TYPE_NAMES = {
    0: "ECG", 1: "ABP", 2: "PPG", 3: "CVP",
    4: "CO2", 5: "AWP", 6: "PAP", 7: "ICP", 8: "RESP",
}


def _scan_subject_dirs(processed: Path) -> list[Path]:
    """processed/ 1 단계 하위 subject 디렉토리의 manifest.json 후보 경로."""
    candidates: list[Path] = []
    with os.scandir(processed) as it:
        for entry in tqdm(it, desc="Scanning subject dirs", unit="dir"):
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            candidates.append(Path(entry.path) / "manifest.json")
    return candidates


def _load_one(manifest_path: Path) -> tuple[str, list[dict]]:
    """manifest 한 개 로드. 손상/누락이면 빈 list + 상태 반환."""
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return "missing", []
    except (json.JSONDecodeError, OSError):
        return "corrupt", []

    subj_dir = manifest_path.parent
    subject_id = data.get("subject_id") or subj_dir.name
    out: list[dict] = []
    sessions = data.get("sessions", [])
    if not sessions:
        return "empty_sessions", []
    for sess in sessions:
        for rec in sess.get("recordings", []):
            out.append({
                "subject_id": subject_id,
                "session_id": sess.get("session_id"),
                "rec": rec,
            })
    return "ok", out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--processed", type=Path, required=True,
                   help="processed 디렉토리 (subject 폴더들 위)")
    p.add_argument("--workers", type=int, default=32,
                   help="manifest 읽기 thread 수 (NAS I/O bound)")
    p.add_argument("--max-examples", type=int, default=10,
                   help="violation 예시 출력 개수")
    p.add_argument("--patch-size", type=int, default=200,
                   help="토큰 수 추정용 (default=200, 100Hz × 2s)")
    args = p.parse_args()

    if not args.processed.is_dir():
        print(f"[ERROR] {args.processed} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {args.processed} ...")
    manifest_paths = _scan_subject_dirs(args.processed)
    print(f"  subject dirs: {len(manifest_paths)}")

    n_ok = n_missing = n_corrupt = n_empty = 0
    all_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_load_one, p) for p in manifest_paths]
        for fut in tqdm(
            as_completed(futures), total=len(futures),
            desc="Reading manifests", unit="file",
        ):
            status, recs = fut.result()
            if status == "ok":
                n_ok += 1
                all_records.extend(recs)
            elif status == "missing":
                n_missing += 1
            elif status == "corrupt":
                n_corrupt += 1
            elif status == "empty_sessions":
                n_empty += 1

    print(f"\n=== Manifest status ===")
    print(f"  ok:             {n_ok}")
    print(f"  missing:        {n_missing}")
    print(f"  corrupt:        {n_corrupt}  ← find_corrupted_subjects.py 로 후처리 필요")
    print(f"  empty_sessions: {n_empty}")

    if not all_records:
        print("\n[ERROR] no recordings to validate")
        sys.exit(1)

    # ── Spec violation 검사 ──
    violations: dict[str, list[str]] = defaultdict(list)
    st_counter: Counter = Counter()
    st_sp_counter: Counter = Counter()
    st_subjects: dict[int, set[str]] = defaultdict(set)
    st_duration_s: dict[int, float] = defaultdict(float)
    total_samples = 0

    for entry in all_records:
        rec = entry["rec"]
        subj = entry["subject_id"]
        sess = entry["session_id"]
        st = rec.get("signal_type")
        sp_list = rec.get("spatial_ids") or []
        n_ch = rec.get("n_channels")
        sr = rec.get("sampling_rate")
        n_t = rec.get("n_timesteps")
        loc = f"{subj}/{sess}/{rec.get('file')}"

        if st not in SPATIAL_MAP:
            if len(violations["signal_type_oor"]) < args.max_examples:
                violations["signal_type_oor"].append(f"{loc}: signal_type={st}")
            continue

        st_counter[st] += 1
        st_subjects[st].add(subj)
        if isinstance(n_t, int) and isinstance(sr, (int, float)) and sr > 0:
            dur = n_t / sr
            st_duration_s[st] += dur
            total_samples += n_t * (n_ch or 1)

        # spatial_id 범위
        valid_local_ids = set(SPATIAL_MAP[st].values())
        for sp in sp_list:
            if sp not in valid_local_ids:
                if len(violations["spatial_id_oor"]) < args.max_examples:
                    violations["spatial_id_oor"].append(
                        f"{loc}: signal_type={st}({SIGNAL_TYPE_NAMES.get(st)}), "
                        f"spatial_id={sp}, valid={sorted(valid_local_ids)}"
                    )
            st_sp_counter[(st, sp)] += 1

        # n_channels ↔ spatial_ids 길이 매칭
        if isinstance(n_ch, int) and n_ch != len(sp_list):
            if len(violations["n_channels_mismatch"]) < args.max_examples:
                violations["n_channels_mismatch"].append(
                    f"{loc}: n_channels={n_ch}, len(spatial_ids)={len(sp_list)}"
                )

        # sampling_rate
        if sr != EXPECTED_SR:
            if len(violations["sampling_rate_unexpected"]) < args.max_examples:
                violations["sampling_rate_unexpected"].append(
                    f"{loc}: sampling_rate={sr} (expected {EXPECTED_SR})"
                )

        # n_timesteps
        if not isinstance(n_t, int) or n_t <= 0:
            if len(violations["n_timesteps_invalid"]) < args.max_examples:
                violations["n_timesteps_invalid"].append(
                    f"{loc}: n_timesteps={n_t}"
                )

    # ── 리포트 ──
    print(f"\n=== Recording stats ===")
    print(f"  total recordings: {len(all_records)}")

    print(f"\n=== Signal type distribution ===")
    print(f"  {'st':>3} {'name':<6} {'recs':>10} {'subjects':>9} {'duration[h]':>13}")
    for st in sorted(st_counter):
        print(
            f"  {st:>3} {SIGNAL_TYPE_NAMES.get(st, '?'):<6} "
            f"{st_counter[st]:>10} {len(st_subjects[st]):>9} "
            f"{st_duration_s[st]/3600:>13.1f}"
        )
    missing_types = sorted(set(SPATIAL_MAP.keys()) - set(st_counter.keys()))
    if missing_types:
        print(f"  [WARN] signal types not seen at all: {missing_types}")

    print(f"\n=== (signal_type, spatial_id) distribution ===")
    print(f"  {'st':>3} {'sp':>3} {'name':<22} {'count':>12}")
    for (st, sp), n in sorted(st_sp_counter.items()):
        name_map = {v: k for k, v in SPATIAL_MAP.get(st, {}).items()}
        sp_name = name_map.get(sp, "?")
        full = f"{SIGNAL_TYPE_NAMES.get(st, '?')}/{sp_name}"
        print(f"  {st:>3} {sp:>3} {full:<22} {n:>12}")

    print(f"\n=== Token estimate (patch_size={args.patch_size}) ===")
    est_tokens = total_samples // args.patch_size
    print(f"  total samples (n_ch * n_t):  {total_samples:,}")
    print(f"  estimated tokens:            {est_tokens:,} ({est_tokens/1e9:.2f}B)")

    print(f"\n=== Violations ===")
    total_violations = sum(len(v) for v in violations.values())
    if total_violations == 0:
        print("  [OK] no spec violations detected")
    else:
        for kind, examples in violations.items():
            print(f"  {kind}: {len(examples)} examples (cap={args.max_examples})")
            for e in examples:
                print(f"    {e}")

    # exit code: 0 if clean, 2 if violations
    if total_violations > 0:
        print(
            f"\n[FAIL] {total_violations} violations detected — sharding 전 수정 필요",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"\n[OK] spec validation passed — sharding 진행 가능")


if __name__ == "__main__":
    main()
