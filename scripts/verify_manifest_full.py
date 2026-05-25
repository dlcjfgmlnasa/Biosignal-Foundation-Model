# -*- coding:utf-8 -*-
"""manifest_full.jsonl 무결성·카운트·토큰 검증 (v2 단일 modality spec).

파싱 종료 직후 sharding 전에, ``manifest_full.jsonl`` 이 학습에 쓰기에
온전한지 확인한다. 두 가지를 한 번에 본다.

1. **카운트 교차검증** — 다음 세 소스의 subject 수가 일치하는지 대조한다.
     (a) manifest_full.jsonl  의 라인 수      (= 학습이 실제로 보는 subject)
     (b) manifest.jsonl       의 인덱스 라인 수 (parser 가 기록한 처리 목록)
     (c) <data_dir>/*/manifest.json 디렉토리 수 (디스크에 실재하는 subject)
   파서 로그의 "manifest_full (N subjects)" 와 "완료 M명" 이 어긋날 때
   (예: 725 vs 109522) manifest_full 이 불완전한 것인지 즉시 판별한다.
   불완전하면 ``scripts.build_manifest_full`` 로 재생성하면 된다.

2. **v2 토큰/시그널 분포** — 각 raw record 를 ``remap_record_v2`` 로 변환한
   *결과* 기준으로 집계한다 (PAP drop, RESP→Impedance(8)/Flow(9) 분리).
   따라서 여기서 나온 토큰 수 == 학습 파이프라인이 실제로 보게 될 양이다.
   ``count_tokens.py`` 는 구 8종 spec 이라 PAP 를 포함하고 RESP 를 안 나눠
   v2 와 어긋난다 — v2 토큰은 이 스크립트를 신뢰할 것.

사용법:
    python -m scripts.verify_manifest_full \\
        --data-dir /home/coder/workspace/updown/bio_fm/data/train/k_mimic_full_v2

    # 디렉토리 스캔(c) 생략하고 manifest_full 만 빠르게 집계
    python -m scripts.verify_manifest_full --data-dir ... --skip-dir-scan
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.spatial_map import (  # noqa: E402
    SIGNAL_TYPE_NAMES,
    remap_record_v2,
)

DEFAULT_PATCH_SIZE = 200
DEFAULT_SR = 100.0
PAP_OLD_ST = 6  # remap_record_v2 가 drop 하는 구 signal_type


def _count_lines(path: Path) -> int:
    """JSONL 라인 수 (빈 줄 제외)."""
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _count_subject_dirs(data_dir: Path) -> int:
    """<data_dir>/*/manifest.json 이 존재하는 subject 디렉토리 수.

    NAS 에서도 os.scandir 단일 listing + per-dir stat 1회로 충분히 빠르다.
    """
    n = 0
    with os.scandir(data_dir) as it:
        for entry in it:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            if (Path(entry.path) / "manifest.json").exists():
                n += 1
    return n


def _iter_records(subject_manifest: dict):
    """subject manifest → (n_channels, n_timesteps, old_signal_type, spatial_ids).

    K-MIMIC 중첩(sessions[].recordings[]) 과 flat(recordings[]/files[]) 모두 지원.
    """
    sessions = subject_manifest.get("sessions")
    if isinstance(sessions, list) and sessions:
        for sess in sessions:
            if not isinstance(sess, dict):
                continue
            for r in sess.get("recordings", []):
                if isinstance(r, dict):
                    yield r
        return
    recs = subject_manifest.get("recordings")
    if recs is None:
        recs = subject_manifest.get("files", [])
    for r in recs:
        if isinstance(r, dict):
            yield r


def _fmt_si(n: int) -> str:
    if n < 1000:
        return str(n)
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if n >= div:
            return f"{n / div:.2f}{suf}"
    return str(n)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", required=True, type=Path,
                   help="processed 디렉토리 (manifest_full.jsonl 포함)")
    p.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
                   help=f"1 token = N samples (default {DEFAULT_PATCH_SIZE}, 2s@100Hz)")
    p.add_argument("--sampling-rate", type=float, default=DEFAULT_SR,
                   help=f"duration 계산용 fallback SR (default {DEFAULT_SR})")
    p.add_argument("--skip-dir-scan", action="store_true",
                   help="subject 디렉토리 스캔(c) 생략 — manifest_full 만 집계")
    args = p.parse_args()

    data_dir: Path = args.data_dir
    full_path = data_dir / "manifest_full.jsonl"
    index_path = data_dir / "manifest.jsonl"

    if not full_path.exists():
        print(f"[ERROR] {full_path} 없음", file=sys.stderr)
        sys.exit(1)

    # ── (1) 카운트 교차검증 ──
    print("=" * 72)
    print("  카운트 교차검증")
    print("=" * 72)
    t0 = time.time()
    n_full = _count_lines(full_path)
    print(f"  (a) manifest_full.jsonl lines : {n_full:>12,}")

    n_index = -1
    if index_path.exists():
        n_index = _count_lines(index_path)
        print(f"  (b) manifest.jsonl  lines     : {n_index:>12,}")
    else:
        print(f"  (b) manifest.jsonl            : (없음)")

    n_dirs = -1
    if not args.skip_dir_scan:
        n_dirs = _count_subject_dirs(data_dir)
        print(f"  (c) */manifest.json dirs      : {n_dirs:>12,}")
    else:
        print(f"  (c) */manifest.json dirs      : (skip)")

    counts = {n for n in (n_full, n_index, n_dirs) if n >= 0}
    incomplete = len(counts) > 1
    if incomplete:
        worst = max(counts)
        print(
            f"\n  [WARN] subject 카운트 불일치 — manifest_full 이 불완전할 수 있음.\n"
            f"         최대 소스 대비 manifest_full 누락: "
            f"{worst - n_full:,} subjects ({n_full}/{worst}).\n"
            f"         → 재생성: python -m scripts.build_manifest_full "
            f"--data-dir {data_dir}"
        )
    else:
        print(f"\n  [OK] 세 소스 카운트 일치 ({n_full:,} subjects)")
    print(f"  (count 단계 {time.time() - t0:.1f}s)")

    # ── (2) v2 토큰/시그널 분포 (manifest_full 본문 집계) ──
    print("\n" + "=" * 72)
    print("  v2 토큰 / signal_type 분포 (remap_record_v2 적용)")
    print("=" * 72)
    t1 = time.time()

    n_subjects = 0
    n_sessions_total = 0
    n_recs_raw = 0
    n_pap_dropped = 0
    st_recs: Counter = Counter()
    st_subjects: dict[int, set] = defaultdict(set)
    st_samples: dict[int, int] = defaultdict(int)
    total_samples = 0

    with open(full_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sm = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_subjects += 1
            subj_id = sm.get("subject_id") or ""
            sessions = sm.get("sessions")
            if isinstance(sessions, list):
                n_sessions_total += len(sessions)

            for r in _iter_records(sm):
                n_recs_raw += 1
                old_st = r.get("signal_type")
                sp = r.get("spatial_ids") or []
                n_t = r.get("n_timesteps", r.get("length", 0))
                n_ch = r.get("n_channels", len(sp) if sp else 1)
                if not isinstance(n_t, int) or n_t <= 0:
                    continue

                remapped = remap_record_v2(old_st, sp)
                if remapped is None:  # PAP drop
                    n_pap_dropped += 1
                    continue
                new_st, _ = remapped
                if new_st not in SIGNAL_TYPE_NAMES:
                    continue

                samples = n_t * (n_ch or 1)
                st_recs[new_st] += 1
                st_subjects[new_st].add(subj_id)
                st_samples[new_st] += samples
                total_samples += samples

    sr = args.sampling_rate
    ps = args.patch_size
    print(f"  subjects (manifest_full)  : {n_subjects:>12,}")
    print(f"  sessions  (sum)           : {n_sessions_total:>12,}")
    print(f"  recordings (raw)          : {n_recs_raw:>12,}")
    print(f"  PAP dropped (v2 제외)     : {n_pap_dropped:>12,}")

    print(f"\n  {'st':>3} {'name':<16} {'recs':>10} {'subj':>9} "
          f"{'dur[h]':>11} {'tokens':>12}")
    for st in sorted(st_recs):
        samp = st_samples[st]
        print(
            f"  {st:>3} {SIGNAL_TYPE_NAMES.get(st, '?'):<16} "
            f"{st_recs[st]:>10,} {len(st_subjects[st]):>9,} "
            f"{samp / sr / 3600:>11,.1f} {_fmt_si(samp // ps):>12}"
        )

    expected_active = set(SIGNAL_TYPE_NAMES) - {PAP_OLD_ST}
    missing = sorted(expected_active - set(st_recs))
    if missing:
        names = [SIGNAL_TYPE_NAMES.get(t, "?") for t in missing]
        print(f"\n  [WARN] 한 번도 안 나온 signal type: {missing} ({names})")

    total_tokens = total_samples // ps
    print(f"\n  total samples (PAP 제외)  : {total_samples:>16,}")
    print(f"  total tokens (patch={ps})  : {total_tokens:>16,} "
          f"({total_tokens / 1e9:.3f}B)")
    print(f"  total duration            : {total_samples / sr / 3600:>16,.1f} h "
          f"({total_samples / sr / 3600 / 24:,.1f} d)")
    print(f"  (token 단계 {time.time() - t1:.1f}s)")

    print("\n" + "=" * 72)
    if incomplete:
        print("  [FAIL] 카운트 불일치 — manifest_full 재생성 후 sharding 권장")
        sys.exit(2)
    print("  [OK] manifest_full 검증 통과")


if __name__ == "__main__":
    main()
