# -*- coding: utf-8 -*-
"""K-MIMIC 파싱 결과의 채널별 누적 시간 집계.

`manifest_full.jsonl` (없으면 per-subject `manifest.json`) 을 스트리밍 파싱하여,
(signal_type, spatial_id) 단위로 아래 항목을 누적한다.

  - recording 수
  - subject 수 (unique)
  - 총 sample 수
  - 총 hours (sampling_rate 기반)

채널명은 `data/spatial_map.py` 의 SPATIAL_MAP 으로 역매핑한다.
sampling_rate 는 manifest record 의 `sampling_rate` 필드를 사용하며,
누락 시 `--default-sr` (기본 100Hz) 로 대체한다.

사용법
------
    python -m scripts.count_kmimic_channel_hours \
        --data-dir datasets/processed/k-mimic-mortal

    # signal_type 필터 (ECG, ABP 만)
    python -m scripts.count_kmimic_channel_hours \
        --data-dir datasets/processed/k-mimic-mortal --signals 0 1

    # CSV 출력
    python -m scripts.count_kmimic_channel_hours \
        --data-dir datasets/processed/k-mimic-mortal --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from data.spatial_map import SPATIAL_MAP


SIGNAL_NAMES: dict[int, str] = {
    0: "ECG",
    1: "ABP",
    2: "PPG",
    3: "CVP",
    4: "CO2",
    5: "AWP",
    6: "PAP",
    7: "ICP",
    8: "RESP",
}


def _local_name(signal_type: int, spatial_id: int) -> str:
    mapping = SPATIAL_MAP.get(signal_type, {})
    for name, lid in mapping.items():
        if lid == spatial_id:
            return name
    return f"local_{spatial_id}"


# ── Manifest 로더 ─────────────────────────────────────────────


def _iter_manifest_full_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_per_subject_manifests(data_dir: Path):
    idx = data_dir / "manifest.jsonl"
    if idx.exists():
        with open(idx, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    meta = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mf_path = data_dir / meta["manifest"]
                if not mf_path.exists():
                    continue
                try:
                    with open(mf_path, encoding="utf-8") as mf:
                        yield json.load(mf)
                except (OSError, json.JSONDecodeError):
                    continue
    else:
        for mf_path in sorted(data_dir.glob("**/manifest.json")):
            try:
                with open(mf_path, encoding="utf-8") as mf:
                    yield json.load(mf)
            except (OSError, json.JSONDecodeError):
                continue


def _iter_records(subject_manifest: dict):
    """(signal_type, spatial_id, n_timesteps, sampling_rate, subject_key) yield."""
    sessions = subject_manifest.get("sessions")
    subj_id = subject_manifest.get("subject_id", "")

    # K-MIMIC: sessions[] 중첩
    if isinstance(sessions, list) and sessions:
        for sess in sessions:
            if not isinstance(sess, dict):
                continue
            sess_id = sess.get("session_id", "")
            subject_key = subj_id if subj_id else sess_id
            for r in sess.get("recordings", []):
                if not isinstance(r, dict):
                    continue
                yield from _expand_record(r, subject_key)
        return

    # flat 구조 (VitalDB 등)
    session_id = subject_manifest.get("session_id", "")
    subject_key = subj_id if subj_id else session_id
    recs = subject_manifest.get("recordings") or subject_manifest.get("files", [])
    for r in recs:
        if not isinstance(r, dict):
            continue
        yield from _expand_record(r, subject_key)


def _expand_record(r: dict, subject_key: str):
    """record 1건 → (multi-channel 가능) (stype, spatial_id, n_ts, sr, subj) 시퀀스."""
    stype = int(r.get("signal_type", 0))
    n_ts = int(r.get("n_timesteps", r.get("length", 0)))
    sr = r.get("sampling_rate", None)
    spatial_ids = r.get("spatial_ids")
    if spatial_ids is None:
        # spatial_ids 누락 시 local 0(Unknown) 단일 채널로 가정
        n_ch = int(r.get("n_channels", 1))
        spatial_ids = [0] * n_ch
    for sid in spatial_ids:
        yield stype, int(sid), n_ts, sr, subject_key


# ── 집계 ──────────────────────────────────────────────────────


def scan(
    data_dir: Path,
    signal_filter: set[int] | None,
    default_sr: float,
    max_subjects: int | None = None,
) -> tuple[dict, dict, dict, dict]:
    """
    Returns
    -------
    per_channel_samples : (stype, sid) → total_samples
    per_channel_count   : (stype, sid) → n_records
    per_channel_subjects: (stype, sid) → set(subject_key)
    per_channel_seconds : (stype, sid) → total_seconds  (sampling_rate 반영)
    """
    per_samples: dict[tuple[int, int], int] = defaultdict(int)
    per_count: dict[tuple[int, int], int] = defaultdict(int)
    per_subjects: dict[tuple[int, int], set] = defaultdict(set)
    per_seconds: dict[tuple[int, int], float] = defaultdict(float)

    full = data_dir / "manifest_full.jsonl"
    if full.exists():
        src = "manifest_full.jsonl"
        it = _iter_manifest_full_jsonl(full)
    else:
        src = "per-subject manifest"
        it = _iter_per_subject_manifests(data_dir)

    print(f"  Scanning {data_dir}  ({src})", file=sys.stderr)

    n_subjects_seen = 0
    for sm in it:
        if max_subjects is not None and n_subjects_seen >= max_subjects:
            print(
                f"  [LIMIT] --max-subjects={max_subjects} 도달, 조기 종료",
                file=sys.stderr,
            )
            break
        n_subjects_seen += 1
        for stype, sid, n_ts, sr, subj in _iter_records(sm):
            if n_ts <= 0:
                continue
            if signal_filter is not None and stype not in signal_filter:
                continue
            key = (stype, sid)
            per_samples[key] += n_ts
            per_count[key] += 1
            if subj:
                per_subjects[key].add(subj)
            srate = float(sr) if sr else default_sr
            per_seconds[key] += n_ts / srate

    print(
        f"  → {n_subjects_seen} subject manifest(s) scanned, "
        f"{sum(per_count.values())} recording(s)",
        file=sys.stderr,
    )
    return per_samples, per_count, per_subjects, per_seconds


# ── 출력 ──────────────────────────────────────────────────────


def _fmt_hr(sec: float) -> str:
    hr = sec / 3600
    if hr >= 24:
        return f"{hr:,.1f} h ({hr/24:,.1f} d)"
    return f"{hr:,.1f} h"


def _fmt_si(n: int) -> str:
    if n < 1000:
        return str(n)
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if n >= div:
            return f"{n / div:.2f}{suf}"
    return str(n)


def print_table(
    per_samples: dict,
    per_count: dict,
    per_subjects: dict,
    per_seconds: dict,
) -> None:
    keys = sorted(per_samples.keys())  # (stype, sid)
    total_sec = sum(per_seconds.values())

    print()
    print("=" * 92)
    print("  채널별 누적 시간 (K-MIMIC processed)")
    print("=" * 92)
    print(
        f"  {'sig':<4s} {'sid':<4s} {'channel':<22s} "
        f"{'recs':>10s} {'subjects':>10s} {'samples':>12s} {'duration':>22s}  {'%':>6s}"
    )
    print("  " + "-" * 90)

    # signal_type별 소계도 같이 출력
    by_stype_sec: dict[int, float] = defaultdict(float)
    by_stype_rec: dict[int, int] = defaultdict(int)
    by_stype_subj: dict[int, set] = defaultdict(set)

    for stype, sid in keys:
        sec = per_seconds[(stype, sid)]
        pct = sec / total_sec * 100 if total_sec > 0 else 0.0
        name = _local_name(stype, sid)
        n_subj = len(per_subjects[(stype, sid)])
        print(
            f"  {stype:<4d} {sid:<4d} "
            f"{f'{SIGNAL_NAMES.get(stype, '?')}/{name}':<22s} "
            f"{_fmt_si(per_count[(stype, sid)]):>10s} "
            f"{_fmt_si(n_subj):>10s} "
            f"{_fmt_si(per_samples[(stype, sid)]):>12s} "
            f"{_fmt_hr(sec):>22s}  {pct:>5.1f}%"
        )
        by_stype_sec[stype] += sec
        by_stype_rec[stype] += per_count[(stype, sid)]
        by_stype_subj[stype].update(per_subjects[(stype, sid)])

    print("  " + "-" * 90)
    print("  Signal type 소계")
    print("  " + "-" * 90)
    for stype in sorted(by_stype_sec.keys()):
        sec = by_stype_sec[stype]
        pct = sec / total_sec * 100 if total_sec > 0 else 0.0
        print(
            f"  {stype:<4d} {'':<4s} "
            f"{SIGNAL_NAMES.get(stype, '?'):<22s} "
            f"{_fmt_si(by_stype_rec[stype]):>10s} "
            f"{_fmt_si(len(by_stype_subj[stype])):>10s} "
            f"{'':>12s} "
            f"{_fmt_hr(sec):>22s}  {pct:>5.1f}%"
        )

    print("  " + "-" * 90)
    print(f"  TOTAL duration: {_fmt_hr(total_sec)}")
    print("=" * 92)


def write_csv(
    csv_path: Path,
    per_samples: dict,
    per_count: dict,
    per_subjects: dict,
    per_seconds: dict,
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["signal_type", "signal_name", "spatial_id", "channel_name",
             "recordings", "subjects", "samples", "seconds", "hours"]
        )
        for stype, sid in sorted(per_samples.keys()):
            w.writerow([
                stype,
                SIGNAL_NAMES.get(stype, "?"),
                sid,
                _local_name(stype, sid),
                per_count[(stype, sid)],
                len(per_subjects[(stype, sid)]),
                per_samples[(stype, sid)],
                f"{per_seconds[(stype, sid)]:.1f}",
                f"{per_seconds[(stype, sid)] / 3600:.3f}",
            ])
    print(f"  CSV 저장: {csv_path}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="K-MIMIC processed 데이터의 채널별(spatial_id 단위) 누적 시간 집계"
    )
    ap.add_argument(
        "--data-dir", "-d", type=str, required=True,
        help="processed 디렉토리 (예: datasets/processed/k-mimic-mortal)",
    )
    ap.add_argument(
        "--signals", type=int, nargs="*", default=None,
        help="signal_type 필터 (0=ECG, 1=ABP, ..., 8=RESP). 미지정 시 전체.",
    )
    ap.add_argument(
        "--default-sr", type=float, default=100.0,
        help="manifest에 sampling_rate 누락 시 fallback (Hz). 기본 100.",
    )
    ap.add_argument(
        "--csv", type=str, default=None,
        help="결과를 CSV 파일로 저장.",
    )
    ap.add_argument(
        "--max-subjects", "-n", type=int, default=None,
        help="테스트용: 처음 N개 subject manifest만 스캔하고 종료. "
             "미지정 시 전체 스캔.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} 없음", file=sys.stderr)
        sys.exit(1)

    signal_filter = set(args.signals) if args.signals else None

    per_samples, per_count, per_subjects, per_seconds = scan(
        data_dir, signal_filter, args.default_sr, max_subjects=args.max_subjects
    )

    if not per_samples:
        print("ERROR: 수집된 record 없음.", file=sys.stderr)
        sys.exit(1)

    print_table(per_samples, per_count, per_subjects, per_seconds)

    if args.csv:
        write_csv(Path(args.csv), per_samples, per_count, per_subjects, per_seconds)


if __name__ == "__main__":
    main()
