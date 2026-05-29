# -*- coding: utf-8 -*-
"""연속 파형(waveform)의 총 시간 집계 — FM 논문 헤드라인용 "X만 시간".

전처리 manifest(`manifest_full.jsonl` 우선, 없으면 per-subject `manifest.json`)를
스캔하여, foundation model 논문에서 데이터 규모를 표현할 때 쓰는 두 가지 시간
수치를 정직하게 분리해 보고한다.

  1. **신호 시간 합계 (signal-hours)** — Σ n_timesteps/sr (채널 합산).
     "X hours of physiological waveforms" 형태로 FM 논문이 통상 보고하는 수치.
     ECG·ABP·PPG 가 같은 시각에 동시 측정돼도 각각을 더한다 (channel-hours).

  2. **모니터링 시간 (wall-clock hours)** — session 단위로 모든 modality 의
     시간 구간 [start_sample, start_sample+n_timesteps) 를 union 해 중복을 제거한
     실제 환자 모니터링 시간. 1번보다 작고 더 보수적·엄격한 수치.
     (start_sample 은 한 session=한 source 파일 내에서만 비교 가능하므로
      union 은 (subject, session) 그룹 안에서만 수행한다.)

두 수치 모두 "만 시간(10k h)" 단위 헤드라인을 함께 출력한다.

사용법
------
    # K-MIMIC 사전학습 데이터
    python -m scripts.count_waveform_hours \
        --data-dir datasets/processed/k-mimic-mortal

    # 여러 dataset 합산
    python -m scripts.count_waveform_hours \
        --data-dir datasets/processed/k-mimic-mortal \
                   datasets/processed/vitaldb

    # 특정 signal_type 만 (0=ECG, 1=ABP, ...)
    python -m scripts.count_waveform_hours -d datasets/processed/vitaldb --signals 0 1

    # CSV 저장
    python -m scripts.count_waveform_hours -d ... --csv hours.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Windows 콘솔(cp949)에서 한글·기호 출력 시 UnicodeEncodeError 방지.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(it, **_kwargs):  # type: ignore[no-redef]
        return it


# signal_type int → 이름 (count_tokens.py 와 동기화한 구 spec 테이블).
# 시간 집계는 signal_type remap 여부와 무관하므로 raw signal_type 기준으로 본다.
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
    9: "RESP_Flow",
}

DEFAULT_SR = 100.0  # project-wide resampling target


# ── 누적 통계 ────────────────────────────────────────────────


@dataclass
class Stats:
    """Per-dataset 시간 통계."""

    n_recordings: int = 0
    signal_seconds: float = 0.0        # Σ n_ts/sr (채널 합산)
    wall_clock_seconds: float = 0.0    # session union 후 합
    per_signal_seconds: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    per_signal_count: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    subjects: set = field(default_factory=set)

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)


# ── Manifest 로더 (count_tokens.py 검증된 패턴 재사용) ────────────


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
                mf_path = data_dir / meta["manifest"]
                if mf_path.exists():
                    paths.append(mf_path)
    else:
        paths = sorted(data_dir.glob("**/manifest.json"))

    for mf_path in tqdm(paths, desc="manifests", unit="file", mininterval=0.5):
        try:
            with open(mf_path, encoding="utf-8") as mf:
                yield json.load(mf)
        except (OSError, json.JSONDecodeError):
            continue


def _iter_sessions(subject_manifest: dict):
    """(subject_key, session_id, recordings[]) yield.

    지원 구조:
      (a) flat (VitalDB) — {"session_id", "recordings"} 또는 {"files"}
      (b) K-MIMIC 중첩 — {"subject_id", "sessions": [{"session_id","recordings"},...]}
    """
    sessions = subject_manifest.get("sessions")
    subj_id = subject_manifest.get("subject_id", "")

    if isinstance(sessions, list) and sessions:
        for sess in sessions:
            if not isinstance(sess, dict):
                continue
            sess_id = sess.get("session_id", "")
            subject_key = subj_id if subj_id else sess_id
            yield subject_key, sess_id, sess.get("recordings", [])
        return

    # flat
    session_id = subject_manifest.get("session_id", "")
    subject_key = subj_id if subj_id else session_id
    recs = subject_manifest.get("recordings")
    if recs is None:
        recs = subject_manifest.get("files", [])
    yield subject_key, session_id, recs


def _union_length_seconds(intervals: list[tuple[float, float]]) -> float:
    """[ (start_sec, end_sec), ... ] → 겹침 제거한 총 길이(초)."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:               # 겹침 → 병합
            if e > cur_e:
                cur_e = e
        else:                        # 새 구간
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def scan_dataset(
    data_dir: Path,
    signal_filter: set[int] | None,
    default_sr: float,
) -> Stats:
    stats = Stats()

    full = data_dir / "manifest_full.jsonl"
    if full.exists():
        source = "manifest_full.jsonl"
        iterator = _iter_manifest_full_jsonl(full)
    else:
        source = "per-subject manifest"
        iterator = _iter_per_subject_manifests(data_dir)

    print(f"  Scanning {data_dir}  ({source}) ...", file=sys.stderr)

    for sm in iterator:
        for subject_key, _sess_id, recs in _iter_sessions(sm):
            # session 단위 wall-clock union 용 구간 누적
            session_intervals: list[tuple[float, float]] = []
            for r in recs:
                if not isinstance(r, dict):
                    continue
                n_ch = int(r.get("n_channels", 1))
                n_ts = int(r.get("n_timesteps", r.get("length", 0)))
                if n_ts <= 0:
                    continue
                stype = int(r.get("signal_type", 0))
                if signal_filter is not None and stype not in signal_filter:
                    continue
                sr = r.get("sampling_rate", None)
                srate = float(sr) if sr else default_sr

                # 1) 신호 시간 합계 (채널 합산)
                rec_seconds = n_ts / srate * n_ch
                stats.n_recordings += 1
                stats.signal_seconds += rec_seconds
                stats.per_signal_seconds[stype] += rec_seconds
                stats.per_signal_count[stype] += 1
                if subject_key:
                    stats.subjects.add(subject_key)

                # 2) wall-clock union 용 구간 (한 session 타임라인 내 비교 가능)
                start = int(r.get("start_sample", 0))
                s_sec = start / srate
                e_sec = (start + n_ts) / srate
                session_intervals.append((s_sec, e_sec))

            stats.wall_clock_seconds += _union_length_seconds(session_intervals)

    return stats


# ── 출력 ─────────────────────────────────────────────────────


def _fmt_hours(sec: float) -> str:
    hr = sec / 3600
    man = hr / 10_000  # 만 시간
    if man >= 1:
        return f"{hr:,.0f} h  ({man:,.2f}만 h, {hr / 24:,.0f} d)"
    return f"{hr:,.0f} h  ({hr / 24:,.0f} d)"


def _fmt_si(n: int) -> str:
    if n < 1000:
        return str(n)
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if n >= div:
            return f"{n / div:.2f}{suf}"
    return str(n)


def print_summary(total: Stats, per_dataset: dict[str, Stats]) -> None:
    print()
    print("=" * 76)
    print("  연속 파형 총 시간 (Waveform Hours)")
    print("=" * 76)

    sig_hr = total.signal_seconds / 3600
    wall_hr = total.wall_clock_seconds / 3600

    print(f"\n## 전체 합산")
    print(f"  Datasets          : {len(per_dataset)}")
    print(f"  Subjects (unique) : {_fmt_si(total.n_subjects)}")
    print(f"  Recordings        : {_fmt_si(total.n_recordings)}")
    print()
    print(f"  ① 신호 시간 합계   : {_fmt_hours(total.signal_seconds)}")
    print(f"     (Σ 채널별 길이 - FM 논문 'X hours of waveforms' 표준)")
    print(f"  ② 모니터링 시간    : {_fmt_hours(total.wall_clock_seconds)}")
    print(f"     (session 내 modality 구간 union, 중복 제거 wall-clock)")

    # 헤드라인 추천
    print()
    print("  ── 논문 헤드라인 추천 ──")
    print(f"    \"{sig_hr / 10_000:,.1f}만 시간의 연속 생체신호 파형\"  (① 기준)")
    print(f"    \"{wall_hr / 10_000:,.1f}만 시간의 연속 모니터링\"        (② 기준)")

    # Per-dataset
    if len(per_dataset) > 1:
        print(f"\n## Dataset별 (① 신호 시간 / ② 모니터링 시간)")
        print(f"  {'Dataset':<42s} {'Subj':>7s} {'Recs':>9s} {'① h':>11s} {'② h':>11s}")
        for name, s in per_dataset.items():
            print(
                f"  {name:<42s} {_fmt_si(s.n_subjects):>7s} "
                f"{_fmt_si(s.n_recordings):>9s} "
                f"{s.signal_seconds / 3600:>11,.0f} "
                f"{s.wall_clock_seconds / 3600:>11,.0f}"
            )

    # Signal type breakdown (① 신호 시간 기준)
    print(f"\n## Signal type 별 (① 신호 시간)")
    print(f"  {'Type':<6s} {'Name':<10s} {'Recordings':>12s} {'Hours':>14s} {'%':>7s}")
    for stype in sorted(total.per_signal_seconds.keys()):
        sec = total.per_signal_seconds[stype]
        hrs = sec / 3600
        pct = sec / total.signal_seconds * 100 if total.signal_seconds > 0 else 0.0
        print(
            f"  {stype:<6d} {SIGNAL_NAMES.get(stype, '?'):<10s} "
            f"{_fmt_si(total.per_signal_count[stype]):>12s} "
            f"{hrs:>14,.1f} {pct:>6.1f}%"
        )

    print("\n" + "=" * 76)


def write_csv(csv_path: Path, total: Stats) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal_type", "signal_name", "recordings", "seconds", "hours"])
        for stype in sorted(total.per_signal_seconds.keys()):
            sec = total.per_signal_seconds[stype]
            w.writerow([
                stype,
                SIGNAL_NAMES.get(stype, "?"),
                total.per_signal_count[stype],
                f"{sec:.1f}",
                f"{sec / 3600:.3f}",
            ])
        # 합계 행
        w.writerow([])
        w.writerow(["TOTAL_signal_hours", "", total.n_recordings,
                    f"{total.signal_seconds:.1f}", f"{total.signal_seconds / 3600:.3f}"])
        w.writerow(["TOTAL_wallclock_hours", "", "",
                    f"{total.wall_clock_seconds:.1f}", f"{total.wall_clock_seconds / 3600:.3f}"])
    print(f"  CSV 저장: {csv_path}", file=sys.stderr)


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="연속 파형의 총 시간 집계 (FM 논문 헤드라인용 '만 시간')"
    )
    ap.add_argument(
        "--data-dir", "-d", type=str, nargs="+", required=True,
        help="processed 디렉토리 (manifest_full.jsonl 또는 manifest.jsonl 포함)",
    )
    ap.add_argument(
        "--signals", type=int, nargs="*", default=None,
        help="signal_type 필터 (0=ECG, 1=ABP, ...). 미지정 시 전체.",
    )
    ap.add_argument(
        "--default-sr", type=float, default=DEFAULT_SR,
        help=f"manifest 에 sampling_rate 누락 시 fallback (Hz). 기본 {DEFAULT_SR:g}.",
    )
    ap.add_argument(
        "--csv", type=str, default=None,
        help="signal_type 별 + 합계를 CSV 로 저장.",
    )
    args = ap.parse_args()

    signal_filter = set(args.signals) if args.signals else None

    total = Stats()
    per_dataset: dict[str, Stats] = {}

    for ddir in args.data_dir:
        path = Path(ddir)
        if not path.exists():
            print(f"WARN: {path} 없음, skip", file=sys.stderr)
            continue
        s = scan_dataset(path, signal_filter, args.default_sr)
        per_dataset[str(path)] = s

        total.n_recordings += s.n_recordings
        total.signal_seconds += s.signal_seconds
        total.wall_clock_seconds += s.wall_clock_seconds
        for k, v in s.per_signal_seconds.items():
            total.per_signal_seconds[k] += v
        for k, v in s.per_signal_count.items():
            total.per_signal_count[k] += v
        total.subjects.update(s.subjects)

    if total.n_recordings == 0:
        print("ERROR: 수집된 record 없음. --data-dir 경로 확인 요망.", file=sys.stderr)
        sys.exit(1)

    print_summary(total, per_dataset)

    if args.csv:
        write_csv(Path(args.csv), total)


if __name__ == "__main__":
    main()