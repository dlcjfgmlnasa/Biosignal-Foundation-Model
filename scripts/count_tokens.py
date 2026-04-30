# -*- coding:utf-8 -*-
"""전처리된 데이터셋의 총 토큰/샘플/용량 집계.

Foundation model의 "token" = patch (기본 patch_size=200 samples @ 100Hz → 2s).
manifest(`manifest_full.jsonl` 우선, 없으면 개별 `manifest.json` fallback)를
스캔하여:
  - 총 recording / subject / samples / tokens
  - signal_type별 breakdown
  - 디스크 footprint 추정
  - LLM 스케일 대비 비교
를 출력한다.

사용법:
    # 단일 dataset
    python -m scripts.count_tokens --data-dir datasets/processed/vitaldb

    # 여러 dataset 합산
    python -m scripts.count_tokens \\
        --data-dir datasets/processed/vitaldb \\
                   datasets/processed/k-mimic-mortal \\
                   datasets/processed/mimic3_waveform

    # Patch size 변경 (e.g., 100 = 1s @ 100Hz)
    python -m scripts.count_tokens --data-dir ... --patch-size 100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# (name, signal_type int) — 코드베이스의 SIGNAL_TYPES와 동기화
SIGNAL_NAMES: dict[int, str] = {
    0: "ECG",
    1: "ABP",
    2: "PPG",
    3: "CVP",
    5: "AWP",
    6: "PAP",
    7: "ICP",
}

DEFAULT_PATCH_SIZE = 200
DEFAULT_SR = 100.0  # project-wide resampling target


@dataclass
class Stats:
    """Per-dataset 누적 통계."""

    n_recordings: int = 0
    n_subjects: int = 0  # unique session_id (subject-level)
    total_samples: int = 0  # sum of n_timesteps across all channels
    total_bytes: int = 0  # estimated disk footprint (float32)
    per_signal_samples: dict[int, int] | None = None  # signal_type → samples
    per_signal_count: dict[int, int] | None = None  # signal_type → n recordings
    missing_files: int = 0
    subjects: set | None = None  # internal: session_id accumulator

    def __post_init__(self) -> None:
        if self.per_signal_samples is None:
            self.per_signal_samples = defaultdict(int)
        if self.per_signal_count is None:
            self.per_signal_count = defaultdict(int)
        if self.subjects is None:
            self.subjects = set()

    def add_record(
        self,
        n_channels: int,
        n_timesteps: int,
        signal_type: int,
        session_id: str = "",
    ) -> None:
        """단일 manifest entry 누적."""
        # n_timesteps가 channel 단위인 경우 vs total samples인 경우 판별:
        # 관례상 manifest의 n_timesteps는 채널당 길이 → 전체 샘플은 × n_channels
        total_channel_samples = n_timesteps * n_channels
        self.n_recordings += 1
        self.total_samples += total_channel_samples
        self.total_bytes += total_channel_samples * 4  # float32
        self.per_signal_samples[signal_type] += total_channel_samples
        self.per_signal_count[signal_type] += 1
        if session_id:
            self.subjects.add(session_id)

    def finalize(self) -> None:
        self.n_subjects = len(self.subjects)


# ── Manifest 로더 ────────────────────────────────────────────


def _iter_manifest_full_jsonl(path: Path):
    """manifest_full.jsonl을 스트리밍 파싱. 각 line = 1 subject's manifest.json 내용."""
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
    """`manifest.jsonl` 인덱스 또는 glob fallback으로 per-subject manifest 로드."""
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


def _extract_records(subject_manifest: dict):
    """Subject manifest에서 개별 recording 엔트리 추출.

    지원 구조:
      (a) flat — {"session_id": ..., "recordings": [...]}, 또는 {"files": [...]}
      (b) K-MIMIC 중첩 — {"subject_id": ..., "sessions": [
              {"session_id": ..., "recordings": [...]}, ...]}
    """
    # (b) K-MIMIC: sessions[] 중첩 구조
    sessions = subject_manifest.get("sessions")
    if isinstance(sessions, list) and sessions:
        subj_id = subject_manifest.get("subject_id", "")
        for sess in sessions:
            if not isinstance(sess, dict):
                continue
            sess_id = sess.get("session_id", subj_id)
            for r in sess.get("recordings", []):
                if not isinstance(r, dict):
                    continue
                yield (
                    int(r.get("n_channels", 1)),
                    int(r.get("n_timesteps", r.get("length", 0))),
                    int(r.get("signal_type", 0)),
                    r.get("session_id", sess_id),
                )
        return

    # (a) flat: recordings 또는 files 키
    session_id = subject_manifest.get("session_id", "")
    recs = subject_manifest.get("recordings")
    if recs is None:
        recs = subject_manifest.get("files", [])
    for r in recs:
        if not isinstance(r, dict):
            continue
        yield (
            int(r.get("n_channels", 1)),
            int(r.get("n_timesteps", r.get("length", 0))),
            int(r.get("signal_type", 0)),
            r.get("session_id", session_id),
        )


def scan_dataset(data_dir: Path, signal_filter: set[int] | None = None) -> Stats:
    """하나의 processed directory를 스캔하여 Stats 집계."""
    stats = Stats()

    full_jsonl = data_dir / "manifest_full.jsonl"
    if full_jsonl.exists():
        source = "manifest_full.jsonl"
        iterator = _iter_manifest_full_jsonl(full_jsonl)
    else:
        source = "manifest.jsonl 또는 glob"
        iterator = _iter_per_subject_manifests(data_dir)

    print(f"  Scanning {data_dir} ({source}) ...")
    for sm in iterator:
        for n_ch, n_ts, stype, sess in _extract_records(sm):
            if n_ts <= 0:
                continue
            if signal_filter is not None and stype not in signal_filter:
                continue
            stats.add_record(n_ch, n_ts, stype, sess)

    stats.finalize()
    return stats


# ── 출력 ─────────────────────────────────────────────────────


def _fmt_si(n: int, unit: str = "") -> str:
    """숫자 → 사람 친화 SI 표기 (1234567 → 1.23M)."""
    if n < 1000:
        return f"{n}{unit}"
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if n >= div:
            return f"{n / div:.2f}{suf}{unit}"
    return f"{n}{unit}"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for div, suf in [
        (1024**5, "PB"),
        (1024**4, "TB"),
        (1024**3, "GB"),
        (1024**2, "MB"),
        (1024**1, "KB"),
    ]:
        if n >= div:
            return f"{n / div:.2f} {suf}"
    return f"{n} B"


def print_summary(
    total: Stats,
    per_dataset: dict[str, Stats],
    patch_size: int,
    sampling_rate: float,
) -> None:
    print()
    print("=" * 72)
    print(f"  Token Count Summary (patch_size={patch_size}, SR={sampling_rate} Hz)")
    print("=" * 72)

    sec_per_token = patch_size / sampling_rate
    total_tokens = total.total_samples // patch_size
    total_duration_sec = total.total_samples / sampling_rate
    total_duration_hr = total_duration_sec / 3600

    print(f"\n## 전체 합산")
    print(f"  Datasets         : {len(per_dataset)}")
    print(f"  Subjects (unique): {_fmt_si(total.n_subjects)}")
    print(f"  Recordings       : {_fmt_si(total.n_recordings)}")
    print(f"  Total samples    : {_fmt_si(total.total_samples)}")
    print(
        f"  Total duration   : "
        f"{total_duration_sec:,.0f} sec "
        f"({total_duration_hr:,.1f} hr, {total_duration_hr/24:,.1f} days)"
    )
    print(f"  Raw disk (float32): {_fmt_bytes(total.total_bytes)}")
    print(
        f"  Tokens (patch={patch_size}, {sec_per_token:.1f}s/token): "
        f"{_fmt_si(total_tokens)}"
    )
    if total.missing_files:
        print(f"  Missing files    : {total.missing_files}")

    # Per-dataset
    if len(per_dataset) > 1:
        print(f"\n## Dataset별")
        print(
            f"  {'Dataset':<40s} {'Subj':>8s} {'Recs':>10s} "
            f"{'Samples':>12s} {'Tokens':>12s}"
        )
        for name, s in per_dataset.items():
            toks = s.total_samples // patch_size
            print(
                f"  {name:<40s} {_fmt_si(s.n_subjects):>8s} "
                f"{_fmt_si(s.n_recordings):>10s} {_fmt_si(s.total_samples):>12s} "
                f"{_fmt_si(toks):>12s}"
            )

    # Signal type breakdown
    print(f"\n## Signal type 별")
    print(
        f"  {'Type':<6s} {'Name':<6s} {'Recordings':>12s} "
        f"{'Samples':>15s} {'Tokens':>14s} {'Hours':>10s}"
    )
    for stype in sorted(total.per_signal_samples.keys()):
        n_rec = total.per_signal_count[stype]
        n_sam = total.per_signal_samples[stype]
        n_tok = n_sam // patch_size
        hrs = n_sam / sampling_rate / 3600
        print(
            f"  {stype:<6d} {SIGNAL_NAMES.get(stype, '?'):<6s} "
            f"{_fmt_si(n_rec):>12s} {_fmt_si(n_sam):>15s} "
            f"{_fmt_si(n_tok):>14s} {hrs:>10,.1f}"
        )

    # LLM 스케일 비교
    print(f"\n## LLM 학습 스케일 대비")
    llm_refs = [
        ("BERT (Wiki+BookCorpus)", 3.3e9),
        ("GPT-2 (WebText)", 40e9),
        ("GPT-3", 300e9),
        ("Llama 2 (pretrain)", 2e12),
        ("Llama 3 (pretrain)", 15e12),
    ]
    print(f"  {'Reference':<28s} {'Tokens':>12s} {'Our %':>10s}")
    for name, ref in llm_refs:
        pct = total_tokens / ref * 100
        ref_str = _fmt_si(int(ref))
        pct_str = f"{pct:.4f}%" if pct < 1 else f"{pct:.2f}%"
        print(f"  {name:<28s} {ref_str:>12s} {pct_str:>10s}")

    # Memory estimates for manifest
    print(f"\n## Manifest 메모리 footprint 추정")
    bytes_per_entry_python = 500  # dataclass overhead
    bytes_per_entry_arrow = 60  # columnar + mmap
    py_mem = total.n_recordings * bytes_per_entry_python
    arrow_mem = total.n_recordings * bytes_per_entry_arrow
    print(
        f"  Python list[RecordingManifest] : {_fmt_bytes(py_mem):>10s}  "
        f"(× num_workers = fork 복제 시)"
    )
    print(
        f"  Arrow mmap (columnar, shared)  : {_fmt_bytes(arrow_mem):>10s}  "
        f"(전 worker 공유)"
    )

    print("\n" + "=" * 72)


# ── CLI ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="전처리된 데이터셋의 토큰/샘플/용량 집계"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str, nargs="+", required=True,
        help="전처리 완료된 디렉토리 (manifest_full.jsonl 또는 manifest.jsonl 포함)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
        help=f"Patch (= 1 token) 샘플 수. 기본 {DEFAULT_PATCH_SIZE} "
             f"({DEFAULT_PATCH_SIZE / DEFAULT_SR:.1f}s @ {DEFAULT_SR:g}Hz)",
    )
    parser.add_argument(
        "--sampling-rate", type=float, default=DEFAULT_SR,
        help=f"공통 샘플링 레이트 (Hz). 기본 {DEFAULT_SR}",
    )
    parser.add_argument(
        "--signals", type=int, nargs="*", default=None,
        help="특정 signal_type만 필터 (0=ECG, 1=ABP, ..., 7=ICP). "
             "미지정 시 전체.",
    )
    args = parser.parse_args()

    signal_filter = set(args.signals) if args.signals else None

    total = Stats()
    per_dataset: dict[str, Stats] = {}

    for ddir in args.data_dir:
        path = Path(ddir)
        if not path.exists():
            print(f"WARN: {path} 없음, skip", file=sys.stderr)
            continue
        s = scan_dataset(path, signal_filter=signal_filter)
        per_dataset[str(path)] = s

        # 합산 — subjects 합집합 유지
        total.n_recordings += s.n_recordings
        total.total_samples += s.total_samples
        total.total_bytes += s.total_bytes
        for k, v in s.per_signal_samples.items():
            total.per_signal_samples[k] += v
        for k, v in s.per_signal_count.items():
            total.per_signal_count[k] += v
        total.subjects.update(s.subjects)

    total.finalize()

    if total.n_recordings == 0:
        print("ERROR: 수집된 manifest 없음. --data-dir 경로 확인 요망.", file=sys.stderr)
        sys.exit(1)

    print_summary(total, per_dataset, args.patch_size, args.sampling_rate)


if __name__ == "__main__":
    main()
