# -*- coding:utf-8 -*-
"""VitalDB 기반 downstream task 의 cohort N 사전 추정.

13-task 중 VitalDB 4개 task (#1 Hypotension / #5 EtCO₂ / #6 Hypoxemia / #9 AKI) 의
입력 modality 가용성을 바탕으로 다음을 추정한다:
  - 모든 필수 signal 이 갖춰진 subject 수
  - 시간 정렬 segment 수
  - 누적 duration (시간)
  - (window_sec, horizon_sec) 조합 별 sliding-window 샘플 수

⚠️ 본 스크립트는 label (MAP/EtCO₂/SpO₂/Cr) 까지 계산하지 않는다.
   Label 은 별도 소스 (raw .vital 또는 clinical_data.csv) 가 필요하며 본 스크립트는
   "input modality 가 동시 존재하는 cohort 의 상한" 을 추정한다.

전제 파일 구조 (`prepare_data.py` 들이 가정하는 형식):
    <data-dir>/<subject_id>/<subject_id>_S{session}_{signal_type}_{spatial_id}_seg{i}_{j}.pt
    예) VDB_0239_S0_abp_1_seg0_0.pt

사용법:
    python -m scripts.estimate_vitaldb_cohort \
        --data-dir /path/to/vitaldb_parsed \
        --windows 60 180 300 600 \
        --horizons 5 10 15 \
        --stride 30

    # 특정 task 만:
    python -m scripts.estimate_vitaldb_cohort \
        --data-dir /path/to/vitaldb_parsed \
        --tasks 1 9
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

TARGET_SR: float = 100.0  # parsed .pt 는 100Hz 통일 (CLAUDE.md 참조)

# ── Task 사양 ─────────────────────────────────────────────────────
#  required_signals: [(signal_type_str, allowed_spatial_ids_or_None), ...]
#    None 이면 spatial_id 무관, set 이면 해당 spatial_id 만 허용
#  label_note: label 계산을 위해 추가로 필요한 소스
#  whole_segment: True 면 sliding window 가 아니라 segment-당-1 case 로 카운트

TASK_SPECS: dict[int, dict] = {
    1: {
        "name": "Intraoperative Hypotension",
        "required": [("abp", None), ("ecg", None), ("ppg", None)],
        "label_note": "MAP derived from ABP (already in input)",
        "horizon_unit": "min",
        "default_horizons_min": (5, 10, 15),
    },
    5: {
        "name": "Intraop EtCO₂ Abnormality",
        # ECG, PPG, AWP, RESP/Flow (spatial_id=2). 추가로 cohort: ventilated.
        "required": [("ecg", None), ("ppg", None), ("awp", None), ("resp", {2})],
        "label_note": "EtCO₂ trend from raw .vital required (--raw-dir)",
        "horizon_unit": "min",
        "default_horizons_min": (5, 10, 15),
    },
    6: {
        "name": "Intraop Hypoxemia (forecasting)",
        # ECG, PPG, CO₂, AWP, RESP/Flow
        "required": [
            ("ecg", None),
            ("ppg", None),
            ("co2", None),
            ("awp", None),
            ("resp", {2}),
        ],
        "label_note": "SpO₂ trend from raw .vital required (--raw-dir)",
        "horizon_unit": "min",
        "default_horizons_min": (5, 10, 15),
    },
    9: {
        "name": "Postop AKI (KDIGO)",
        "required": [("abp", None), ("ecg", None), ("ppg", None)],
        "label_note": "preop_cr + postop Cr from clinical_data.csv + lab_data.csv",
        "horizon_unit": None,  # AKI 는 horizon 없음 (whole intraop)
        "default_horizons_min": (0,),  # placeholder
        "default_window_sec": 600.0,
        "default_stride_sec": 300.0,
    },
}


# ── 파일명 파서 ───────────────────────────────────────────────────

_FNAME_RE = re.compile(
    r"^(.+?)_S(\d+)_([a-z0-9]+)_(\d+)_seg(\d+)_(\d+)\.pt$"
)


def parse_pt_filename(name: str) -> dict | None:
    m = _FNAME_RE.match(name)
    if m is None:
        return None
    return {
        "subject_id": m.group(1),
        "session_id": int(m.group(2)),
        "signal_type": m.group(3),
        "spatial_id": int(m.group(4)),
        "seg_i": int(m.group(5)),
        "seg_j": int(m.group(6)),
    }


# ── duration 추정 (file size 기반, 1회 calibration) ─────────────

def calibrate_bytes_per_sample(sample_pt: Path) -> tuple[int, int]:
    """대표 .pt 1개를 로드해서 (overhead_bytes, bytes_per_sample) 추정."""
    t = torch.load(sample_pt, weights_only=True, map_location="cpu")
    if t.dim() > 1:
        t = t.squeeze(0)
    n = int(t.numel())
    sz = sample_pt.stat().st_size
    dtype_bytes = int(t.element_size())
    overhead = max(sz - n * dtype_bytes, 0)
    return overhead, dtype_bytes


def estimate_n_samples(file_size: int, overhead: int, dtype_bytes: int) -> int:
    """file size → tensor sample 수 추정 (overhead 보정)."""
    return max(0, (file_size - overhead) // dtype_bytes)


# ── 신호 가용성 + segment 집계 ──────────────────────────────────

def scan_subjects(
    root: Path,
    max_subjects: int | None = None,
) -> tuple[dict, int, int]:
    """전체 .pt 디렉토리 스캔.

    Returns
    -------
    subjects : {subject_id: {(session, seg_i, seg_j): {sig_type_str: (path, file_size, spatial_id)}}}
    n_files  : 스캔한 총 .pt 파일 수
    n_unparseable : 파일명 매칭 실패 수
    """
    print(f"  Listing {root} ...", flush=True)
    subject_dirs: list[Path] = []
    list_pbar = tqdm(
        root.iterdir(),
        desc="  Listing entries",
        unit="entry",
        dynamic_ncols=True,
    )
    for entry in list_pbar:
        if entry.is_dir():
            subject_dirs.append(entry)
        list_pbar.set_postfix(dirs=len(subject_dirs), refresh=False)
    list_pbar.close()
    subject_dirs.sort()
    print(f"  → {len(subject_dirs)} subject directory(ies) found", flush=True)
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    subjects: dict[str, dict[tuple[int, int, int], dict[str, tuple[Path, int, int]]]] = {}
    n_files = 0
    n_bad = 0

    pbar = tqdm(
        subject_dirs,
        desc="Scanning subjects",
        unit="subj",
        dynamic_ncols=True,
    )
    for subj_dir in pbar:
        sid = subj_dir.name
        seg_map: dict[tuple[int, int, int], dict[str, tuple[Path, int, int]]] = {}
        for pt in subj_dir.glob("*.pt"):
            n_files += 1
            meta = parse_pt_filename(pt.name)
            if meta is None:
                n_bad += 1
                continue
            seg_key = (meta["session_id"], meta["seg_i"], meta["seg_j"])
            seg_map.setdefault(seg_key, {})[meta["signal_type"]] = (
                pt,
                pt.stat().st_size,
                meta["spatial_id"],
            )
        if seg_map:
            subjects[sid] = seg_map
        pbar.set_postfix(files=n_files, kept=len(subjects), refresh=False)
    pbar.close()

    return subjects, n_files, n_bad


# ── Task 별 cohort 추정 ──────────────────────────────────────────

def check_segment_satisfies(
    seg_signals: dict[str, tuple[Path, int, int]],
    required: list[tuple[str, set[int] | None]],
) -> bool:
    """segment 의 신호 dict 가 required 조건을 모두 만족하는지."""
    for sig_type, allowed_sids in required:
        if sig_type not in seg_signals:
            return False
        if allowed_sids is not None:
            _, _, sid = seg_signals[sig_type]
            if sid not in allowed_sids:
                return False
    return True


def aggregate_task_cohort(
    subjects: dict,
    task_id: int,
    overhead: int,
    dtype_bytes: int,
) -> dict:
    """task 의 required signal 을 만족하는 cohort 통계 집계."""
    spec = TASK_SPECS[task_id]
    required = spec["required"]
    req_types = [t for t, _ in required]

    n_subjects = 0
    n_segments = 0
    total_samples = 0  # 100Hz 기준 sample 합 (가장 짧은 신호 기준)
    seg_durations_sec: list[float] = []  # window 계산용

    for sid, seg_map in subjects.items():
        subj_has_any = False
        for _, seg_signals in seg_map.items():
            if not check_segment_satisfies(seg_signals, required):
                continue
            # 가장 짧은 신호 기준 (aligned 자르기)
            min_samples = min(
                estimate_n_samples(seg_signals[t][1], overhead, dtype_bytes)
                for t in req_types
            )
            if min_samples <= 0:
                continue
            seg_dur = min_samples / TARGET_SR
            seg_durations_sec.append(seg_dur)
            total_samples += min_samples
            n_segments += 1
            subj_has_any = True
        if subj_has_any:
            n_subjects += 1

    return {
        "n_subjects": n_subjects,
        "n_segments": n_segments,
        "total_hours": total_samples / TARGET_SR / 3600.0,
        "seg_durations_sec": seg_durations_sec,
    }


def count_sliding_windows(
    seg_durations_sec: list[float],
    window_sec: float,
    horizon_sec: float,
    stride_sec: float,
) -> int:
    """segment 별 sliding window 샘플 수 합산.

    유효 window 조건: win_start + window_sec + horizon_sec <= seg_dur
        → n = floor((seg_dur - window_sec - horizon_sec) / stride_sec) + 1
    """
    total = 0
    needed = window_sec + horizon_sec
    for dur in seg_durations_sec:
        if dur < needed:
            continue
        n = int((dur - needed) // stride_sec) + 1
        total += max(n, 0)
    return total


# ── 출력 ─────────────────────────────────────────────────────────

def print_task_report(
    task_id: int,
    cohort: dict,
    windows_sec: list[float],
    horizons_min: list[float],
    stride_sec: float,
) -> None:
    spec = TASK_SPECS[task_id]
    bar = "─" * 92
    print(f"\n{bar}")
    print(f"  Task #{task_id}: {spec['name']}")
    print(f"{bar}")

    req_str = ", ".join(
        f"{t}{'' if s is None else '(' + ','.join(map(str, sorted(s))) + ')'}"
        for t, s in spec["required"]
    )
    print(f"  Required input signals : {req_str}")
    print(f"  Label source           : {spec['label_note']}")
    print()
    print(f"  Subjects (all signals present) : {cohort['n_subjects']:>10,}")
    print(f"  Aligned segments               : {cohort['n_segments']:>10,}")
    print(f"  Total duration                 : {cohort['total_hours']:>10,.1f} h"
          f"  ({cohort['total_hours'] / 24:.1f} d)")

    if cohort["n_segments"] == 0:
        print("  ⚠ No aligned segments — required modality 가 한 case 도 없음.")
        return

    # window × horizon grid
    if task_id == 9:
        # AKI: horizon 없음, window/stride 만
        win = spec.get("default_window_sec", 600.0)
        strd = spec.get("default_stride_sec", 300.0)
        n_win = count_sliding_windows(cohort["seg_durations_sec"], win, 0.0, strd)
        print()
        print(f"  AKI sliding windows (win={int(win)}s, stride={int(strd)}s) :"
              f" {n_win:>10,}")
        print("    (라벨 KDIGO 는 patient-level → 각 환자당 1개 binary, "
              "window 는 representation aggregation 용)")
        return

    print()
    print("  Sliding-window 샘플 수 (label 미포함, input 가용성만 기준)")
    print(f"  stride = {int(stride_sec)} s")
    print()
    header = "    window\\horizon  |"
    for h in horizons_min:
        header += f"  {int(h):>3} min      "
    print(header)
    print("    " + "-" * (len(header) - 4))
    for w in windows_sec:
        row = f"    {int(w):>4} s          |"
        for h in horizons_min:
            n = count_sliding_windows(
                cohort["seg_durations_sec"], w, h * 60.0, stride_sec,
            )
            row += f"  {n:>10,}  "
        print(row)


# ── 메인 ────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True,
                    help="parsed .pt 디렉토리 (subject 별 서브디렉토리 포함)")
    ap.add_argument("--tasks", nargs="+", type=int,
                    default=[1, 5, 6, 9],
                    choices=list(TASK_SPECS.keys()),
                    help="추정할 task id (default: 1 5 6 9)")
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="최대 subject 수 (디버깅용)")
    ap.add_argument("--windows", nargs="+", type=float,
                    default=[60.0, 180.0, 300.0, 600.0],
                    help="input window 길이 (초) — sliding-window 카운트 grid")
    ap.add_argument("--horizons", nargs="+", type=float,
                    default=[5.0, 10.0, 15.0],
                    help="prediction horizon (분) — sliding-window 카운트 grid")
    ap.add_argument("--stride", type=float, default=30.0,
                    help="sliding stride (초). 기본 30s (prepare_data.py 와 동일)")
    args = ap.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        print(f"ERROR: data-dir not found: {root}", file=sys.stderr)
        return 1

    print(f"  Scanning {root}  (max_subjects={args.max_subjects})", flush=True)
    subjects, n_files, n_bad = scan_subjects(root, args.max_subjects)
    print(f"  → {len(subjects)} subject(s), {n_files:,} .pt file(s),"
          f" {n_bad:,} unparseable")

    if not subjects:
        print("ERROR: 스캔된 subject 가 없습니다.", file=sys.stderr)
        return 1

    # calibration: 첫 subject 의 첫 segment 첫 signal
    first_pt = None
    for _, seg_map in subjects.items():
        for _, sigs in seg_map.items():
            for _, (p, _, _) in sigs.items():
                first_pt = p
                break
            if first_pt:
                break
        if first_pt:
            break
    overhead, dtype_bytes = calibrate_bytes_per_sample(first_pt)
    print(f"  Calibration ({first_pt.name}): overhead={overhead} B,"
          f" {dtype_bytes} B/sample")
    print(f"  (file-size 기반 duration 추정 — 정확도 ±1 sample)")

    for task_id in args.tasks:
        cohort = aggregate_task_cohort(subjects, task_id, overhead, dtype_bytes)
        print_task_report(
            task_id, cohort,
            windows_sec=args.windows,
            horizons_min=args.horizons,
            stride_sec=args.stride,
        )

    print("\n  Notes:")
    print("  • 본 추정은 input modality 가용성 기준의 상한이다.")
    print("  • 실제 N 은 label (raw .vital trend / clinical CSV) 가용성으로 추가 감소한다.")
    print("  • Hypotension : Invasive ABP 환자 cohort 와 거의 일치 (ABP 자체가 invasive)")
    print("  • EtCO₂ / Hypoxemia : ventilated cohort 는 AWP+RESP/Flow 가용성으로 proxy")
    print("  • AKI : patient-level 1 binary label — 위 window 수는 representation 용")
    return 0


if __name__ == "__main__":
    sys.exit(main())
