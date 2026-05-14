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
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # 2026-05-14 v2: AWP only (RESP/Flow 제거 — VitalDB 0건 결과 반영)
        "required": [("ecg", None), ("ppg", None), ("awp", None)],
        "label_note": "EtCO₂ trend from raw .vital required (--raw-dir)",
        "horizon_unit": "min",
        "default_horizons_min": (5, 10, 15),
    },
    6: {
        "name": "Intraop Hypoxemia (forecasting)",
        # 2026-05-14 v2: AWP only (RESP/Flow 제거)
        "required": [("ecg", None), ("ppg", None), ("co2", None), ("awp", None)],
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

def _scan_one_subject(
    subj_dir: Path,
) -> tuple[str, dict[tuple[int, int, int], dict[str, tuple[Path, int, int]]], int, int]:
    """한 subject 디렉토리 스캔 — worker thread 용.

    os.scandir 를 쓰면 DirEntry 가 size 를 cache 하므로 별도 stat() 호출이
    줄어든다 (Linux 에서도 다음 호출시 캐시 사용).
    """
    sid = subj_dir.name
    seg_map: dict[tuple[int, int, int], dict[str, tuple[Path, int, int]]] = {}
    n_files = 0
    n_bad = 0
    try:
        with os.scandir(subj_dir) as it:
            for entry in it:
                name = entry.name
                if not name.endswith(".pt"):
                    continue
                n_files += 1
                meta = parse_pt_filename(name)
                if meta is None:
                    n_bad += 1
                    continue
                size = entry.stat().st_size  # DirEntry.stat caches
                seg_key = (meta["session_id"], meta["seg_i"], meta["seg_j"])
                seg_map.setdefault(seg_key, {})[meta["signal_type"]] = (
                    Path(entry.path),
                    size,
                    meta["spatial_id"],
                )
    except OSError:
        pass
    return sid, seg_map, n_files, n_bad


def scan_subjects(
    root: Path,
    max_subjects: int | None = None,
    workers: int = 16,
) -> tuple[dict, int, int]:
    """전체 .pt 디렉토리 병렬 스캔.

    Returns
    -------
    subjects : {subject_id: {(session, seg_i, seg_j): {sig_type_str: (path, file_size, spatial_id)}}}
    n_files  : 스캔한 총 .pt 파일 수
    n_unparseable : 파일명 매칭 실패 수
    """
    print(f"  Listing {root} ...", flush=True)
    subject_dirs: list[Path] = []
    list_pbar = tqdm(
        os.scandir(root),
        desc="  Listing entries",
        unit="entry",
        dynamic_ncols=True,
    )
    for entry in list_pbar:
        if entry.is_dir():
            subject_dirs.append(Path(entry.path))
        list_pbar.set_postfix(dirs=len(subject_dirs), refresh=False)
    list_pbar.close()
    subject_dirs.sort()
    print(f"  → {len(subject_dirs)} subject directory(ies) found", flush=True)
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    subjects: dict[str, dict[tuple[int, int, int], dict[str, tuple[Path, int, int]]]] = {}
    n_files_total = 0
    n_bad_total = 0

    print(f"  Scanning with {workers} parallel workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_scan_one_subject, sd): sd for sd in subject_dirs}
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Scanning subjects",
            unit="subj",
            dynamic_ncols=True,
        )
        for fut in pbar:
            sid, seg_map, n_files, n_bad = fut.result()
            n_files_total += n_files
            n_bad_total += n_bad
            if seg_map:
                subjects[sid] = seg_map
            pbar.set_postfix(
                files=n_files_total, kept=len(subjects), refresh=False,
            )
        pbar.close()

    return subjects, n_files_total, n_bad_total


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
    """task 의 required signal 을 만족하는 cohort 통계 집계.

    Returns
    -------
    dict with:
        n_subjects, n_segments, total_hours
        seg_records: list of (subject_id, duration_sec)
            → per-cell effective subject count + universal-cohort filter 에 사용
    """
    spec = TASK_SPECS[task_id]
    required = spec["required"]
    req_types = [t for t, _ in required]

    n_subjects = 0
    n_segments = 0
    total_samples = 0
    seg_records: list[tuple[str, float]] = []  # (subject_id, duration_sec)

    for sid, seg_map in subjects.items():
        subj_has_any = False
        for _, seg_signals in seg_map.items():
            if not check_segment_satisfies(seg_signals, required):
                continue
            min_samples = min(
                estimate_n_samples(seg_signals[t][1], overhead, dtype_bytes)
                for t in req_types
            )
            if min_samples <= 0:
                continue
            seg_dur = min_samples / TARGET_SR
            seg_records.append((sid, seg_dur))
            total_samples += min_samples
            n_segments += 1
            subj_has_any = True
        if subj_has_any:
            n_subjects += 1

    return {
        "n_subjects": n_subjects,
        "n_segments": n_segments,
        "total_hours": total_samples / TARGET_SR / 3600.0,
        "seg_records": seg_records,
    }


def count_windows_with_subjects(
    seg_records: list[tuple[str, float]],
    window_sec: float,
    horizon_sec: float,
    stride_sec: float,
) -> tuple[int, int, int]:
    """sliding window 수 + 유효 segment/subject 수.

    Returns
    -------
    n_windows : 총 window 수
    n_eff_segments : window 가 최소 1개 잡히는 segment 수
    n_eff_subjects : window 가 최소 1개 잡히는 subject 수
    """
    needed = window_sec + horizon_sec
    n_windows = 0
    n_eff_segments = 0
    eff_subjects: set[str] = set()
    for sid, dur in seg_records:
        if dur < needed:
            continue
        n = int((dur - needed) // stride_sec) + 1
        if n <= 0:
            continue
        n_windows += n
        n_eff_segments += 1
        eff_subjects.add(sid)
    return n_windows, n_eff_segments, len(eff_subjects)


def filter_records_universal(
    seg_records: list[tuple[str, float]],
    min_dur_sec: float,
) -> tuple[list[tuple[str, float]], int]:
    """universal cohort: 모든 segment 가 min_dur_sec 이상.

    Returns (filtered_records, n_unique_subjects)
    """
    filtered = [(s, d) for s, d in seg_records if d >= min_dur_sec]
    return filtered, len({s for s, _ in filtered})


# ── 출력 ─────────────────────────────────────────────────────────

def print_task_report(
    task_id: int,
    cohort: dict,
    windows_sec: list[float],
    horizons_min: list[float],
    stride_sec: float,
    unified_cohort: bool = False,
) -> None:
    spec = TASK_SPECS[task_id]
    bar = "─" * 100
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
    print(f"  Subjects (signal availability only) : {cohort['n_subjects']:>10,}")
    print(f"  Aligned segments                    : {cohort['n_segments']:>10,}")
    print(f"  Total duration                      : {cohort['total_hours']:>10,.1f} h"
          f"  ({cohort['total_hours'] / 24:.1f} d)")

    if cohort["n_segments"] == 0:
        print("  ⚠ No aligned segments — required modality 가 한 case 도 없음.")
        return

    seg_records = cohort["seg_records"]

    # window × horizon grid
    if task_id == 9:
        # AKI: horizon 없음, window/stride 만
        win = spec.get("default_window_sec", 600.0)
        strd = spec.get("default_stride_sec", 300.0)
        n_win, n_seg, n_subj = count_windows_with_subjects(
            seg_records, win, 0.0, strd,
        )
        print()
        print(f"  AKI sliding windows (win={int(win)}s, stride={int(strd)}s)")
        print(f"    windows  : {n_win:>10,}")
        print(f"    segments : {n_seg:>10,}")
        print(f"    subjects : {n_subj:>10,}")
        print("    (라벨 KDIGO 는 patient-level → 각 환자당 1개 binary, "
              "window 는 representation aggregation 용)")
        return

    # Universal cohort: 모든 config 가 동일 cohort 사용 (가장 긴 win+horizon 기준)
    if unified_cohort:
        max_needed = max(windows_sec) + max(horizons_min) * 60.0
        filtered_records, n_filt_subj = filter_records_universal(
            seg_records, max_needed,
        )
        n_filt_seg = len(filtered_records)
        print()
        print(f"  [UNIFIED COHORT] min duration ≥ {int(max_needed)} s "
              f"(max win + max horizon = {int(max(windows_sec))}s "
              f"+ {int(max(horizons_min))}min)")
        print(f"    fixed subjects : {n_filt_subj:>10,}  "
              f"({100 * n_filt_subj / max(cohort['n_subjects'], 1):.1f}%)")
        print(f"    fixed segments : {n_filt_seg:>10,}  "
              f"({100 * n_filt_seg / max(cohort['n_segments'], 1):.1f}%)")
        use_records = filtered_records
    else:
        use_records = seg_records

    print()
    print(f"  Sliding-window grid  (stride = {int(stride_sec)} s)")
    print("  cell = windows / segments / subjects")
    print()
    col_w = 24
    header = "    win\\horizon |"
    for h in horizons_min:
        header += f" {int(h):>3} min".ljust(col_w) + "|"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for w in windows_sec:
        row = f"    {int(w):>4} s      |"
        for h in horizons_min:
            n_win, n_seg, n_subj = count_windows_with_subjects(
                use_records, w, h * 60.0, stride_sec,
            )
            cell = f" {n_win:,}/{n_seg:,}/{n_subj:,}"
            row += cell.ljust(col_w) + "|"
        print(row)

    if not unified_cohort:
        # 경고: per-config cohort 차이
        max_subj_cell = 0
        min_subj_cell = 10**9
        for w in windows_sec:
            for h in horizons_min:
                _, _, n_subj = count_windows_with_subjects(
                    seg_records, w, h * 60.0, stride_sec,
                )
                max_subj_cell = max(max_subj_cell, n_subj)
                min_subj_cell = min(min_subj_cell, n_subj)
        if max_subj_cell - min_subj_cell > 0:
            print()
            print(f"  ⚠ config 간 subject 수 편차: {min_subj_cell:,} ~ {max_subj_cell:,} "
                  f"(차이 {max_subj_cell - min_subj_cell:,})")
            print(f"     → 동등 비교 원하면 --unified-cohort 옵션 권장")


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
    ap.add_argument("--workers", type=int, default=16,
                    help="병렬 scan thread 수 (default 16). 스토리지가 빠르면 8, 느리면 32~64")
    ap.add_argument("--unified-cohort", action="store_true",
                    help="모든 (window, horizon) config 가 동일 cohort 사용 — "
                         "가장 긴 (max_win + max_horizon) 충족 segment 만 통과. "
                         "config 간 동등 비교 (paper-fair)")
    args = ap.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        print(f"ERROR: data-dir not found: {root}", file=sys.stderr)
        return 1

    print(f"  Scanning {root}  (max_subjects={args.max_subjects})", flush=True)
    subjects, n_files, n_bad = scan_subjects(
        root, args.max_subjects, workers=args.workers,
    )
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
            unified_cohort=args.unified_cohort,
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
