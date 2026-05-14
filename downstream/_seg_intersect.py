# -*- coding:utf-8 -*-
"""Downstream prep 용 manifest 기반 시간선 intersection 로더.

기존 prep 들의 ``_load_local_pt_aligned_signals`` (filename seg_key 매칭) 를
대체하는 drop-in utility. 같은 (subject, session) 의 required signal 들을
manifest.json 의 ``start_sample`` + ``n_timesteps`` 로 시간선 정렬 후
모든 신호가 동시 valid 한 구간 (intersection) 만 추출한다.

기존 seg_key 매칭 대비 cohort N 이 13-124× 회복됨이 확인됨 (2026-05-14
estimate_vitaldb_cohort.py 실측). multi-modal alignment 의 실제 한계 반영.

출력 포맷은 기존 함수와 동일:
    [{"case_id": str, "patient_id": str, "session_id": str,
      "start_sample": int, "signals": {sig_str: np.ndarray (T,)}}, ...]
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

TARGET_SR: float = 100.0

SIGNAL_TYPE_STR: dict[int, str] = {
    0: "ecg", 1: "abp", 2: "ppg", 3: "cvp",
    4: "co2", 5: "awp", 6: "pap", 7: "icp", 8: "resp",
}


@dataclass
class _SegmentRef:
    """Parser manifest 의 한 recording 메타."""
    start_sample: int  # 절대 시작 (TARGET_SR 기준)
    n_timesteps: int
    spatial_id: int
    file: str          # subject_dir 기준 상대 파일명


def _merge_overlapping_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """겹치는 [start, end) interval 들 병합 (multi-lead ECG 등)."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _intersect_intervals(
    intervals_by_sig: dict[str, list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """모든 signal 동시 valid 한 구간 sweep-line intersection."""
    if not intervals_by_sig:
        return []
    n_req = len(intervals_by_sig)
    events: list[tuple[int, int, str]] = []
    for sig, intervals in intervals_by_sig.items():
        for s, e in intervals:
            events.append((s, 0, sig))
            events.append((e, 1, sig))
    events.sort(key=lambda x: (x[0], x[1]))

    active: dict[str, int] = defaultdict(int)
    n_active = 0
    result: list[tuple[int, int]] = []
    current_start: int | None = None
    for t, kind, sig in events:
        if kind == 0:
            if active[sig] == 0:
                n_active += 1
                if n_active == n_req:
                    current_start = t
            active[sig] += 1
        else:
            active[sig] -= 1
            if active[sig] == 0:
                if n_active == n_req and current_start is not None:
                    if t > current_start:
                        result.append((current_start, t))
                    current_start = None
                n_active -= 1
    return result


def _normalize_required(
    required_signals,
) -> list[tuple[str, set[int] | None]]:
    """str 또는 (str, allowed_sids) 형태 입력을 (str, set|None) 으로 통일."""
    out: list[tuple[str, set[int] | None]] = []
    for item in required_signals:
        if isinstance(item, tuple):
            sig, sids = item
            out.append((sig, None if sids is None else set(sids)))
        else:
            out.append((item, None))
    return out


def _find_containing_seg(
    segs: list[_SegmentRef],
    allowed_sids: set[int] | None,
    start: int,
    end: int,
) -> _SegmentRef | None:
    """[start, end) 를 fully 포함하는 segment 탐색."""
    for s in segs:
        if allowed_sids is not None and s.spatial_id not in allowed_sids:
            continue
        s_end = s.start_sample + s.n_timesteps
        if s.start_sample <= start and s_end >= end:
            return s
    return None


def _load_one_subject(
    subj_dir: Path,
    required: list[tuple[str, set[int] | None]],
    min_duration_samples: int,
) -> list[dict]:
    """Subject manifest 로드 → 시간선 intersection → 각 interval 의 신호 slice."""
    mp = subj_dir / "manifest.json"
    if not mp.is_file():
        return []
    try:
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    cases: list[dict] = []
    subject_id = subj_dir.name

    for sess in m.get("sessions", []):
        sess_id = str(sess.get("session_id", ""))

        # signal_type_str 별 segment 리스트
        sig_segs: dict[str, list[_SegmentRef]] = defaultdict(list)
        for rec in sess.get("recordings", []):
            st_int = rec.get("signal_type")
            ss = rec.get("start_sample")
            nt = rec.get("n_timesteps")
            sp_ids = rec.get("spatial_ids", [])
            fname = rec.get("file")
            if st_int is None or ss is None or nt is None or not fname:
                continue
            st_str = SIGNAL_TYPE_STR.get(int(st_int))
            if st_str is None:
                continue
            sp_id = int(sp_ids[0]) if sp_ids else 0
            sig_segs[st_str].append(_SegmentRef(
                start_sample=int(ss),
                n_timesteps=int(nt),
                spatial_id=sp_id,
                file=str(fname),
            ))
        for k in sig_segs:
            sig_segs[k].sort(key=lambda x: x.start_sample)

        # required signal 별 intervals (overlap 병합)
        intervals_by_sig: dict[str, list[tuple[int, int]]] = {}
        skip = False
        for sig_str, allowed_sids in required:
            recs = sig_segs.get(sig_str, [])
            sig_intervals: list[tuple[int, int]] = []
            for seg in recs:
                if allowed_sids is not None and seg.spatial_id not in allowed_sids:
                    continue
                sig_intervals.append((seg.start_sample, seg.start_sample + seg.n_timesteps))
            if not sig_intervals:
                skip = True
                break
            intervals_by_sig[sig_str] = _merge_overlapping_intervals(sig_intervals)
        if skip:
            continue

        intersections = _intersect_intervals(intervals_by_sig)
        if not intersections:
            continue

        # 각 intersection interval 의 신호 slice 로드
        for int_idx, (i_start, i_end) in enumerate(intersections):
            if (i_end - i_start) < min_duration_samples:
                continue

            signals: dict[str, np.ndarray] = {}
            ok = True
            for sig_str, allowed_sids in required:
                seg = _find_containing_seg(
                    sig_segs[sig_str], allowed_sids, i_start, i_end,
                )
                if seg is None:
                    # multi-lead overlap 으로 intersection 이 잡혔으나 단일 seg 로
                    # full-cover 가 안 되는 경우 — 해당 interval drop (drop 정책)
                    ok = False
                    break
                try:
                    pt_path = subj_dir / seg.file
                    arr = torch.load(pt_path, weights_only=True)
                except Exception:
                    ok = False
                    break
                if arr.dim() > 1:
                    arr = arr.squeeze(0)
                # OOM 회피: parser 가 float32 로 저장한 신호를 즉시 float16 로 캐스팅.
                # 6000+ case 전체를 memory 적재 → float32 (4B) → float16 (2B) → 50%
                # 절감. NaN 보존 (np.nan.astype(float16)=nan), 정확도 영향 0 (저장
                # dtype 정책과 일관 — _save_utils 정책).
                arr_np = arr.numpy().astype(np.float16, copy=False)
                rel_start = i_start - seg.start_sample
                rel_end = i_end - seg.start_sample
                rel_end = min(rel_end, len(arr_np))
                if rel_end - rel_start < min_duration_samples:
                    ok = False
                    break
                signals[sig_str] = arr_np[rel_start:rel_end]

            if not ok:
                continue

            # 공통 길이로 trim (이미 같은 길이여야 하지만 안전장치)
            min_len = min(len(v) for v in signals.values())
            signals = {k: v[:min_len] for k, v in signals.items()}

            cases.append({
                "case_id": f"{subject_id}_s{sess_id}_int{int_idx}",
                "patient_id": subject_id,
                "session_id": sess_id,
                "start_sample": int(i_start),
                "signals": signals,
            })

    return cases


def load_aligned_signals_intersection(
    data_dir: str,
    required_signals,  # list of str 또는 list of (str, allowed_spatial_ids_or_None)
    *,
    min_duration_sec: float = 60.0,
    max_subjects: int | None = None,
    workers: int = 16,
) -> list[dict]:
    """Manifest 기반 시간선 intersection 으로 다신호 정렬 데이터 로드.

    기존 ``_load_local_pt_aligned_signals`` 의 drop-in replacement.

    Parameters
    ----------
    data_dir : 파싱된 .pt 디렉토리 (각 subject_dir 안에 manifest.json 존재).
    required_signals : 필수 signal type 리스트.
        - ``["abp", "ecg", "ppg"]`` : spatial_id 무관
        - ``[("resp", [2]), "ecg"]`` : resp 는 spatial_id=2(Flow) 만 허용 등
        - 본 함수가 자동으로 abp 를 추가하지 않으니, label 계산에 필요하면
          호출자가 명시적으로 포함시킬 것.
    min_duration_sec : intersection interval 최소 길이 (초).
    max_subjects : 처리 subject 상한 (디버깅용).
    workers : 병렬 worker 수.

    Returns
    -------
    list of {
        "case_id": "<subj>_s<sess>_int<i>",
        "patient_id": "<subj>",
        "session_id": "<sess>",
        "start_sample": int (절대 시작 sample, TARGET_SR 기준),
        "signals": {sig_str: np.ndarray (T,)},  # 한 case 의 모든 신호 같은 길이 T
    }
    """
    required = _normalize_required(required_signals)
    root = Path(data_dir)
    if not root.is_dir():
        print(f"  ERROR: Data directory not found: {root}")
        return []

    subject_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    min_samples = int(min_duration_sec * TARGET_SR)
    req_str = ", ".join(
        f"{s}{'' if a is None else '(' + ','.join(map(str, sorted(a))) + ')'}"
        for s, a in required
    )
    print(f"  Loading aligned signals via timeline intersection")
    print(f"    data_dir : {root}")
    print(f"    required : {req_str}")
    print(f"    min_dur  : {min_duration_sec:.0f}s")
    print(f"    subjects : {len(subject_dirs)}")

    cases: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_load_one_subject, sd, required, min_samples): sd
            for sd in subject_dirs
        }
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="  Loading subjects", unit="subj", dynamic_ncols=True)
        for fut in pbar:
            try:
                subj_cases = fut.result()
                cases.extend(subj_cases)
            except Exception as exc:
                # 한 subject 실패해도 나머지 진행
                print(f"  [WARN] subject load failed: {exc}")
            pbar.set_postfix(cases=len(cases), refresh=False)
        pbar.close()

    n_subj = len({c["patient_id"] for c in cases})
    print(f"  → {len(cases):,} intersection intervals / {n_subj:,} subject(s)")
    return cases
