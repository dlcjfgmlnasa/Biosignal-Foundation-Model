# -*- coding:utf-8 -*-
"""Task 8: Any-to-Any Cross-modal — 데이터 준비 스크립트.

다채널 시간 정렬 윈도우를 추출하여 .pt로 저장한다.
run.py에서 오프라인으로 로드하여 평가에 사용.

데이터 소스:
  - VitalDB (수술중 모니터링, 내부 평가)
  - MIMIC-III Waveform (ICU, 외부 평가)
  - local: 미리 파싱된 .pt 디렉토리 (parser 출력물)

사용법:
    # VitalDB, 기본 4채널 (ecg, abp, ppg, cvp), 10케이스
    python -m downstream.generation.cross_modal.prepare_data \
        --source vitaldb --n-cases 10

    # MIMIC-III, 3채널 (ecg, abp, ppg), 5케이스
    python -m downstream.generation.cross_modal.prepare_data \
        --source mimic3 --signal-types ecg abp ppg --n-cases 5

    # Local .pt directory (e.g. mimic3_ext_ppg parser output)
    python -m downstream.generation.cross_modal.prepare_data \
        --source local --data-dir /path/to/parsed_dir \
        --signal-types ecg abp ppg --n-cases 20

    # 짧은 윈도우, 좁은 stride
    python -m downstream.generation.cross_modal.prepare_data \
        --source vitaldb --window-sec 10 --stride-sec 5 --n-cases 20
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

from downstream._save_utils import (
    add_signal_dtype_arg,
    stack_arrays_destructive,
)
from downstream._kfold_utils import (
    stratified_kfold_patient_splits,
    summarize_splits,
)


TARGET_SR: float = 100.0


# ---- 절대압(calibrated) 게이팅 / 계약 유틸 ----

# 절대 mmHg ABP window 생리 게이트 파라미터 (chunk meta 에 그대로 기록).
# pointwise 하드리밋(20/300)은 data_utils.joint_clean_and_slice 가 이미 적용.
ABP_GATE: dict[str, object] = {
    "abp_hard": [20.0, 300.0],  # pointwise (joint_clean_and_slice 에서 적용)
    "map": [30.0, 180.0],
    "sbp": [40.0, 260.0],
    "dbp": [20.0, 160.0],
    "pp": [10.0, 150.0],
    "flatline_std_min": 1.0,   # window std(mmHg) 하한 → 미만이면 zeroed/분리
    "flush_clip_val": 250.0,   # 고압 포화 표본 판정값
    "flush_clip_ratio": 0.02,  # 포화 표본 비율 상한 (fast-flush 스파이크)
}


def _abp_window_ok(abp: np.ndarray) -> bool:
    """절대 mmHg ABP window 의 생리 타당성 게이트. 불량이면 False → window drop.

    damped-trace(DBP↑·PP↓), flush-spike(PP↑·포화), flat-line/분리(std≈0),
    비생리 MAP/SBP/DBP 를 걸러 절대압 라벨 오염을 방지한다. clip 이 아니라 drop.
    """
    if abp.size == 0 or np.isnan(abp).any():
        return False
    if float(np.std(abp)) < float(ABP_GATE["flatline_std_min"]):
        return False
    sbp = float(np.percentile(abp, 95))  # 강건 SBP/DBP/MAP 근사
    dbp = float(np.percentile(abp, 5))
    mapv = float(np.mean(abp))
    pp = sbp - dbp
    lo, hi = ABP_GATE["map"]
    ok = lo <= mapv <= hi
    lo, hi = ABP_GATE["sbp"]
    ok = ok and (lo <= sbp <= hi)
    lo, hi = ABP_GATE["dbp"]
    ok = ok and (lo <= dbp <= hi)  # damped → DBP↑ 로 여기서 탈락
    lo, hi = ABP_GATE["pp"]
    ok = ok and (lo <= pp <= hi)   # damped → PP↓, flush → PP↑
    clip_ratio = float(np.mean(abp >= float(ABP_GATE["flush_clip_val"])))
    ok = ok and (clip_ratio <= float(ABP_GATE["flush_clip_ratio"]))
    return bool(ok)


def _mark_anchors(windows: list[dict]) -> np.ndarray:
    """환자별 첫 gated-valid window(causal, 가장 이른 시각) 를 is_anchor=True 로.

    (patient_id, case_id, win_start) 오름차순에서 각 환자의 첫 window 를 anchor 로
    표시하되, 반환 배열은 입력 ``windows`` 원래 순서에 정렬된다. patient-disjoint
    split 전제에서 split 당 환자마다 정확히 1개의 anchor 가 생긴다.
    """
    is_anchor = np.zeros(len(windows), dtype=bool)
    seen: set[str] = set()
    order = sorted(
        range(len(windows)),
        key=lambda i: (
            str(windows[i]["patient_id"]),
            str(windows[i]["case_id"]),
            int(windows[i]["win_start"]),
        ),
    )
    for i in order:
        pid = str(windows[i]["patient_id"])
        if pid not in seen:
            is_anchor[i] = True
            seen.add(pid)
    return is_anchor


def _abp_train_scale(train_windows: list[dict]) -> tuple[float, float]:
    """train split window 의 raw ABP(mmHg) 표본에서만 전역 고정 정규화 스케일 산출.

    val/test 에 동일 (mu, sd) 를 적용(가역 affine) → leakage 없이 절대압 보존.
    ABP 가 없으면 (0.0, 1.0).
    """
    if not train_windows or "abp" not in train_windows[0]["signals"]:
        return 0.0, 1.0
    vals = np.concatenate([w["signals"]["abp"].ravel() for w in train_windows])
    return float(vals.mean()), float(vals.std() + 1e-6)


def _assert_no_patient_leak(splits_windows: dict[str, list[dict]]) -> None:
    """공정성 게이트: split 간 환자 disjoint + split 내 환자당 anchor 정확히 1개.

    실패 시 AssertionError 로 abort (착시 승리 방지).
    """
    pid_sets = {
        name: {str(w["patient_id"]) for w in ws}
        for name, ws in splits_windows.items()
        if ws
    }
    names = list(pid_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = pid_sets[names[i]] & pid_sets[names[j]]
            assert not inter, (
                f"patient leak {names[i]}∩{names[j]}: {sorted(inter)[:5]}"
            )
    for name, ws in splits_windows.items():
        if not ws:
            continue
        n_anchor = int(_mark_anchors(ws).sum())
        n_pat = len(pid_sets[name])
        assert n_anchor == n_pat, f"{name}: anchors({n_anchor}) != patients({n_pat})"


# ---- MIMIC-III 로더 ----


def _load_mimic3_cases(
    n_cases: int,
    signal_types: list[str],
    min_duration_sec: float = 600.0,
    manifest_path: str | None = None,
    out_dir: str = "outputs/downstream/mimic3",
) -> list[dict]:
    """MIMIC-III에서 시간 정렬된 다채널 데이터를 로드한다.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signals": {stype: array}}
    """
    import wfdb
    from data.parser.mimic3_waveform import (
        scan_abp_records,
        load_manifest,
        MIMIC3_NATIVE_SR,
    )
    from downstream.data_utils import joint_clean_and_slice

    # manifest 로드 또는 스캔
    manifest_file = Path(out_dir) / "mimic3_abp_manifest.json"
    if manifest_path and Path(manifest_path).exists():
        records = load_manifest(manifest_path)
    elif manifest_file.exists():
        records = load_manifest(str(manifest_file))
    else:
        print("  Scanning for ABP records...")
        records = scan_abp_records(
            max_records=n_cases * 5,
            save_path=str(manifest_file),
        )

    # 필요 채널이 모두 있는 레코드 필터링
    filtered = []
    for r in records:
        has_all = True
        if "ecg" in signal_types and not r.has_ecg:
            has_all = False
        if "ppg" in signal_types and not r.has_ppg:
            has_all = False
        if has_all:
            filtered.append(r)

    if not filtered:
        print(f"  WARNING: No records with all required signals {signal_types}")
        return []

    filtered = filtered[:n_cases]
    print(f"  Found {len(filtered)} records with {signal_types}")

    # 각 레코드에서 시간 정렬된 다채널 데이터 추출
    cases = []
    for i, info in enumerate(filtered):
        print(f"  [{i + 1}/{len(filtered)}] {info.record_name}...", end=" ")
        t0 = time.time()

        try:
            hdr = wfdb.rdheader(info.record_name, pn_dir=info.pn_dir)
        except Exception as e:
            print(f"SKIP (header error: {e})")
            continue

        if not hasattr(hdr, "seg_name") or not hdr.seg_name:
            print("SKIP (no segments)")
            continue

        # 채널 매핑
        ch_map = {}
        if "abp" in signal_types:
            ch_map["abp"] = info.abp_channel
        if "ecg" in signal_types and info.ecg_channel:
            ch_map["ecg"] = info.ecg_channel
        if "ppg" in signal_types and info.ppg_channel:
            ch_map["ppg"] = info.ppg_channel

        if len(ch_map) < len(signal_types):
            print(
                f"SKIP (missing channels: need {signal_types}, have {list(ch_map.keys())})"
            )
            continue

        # 모든 채널이 동시에 존재하는 가장 긴 세그먼트 찾기
        best_seg = None
        best_len = 0

        for seg_name, seg_len in zip(hdr.seg_name, hdr.seg_len):
            if seg_name == "~" or seg_name.endswith("_layout") or seg_len <= 0:
                continue
            try:
                seg_hdr = wfdb.rdheader(seg_name, pn_dir=info.pn_dir)
                if seg_hdr.sig_name is None:
                    continue
                all_present = all(
                    ch_name in seg_hdr.sig_name for ch_name in ch_map.values()
                )
                if all_present and seg_len > best_len:
                    best_seg = seg_name
                    best_len = seg_len
            except Exception:
                continue

        if best_seg is None or best_len < int(min_duration_sec * MIMIC3_NATIVE_SR):
            print(f"SKIP (no aligned segment >= {min_duration_sec / 60:.0f}min)")
            continue

        # 가장 긴 정렬 세그먼트 로드
        try:
            seg = wfdb.rdrecord(best_seg, pn_dir=info.pn_dir)
        except Exception as e:
            print(f"SKIP (read error: {e})")
            continue

        if seg.p_signal is None:
            print("SKIP (no signal)")
            continue

        # 각 채널을 같은 segment(공통 125Hz 그리드)에서 raw 로 추출 → joint 정렬.
        # 채널별 독립 전처리(구 _apply_pipeline)는 desync 를 유발하므로 사용 안 함.
        raw_cols: dict[str, np.ndarray] = {}
        valid = True
        for stype, ch_name in ch_map.items():
            if ch_name not in seg.sig_name:
                valid = False
                break
            ch_idx = seg.sig_name.index(ch_name)
            raw_cols[stype] = seg.p_signal[:, ch_idx].astype(np.float64)

        if not valid:
            print("SKIP (channel missing)")
            continue

        # joint valid mask + 최장 공통 구간 → 등길이·100Hz (절대 mmHg 보존)
        signals = joint_clean_and_slice(
            raw_cols, MIMIC3_NATIVE_SR, list(ch_map.keys())
        )
        if not signals or any(
            len(v) < int(min_duration_sec * TARGET_SR) for v in signals.values()
        ):
            print("SKIP (no joint-aligned segment)")
            continue

        elapsed = time.time() - t0
        n_aligned = len(next(iter(signals.values())))  # joint 후 모든 신호 등길이
        dur_min = n_aligned / TARGET_SR / 60
        print(f"OK ({dur_min:.1f}min, {len(signals)} ch) [{elapsed:.1f}s]")

        cases.append(
            {
                "case_id": info.record_name,
                "patient_id": info.patient_id,
                "signals": signals,
            }
        )

    return cases


# ---- VitalDB 로더 ----


def _vitaldb_caseid_to_subject(clinical_csv: str | None = None) -> dict[int, int]:
    """VitalDB Open clinical info 에서 caseid→subjectid 매핑을 얻는다.

    우선순위: (1) ``clinical_csv`` 로컬 파일(사내망 API 차단 환경), (2) 공개 API
    ``https://api.vitaldb.net/cases``. 둘 다 실패하면 빈 dict (호출부에서
    ``patient_id=case_id`` fallback + WARN).
    """
    import pandas as pd

    # (1) 로컬 clinical CSV 주입
    if clinical_csv:
        p = Path(clinical_csv)
        if p.exists():
            try:
                df = pd.read_csv(str(p))
                return dict(
                    zip(df["caseid"].astype(int), df["subjectid"].astype(int))
                )
            except Exception as e:  # noqa: BLE001
                print(f"  WARN: clinical-csv 파싱 실패({e}) → API/fallback 시도")
        else:
            print(f"  WARN: clinical-csv 없음: {clinical_csv} → API/fallback 시도")

    # (2) 공개 API
    try:
        df = pd.read_csv("https://api.vitaldb.net/cases")
        return dict(zip(df["caseid"].astype(int), df["subjectid"].astype(int)))
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: api.vitaldb.net/cases 접근 실패({e}) → patient_id=case_id fallback")
        return {}


def _load_vitaldb_cases(
    n_cases: int,
    signal_types: list[str],
    min_duration_sec: float = 600.0,
    offset_from_end: int = 200,
    vital_dir: str | None = None,
    clinical_csv: str | None = None,
) -> list[dict]:
    """VitalDB에서 joint 시간축 정렬된 다채널 데이터를 로드한다.

    ``load_joint_cases`` 로 모든 신호를 공통 100Hz 그리드에서 함께 로드·정렬하며
    (신호별 독립 세그먼트 선택 없음), caseid→subjectid 매핑으로 실제 환자
    ``patient_id`` 를 부여한다.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signals": {stype: array}}
    """
    from downstream.data_utils import load_joint_cases

    print(f"  Loading {n_cases} VitalDB cases joint-aligned (signals={signal_types})...")
    raw_cases = load_joint_cases(
        n_cases=n_cases,
        offset_from_end=offset_from_end,
        signal_types=signal_types,
        vital_dir=vital_dir,
    )

    # 실제 환자 매핑 (로컬 .vital 은 subjectid 부재 → fallback)
    cid2subj: dict[int, int] = {}
    if vital_dir is None:
        cid2subj = _vitaldb_caseid_to_subject(clinical_csv)

    cases = []
    n_fallback = 0
    for rc in raw_cases:
        # 모든 필요 채널이 있는지 확인
        if not all(st in rc.tracks for st in signal_types):
            continue

        # joint 로더가 이미 등길이·정렬 → min_len 절단 불필요 (안전용 assert)
        lens = {len(rc.tracks[st]) for st in signal_types}
        assert len(lens) == 1, f"joint misaligned lengths for case {rc.case_id}: {lens}"
        n = lens.pop()
        if n < int(min_duration_sec * TARGET_SR):
            continue

        # 실제 환자 id 결정 (없으면 case_id fallback)
        pid: str
        if cid2subj and str(rc.case_id).isdigit() and int(rc.case_id) in cid2subj:
            pid = str(cid2subj[int(rc.case_id)])
        else:
            pid = f"vitaldb_{rc.case_id}"
            n_fallback += 1

        signals = {st: rc.tracks[st] for st in signal_types}
        cases.append(
            {
                "case_id": f"vitaldb_{rc.case_id}",
                "patient_id": pid,
                "signals": signals,
            }
        )

    if n_fallback:
        print(
            f"  WARN: patient_id fallback to case_id ({n_fallback}/{len(cases)} cases) "
            f"— 같은 환자 다중 case 누수 방어 불완전 (subjectid 매핑 필요)"
        )
    print(f"  Loaded {len(cases)} cases with all required signals")
    return cases


# ---- Local .pt directory 로더 ----


def _load_local_pt_cases(
    data_dir: str,
    signal_types: list[str],
    min_duration_sec: float = 600.0,
    max_subjects: int | None = None,
) -> list[dict]:
    """로컬 .pt 디렉토리에서 시간 정렬된 다채널 데이터를 로드한다.

    manifest.json을 읽고, 각 subject의 signal type별 .pt 파일을 로드한다.
    parser 출력 디렉토리 구조::

        data_dir/
            manifest.json     # [{"subject_id": "...", "files": {"ecg": "path", ...}, ...}, ...]
            subjectA/
                ecg.pt        # {"values": Tensor(T,), "sampling_rate": 100.0, ...}
                abp.pt
                ppg.pt

    manifest.json이 없으면, 디렉토리 구조를 직접 스캔한다.
    (subject 디렉토리 아래 {signal_type}.pt 파일)

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signals": {stype: np.array}}
    """
    import json as _json

    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"  ERROR: data_dir not found: {data_dir}")
        return []

    min_samples = int(min_duration_sec * TARGET_SR)
    manifest_file = data_path / "manifest.json"

    subjects: list[dict] = []

    if manifest_file.exists():
        # manifest.json 기반 로드
        with open(manifest_file) as f:
            manifest = _json.load(f)

        for entry in manifest:
            subject_id = entry.get("subject_id", entry.get("record_id", "unknown"))
            files = entry.get("files", {})

            # signal_types 중 필요한 파일이 모두 있는지 확인
            has_all = True
            for st in signal_types:
                if st not in files:
                    has_all = False
                    break
            if not has_all:
                continue

            subjects.append({"subject_id": str(subject_id), "files": files})
    else:
        # 디렉토리 스캔: subject_dir/{signal_type}.pt
        print(f"  No manifest.json, scanning directories...")
        for sub_dir in sorted(data_path.iterdir()):
            if not sub_dir.is_dir():
                continue
            files = {}
            has_all = True
            for st in signal_types:
                pt_file = sub_dir / f"{st}.pt"
                if pt_file.exists():
                    files[st] = str(pt_file)
                else:
                    has_all = False
                    break
            if has_all:
                subjects.append({"subject_id": sub_dir.name, "files": files})

    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    print(f"  Found {len(subjects)} subjects with {signal_types}")

    cases = []
    for i, subj in enumerate(subjects):
        subject_id = subj["subject_id"]
        files = subj["files"]

        signals = {}
        valid = True

        for st in signal_types:
            file_path = files[st]
            # Resolve relative paths against data_dir
            fp = Path(file_path)
            if not fp.is_absolute():
                fp = data_path / fp

            try:
                pt_data = torch.load(str(fp), weights_only=False)
            except Exception as e:
                print(f"  [{i + 1}] {subject_id}: SKIP ({st} load error: {e})")
                valid = False
                break

            # Extract values tensor
            if isinstance(pt_data, dict):
                values = pt_data.get("values", pt_data.get("signal", None))
                if values is None:
                    # Try first tensor value
                    for v in pt_data.values():
                        if isinstance(v, torch.Tensor):
                            values = v
                            break
            elif isinstance(pt_data, torch.Tensor):
                values = pt_data
            else:
                print(f"  [{i + 1}] {subject_id}: SKIP ({st} unknown format)")
                valid = False
                break

            if values is None or values.numel() < min_samples:
                valid = False
                break

            signals[st] = values.numpy().astype(np.float64).ravel()

        if not valid:
            continue

        # Align to same length
        min_len = min(len(s) for s in signals.values())
        if min_len < min_samples:
            continue
        signals = {k: v[:min_len] for k, v in signals.items()}

        dur_min = min_len / TARGET_SR / 60
        print(f"  [{i + 1}] {subject_id}: OK ({dur_min:.1f}min, {len(signals)} ch)")

        cases.append({
            "case_id": f"local_{subject_id}",
            "patient_id": str(subject_id),
            "signals": signals,
        })

    print(f"  Loaded {len(cases)} cases")
    return cases


# ---- 윈도우 추출 ----


def extract_aligned_windows(
    cases: list[dict],
    signal_types: list[str],
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
) -> list[dict]:
    """각 케이스에서 시간 정렬된 다채널 윈도우를 슬라이딩 추출한다.

    Returns
    -------
    list of {"case_id": str, "signals": {stype: array(win_samples,)}}
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)

    windows = []
    for case in cases:
        signals = case["signals"]
        n_total = min(len(signals[st]) for st in signal_types)

        if n_total < win_samples:
            continue

        for start in range(0, n_total - win_samples + 1, stride_samples):
            win = {}
            valid = True
            for st in signal_types:
                seg = signals[st][start : start + win_samples]
                # NaN 체크
                if np.isnan(seg).any():
                    valid = False
                    break
                win[st] = seg

            # 절대압 window 생리 게이트 (ABP 포함 시). clip 아니라 drop.
            if valid and "abp" in win and not _abp_window_ok(win["abp"]):
                valid = False

            if valid:
                windows.append(
                    {
                        "case_id": case["case_id"],
                        "patient_id": case["patient_id"],
                        "win_start": int(start),  # anchor 인과성 / 재현용
                        "signals": win,
                    }
                )

    return windows


# ---- 저장 ----


def save_dataset(
    train_windows: list[dict],
    test_windows: list[dict],
    signal_types: list[str],
    window_sec: float,
    source: str,
    out_dir: str,
    signal_dtype: torch.dtype = torch.float16,
    chunk_windows: int = 2000,
    val_windows: list[dict] | None = None,
    fold_idx: int | None = None,
    n_folds: int = 1,
) -> list[Path]:
    """윈도우 리스트를 split별 chunk .pt로 저장한다 (sepsis chunk 패턴과 동일).

    파일명:
      - 단일(fold 미사용): ``cross_modal_{source}_{types}_w{win}s_{split}_chunk{C}.pt``
      - K-fold:           ``cross_modal_{source}_{types}_w{win}s_fold{F}_{split}_chunk{C}.pt``

    각 chunk 는 최대 ``chunk_windows`` 개 윈도우를 담으며, chunk 단위로 텐서를
    destructive build → save → ``del + gc.collect()`` 하여 피크 메모리를 낮춘다.
    run.py 의 ``load_prepared_chunked()`` 가 split별로 다시 concat 한다.

    ``fold_idx`` 가 주어지면 파일명에 ``_fold{F}`` 가 붙고, ``val_windows`` 가
    있으면 train/val/test 3 split 을 저장한다 (classification fine-tuning 비교용).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    types_str = "_".join(signal_types)
    win_str = int(window_sec)
    base = f"cross_modal_{source}_{types_str}_w{win_str}s"
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""

    split_items: list[tuple[str, list[dict]]] = [("train", train_windows)]
    if val_windows is not None:
        split_items.append(("val", val_windows))
    split_items.append(("test", test_windows))

    # 전역 고정 ABP 정규화 스케일 = train split 에서만 산출 (val/test 동일 적용).
    # prepare 는 raw mmHg 를 저장하고, 이 (mu, sd) 는 meta 로만 전달 → 모델/eval 에서
    # 가역 affine 정규화. train ABP 를 pop 전에 계산해야 하므로 여기서 먼저 수행.
    abp_mu, abp_sd = _abp_train_scale(train_windows)

    saved: list[Path] = []
    for split_name, windows in split_items:
        if not windows:
            continue
        # 환자별 첫 gated-valid window(causal) = anchor. 원래 순서 정렬 배열.
        is_anchor_full = _mark_anchors(windows)
        chunk_idx = 0
        total = 0
        for bstart in range(0, len(windows), chunk_windows):
            batch = windows[bstart:bstart + chunk_windows]
            n_w = len(batch)
            case_ids = [w["case_id"] for w in batch]
            patient_ids = [str(w["patient_id"]) for w in batch]
            win_starts = [int(w["win_start"]) for w in batch]
            is_anchor_b = is_anchor_full[bstart:bstart + n_w].tolist()
            sig_tensors: dict[str, torch.Tensor] = {}
            for st in signal_types:
                # 각 윈도우 dict 에서 stype pop → numpy 즉시 해제하며 (n_w, T) build
                arrs = [w["signals"].pop(st) for w in batch]
                sig_tensors[st] = stack_arrays_destructive(arrs, signal_dtype)

            chunk = {
                "split": split_name,
                "signals": sig_tensors,          # {stype: (n_w, win_samples)} raw mmHg(ABP)
                "case_ids": case_ids,            # list[str], len == n_w
                "patient_ids": patient_ids,      # list[str] 실제 환자, len == n_w
                "win_starts": win_starts,        # list[int] 100Hz sample idx
                "is_anchor": is_anchor_b,        # list[bool] 환자별 첫 window
                "metadata": {
                    "task": "any_to_any_cross_modal",
                    "source": source,
                    "signal_types": signal_types,
                    "window_sec": window_sec,
                    "sampling_rate": TARGET_SR,
                    "fold_idx": fold_idx,
                    "n_folds": n_folds,
                    "split": split_name,
                    "chunk_idx": chunk_idx,
                    "n_windows": n_w,
                    "signal_dtype": str(signal_dtype).replace("torch.", ""),
                    "abp_mu_train": abp_mu,      # mmHg, train-only, freeze
                    "abp_sd_train": abp_sd,      # mmHg, train-only, freeze
                    "gating": ABP_GATE,          # 절대압 게이트 파라미터(재현성)
                    "align": "joint_load_v1",    # 정렬 프로비넌스
                    "abp_units": "mmHg_raw",     # ABP 는 raw mmHg (z-norm 안 함)
                },
            }
            out_file = (
                out_path / f"{base}{fold_suffix}_{split_name}_chunk{chunk_idx}.pt"
            )
            torch.save(chunk, out_file)
            file_size_mb = out_file.stat().st_size / (1024 * 1024)
            print(
                f"  Saved: {out_file.name} "
                f"({n_w} windows, {file_size_mb:.2f} MB)"
            )
            saved.append(out_file)
            total += n_w
            chunk_idx += 1
            del batch, sig_tensors, chunk
            gc.collect()
        print(f"  {split_name}: {chunk_idx} chunk(s), {total} windows")

    train_windows.clear()
    test_windows.clear()
    if val_windows is not None:
        val_windows.clear()
    gc.collect()
    return saved


# ---- 통계 출력 ----


def print_stats(
    name: str,
    windows: list[dict],
    signal_types: list[str],
) -> None:
    """윈도우 통계 출력."""
    if not windows:
        print(f"  {name}: 0 windows")
        return

    n = len(windows)
    case_ids = set(w["case_id"] for w in windows)
    print(f"  {name}: {n} windows from {len(case_ids)} cases")

    for st in signal_types:
        vals = np.concatenate([w["signals"][st] for w in windows])
        print(
            f"    {st:5s}: range=[{vals.min():.2f}, {vals.max():.2f}], "
            f"mean={vals.mean():.2f}, std={vals.std():.2f}"
        )


# ---- 메인 ----


def prepare_any_to_any(
    source: str = "vitaldb",
    signal_types: list[str] | None = None,
    n_cases: int = 10,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/any_to_any",
    manifest_path: str | None = None,
    data_dir: str | None = None,
    vital_dir: str | None = None,
    signal_dtype: torch.dtype = torch.float16,
    chunk_windows: int = 2000,
    n_folds: int = 1,
    seed: int = 42,
    clinical_csv: str | None = None,
) -> list[Path]:
    """Any-to-Any cross-modal 평가 데이터를 준비한다.

    Parameters
    ----------
    source : "vitaldb", "mimic3", 또는 "local".
    signal_types : 추출할 signal type 목록. None이면 ["ecg", "abp", "ppg", "cvp"].
    n_cases : 로드할 케이스 수.
    window_sec : 윈도우 길이 (초).
    stride_sec : 슬라이드 보폭 (초).
    train_ratio : patient 단위 train/test 분할 비율 (n_folds==1 일 때만).
    out_dir : 저장 디렉토리.
    manifest_path : MIMIC-III manifest 경로.
    data_dir : local source 시 .pt 디렉토리 경로.
    n_folds : >=2 면 patient-level K-fold CV (fold별 train/val/test chunk 저장).
        1 이면 기존 단일 train/test (fold suffix 없음, back-compat).
    seed : patient split RNG seed.
    clinical_csv : VitalDB caseid→subjectid 매핑용 로컬 clinical CSV 경로
        (source=vitaldb API 경로, 사내망 API 차단 시). None 이면 API 시도 후 fallback.
    """
    if signal_types is None:
        signal_types = ["ecg", "abp", "ppg", "cvp"]

    min_duration_sec = window_sec + stride_sec  # 최소 1 윈도우 + 1 stride
    types_str = " + ".join(s.upper() for s in signal_types)

    print(f"{'=' * 60}")
    print("  Task 8: Any-to-Any Cross-modal — Data Preparation")
    print(f"  Source:  {source}")
    print(f"  Signals: {types_str}")
    print(f"  Window:  {window_sec}s, Stride: {stride_sec}s")
    print(f"{'=' * 60}")

    # 1. 데이터 로드
    print("\n[1/4] Loading aligned multi-channel data...")
    if source == "mimic3":
        cases = _load_mimic3_cases(
            n_cases,
            signal_types,
            min_duration_sec,
            manifest_path,
            out_dir=str(Path(out_dir).parent / "mimic3"),
        )
    elif source == "vitaldb":
        cases = _load_vitaldb_cases(
            n_cases,
            signal_types,
            min_duration_sec,
            vital_dir=vital_dir,
            clinical_csv=clinical_csv,
        )
    elif source == "local":
        if not data_dir:
            print("ERROR: --data-dir required for source=local", file=sys.stderr)
            sys.exit(1)
        cases = _load_local_pt_cases(
            data_dir,
            signal_types,
            min_duration_sec,
            max_subjects=n_cases,
        )
    else:
        print(f"ERROR: Unknown source '{source}'", file=sys.stderr)
        sys.exit(1)

    if not cases:
        print("ERROR: No valid cases loaded.", file=sys.stderr)
        sys.exit(1)

    # ── K-fold CV (n_folds>=2): patient-level group K-fold ──────────────
    if n_folds >= 2:
        # generation 은 class label 이 없으므로 더미 uniform label →
        # stratified_kfold_patient_splits 가 plain group K-fold (60/20/20,
        # 모든 환자 정확히 1번 test) 로 동작. (sepsis prepare_data 호출 패턴 동일.)
        print(f"\n[2/4] Patient-level {n_folds}-fold CV split...")
        patient_ids = sorted({str(c["patient_id"]) for c in cases})
        patient_to_labels = {pid: [0] for pid in patient_ids}
        splits = stratified_kfold_patient_splits(
            patient_ids, patient_to_labels, n_folds=n_folds, seed=seed,
        )
        print(summarize_splits(splits))

        all_saved: list[Path] = []
        for fold_idx, (train_p, val_p, test_p) in enumerate(splits):
            print(f"\n[3-4/4] Fold {fold_idx}: extract windows + save...")
            fold_windows: dict[str, list[dict]] = {}
            for split_name, pset in (
                ("train", train_p), ("val", val_p), ("test", test_p),
            ):
                split_cases = [c for c in cases if str(c["patient_id"]) in pset]
                # 환자 단위로 case 분리 후 윈도우 추출 → fold 간 누수 없음.
                fold_windows[split_name] = extract_aligned_windows(
                    split_cases, signal_types, window_sec, stride_sec
                )
            if not any(fold_windows.values()):
                print(f"  WARN: fold {fold_idx} produced no windows — skip.")
                continue
            # 공정성 게이트: 환자 disjoint + 환자당 anchor 1개 (실패=abort)
            _assert_no_patient_leak(fold_windows)
            saved = save_dataset(
                fold_windows["train"],
                fold_windows["test"],
                signal_types,
                window_sec,
                source,
                out_dir,
                signal_dtype=signal_dtype,
                chunk_windows=chunk_windows,
                val_windows=fold_windows["val"],
                fold_idx=fold_idx,
                n_folds=n_folds,
            )
            all_saved.extend(saved)

        print(f"\n{'=' * 60}")
        print(f"  Done! {n_folds}-fold CV, {len(all_saved)} chunk file(s) in {out_dir}")
        print(f"{'=' * 60}")
        return all_saved

    # ── 단일 train/test (n_folds==1, back-compat) ───────────────────────
    # 2. Patient 단위 train/test 분할
    print(f"\n[2/4] Splitting by patient (ratio={train_ratio})...")
    rng = np.random.default_rng(seed)
    patient_ids = list({c["patient_id"] for c in cases})
    rng.shuffle(patient_ids)
    n_train_patients = max(1, int(len(patient_ids) * train_ratio))
    train_patients = set(patient_ids[:n_train_patients])

    train_cases = [c for c in cases if c["patient_id"] in train_patients]
    test_cases = [c for c in cases if c["patient_id"] not in train_patients]
    print(f"  Train: {len(train_cases)} cases ({len(train_patients)} patients)")
    print(
        f"  Test:  {len(test_cases)} cases ({len(patient_ids) - len(train_patients)} patients)"
    )

    # 3. 윈도우 추출
    print(
        f"\n[3/4] Extracting aligned windows (window={window_sec}s, stride={stride_sec}s)..."
    )
    train_windows = extract_aligned_windows(
        train_cases, signal_types, window_sec, stride_sec
    )
    test_windows = extract_aligned_windows(
        test_cases, signal_types, window_sec, stride_sec
    )

    print_stats("Train", train_windows, signal_types)
    print_stats("Test", test_windows, signal_types)

    if not train_windows and not test_windows:
        print("ERROR: No windows extracted.", file=sys.stderr)
        sys.exit(1)

    # 공정성 게이트: 환자 disjoint + 환자당 anchor 1개 (실패=abort)
    _assert_no_patient_leak({"train": train_windows, "test": test_windows})

    # 4. 저장 (split별 chunk)
    print("\n[4/4] Saving...")
    n_train, n_test = len(train_windows), len(test_windows)
    saved_paths = save_dataset(
        train_windows,
        test_windows,
        signal_types,
        window_sec,
        source,
        out_dir,
        signal_dtype=signal_dtype,
        chunk_windows=chunk_windows,
    )

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(saved_paths)} chunk file(s) in {out_dir}")
    print(f"  Train: {n_train} windows, Test: {n_test} windows")
    print(f"{'=' * 60}")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 8: Any-to-Any Cross-modal — Data Preparation",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="vitaldb",
        choices=["mimic3", "vitaldb", "local"],
        help="Data source",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local .pt directory path (required for source=local)",
    )
    parser.add_argument(
        "--vital-dir",
        type=str,
        default=None,
        help="로컬 VitalDB Open .vital 파일 디렉토리 (source=vitaldb 시 API 대신 사용)",
    )
    parser.add_argument(
        "--signal-types",
        nargs="+",
        default=["ecg", "abp", "ppg", "cvp", "co2", "resp_impedance", "icp"],
        choices=["ecg", "abp", "ppg", "cvp", "co2", "awp",
                 "resp_impedance", "resp_flow", "icp"],
        help="Signal types to extract (Task #12 default: 7 modality — "
             "ECG, ABP, PPG, CVP, CO2, RESP_Impedance, ICP). "
             "v2: RESP 가 resp_impedance / resp_flow 로 분리, PAP 제거됨.",
    )
    parser.add_argument(
        "--n-cases", type=int, default=10, help="Number of cases to load"
    )
    parser.add_argument(
        "--window-sec", type=float, default=30.0, help="Window length in seconds"
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=15.0,
        help="Sliding window stride in seconds",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train/test split ratio (patient-level)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/downstream/any_to_any",
        help="Output directory",
    )
    parser.add_argument(
        "--manifest", type=str, default=None, help="MIMIC-III manifest JSON path"
    )
    parser.add_argument(
        "--chunk-windows",
        type=int,
        default=2000,
        help="Windows per split chunk file (sepsis-style chunked save, "
             "lowers peak RAM/disk). Default 2000.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Patient-level K-fold CV. >=2 면 fold별 train/val/test chunk 저장 "
             "(파일명에 _fold{F}). 1 이면 단일 train/test (train_ratio, back-compat). "
             "Default 5.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Patient split RNG seed.",
    )
    parser.add_argument(
        "--clinical-csv",
        type=str,
        default=None,
        help="VitalDB caseid→subjectid 매핑용 로컬 clinical CSV 경로 "
             "(사내망 api.vitaldb.net 차단 시). 없으면 API 시도 후 case_id fallback.",
    )
    dtype_map = add_signal_dtype_arg(parser)
    args = parser.parse_args()

    prepare_any_to_any(
        source=args.source,
        signal_types=args.signal_types,
        n_cases=args.n_cases,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        vital_dir=args.vital_dir,
        signal_dtype=dtype_map(args.signal_dtype),
        chunk_windows=args.chunk_windows,
        n_folds=args.n_folds,
        seed=args.seed,
        clinical_csv=args.clinical_csv,
    )


if __name__ == "__main__":
    main()
