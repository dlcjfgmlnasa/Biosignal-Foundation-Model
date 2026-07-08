# -*- coding:utf-8 -*-
"""Intracranial Hypertension **Prediction** — 양성 표본 희소성 빠른 스캔 (Güiza식).

full `prepare_data.py` 를 돌리기 전에, ICP 채널만 읽어서 (ABP/ECG 100Hz 파이프라인
스킵 → 수십 배 빠름) 주어진 (window, horizon) 스펙에서 양성 window 가 실제로 몇 개나
나오는지 **상한**을 빠르게 추정한다.

라벨 엔진은 Güiza et al. (Crit Care Med 2013) ICP-crisis 정의를 따르는 2단 구조:
  [1단 · 이벤트 확정]  ICP 를 1분 **median** 블록으로 요약(아티팩트 견고) → 5블록
      연속(=5분) median > 20 mmHg 이면 ICP crisis 로 확정. run 의 첫 블록 = **onset**.
      (평균 대신 median: 석션·플러시·체위변경 순간 스파이크에 오염되지 않음.)
  [2단 · 예측 라벨]   입력 window 끝 T 에서 이미 crisis(≤20 아님) 면 제외(현재-ICH
      제외). 이후 horizon H분 안에 **신규 onset** 이 있으면 positive, 없으면 negative.
      onset anchor 로 진행 중 위기의 중복 계수를 차단 → 양성 상한이 명확.

⚠️ 상한 주의: 여기서는 ICP 유효성만 본다. 실제 prepare_data 는 입력 채널(ABP 등)
valid_ratio 도 요구하므로, 실제 양성 수는 이 값보다 **작거나 같다**. "이 스펙으로
양성이 애초에 나오긴 하나"를 몇 분 안에 판단하는 용도.

사용법:
    # 새 스펙 (20min window, horizon 5/10/15, event=5×1min median >20) 전체 스캔
    python -m downstream.acute_event.intracranial_hypertension.scan_positives \
        --waveform-dir datasets/raw/mimic3-waveform-ich \
        --window-secs 1200 --horizon-mins 5 10 15 --stride-sec 60

    # 초고속 첫 확인 — 앞 30개 레코드만
    python -m downstream.acute_event.intracranial_hypertension.scan_positives \
        --waveform-dir datasets/raw/mimic3-waveform-ich \
        --window-secs 1200 --horizon-mins 5 10 15 --max-records 30
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from downstream.acute_event.intracranial_hypertension.prepare_data import (
    ICP_THRESHOLD,
)

BLOCK_SEC: float = 60.0   # Güiza 1분 median 블록
EVENT_CONSEC: int = 5     # 5블록 연속(=5분) > threshold = crisis 확정


# ── ICP-only 빠른 로딩 (1분 median 블록) ────────────────────


def read_icp_blocks(
    record_dir: Path, record_name: str,
) -> tuple[np.ndarray | None, str]:
    """WFDB 레코드에서 ICP 채널만 읽어 1분-median 블록 timeline 을 반환한다.

    ⚠️ MIMIC waveform 은 multi-segment 라 master .hea 엔 sig_name 이 없고 신호는
    _layout/세그먼트에 정의된다. 따라서 rdheader 게이트로 채널을 확인하면 전부
    걸러진다 → 게이트 없이 rdrecord(channel_names=["ICP"]) 로 직접 조립한다
    (ICP-RECORDS 는 이미 ICP 보유 레코드만 추린 목록).

    Returns
    -------
    (blocks, reason) — 성공 시 (ndarray(n_blocks,), "ok"), 실패 시 (None, reason).
    reason ∈ {no_hea, read_fail, empty, short}.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    hea_path = record_dir / f"{record_name}.hea"
    if not hea_path.exists():
        return None, "no_hea"

    try:
        rec = wfdb.rdrecord(str(record_dir / record_name), channel_names=["ICP"])
    except Exception:
        return None, "read_fail"  # ICP 채널 없음 / .dat 누락 / 파싱 오류 포함

    if rec.p_signal is None or rec.sig_len == 0 or rec.p_signal.shape[1] == 0:
        return None, "empty"

    fs = float(rec.fs)
    icp = rec.p_signal[:, 0].astype(np.float64)  # (N,)

    block_samps = max(1, int(round(BLOCK_SEC * fs)))
    n_blocks = len(icp) // block_samps
    if n_blocks == 0:
        return None, "short"

    trimmed = icp[: n_blocks * block_samps].reshape(n_blocks, block_samps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN 블록
        block_med = np.nanmedian(trimmed, axis=1)  # (n_blocks,)
    return block_med.astype(np.float32), "ok"


# ── 레코드 목록 (load_patient_signals 와 동일 규칙) ──────────


def _patient_id_from_parts(parts: tuple[str, ...], rec_name: str) -> str:
    """경로 조각에서 7자리 pXXXXXX subject dir 를 찾는다(없으면 rec_name prefix)."""
    for part in parts:
        if part.startswith("p") and len(part) == 7 and part[1:].isdigit():
            return part
    return rec_name.split("-", 1)[0]


_SEG_RE = re.compile(r"_\d{4}$")           # 세그먼트 헤더 접미(_0001)
_MASTER_NUM_RE = re.compile(r"^\d+$")       # 일반 mimic3wdb 숫자 레코드(3008477)
_MASTER_MATCHED_RE = re.compile(            # matched subset 레코드(p000907-2163-...)
    r"^p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$"
)


def list_records_scandir(waveform_dir: Path) -> list[tuple[str, Path, str]]:
    """waveform_dir **최상위 한 단계**를 scandir(재귀·stat 없음)해 마스터 헤더 열거.

    ICP-RECORDS(matched 경로)와 실제 디스크 레이아웃이 다를 때 사용 — 특히 flat
    numeric(3008477.hea/_0001.dat, 일반 mimic3wdb) 레이아웃. matched(pXXXXXX-...)
    도 함께 인식한다. _layout / _NNNN 세그먼트 헤더는 제외.

    patient_id: matched 는 pXXXXXX 추출, numeric 은 subject 링크가 없어 record
    stem 을 그대로 pseudo-patient 로 둔다(스캔 feasibility 용 — CV 그룹핑엔 부적합).
    """
    import os

    out: list[tuple[str, Path, str]] = []
    with os.scandir(waveform_dir) as it:
        for entry in it:
            name = entry.name
            if not name.endswith(".hea"):
                continue
            stem = name[:-4]
            if stem.endswith("_layout") or _SEG_RE.search(stem):
                continue
            m = _MASTER_MATCHED_RE.match(stem)
            if m:
                pid = stem.split("-", 1)[0]  # pXXXXXX
            elif _MASTER_NUM_RE.match(stem):
                pid = stem                    # 숫자 레코드 → 자기 자신이 group
            else:
                continue
            out.append((pid, waveform_dir, stem))
    return sorted(out)


def list_records_from_file(
    waveform_dir: Path, records_file: Path,
) -> list[tuple[str, Path, str]]:
    """RECORDS 파일(각 줄=`p00/p000907/p000907-2163-...`)에서 목록 구성.

    네트워크 마운트 전체 트리를 rglob 하지 않아 **즉시 시작**한다(rglob 은 마운트
    에서 재귀 디렉토리 스캔이 수 분 걸려 "멈춘 것처럼" 보이는 주범).
    """
    out: list[tuple[str, Path, str]] = []
    for line in records_file.read_text(encoding="utf-8").splitlines():
        rec_path = line.strip()
        if not rec_path:
            continue
        rel = Path(rec_path)
        rec_name = rel.name
        rec_dir = waveform_dir / rel.parent
        patient_id = _patient_id_from_parts(rel.parts, rec_name)
        out.append((patient_id, rec_dir, rec_name))
    return out


def list_records(waveform_dir: Path) -> list[tuple[str, Path, str]]:
    """(patient_id, record_dir, record_name) 목록. master .hea 만 선별 (rglob).

    ⚠️ 네트워크 마운트에서 느림 — 가능하면 list_records_from_file 을 쓸 것.
    """
    master_re = re.compile(r"^p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$")
    all_hea = sorted(
        h for h in waveform_dir.rglob("*.hea") if master_re.match(h.stem)
    )
    out: list[tuple[str, Path, str]] = []
    for hea in all_hea:
        rec_name = hea.stem
        patient_id = _patient_id_from_parts(hea.parts, rec_name)
        out.append((patient_id, hea.parent, rec_name))
    return out


# ── 1단: 이벤트 확정 (Güiza onset) ──────────────────────────


def find_onsets(blocks: np.ndarray, threshold: float, consec: int) -> list[int]:
    """5블록 연속 median>threshold 인 crisis run 의 onset(첫 블록) index 목록.

    유효하지 않은(NaN) 블록은 run 을 끊는다(보수적). run 이 consec 에 도달하는
    순간 run_start 를 onset 으로 확정(= ICP 가 처음 임계를 넘은 시점, 5분 전).
    한 run 은 onset 하나만 기여 → 진행 중 위기 중복 계수 차단.
    """
    onsets: list[int] = []
    run = 0
    run_start: int | None = None
    for i, v in enumerate(blocks):
        if not np.isnan(v) and v > threshold:
            if run == 0:
                run_start = i
            run += 1
            if run == consec:
                onsets.append(run_start)  # type: ignore[arg-type]
        else:
            run = 0
            run_start = None
    return onsets


# ── 2단: 예측 라벨 window 시뮬레이션 ────────────────────────


@dataclass
class HorizonStat:
    n_windows: int = 0
    n_pos: int = 0
    n_neg: int = 0
    n_dropped_baseline: int = 0  # 현재-ICH 제외 or baseline 무효로 drop
    pos_records: set = field(default_factory=set)
    pos_patients: set = field(default_factory=set)


def _baseline_ok(blocks: np.ndarray, win_end: int, threshold: float) -> bool:
    """현재-ICH 제외: 입력 끝(마지막 window 블록) 유효 + median ≤ threshold."""
    base = blocks[win_end - 1]
    return not np.isnan(base) and float(base) <= threshold


def scan_record(
    blocks: np.ndarray,
    patient_id: str,
    record_id: str,
    onsets: list[int],
    window_sec: float,
    horizon_secs: list[float],
    stride_sec: float,
    threshold: float,
    stats: dict[float, HorizonStat],
    sampling: str = "dense",
    neg_stride_sec: float = 300.0,
) -> None:
    """1분-median 블록 timeline 에서 horizon 별 양/음성 window 를 센다.

    sampling:
      "dense"    — stride_sec 마다 dense sliding. positive = horizon 안에 onset,
                   나머지 전부 negative(현실 유병률 그대로, 중복·불균형 큼).
      "anchored" — 양성은 onset 마다 목표 lead(horizon) 에서 끝나는 window 1개로
                   고정(중복 제거, onset 보존). 음성은 crisis 없는 구간에서
                   neg_stride_sec 균일 stride 로만 추출(경계 15~20 도 음성 유지 →
                   정직). biased 모드(clean-<15 필터·cherry-pick)와 다름.
    """
    win_b = max(1, int(round(window_sec / BLOCK_SEC)))
    L = len(blocks)
    onset_list = sorted(onsets)

    for horizon_sec in horizon_secs:
        hor_b = max(1, int(round(horizon_sec / BLOCK_SEC)))
        st = stats[horizon_sec]
        total_needed = win_b + hor_b
        if L < total_needed:
            continue

        if sampling == "anchored":
            # ── 양성: onset 당 1개, horizon 끝 블록 = onset (정확히 H분 전 예측) ──
            for o in onset_list:
                win_end = o - hor_b + 1        # horizon last block == o
                start = win_end - win_b
                if start < 0:
                    continue                    # 앞 데이터 부족 → anchor 불가
                if not _baseline_ok(blocks, win_end, threshold):
                    st.n_dropped_baseline += 1   # 예측 시점 이미 crisis → 제외
                    continue
                st.n_pos += 1
                st.pos_records.add(record_id)
                st.pos_patients.add(patient_id)
            # ── 음성: crisis 없는 구간, 균일 stride, 경계 유지(정직) ──
            neg_stride_b = max(1, int(round(neg_stride_sec / BLOCK_SEC)))
            for start in range(0, L - total_needed + 1, neg_stride_b):
                win_end = start + win_b
                if not _baseline_ok(blocks, win_end, threshold):
                    continue
                if any(win_end <= o < win_end + hor_b for o in onset_list):
                    continue                    # horizon 에 onset → 음성 아님
                horizon = blocks[win_end: win_end + hor_b]
                if np.isnan(horizon).all():
                    continue                    # 미래 데이터 전무 → skip
                st.n_neg += 1
            continue

        # ── dense (기본) ──
        stride_b = max(1, int(round(stride_sec / BLOCK_SEC)))
        for start in range(0, L - total_needed + 1, stride_b):
            win_end = start + win_b
            if not _baseline_ok(blocks, win_end, threshold):
                st.n_dropped_baseline += 1
                continue
            st.n_windows += 1
            is_pos = any(win_end <= o < win_end + hor_b for o in onset_list)
            if is_pos:
                st.n_pos += 1
                st.pos_records.add(record_id)
                st.pos_patients.add(patient_id)
            else:
                st.n_neg += 1


# ── 리포트 ───────────────────────────────────────────────────


def print_report(
    stats: dict[float, HorizonStat],
    window_sec: float,
    threshold: float,
    stride_sec: float,
    n_records_scanned: int,
    n_records_with_icp: int,
    n_patients_with_icp: int,
    total_onsets: int,
    onset_records: int,
    onset_patients: int,
    sampling: str = "dense",
    neg_stride_sec: float = 300.0,
    neg_ratio: float | None = None,
) -> None:
    print(f"\n{'=' * 78}")
    print("  ICH Prediction — 양성 희소성 스캔 (Güiza식, ICP-only 상한)")
    print(f"{'=' * 78}")
    smp = (f"sampling={sampling}"
           + (f"(neg stride={neg_stride_sec:.0f}s)" if sampling == "anchored"
              else f"(dense stride={stride_sec:.0f}s)"))
    cap = f"  neg-ratio cap=1:{neg_ratio:.0f}" if neg_ratio else ""
    print(f"  Window={window_sec:.0f}s ({window_sec / 60:.0f}min)  "
          f"event=5×1min median >{threshold:.0f}mmHg  {smp}{cap}")
    print(f"  Records scanned={n_records_scanned}  with ICP={n_records_with_icp}  "
          f"patients with ICP={n_patients_with_icp}")
    print(f"{'-' * 78}")
    # [1단] 확정 이벤트 = 양성의 진짜 천장
    print(f"  [이벤트 확정]  총 crisis onset={total_onsets}  "
          f"onset 보유 레코드={onset_records}  onset 보유 환자={onset_patients}")
    print(f"{'-' * 78}")
    # [2단] horizon 별 양/음성 (NEG(pool)=전체 crisis-free 풀, NEG(kept)=cap 적용 후)
    print(f"  {'horizon':>8} | {'POS':>7} | {'NEG pool':>9} | {'NEG kept':>9} | "
          f"{'ratio':>8} | {'pos pt':>6}")
    print(f"  {'-' * 8} | {'-' * 7} | {'-' * 9} | {'-' * 9} | "
          f"{'-' * 8} | {'-' * 6}")
    for horizon_sec in sorted(stats):
        st = stats[horizon_sec]
        # cap: 양성당 최대 neg_ratio 개 (전역 균일 서브샘플의 결과 개수)
        if neg_ratio and st.n_pos:
            kept = min(st.n_neg, int(round(neg_ratio * st.n_pos)))
        else:
            kept = st.n_neg
        ratio = (kept / st.n_pos) if st.n_pos else float("inf")
        ratio_s = f"1:{ratio:.1f}" if st.n_pos else "1:inf"
        print(f"  {horizon_sec / 60:6.0f}min | {st.n_pos:7d} | {st.n_neg:9d} | "
              f"{kept:9d} | {ratio_s:>8} | {len(st.pos_patients):6d}")
    print(f"{'-' * 78}")
    print("  주: POS 는 ICP 유효성만 반영한 상한. 실제 prepare_data 는 입력 채널")
    print("      (ABP) valid_ratio 도 요구 → 실제 양성 ≤ 위 값.")
    if sampling == "anchored":
        print("  anchored: POS=onset 당 1개(중복 제거), NEG pool=crisis-free 균일 stride.")
    if neg_ratio:
        print(f"  NEG kept = min(pool, {neg_ratio:.0f}×POS) — prepare_data 가 풀에서 "
              "전역 균일 서브샘플해 이 개수만 저장(정직: 경계 15~20 도 후보 포함).")
    print("  판단 가이드: 5-fold stratified CV 안정성엔 'onset 보유 환자' ≳ 15~20 권장.")
    print(f"{'=' * 78}")


# ── 병렬 워커 (네트워크 마운트 I/O 겹치기) ──────────────────


def process_record(
    item: tuple[str, Path, str],
    window_sec: float,
    horizon_secs: list[float],
    stride_sec: float,
    threshold: float,
    event_consec: int,
    sampling: str = "dense",
    neg_stride_sec: float = 300.0,
) -> tuple[str, str, str, int, dict[float, HorizonStat] | None]:
    """레코드 1개를 읽어 onset + horizon 별 window 통계를 독립 계산(스레드 안전).

    read_icp_blocks(느린 마운트 I/O) 를 worker 스레드로 겹치고, 순수 계산
    (find_onsets/scan_record) 결과를 per-record 로 반환 → 메인이 병합.

    Returns (reason, rec_id, pid, n_onsets, local). 실패 시 local=None,
    reason ∈ {no_hea, read_fail, empty, short}.
    """
    pid, rec_dir, rec_name = item
    rec_id = f"{pid}_{rec_name}"
    blocks, reason = read_icp_blocks(rec_dir, rec_name)
    if blocks is None:
        return reason, rec_id, pid, 0, None
    onsets = find_onsets(blocks, threshold, event_consec)
    local: dict[float, HorizonStat] = {h: HorizonStat() for h in horizon_secs}
    scan_record(
        blocks, pid, rec_id, onsets,
        window_sec, horizon_secs, stride_sec, threshold, local,
        sampling=sampling, neg_stride_sec=neg_stride_sec,
    )
    return "ok", rec_id, pid, len(onsets), local


def _merge(dst: HorizonStat, src: HorizonStat) -> None:
    dst.n_windows += src.n_windows
    dst.n_pos += src.n_pos
    dst.n_neg += src.n_neg
    dst.n_dropped_baseline += src.n_dropped_baseline
    dst.pos_records |= src.pos_records
    dst.pos_patients |= src.pos_patients


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICH Prediction 양성 희소성 빠른 스캔 (Güiza 1분-median onset)",
    )
    parser.add_argument(
        "--waveform-dir", type=str,
        default="datasets/raw/mimic3-waveform-ich",
    )
    parser.add_argument("--window-secs", type=float, default=1200.0,
                        help="입력 window 길이(초). 기본 1200(20min).")
    parser.add_argument("--horizon-mins", nargs="+", type=float,
                        default=[5.0, 10.0, 15.0])
    parser.add_argument("--stride-sec", type=float, default=60.0,
                        help="dense 모드 sliding stride(초). 기본 60.")
    parser.add_argument("--sampling", type=str, default="dense",
                        choices=["dense", "anchored"],
                        help="dense(현실 유병률·중복 큼) | anchored(onset 당 양성 1개 "
                             "+ crisis-free 균일 stride 음성; 제안 설계).")
    parser.add_argument("--neg-stride-sec", type=float, default=300.0,
                        help="anchored 모드 음성 균일 stride(초). 기본 300(5min).")
    parser.add_argument("--neg-ratio", type=float, default=None,
                        help="양성당 음성 캡(예: 10 → 1:10). prepare_data 가 풀에서 "
                             "전역 균일 서브샘플해 저장할 최종 개수. 미지정 시 풀 전체.")
    parser.add_argument("--icp-threshold", type=float, default=ICP_THRESHOLD)
    parser.add_argument("--event-consec", type=int, default=EVENT_CONSEC,
                        help="crisis 확정 연속 1분블록 수. 기본 5(=5min). "
                             "완화 시(예: 3=3min) 양성 확보.")
    parser.add_argument("--max-records", type=int, default=None,
                        help="앞 N개 레코드만 스캔(초고속 첫 확인).")
    parser.add_argument("--workers", type=int, default=8,
                        help="병렬 읽기 스레드 수(네트워크 마운트 I/O 겹치기). "
                             "기본 8. 마운트가 느리면 16~32 로 올려도 됨.")
    parser.add_argument(
        "--records-file", type=str,
        default=str(Path(__file__).parent / "ICP-RECORDS"),
        help="레코드 경로 목록 파일(각 줄 p00/p000907/p000907-...). 기본=번들 "
             "ICP-RECORDS. 마운트 전체 rglob 을 회피해 즉시 시작(권장).",
    )
    parser.add_argument("--rglob", action="store_true",
                        help="records-file 대신 waveform-dir 를 rglob 스캔(느림).")
    parser.add_argument("--scan-dir", action="store_true",
                        help="waveform-dir 최상위를 scandir 로 마스터 헤더 직접 열거"
                             "(재귀 없음). flat numeric(3008477) 등 ICP-RECORDS 경로와"
                             " 레이아웃이 다를 때 사용.")
    args = parser.parse_args()

    waveform_dir = Path(args.waveform_dir)
    if not waveform_dir.exists():
        print(f"ERROR: waveform-dir 없음: {waveform_dir}", file=sys.stderr)
        sys.exit(1)

    horizon_secs = [m * 60.0 for m in args.horizon_mins]
    records_file = Path(args.records_file)
    if args.scan_dir:
        print(f"  scandir 마스터 헤더 열거: {waveform_dir} (재귀 없음)...", flush=True)
        records = list_records_scandir(waveform_dir)
    elif not args.rglob and records_file.is_file():
        print(f"  Reading record list from {records_file} (rglob 회피, 즉시 시작)",
              flush=True)
        records = list_records_from_file(waveform_dir, records_file)
    else:
        if not args.rglob:
            print(f"  WARN: records-file 없음({records_file}) → rglob 폴백",
                  file=sys.stderr, flush=True)
        print(f"  rglob scanning {waveform_dir} (마운트에서 수 분 걸릴 수 있음)...",
              flush=True)
        records = list_records(waveform_dir)
    if args.max_records is not None:
        records = records[: args.max_records]
    print(f"  {len(records)} records to scan (workers={args.workers})", flush=True)

    stats: dict[float, HorizonStat] = {h: HorizonStat() for h in horizon_secs}
    icp_records: set[str] = set()
    icp_patients: set[str] = set()
    total_onsets = 0
    onset_records: set[str] = set()
    onset_patients: set[str] = set()
    per_patient_onsets: dict[str, int] = defaultdict(int)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    def _submit(ex: ThreadPoolExecutor, item: tuple[str, Path, str]):
        return ex.submit(
            process_record, item,
            args.window_secs, horizon_secs, args.stride_sec,
            args.icp_threshold, args.event_consec,
            args.sampling, args.neg_stride_sec,
        )

    reason_counts: dict[str, int] = defaultdict(int)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {_submit(ex, item): item for item in records}
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="Scanning ICP", unit="rec")
        for fut in pbar:
            reason, rec_id, pid, n_onsets, local = fut.result()
            reason_counts[reason] += 1
            if reason != "ok":
                continue

            icp_records.add(rec_id)
            icp_patients.add(pid)
            if n_onsets:
                total_onsets += n_onsets
                onset_records.add(rec_id)
                onset_patients.add(pid)
                per_patient_onsets[pid] += n_onsets
            for h in horizon_secs:
                _merge(stats[h], local[h])

            # 진행 중 누적 지표를 progress bar 오른쪽에 표시
            pbar.set_postfix(
                icp=len(icp_records),
                onset=total_onsets,
                onset_pt=len(onset_patients),
            )

    # 로딩 사유 요약 (0 이 나올 때 원인 파악용: no_hea=경로/레이아웃 불일치 등)
    print("  load reasons: " + "  ".join(
        f"{k}={reason_counts[k]}" for k in sorted(reason_counts)
    ), flush=True)

    print_report(
        stats, args.window_secs, args.icp_threshold, args.stride_sec,
        len(records), len(icp_records), len(icp_patients),
        total_onsets, len(onset_records), len(onset_patients),
        sampling=args.sampling, neg_stride_sec=args.neg_stride_sec,
        neg_ratio=args.neg_ratio,
    )


if __name__ == "__main__":
    main()
