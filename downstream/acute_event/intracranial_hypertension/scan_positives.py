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


def read_icp_blocks(record_dir: Path, record_name: str) -> np.ndarray | None:
    """WFDB 레코드에서 ICP 채널만 읽어 1분-median 블록 timeline 을 반환한다.

    Returns
    -------
    np.ndarray (n_blocks,) — 각 원소는 1분 구간 median ICP(mmHg), 전부 NaN 인
    구간은 NaN. ICP 채널이 없으면 None.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    hea_path = record_dir / f"{record_name}.hea"
    if not hea_path.exists():
        return None

    # 헤더만 먼저 읽어 ICP 채널 존재 확인 (없으면 무거운 rdrecord 스킵)
    try:
        hdr = wfdb.rdheader(str(record_dir / record_name), rd_segments=False)
    except Exception:
        return None
    if "ICP" not in (hdr.sig_name or []):
        return None

    try:
        rec = wfdb.rdrecord(str(record_dir / record_name), channel_names=["ICP"])
    except Exception:
        return None

    if rec.p_signal is None or rec.sig_len == 0:
        return None

    fs = float(rec.fs)
    icp = rec.p_signal[:, 0].astype(np.float64)  # (N,)

    block_samps = max(1, int(round(BLOCK_SEC * fs)))
    n_blocks = len(icp) // block_samps
    if n_blocks == 0:
        return None

    trimmed = icp[: n_blocks * block_samps].reshape(n_blocks, block_samps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN 블록
        block_med = np.nanmedian(trimmed, axis=1)  # (n_blocks,)
    return block_med.astype(np.float32)


# ── 레코드 목록 (load_patient_signals 와 동일 규칙) ──────────


def list_records(waveform_dir: Path) -> list[tuple[str, Path, str]]:
    """(patient_id, record_dir, record_name) 목록. master .hea 만 선별."""
    master_re = re.compile(r"^p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$")
    all_hea = sorted(
        h for h in waveform_dir.rglob("*.hea") if master_re.match(h.stem)
    )
    out: list[tuple[str, Path, str]] = []
    for hea in all_hea:
        rec_name = hea.stem
        patient_id = "unknown"
        for part in hea.parts:
            if part.startswith("p") and len(part) == 7 and part[1:].isdigit():
                patient_id = part
                break
        if patient_id == "unknown":
            patient_id = rec_name.split("-", 1)[0]
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
) -> None:
    """1분-median 블록 timeline 에서 horizon 별 양/음성 window 를 센다."""
    win_b = max(1, int(round(window_sec / BLOCK_SEC)))
    stride_b = max(1, int(round(stride_sec / BLOCK_SEC)))
    L = len(blocks)
    onset_set = set(onsets)

    for horizon_sec in horizon_secs:
        hor_b = max(1, int(round(horizon_sec / BLOCK_SEC)))
        st = stats[horizon_sec]
        total_needed = win_b + hor_b
        if L < total_needed:
            continue

        for start in range(0, L - total_needed + 1, stride_b):
            win_end = start + win_b  # horizon 시작 블록 (exclusive window 끝)
            # 현재-ICH 제외: 입력 끝(마지막 window 블록)이 유효 + median ≤ threshold
            base = blocks[win_end - 1]
            if np.isnan(base) or float(base) > threshold:
                st.n_dropped_baseline += 1
                continue

            st.n_windows += 1
            # positive: horizon [win_end, win_end+hor_b) 안에 신규 onset 존재
            is_pos = any(win_end <= o < win_end + hor_b for o in onset_set)
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
) -> None:
    print(f"\n{'=' * 70}")
    print("  ICH Prediction — 양성 희소성 스캔 (Güiza식, ICP-only 상한)")
    print(f"{'=' * 70}")
    print(f"  Window={window_sec:.0f}s ({window_sec / 60:.0f}min)  "
          f"event=5×1min median >{threshold:.0f}mmHg  stride={stride_sec:.0f}s")
    print(f"  Records scanned={n_records_scanned}  with ICP={n_records_with_icp}  "
          f"patients with ICP={n_patients_with_icp}")
    print(f"{'-' * 70}")
    # [1단] 확정 이벤트 = 양성의 진짜 천장
    print(f"  [이벤트 확정]  총 crisis onset={total_onsets}  "
          f"onset 보유 레코드={onset_records}  onset 보유 환자={onset_patients}")
    print(f"  → 양성 window 는 이 onset 들 주변에서만 나온다(각 onset ≤ 1개 양성 band).")
    print(f"{'-' * 70}")
    # [2단] horizon 별 양성 window
    print(f"  {'horizon':>8} | {'windows':>9} | {'POS':>6} | {'pos%':>6} | "
          f"{'pos rec':>7} | {'pos pt':>6}")
    print(f"  {'-' * 8} | {'-' * 9} | {'-' * 6} | {'-' * 6} | "
          f"{'-' * 7} | {'-' * 6}")
    for horizon_sec in sorted(stats):
        st = stats[horizon_sec]
        pos_rate = (100.0 * st.n_pos / st.n_windows) if st.n_windows else 0.0
        print(f"  {horizon_sec / 60:6.0f}min | {st.n_windows:9d} | "
              f"{st.n_pos:6d} | {pos_rate:5.2f}% | "
              f"{len(st.pos_records):7d} | {len(st.pos_patients):6d}")
    print(f"{'-' * 70}")
    print("  주: POS 는 ICP 유효성만 반영한 상한. 실제 prepare_data 는 입력 채널")
    print("      (ABP) valid_ratio 도 요구 → 실제 양성 ≤ 위 값.")
    print("  판단 가이드: 5-fold stratified CV 안정성엔 'onset 보유 환자' ≳ 15~20 권장.")
    print(f"{'=' * 70}")


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
    parser.add_argument("--stride-sec", type=float, default=60.0)
    parser.add_argument("--icp-threshold", type=float, default=ICP_THRESHOLD)
    parser.add_argument("--event-consec", type=int, default=EVENT_CONSEC,
                        help="crisis 확정 연속 1분블록 수. 기본 5(=5min). "
                             "완화 시(예: 3=3min) 양성 확보.")
    parser.add_argument("--max-records", type=int, default=None,
                        help="앞 N개 레코드만 스캔(초고속 첫 확인).")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    waveform_dir = Path(args.waveform_dir)
    if not waveform_dir.exists():
        print(f"ERROR: waveform-dir 없음: {waveform_dir}", file=sys.stderr)
        sys.exit(1)

    horizon_secs = [m * 60.0 for m in args.horizon_mins]
    records = list_records(waveform_dir)
    if args.max_records is not None:
        records = records[: args.max_records]
    print(f"  Found {len(records)} master records under {waveform_dir}")

    stats: dict[float, HorizonStat] = {h: HorizonStat() for h in horizon_secs}
    icp_records: set[str] = set()
    icp_patients: set[str] = set()
    total_onsets = 0
    onset_records: set[str] = set()
    onset_patients: set[str] = set()
    per_patient_onsets: dict[str, int] = defaultdict(int)

    for i, (pid, rec_dir, rec_name) in enumerate(records):
        blocks = read_icp_blocks(rec_dir, rec_name)
        if blocks is None:
            if (i + 1) % args.progress_every == 0:
                print(f"    [{i + 1}/{len(records)}] {pid}/{rec_name}  (no ICP)")
            continue

        rec_id = f"{pid}_{rec_name}"
        icp_records.add(rec_id)
        icp_patients.add(pid)

        onsets = find_onsets(blocks, args.icp_threshold, args.event_consec)
        if onsets:
            total_onsets += len(onsets)
            onset_records.add(rec_id)
            onset_patients.add(pid)
            per_patient_onsets[pid] += len(onsets)

        scan_record(
            blocks, pid, rec_id, onsets,
            args.window_secs, horizon_secs, args.stride_sec,
            args.icp_threshold, stats,
        )

        if (i + 1) % args.progress_every == 0:
            dur_min = len(blocks) * BLOCK_SEC / 60.0
            print(f"    [{i + 1}/{len(records)}] {pid}/{rec_name}  "
                  f"dur={dur_min:.0f}min  onsets={len(onsets)}  "
                  f"cohort onsets so far={total_onsets}")

    print_report(
        stats, args.window_secs, args.icp_threshold, args.stride_sec,
        len(records), len(icp_records), len(icp_patients),
        total_onsets, len(onset_records), len(onset_patients),
    )


if __name__ == "__main__":
    main()
