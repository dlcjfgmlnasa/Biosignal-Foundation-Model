# -*- coding:utf-8 -*-
"""MIMIC-III Waveform 선택적 다운로드 — Cardiac Arrest (Acute Event 방식).

cardiac_arrest_cohort CSV + RECORDS-waveforms → 예측 시점 기준 윈도우 다운로드.

Framework: **Acute Event Detection (Approach 1, Future Prediction)**
  - 5/15/30분 horizon으로 곧 일어날 cardiac arrest를 조기 경보
  - Hypotension(1.1.1) / ICH(1.1.3)와 동일한 framework

각 환자의 다운로드 윈도우:
  - Cardiac Arrest+: horizons의 union window — [onset - pre - max(h), onset - min(h) + post]
    여기서 onset = first arrest documented time (chartevents asystole/VF/VT 우선,
    부재 시 procedureevents CPR 시작).
    → 한 번 다운로드로 여러 horizon downstream 평가 가능 (multi-horizon reporting).
  - Cardiac Arrest- (risk-set matching): positive의 (t_arrest - icu_intime) Δt 분포에서
    무작위 Δt 샘플링하여 anchor = icu_intime + Δt 설정. icu_los가 Δt + max(h) + pre를
    수용하는 negative만 채택. 결과적으로 positive와 동일 시간 컨텍스트(ICU 입실 후 동일
    시간)에서 윈도우를 추출하여 shortcut learning ("입실 직후 stable vs. 후기 deterio-
    rating") 방지.

Cohort 로드:
  - Subject-level dedup (multi-stay → 1명)
  - Seeded shuffle (top-N slice bias 제거)

기본 horizons (Acute Event scale): **5 / 15 / 30 분** (= 0.0833 / 0.25 / 0.5 hours)
  (Pioneer/JAHA 2024, Tonekaboni/Lancet Digit Health 2020, Kwon/JAHA 2018 계보)
  --pre-hours 0.1667 (= 10분 input window) 권장

사용법:
    # 기본 (5/15/30 min horizon, 10분 input window)
    python -m downstream.acute_event.cardiac_arrest.download_waveforms \\
        --cohort-csv downstream/acute_event/cardiac_arrest/bquxjob_cardiac_arrest_TODO.csv \\
        --records-file downstream/acute_event/cardiac_arrest/RECORDS-waveforms \\
        --out-dir datasets/raw/mimic3-waveform-cardiac-arrest \\
        --max-arrest-pos 500 --max-arrest-neg 2500 \\
        --horizons 0.0833 0.25 0.5 --pre-hours 0.1667 --seed 42

    # 단일 horizon fallback (e.g., 15분만)
    python -m downstream.acute_event.cardiac_arrest.download_waveforms \\
        --cohort-csv downstream/acute_event/cardiac_arrest/bquxjob_cardiac_arrest_TODO.csv \\
        --records-file downstream/acute_event/cardiac_arrest/RECORDS-waveforms \\
        --out-dir datasets/raw/mimic3-waveform-cardiac-arrest \\
        --prediction-horizon-h 0.25 --pre-hours 0.1667 --seed 42

Cohort 크기 권고:
    - arrest+ (positive): ~500-1500 (MIMIC-III waveform 교집합 기준, rare event)
    - arrest- (negative): ~2500 (natural prevalence ~1:5) — risk-set matching 후 일부 손실 가능
    - Tonekaboni 2020 / Kwon 2018 MIMIC-III와 유사 scale
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path


PHYSIONET_BASE = "https://physionet.org/files/mimic3wdb-matched/1.0"


def parse_record_datetime(record_path: str) -> tuple[int, datetime | None]:
    """레코드 경로에서 subject_id와 시작 시간을 추출한다.

    예: p00/p000020/p000020-2183-04-28-17-47
    → subject_id=20, datetime=2183-04-28 17:47
    """
    parts = record_path.strip().split("/")
    if len(parts) < 3:
        return 0, None

    folder = parts[1]  # p000020
    sid = int(folder[1:])

    # 파일명에서 날짜 추출: p000020-2183-04-28-17-47
    fname = parts[-1]
    m = re.match(r"p\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", fname)
    if m is None:
        return sid, None

    try:
        dt = datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)),
        )
    except ValueError:
        return sid, None

    return sid, dt


def load_waveform_index(records_file: str) -> dict[int, list[tuple[str, datetime]]]:
    """RECORDS-waveforms → {subject_id: [(record_path, datetime), ...]}"""
    index: dict[int, list[tuple[str, datetime]]] = {}

    with open(records_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, dt = parse_record_datetime(line)
            if sid == 0 or dt is None:
                continue
            if sid not in index:
                index[sid] = []
            index[sid].append((line, dt))

    # 시간순 정렬
    for sid in index:
        index[sid].sort(key=lambda x: x[1])

    return index


def load_cohort(
    cohort_csv: str,
    wf_index: dict[int, list],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """cohort CSV에서 waveform이 있는 arrest+/arrest- 환자를 분리한다.

    - Subject-level dedup: 같은 subject_id의 여러 ICU stay는 1명으로 집계.
      한 번이라도 arrest+면 arrest+로 분류.
    - Seeded shuffle: CSV 정렬 순서(subject_id ASC)로 인한 selection bias 제거.
    """
    subj_rows: dict[int, list[dict]] = {}

    with open(cohort_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["subject_id"])
            if sid not in wf_index:
                continue
            subj_rows.setdefault(sid, []).append(row)

    arrest_pos: list[dict] = []
    arrest_neg: list[dict] = []

    for sid, rows in subj_rows.items():
        pos_stays = [r for r in rows if int(r["cardiac_arrest"]) == 1]
        if pos_stays:
            row = pos_stays[0]
            label = 1
        else:
            row = rows[0]
            label = 0

        rec = {
            "subject_id": sid,
            "icustay_id": row["icustay_id"],
            "cardiac_arrest": label,
            "icu_intime": row["icu_intime"],
            "icu_outtime": row["icu_outtime"],
            "first_arrest_time": row.get("first_arrest_time", ""),
            "sofa_total": row.get("sofa_total", ""),
            "hospital_expire_flag": row.get("hospital_expire_flag", "0"),
            "age": row.get("age", ""),
            "gender": row.get("gender", ""),
        }

        if label == 1:
            arrest_pos.append(rec)
        else:
            arrest_neg.append(rec)

    rng = random.Random(seed)
    rng.shuffle(arrest_pos)
    rng.shuffle(arrest_neg)

    return arrest_pos, arrest_neg


def parse_datetime_str(s: str) -> datetime | None:
    """다양한 포맷의 datetime 문자열을 파싱한다."""
    if not s or s.strip() == "":
        return None
    s = s.strip().replace("T", " ").split(".")[0].replace(" UTC", "")
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def assign_negative_anchors(
    arrest_pos: list[dict],
    arrest_neg: list[dict],
    pre_hours: float,
    post_hours: float,
    horizons: list[float],
    seed: int = 42,
) -> list[dict]:
    """Risk-set matching으로 negative 환자에게 anchor_time을 할당한다.

    Approach 1 (Future Prediction)에서 positive와 negative의 시간 컨텍스트를 맞추기
    위한 처리. 단순히 icu_intime을 anchor로 쓰면 positive(arrest 직전, ICU 후기)와
    negative(ICU 입실 직후)가 ICU 체류 시점이 달라 모델이 "체류 시점"을 shortcut
    feature로 학습할 수 있다.

    Algorithm:
      1. positive별 Δt_p = (first_arrest_time - icu_intime) 계산.
      2. negative별 icu_los = (icu_outtime - icu_intime) 계산.
      3. negative마다 {Δt_p : Δt_p + post_hours ≤ icu_los AND
                       Δt_p ≥ pre_hours + max(horizons)} 후보군에서 무작위 1개 샘플링.
         (= "이 환자가 동일 Δt 시점에 ICU에 있었고, 윈도우가 stay 안에 들어가는 경우")
      4. 후보군이 비어있는 negative는 매칭 실패 → 제외.
      5. anchor_time = icu_intime + Δt 를 patient dict에 기록.

    Returns
    -------
    Risk-set matching에 성공한 negative 환자 리스트 (실패한 환자는 제외됨).
    각 dict에 새 키 "anchor_time" (ISO datetime 문자열) 추가.
    """
    rng = random.Random(seed)
    max_h = max(horizons) if horizons else 0.0
    min_window_required = pre_hours + max_h  # Δt가 최소 이만큼은 되어야 윈도우가 stay 안에 들어감

    # 1) positive의 Δt 분포 계산
    delta_t_pool: list[float] = []
    for p in arrest_pos:
        intime = parse_datetime_str(p["icu_intime"])
        arrest_t = parse_datetime_str(p["first_arrest_time"])
        if intime is None or arrest_t is None:
            continue
        dt_h = (arrest_t - intime).total_seconds() / 3600.0
        if dt_h < min_window_required:
            continue  # 윈도우가 stay 시작 전으로 빠지면 사용 불가
        delta_t_pool.append(dt_h)

    if not delta_t_pool:
        print(
            "  WARN: positive Δt pool empty — risk-set matching unavailable. "
            "Falling back to icu_intime anchor for all negatives.",
            file=sys.stderr,
        )
        for n in arrest_neg:
            n["anchor_time"] = n["icu_intime"]
        return arrest_neg

    print(f"  Positive Δt pool: n={len(delta_t_pool)}, "
          f"min={min(delta_t_pool):.2f}h, max={max(delta_t_pool):.2f}h, "
          f"median={sorted(delta_t_pool)[len(delta_t_pool)//2]:.2f}h")

    # 2) negative마다 risk-set 매칭
    matched: list[dict] = []
    n_skip_no_los = 0
    n_skip_no_match = 0

    for n in arrest_neg:
        intime = parse_datetime_str(n["icu_intime"])
        outtime = parse_datetime_str(n["icu_outtime"])
        if intime is None or outtime is None:
            n_skip_no_los += 1
            continue
        icu_los_h = (outtime - intime).total_seconds() / 3600.0

        # 후보: pre_window가 stay 안에 들어가고, post가 outtime을 넘지 않음
        candidates = [
            dt for dt in delta_t_pool
            if dt >= min_window_required and dt + post_hours <= icu_los_h
        ]
        if not candidates:
            n_skip_no_match += 1
            continue

        chosen_dt = rng.choice(candidates)
        anchor = intime + timedelta(hours=chosen_dt)
        n["anchor_time"] = anchor.strftime("%Y-%m-%d %H:%M:%S")
        n["matched_delta_t_h"] = chosen_dt
        matched.append(n)

    print(f"  Risk-set matching: kept {len(matched)}/{len(arrest_neg)} negatives "
          f"(skip: invalid_stay={n_skip_no_los}, no_match={n_skip_no_match})")

    return matched


def select_records_for_patient(
    patient: dict,
    wf_records: list[tuple[str, datetime]],
    pre_hours: float = 0.1667,  # 10 min default (acute event scale)
    post_hours: float = 0.0,
    horizons: list[float] | None = None,
) -> list[str]:
    """환자의 관심 시간 윈도우에 해당하는 waveform 레코드를 선택한다.

    Cardiac Arrest+ window (horizons의 union):
      주어진 horizons = [h_1, ..., h_n]에 대해, 각 horizon의
      [onset - pre_hours - h_i, onset - h_i + post_hours]의 union을 계산:
          [onset - pre_hours - max(h), onset - min(h) + post_hours]
      → 한 번 다운로드로 여러 horizon downstream 평가 가능.
      → horizons=[0]이면 legacy 동작 ([onset - pre, onset + post]).

    Cardiac Arrest- window (risk-set matched):
      assign_negative_anchors()로 미리 부여된 anchor_time을 positive와 동일 구조로 사용:
          [anchor - pre_hours - max(h), anchor - min(h) + post_hours]
      anchor_time 키가 없으면 icu_intime fallback (legacy).
    """
    if horizons is None or len(horizons) == 0:
        horizons = [0.0]

    if patient["cardiac_arrest"] == 1:
        center = parse_datetime_str(patient["first_arrest_time"])
    else:
        # Risk-set matched anchor (assign_negative_anchors가 부여) 또는 fallback
        center = parse_datetime_str(
            patient.get("anchor_time") or patient["icu_intime"]
        )

    if center is None:
        return []

    max_h = max(horizons)
    min_h = min(horizons)
    window_start = center - timedelta(hours=pre_hours + max_h)
    window_end = center - timedelta(hours=min_h) + timedelta(hours=post_hours)

    selected = []
    for rec_path, rec_dt in wf_records:
        if window_start <= rec_dt <= window_end:
            selected.append(rec_path)

    return selected


def download_record(record_path: str, out_dir: Path) -> bool:
    """wfdb.dl_database로 multi-segment 레코드를 다운로드한다.

    MIMIC-III Waveform은 MultiRecord 포맷이라 세그먼트별 .hea/.dat가 있음.
    wfdb.dl_database가 자동으로 모든 세그먼트를 받아줌.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        return False

    # record_path: p00/p000052/p000052-2191-01-10-02-21
    parts = record_path.split("/")
    # wfdb.dl_database의 db 인자는 버전 없이: mimic3wdb-matched/p00/p000052
    db_subdir = f"mimic3wdb-matched/{parts[0]}/{parts[1]}"
    patient_dir = out_dir / parts[0] / parts[1]

    # 이미 다운된 경우 스킵
    rec_name = parts[-1]
    if (patient_dir / f"{rec_name}.hea").exists():
        return True

    try:
        wfdb.dl_database(
            db_subdir,
            dl_dir=str(out_dir),
            records=[rec_name],
            overwrite=False,
        )
        return True
    except Exception as e:
        print(f"  FAIL {record_path}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-III Waveform selective download for cardiac arrest prediction"
    )
    parser.add_argument(
        "--cohort-csv", type=str, required=True,
        help="cardiac_arrest_cohort CSV (BigQuery 결과)",
    )
    parser.add_argument(
        "--records-file", type=str, required=True,
        help="RECORDS-waveforms file",
    )
    parser.add_argument(
        "--out-dir", type=str, default="datasets/raw/mimic3-waveform-cardiac-arrest",
    )
    parser.add_argument(
        "--max-arrest-pos", type=int, default=None,
        help="최대 arrest+ 환자 수 (None=전부)",
    )
    parser.add_argument(
        "--max-arrest-neg", type=int, default=None,
        help="최대 arrest- 환자 수 (None=전부)",
    )
    parser.add_argument(
        "--pre-hours", type=float, default=0.1667,  # 10 min
        help="입력 윈도우 길이(시간). 기본 0.1667h = 10분 (Acute Event scale). "
             "Outcome-style이면 24.0 등 사용 가능.",
    )
    parser.add_argument(
        "--post-hours", type=float, default=0.0,
        help="윈도우 duration 후반부 (기본 0 — onset 이후 구간 제외).",
    )
    parser.add_argument(
        "--prediction-horizon-h", type=float, default=0.0,
        help="단일 prediction horizon(시간). --horizons가 지정되면 무시됨. "
             "> 0이면 onset-horizon 이전 신호만 사용 (label leakage 완화). "
             "예: 0.0833 (=5분), 0.25 (=15분), 0.5 (=30분).",
    )
    parser.add_argument(
        "--horizons", type=float, nargs="+",
        default=[0.0833, 0.25, 0.5],  # 5/15/30 min (Acute Event default)
        help="여러 prediction horizon (시간) 다중 채택. 지정 시 모든 horizon을 "
             "커버하는 union window 한 번에 다운로드 → prepare_data.py에서 "
             "horizon별로 cut 가능. 기본 [0.0833, 0.25, 0.5] = 5/15/30 min "
             "(Acute Event framework).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="cohort random sampling seed (top-N slice bias 제거용)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="병렬 다운로드 worker 수 (기본 8). bandwidth 한계로 16 이상은 비추천.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve horizons: --horizons가 있으면 그것 사용, 없으면 단일값 fallback
    if args.horizons:
        horizons = sorted(set(args.horizons))
    else:
        horizons = [args.prediction_horizon_h]
    print(f"Prediction horizons: {horizons}")

    # 1. Waveform 인덱스 로드
    print("Loading waveform index...")
    wf_index = load_waveform_index(args.records_file)
    print(f"  {len(wf_index)} subjects with waveforms")

    # 2. Cohort 로드 + 매칭 (subject-level dedup + seeded shuffle)
    print(f"Loading cohort (seed={args.seed})...")
    arrest_pos, arrest_neg = load_cohort(
        args.cohort_csv, wf_index, seed=args.seed,
    )
    print(f"  Cardiac Arrest+ with waveform: {len(arrest_pos)} (unique subjects)")
    print(f"  Cardiac Arrest- with waveform: {len(arrest_neg)} (unique subjects)")

    # 3. 샘플 제한
    if args.max_arrest_pos is not None:
        arrest_pos = arrest_pos[: args.max_arrest_pos]
    if args.max_arrest_neg is not None:
        arrest_neg = arrest_neg[: args.max_arrest_neg]

    # 3.5 Risk-set matching — negative에 anchor_time 부여 (Acute Event Approach 1)
    print(f"\nRisk-set matching negatives (pre={args.pre_hours}h, "
          f"max_horizon={max(horizons)}h)...")
    arrest_neg = assign_negative_anchors(
        arrest_pos, arrest_neg,
        pre_hours=args.pre_hours,
        post_hours=args.post_hours,
        horizons=horizons,
        seed=args.seed,
    )

    all_patients = arrest_pos + arrest_neg
    print(f"\nTarget: {len(arrest_pos)} arrest+ + {len(arrest_neg)} arrest- "
          f"= {len(all_patients)} patients")

    # 4. 환자별 레코드 선택 → 모든 (patient, rec_path) 쌍 평탄화
    total_records = 0
    skipped_patients = 0
    patient_to_selected: dict[int, list[str]] = {}
    patient_meta: dict[int, dict] = {}

    for patient in all_patients:
        sid = patient["subject_id"]
        wf_records = wf_index.get(sid, [])
        selected = select_records_for_patient(
            patient, wf_records,
            pre_hours=args.pre_hours,
            post_hours=args.post_hours,
            horizons=horizons,
        )
        if not selected:
            skipped_patients += 1
            continue
        total_records += len(selected)
        patient_to_selected[sid] = selected
        patient_meta[sid] = patient

    # 평탄화: [(sid, rec_path), ...]
    pending = [
        (sid, rec) for sid, recs in patient_to_selected.items() for rec in recs
    ]
    print(
        f"\nDownloading {len(pending)} records for {len(patient_to_selected)} "
        f"patients with {args.workers} workers (skipped {skipped_patients} "
        f"patients with no records in window)..."
    )

    downloaded = 0
    failed = 0
    n_done = 0
    success_by_patient: dict[int, list[str]] = {sid: [] for sid in patient_to_selected}

    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {
            executor.submit(download_record, rec, out_dir): (sid, rec)
            for sid, rec in pending
        }
        for fut in as_completed(futures):
            sid, rec = futures[fut]
            try:
                ok = fut.result()
            except Exception as e:
                ok = False
                print(f"  FAIL {rec}: {e}")
            if ok:
                downloaded += 1
                success_by_patient[sid].append(rec)
            else:
                failed += 1
            n_done += 1
            if n_done % 10 == 0 or n_done == 1:
                print(
                    f"  [{n_done}/{len(pending)}] downloaded={downloaded}, "
                    f"failed={failed}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down workers...", file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=True)

    skipped = skipped_patients
    manifest = []
    for sid, patient in patient_meta.items():
        recs = success_by_patient[sid]
        manifest.append({
            **patient,
            "waveform_records": recs,
            "n_records": len(recs),
        })

    # 5. Manifest 저장
    manifest_path = out_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "framework": "Acute Event Detection (Approach 1, Future Prediction)",
            "negative_strategy": "risk-set matching (time-from-admission)",
            "n_patients": len(manifest),
            "n_arrest_pos": sum(1 for m in manifest if m["cardiac_arrest"] == 1),
            "n_arrest_neg": sum(1 for m in manifest if m["cardiac_arrest"] == 0),
            "total_records": total_records,
            "downloaded": downloaded,
            "failed": failed,
            "skipped_no_records": skipped,
            "pre_hours": args.pre_hours,
            "post_hours": args.post_hours,
            "horizons": horizons,
            "seed": args.seed,
            "patients": manifest,
        }, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  Download Complete")
    print(f"  Patients: {len(manifest)} ({sum(1 for m in manifest if m['cardiac_arrest']==1)} arrest+)")
    print(f"  Records: {downloaded} downloaded, {failed} failed, {skipped} skipped")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
