# -*- coding:utf-8 -*-
"""MIMIC-III-Ext-PPG 파서.

PhysioNet MIMIC-III-Ext-PPG 데이터셋을 파싱하여 downstream task용 .pt 파일로 저장한다.
30초 WFDB 세그먼트 (125Hz) → 100Hz 리샘플링 → .pt 저장.

데이터 구조:
    raw/mimic-iii-ext-ppg/1.1.0/
    ├── metadata.csv          ← 라벨 (rhythm, BP, HR, RR, SQI)
    └── p00/p000052/
        ├── 3238451_0005_0_1.hea
        └── 3238451_0005_0_1.dat

사용법:
    python -m data.parser.mimic3_ext_ppg \
        --raw-dir datasets/raw/mimic-iii-ext-ppg/1.1.0 \
        --out-dir datasets/processed/mimic3_ext_ppg \
        --max-records 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from data.parser._common import resample_to_target

TARGET_SR: float = 100.0
NATIVE_SR: float = 125.0

# 신호명 → signal_type 매핑
SIGNAL_NAME_MAP: dict[str, str] = {
    "PLETH": "ppg",
    "II": "ecg",
    "II+": "ecg",
    "ABP": "abp",
    "ART": "abp",
}

# signal_type → 정수 코드 (spatial_map.py 기준)
SIGNAL_TYPE_INT: dict[str, int] = {
    "ecg": 0,
    "abp": 1,
    "ppg": 2,
}


@dataclass
class SegmentMeta:
    """metadata.csv의 한 행."""

    signal_file_name: str
    folder_path: str
    patient: str
    event_rhythm: str
    median_30s_sbp: float | None
    median_30s_dbp: float | None
    median_30s_hr: float | None
    median_30s_rr: float | None
    subject_id: str
    hadm_id: str
    age: float | None
    gender: str
    strat_fold: int | None


def _safe_float(val: str) -> float | None:
    """빈 문자열이나 비정상 값을 None으로 변환."""
    if not val or val.strip() == "":
        return None
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except ValueError:
        return None


def _safe_int(val: str) -> int | None:
    if not val or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_metadata(csv_path: Path) -> dict[str, SegmentMeta]:
    """metadata.csv를 로드하여 signal_file_name → SegmentMeta 딕셔너리 반환."""
    metadata: dict[str, SegmentMeta] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("signal_file_name", "").strip()
            if not fname:
                continue

            meta = SegmentMeta(
                signal_file_name=fname,
                folder_path=row.get("folder_path", "").strip(),
                patient=row.get("patient", "").strip(),
                event_rhythm=row.get("event_rhythm", "").strip(),
                median_30s_sbp=_safe_float(row.get("median_30s_sbp", "")),
                median_30s_dbp=_safe_float(row.get("median_30s_dbp", "")),
                median_30s_hr=_safe_float(row.get("median_30s_hr", "")),
                median_30s_rr=_safe_float(row.get("median_30s_rr", "")),
                subject_id=row.get("subject_id", "").strip(),
                hadm_id=row.get("hadm_id", "").strip(),
                age=_safe_float(row.get("age", "")),
                gender=row.get("gender", "").strip(),
                strat_fold=_safe_int(row.get("strat_fold", "")),
            )
            metadata[fname] = meta

    return metadata


def parse_record(
    record_path: Path,
    meta: SegmentMeta | None,
    out_dir: Path,
    min_quality: float = 0.5,
) -> dict | None:
    """단일 WFDB 레코드를 파싱하여 .pt로 저장한다.

    Parameters
    ----------
    record_path: .hea 파일 경로 (확장자 제외한 stem).
    meta: metadata.csv에서 읽은 메타데이터.
    out_dir: 출력 디렉토리.
    min_quality: 최소 품질 (NaN 비율 기준).

    Returns
    -------
    저장된 레코드 정보 딕셔너리, 실패 시 None.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    try:
        rec = wfdb.rdrecord(str(record_path))
    except Exception as e:
        print(f"  SKIP {record_path.name}: {e}")
        return None

    if rec.p_signal is None or rec.sig_len == 0:
        return None

    signals_saved = []
    record_name = record_path.name

    for ch_idx, sig_name in enumerate(rec.sig_name):
        sig_type = SIGNAL_NAME_MAP.get(sig_name)
        if sig_type is None:
            continue

        signal = rec.p_signal[:, ch_idx].astype(np.float64)

        # NaN 체크
        nan_ratio = np.isnan(signal).mean()
        if nan_ratio > (1.0 - min_quality):
            continue

        # NaN 보간 (선형)
        if nan_ratio > 0:
            nans = np.isnan(signal)
            x = np.arange(len(signal))
            signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])

        # 리샘플링 125Hz → 100Hz
        if rec.fs != TARGET_SR:
            signal = resample_to_target(signal, float(rec.fs), TARGET_SR)

        # 저장
        tensor = torch.from_numpy(signal).float().unsqueeze(0)  # (1, T)
        stype_int = SIGNAL_TYPE_INT.get(sig_type, -1)

        out_name = f"{record_name}_{sig_type}.pt"
        torch.save(tensor, out_dir / out_name)

        signals_saved.append({
            "file": out_name,
            "signal_type": sig_type,
            "signal_type_int": stype_int,
            "signal_name": sig_name,
            "n_samples": len(signal),
            "duration_sec": len(signal) / TARGET_SR,
        })

    if not signals_saved:
        return None

    # 라벨 정보
    label_info = {}
    if meta is not None:
        label_info = {
            "rhythm": meta.event_rhythm,
            "sbp": meta.median_30s_sbp,
            "dbp": meta.median_30s_dbp,
            "hr": meta.median_30s_hr,
            "rr": meta.median_30s_rr,
            "age": meta.age,
            "gender": meta.gender,
            "subject_id": meta.subject_id,
            "strat_fold": meta.strat_fold,
        }

    return {
        "record": record_name,
        "patient": meta.patient if meta else record_path.parent.name,
        "signals": signals_saved,
        "labels": label_info,
        "sampling_rate": TARGET_SR,
    }


def parse_dataset(
    raw_dir: str,
    out_dir: str,
    max_records: int | None = None,
) -> None:
    """MIMIC-III-Ext-PPG 전체 파싱.

    Parameters
    ----------
    raw_dir: raw 데이터 루트 (metadata.csv + pXX/ 디렉토리가 있는 경로).
    out_dir: 출력 디렉토리.
    max_records: 최대 레코드 수 (디버깅용).
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # metadata.csv 로드
    csv_path = raw_path / "metadata.csv"
    metadata: dict[str, SegmentMeta] = {}
    if csv_path.exists():
        print(f"Loading metadata: {csv_path}")
        metadata = load_metadata(csv_path)
        print(f"  {len(metadata)} entries loaded")
    else:
        print(f"  WARNING: metadata.csv not found at {csv_path}")

    # .hea 파일 탐색
    hea_files = sorted(raw_path.rglob("*.hea"))
    print(f"Found {len(hea_files)} records")

    if max_records is not None:
        hea_files = hea_files[:max_records]
        print(f"  Processing first {max_records} records")

    manifest = []
    rhythm_counts: dict[str, int] = {}
    n_success = 0
    n_skip = 0

    for i, hea_path in enumerate(hea_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{len(hea_files)}] {hea_path.stem}...")

        record_stem = str(hea_path.with_suffix(""))
        fname = hea_path.stem

        # metadata 매칭
        meta = metadata.get(fname)

        # 환자별 출력 디렉토리
        patient_id = meta.patient if meta else hea_path.parent.name
        patient_dir = out_path / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        # 이미 파싱된 경우 스킵
        existing_pts = list(patient_dir.glob(f"{fname}_*.pt"))
        if existing_pts:
            n_skip += 1
            # manifest에 기존 데이터 추가
            label_info = {}
            if meta:
                label_info = {
                    "rhythm": meta.event_rhythm,
                    "sbp": meta.median_30s_sbp,
                    "dbp": meta.median_30s_dbp,
                    "hr": meta.median_30s_hr,
                    "rr": meta.median_30s_rr,
                    "age": meta.age,
                    "gender": meta.gender,
                    "subject_id": meta.subject_id,
                    "strat_fold": meta.strat_fold,
                }
                rhythm = meta.event_rhythm
                rhythm_counts[rhythm] = rhythm_counts.get(rhythm, 0) + 1
            manifest.append({
                "record": fname,
                "patient": patient_id,
                "signals": [{"file": p.name} for p in existing_pts],
                "labels": label_info,
                "sampling_rate": TARGET_SR,
            })
            continue

        result = parse_record(
            Path(record_stem), meta, patient_dir,
        )

        if result is None:
            n_skip += 1
            continue

        manifest.append(result)
        n_success += 1

        # 리듬 카운트
        if meta and meta.event_rhythm:
            rhythm = meta.event_rhythm
            rhythm_counts[rhythm] = rhythm_counts.get(rhythm, 0) + 1

    # manifest 저장
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "dataset": "MIMIC-III-Ext-PPG",
            "version": "1.1.0",
            "sampling_rate": TARGET_SR,
            "n_records": len(manifest),
            "rhythm_distribution": rhythm_counts,
            "records": manifest,
        }, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  MIMIC-III-Ext-PPG Parsing Complete")
    print(f"  Success: {n_success}, Skip: {n_skip}")
    print(f"  Total records: {len(manifest)}")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  Rhythm distribution:")
    for rhythm, count in sorted(rhythm_counts.items(), key=lambda x: -x[1]):
        print(f"    {rhythm}: {count}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-III-Ext-PPG Parser",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Raw data directory (contains metadata.csv + pXX/)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="datasets/processed/mimic3_ext_ppg",
        help="Output directory for processed .pt files",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Max records to process (None=all)",
    )
    args = parser.parse_args()

    parse_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()
