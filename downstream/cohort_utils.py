# -*- coding:utf-8 -*-
"""환자 단위 cohort CSV 로더.

Mortality / Sepsis / ICH 등에서 공통으로 쓰는
cohort CSV → {subject_id: {label, metadata}} 변환.

각 task는 BigQuery / 자체 script로 생성된 CSV를 가지며, 다음 포맷을 공유한다:

    subject_id, hadm_id, icustay_id, <label_col>, icu_intime, icu_outtime, ...

Public API
----------
- load_patient_cohort(csv_path, label_column, ...) -> list[dict]
"""
from __future__ import annotations

import csv
from pathlib import Path


def load_patient_cohort(
    csv_path: str | Path,
    label_column: str,
    positive_value: str | int = "1",
    subject_column: str = "subject_id",
    extra_columns: list[str] | None = None,
) -> list[dict]:
    """Generic cohort CSV 로더.

    Parameters
    ----------
    csv_path:
        CSV 경로 (예: sepsis3_cohort.csv, mortality_cohort.csv).
    label_column:
        이진 라벨 컬럼명 (예: "sepsis3", "hospital_expire_flag").
    positive_value:
        양성으로 간주할 값 (기본 "1"; CSV는 문자열로 읽힘).
    subject_column:
        subject_id에 해당하는 컬럼명 (기본 "subject_id").
    extra_columns:
        추가로 보존할 컬럼명들 (없으면 None). 있으면 각 row dict에 포함.

    Returns
    -------
    list of dict: [{subject_id: int, label: int, <extra>: ...}, ...]

    Notes
    -----
    - 파일이 없거나 column이 없으면 즉시 FileNotFoundError/KeyError.
    - subject_id는 int로 변환 (숫자 아니면 str로 유지).
    - 빈 label cell은 skip (경고 없음).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cohort CSV not found: {csv_path}")

    extras = extra_columns or []
    records: list[dict] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for col in [subject_column, label_column]:
            if col not in fieldnames:
                raise KeyError(f"Column '{col}' not found in {csv_path}")

        pos_str = str(positive_value)

        for row in reader:
            label_raw = row.get(label_column, "").strip()
            if not label_raw:
                continue
            label = 1 if label_raw == pos_str else 0

            sid_raw = row.get(subject_column, "").strip()
            try:
                sid = int(sid_raw)
            except ValueError:
                sid = sid_raw  # type: ignore[assignment]

            rec = {"subject_id": sid, "label": label}
            for c in extras:
                rec[c] = row.get(c, "")
            records.append(rec)

    return records


def split_cohort_by_subject(
    cohort: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """환자 단위 (subject_id 기준) train/test split.

    같은 subject_id가 양쪽에 나타나지 않도록 분할.
    """
    import random

    unique_sids = sorted({r["subject_id"] for r in cohort})
    rng = random.Random(seed)
    rng.shuffle(unique_sids)

    n_test = max(1, int(len(unique_sids) * test_ratio))
    test_sids = set(unique_sids[:n_test])

    train = [r for r in cohort if r["subject_id"] not in test_sids]
    test = [r for r in cohort if r["subject_id"] in test_sids]
    return train, test
