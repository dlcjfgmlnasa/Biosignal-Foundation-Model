"""PTB-XL ECG Database 파서.

PTB-XL 12-lead ECG를 로드하여 downstream classification 평가용 데이터를 생성한다.
100Hz 버전(records100/)을 직접 사용하므로 리샘플링 불필요.

Usage
-----
    from data.parser.ptbxl import load_ptbxl

    samples, labels, label_names = load_ptbxl(
        data_dir="datasets/ptb-xl/1.0.3",
        lead="II",           # Lead II만 사용 (VitalDB ECG_II와 동일)
        superclass=True,     # 5-class superclass 분류
    )

5-class superclass:
    NORM  — Normal ECG
    MI    — Myocardial Infarction
    STTC  — ST/T Change
    CD    — Conduction Disturbance
    HYP   — Hypertrophy
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb


# 12-lead 순서 (PTB-XL 표준)
LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# 5-class superclass 매핑
SUPERCLASS_MAP = {
    "NORM": 0,
    "MI": 1,
    "STTC": 2,
    "CD": 3,
    "HYP": 4,
}
SUPERCLASS_NAMES = list(SUPERCLASS_MAP.keys())


def load_ptbxl(
    data_dir: str | Path,
    lead: str = "II",
    superclass: bool = True,
    sampling_rate: int = 100,
    min_label_confidence: float = 100.0,
) -> tuple[list[torch.Tensor], list[int], list[str]]:
    """PTB-XL 데이터를 로드한다.

    Parameters
    ----------
    data_dir:
        PTB-XL 루트 디렉토리 (ptbxl_database.csv가 있는 곳).
    lead:
        사용할 리드 이름 (기본 "II"). VitalDB ECG Lead II와 동일.
    superclass:
        True면 5-class superclass, False면 원본 SCP code.
    sampling_rate:
        100 또는 500. 100Hz 권장 (우리 모델과 동일).
    min_label_confidence:
        SCP 라벨 최소 confidence (기본 100 = 확실한 것만).

    Returns
    -------
    (samples, labels, label_names)
        samples: list[torch.Tensor] — 각 (time,) 1D ECG
        labels: list[int] — 정수 클래스 라벨
        label_names: list[str] — 클래스 이름 목록
    """
    data_dir = Path(data_dir)

    # 1. 메타데이터 로드
    df = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # 2. SCP statements 로드 (superclass 매핑용)
    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]  # diagnostic statements만

    # 3. Superclass 라벨 추출
    def _get_superclass(scp_dict: dict) -> str | None:
        """SCP code dict → 가장 높은 confidence의 superclass 반환."""
        best_class = None
        best_conf = 0
        for code, conf in scp_dict.items():
            if conf < min_label_confidence:
                continue
            if code in scp.index:
                sc = scp.loc[code, "diagnostic_class"]
                if isinstance(sc, str) and sc in SUPERCLASS_MAP:
                    if conf > best_conf:
                        best_conf = conf
                        best_class = sc
        return best_class

    # 4. 라벨 할당
    df["superclass"] = df["scp_codes"].apply(_get_superclass)
    df = df.dropna(subset=["superclass"])  # 라벨 없는 것 제거

    # 5. 리드 인덱스
    if lead not in LEAD_NAMES:
        raise ValueError(f"Unknown lead: {lead}. Available: {LEAD_NAMES}")
    lead_idx = LEAD_NAMES.index(lead)

    # 6. 데이터 로드
    record_dir = f"records{sampling_rate}"
    samples: list[torch.Tensor] = []
    labels: list[int] = []

    n_loaded = 0
    n_failed = 0

    for ecg_id, row in df.iterrows():
        filename = row["filename_hr" if sampling_rate == 500 else "filename_lr"]
        record_path = str(data_dir / filename)

        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, lead_idx]  # (time,)
            signal = signal.astype(np.float32)

            # NaN 체크
            if np.isnan(signal).any():
                n_failed += 1
                continue

            samples.append(torch.from_numpy(signal))
            labels.append(SUPERCLASS_MAP[row["superclass"]])
            n_loaded += 1

        except Exception:
            n_failed += 1
            continue

    print(f"PTB-XL loaded: {n_loaded} samples ({n_failed} failed)")
    print(
        f"  Lead: {lead}, SR: {sampling_rate}Hz, Length: {samples[0].shape[0] if samples else 0} samples"
    )
    print(f"  Classes: {SUPERCLASS_NAMES}")

    # 클래스 분포
    from collections import Counter

    dist = Counter(labels)
    for cls_name, cls_id in SUPERCLASS_MAP.items():
        print(f"    {cls_name} ({cls_id}): {dist.get(cls_id, 0)}")

    return samples, labels, SUPERCLASS_NAMES


def load_ptbxl_split(
    data_dir: str | Path,
    lead: str = "II",
    sampling_rate: int = 100,
    min_label_confidence: float = 100.0,
) -> dict[str, tuple[list[torch.Tensor], list[int]]]:
    """PTB-XL을 공식 train/val/test split으로 로드한다.

    PTB-XL은 strat_fold 1-10으로 나뉘어 있고, 공식 split:
        train: fold 1-8
        val:   fold 9
        test:  fold 10

    Returns
    -------
    {"train": (samples, labels), "val": (samples, labels), "test": (samples, labels)}
    """
    data_dir = Path(data_dir)

    df = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]

    def _get_superclass(scp_dict: dict) -> str | None:
        best_class = None
        best_conf = 0
        for code, conf in scp_dict.items():
            if conf < min_label_confidence:
                continue
            if code in scp.index:
                sc = scp.loc[code, "diagnostic_class"]
                if isinstance(sc, str) and sc in SUPERCLASS_MAP:
                    if conf > best_conf:
                        best_conf = conf
                        best_class = sc
        return best_class

    df["superclass"] = df["scp_codes"].apply(_get_superclass)
    df = df.dropna(subset=["superclass"])

    if lead not in LEAD_NAMES:
        raise ValueError(f"Unknown lead: {lead}. Available: {LEAD_NAMES}")
    lead_idx = LEAD_NAMES.index(lead)

    splits = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }

    for ecg_id, row in df.iterrows():
        fold = row["strat_fold"]
        if fold <= 8:
            split = "train"
        elif fold == 9:
            split = "val"
        else:
            split = "test"

        filename = row["filename_hr" if sampling_rate == 500 else "filename_lr"]
        record_path = str(data_dir / filename)

        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, lead_idx].astype(np.float32)

            if np.isnan(signal).any():
                continue

            splits[split][0].append(torch.from_numpy(signal))
            splits[split][1].append(SUPERCLASS_MAP[row["superclass"]])

        except Exception:
            continue

    for split_name, (samps, labs) in splits.items():
        from collections import Counter

        dist = Counter(labs)
        print(f"  {split_name}: {len(samps)} samples - {dict(dist)}")

    return splits
