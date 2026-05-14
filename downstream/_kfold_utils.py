# -*- coding: utf-8 -*-
"""Downstream task 공용 Stratified Patient-level K-fold CV utilities.

Clinical AI 표준 protocol (memory: project_classification_eval_protocol):
- Patient-level split (같은 환자의 모든 window 같은 fold)
- Stratified by outcome label (positive/negative)
- Fold rotation: test=fold[i], val=fold[(i+1)%k], train=나머지

11개 classification task 가 모두 이 helper 를 import 해서 사용.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _patient_majority_labels(
    patient_ids: list[str],
    patient_to_labels: dict[str, list[int]],
) -> dict[str, int]:
    """환자별 majority label 계산 (binary/multi-class 모두 지원).

    Tie 시 가장 큰 label 우선 (positive 우선 → 희귀 클래스 보호).
    """
    out: dict[str, int] = {}
    for pid in patient_ids:
        labels = patient_to_labels.get(pid, [])
        if not labels:
            out[pid] = 0
            continue
        counts: dict[int, int] = defaultdict(int)
        for lb in labels:
            counts[lb] += 1
        # tie-break: 가장 큰 label 선호 (positive class 우선)
        out[pid] = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return out


def stratified_kfold_patient_splits(
    patient_ids: list[str],
    patient_to_labels: dict[str, list[int]],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[set[str], set[str], set[str]]]:
    """Stratified patient-level K-fold CV split.

    각 fold i:
        test  = fold[i]
        val   = fold[(i+1) % n_folds]
        train = 나머지 (n_folds - 2) folds
    → n_folds=5 면 60/20/20, 모든 환자 정확히 1번씩 test.

    Stratification: 환자별 majority label 로 sklearn-style stratified split.
    Class imbalance 심한 task (positive < 5%) 에서 1 fold positive 0 사고 방지.

    Args:
        patient_ids: 전체 환자 id 리스트 (중복 없음).
        patient_to_labels: {patient_id: [label0, label1, ...]} 환자의 모든 window label.
        n_folds: K (default 5).
        seed: RNG seed.

    Returns:
        [(train_set, val_set, test_set), ...] 길이 n_folds.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    rng = np.random.default_rng(seed)
    pids = list(patient_ids)

    # 환자별 majority label
    pid_to_label = _patient_majority_labels(pids, patient_to_labels)

    # Class 별 환자 그룹화
    class_to_pids: dict[int, list[str]] = defaultdict(list)
    for pid in pids:
        class_to_pids[pid_to_label[pid]].append(pid)

    # 각 class 내에서 shuffle → n_folds 균등 분할 → fold 별 concat
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for cls, cls_pids in class_to_pids.items():
        shuffled = list(cls_pids)
        rng.shuffle(shuffled)
        for i, pid in enumerate(shuffled):
            folds[i % n_folds].append(pid)

    splits: list[tuple[set[str], set[str], set[str]]] = []
    for i in range(n_folds):
        test_set = set(folds[i])
        val_set = set(folds[(i + 1) % n_folds])
        train_set: set[str] = set()
        for j in range(n_folds):
            if j != i and j != (i + 1) % n_folds:
                train_set.update(folds[j])
        splits.append((train_set, val_set, test_set))
    return splits


def kfold_patient_splits(
    patient_ids: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[set[str], set[str], set[str]]]:
    """Non-stratified patient-level K-fold CV (label 정보 없을 때 fallback).

    Stratified 버전 사용 권장. 이 함수는 label 이 없거나 회귀 task 용.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    rng = np.random.default_rng(seed)
    pids = list(patient_ids)
    rng.shuffle(pids)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, pid in enumerate(pids):
        folds[i % n_folds].append(pid)

    splits: list[tuple[set[str], set[str], set[str]]] = []
    for i in range(n_folds):
        test_set = set(folds[i])
        val_set = set(folds[(i + 1) % n_folds])
        train_set: set[str] = set()
        for j in range(n_folds):
            if j != i and j != (i + 1) % n_folds:
                train_set.update(folds[j])
        splits.append((train_set, val_set, test_set))
    return splits


def summarize_splits(
    splits: list[tuple[set[str], set[str], set[str]]],
    patient_to_labels: dict[str, list[int]] | None = None,
) -> str:
    """Split 요약 문자열 (fold 별 환자수 + class 분포)."""
    lines: list[str] = []
    for i, (tr, va, te) in enumerate(splits):
        line = f"  Fold {i}: train={len(tr)} val={len(va)} test={len(te)} patients"
        if patient_to_labels is not None:
            def _pos_rate(pids: set[str]) -> str:
                all_labels: list[int] = []
                for pid in pids:
                    all_labels.extend(patient_to_labels.get(pid, []))
                if not all_labels:
                    return "0.0%"
                pos = sum(1 for x in all_labels if x > 0)
                return f"{100.0 * pos / len(all_labels):.1f}%"
            line += (
                f"  pos_rate train={_pos_rate(tr)} "
                f"val={_pos_rate(va)} test={_pos_rate(te)}"
            )
        lines.append(line)
    return "\n".join(lines)
