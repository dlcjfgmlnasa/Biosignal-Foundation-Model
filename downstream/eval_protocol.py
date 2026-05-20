# -*- coding:utf-8 -*-
"""11 classification task 통합 평가 protocol (OOF + patient-level bootstrap + 5-fold SD).

memory: project_classification_eval_protocol (2026-05-14 확정)

배경
----
각 task run.py 는 fold 별 test 예측 (`y_true`, `y_score`, `patient_id`) 을 저장한다.
이 모듈은 fold 결과를 모아:
  1. OOF (out-of-fold) prediction 으로 concat (각 환자 정확히 1번)
  2. Primary metric (AUROC/AUPRC) — 전체 OOF 한 번에 계산
  3. **Patient-level** bootstrap 95% CI (1,000 iter) — 환자 단위 resampling
  4. 5-fold mean ± SD (secondary stability)
  5. paper 용 reporting 표 생성

왜 patient-level bootstrap 인가
------------------------------
`metrics.bootstrap_ci` 는 sample(window)-level resampling 이라, 같은 환자의 여러
window 를 독립으로 취급 → 상관된 샘플을 독립 가정 → CI 가 비현실적으로 좁아진다.
환자 단위로 resample 해야 임상 AI 표준에 맞는 보수적 CI 가 나온다.

metric_fn convention
--------------------
threshold-free metric (AUROC/AUPRC) 는 (y_true, y_score) 호출.
다른 시그니처가 필요하면 caller 가 closure 로 감싼다.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from downstream.metrics import compute_auroc, compute_auprc


# ── Patient-level bootstrap ──────────────────────────────────────


def patient_bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_ids: np.ndarray,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Patient-level percentile bootstrap CI.

    각 iteration: 고유 patient_id 풀에서 with-replacement 로 환자를 뽑고,
    그 환자들의 **모든** prediction 을 모아 metric 재계산.
    같은 환자가 여러 번 뽑히면 그 환자 window 가 중복 포함된다 (환자 단위 분산 반영).

    Parameters
    ----------
    metric_fn:
        ``metric_fn(y_true, y_score) -> float``.
    y_true, y_score, patient_ids:
        길이 N (= window 수) 의 1-D array. patient_ids[i] = i번째 예측의 환자.
    n_iter:
        bootstrap iteration (default 1000, protocol spec).
    alpha:
        significance (default 0.05 → 95% CI).
    seed:
        RNG seed.

    Returns
    -------
    (lower, upper) percentile CI.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    patient_ids = np.asarray(patient_ids)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0

    # 환자 → 해당 window 인덱스 배열 (한 번만 구축)
    pid_to_idx: dict = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        pid_to_idx[pid].append(i)
    unique_pids = np.array(list(pid_to_idx.keys()), dtype=object)
    pid_idx_arrays = [np.asarray(pid_to_idx[p]) for p in unique_pids]
    n_patients = len(unique_pids)
    if n_patients == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_iter):
        # 환자 풀에서 with-replacement
        chosen = rng.integers(0, n_patients, size=n_patients)
        idx = np.concatenate([pid_idx_arrays[c] for c in chosen])
        yt_b = y_true[idx]
        ys_b = y_score[idx]
        try:
            v = float(metric_fn(yt_b, ys_b))
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        values.append(v)

    arr = np.asarray(values, dtype=np.float64)
    lower = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


# ── OOF aggregation ──────────────────────────────────────────────


@dataclass
class FoldResult:
    """단일 fold 의 test 예측."""

    y_true: np.ndarray  # (n_i,)
    y_score: np.ndarray  # (n_i,)
    patient_ids: np.ndarray  # (n_i,) — 환자 단위 bootstrap·leakage 검증용
    fold_idx: int = -1


@dataclass
class TaskEvalResult:
    """한 task 의 통합 평가 결과 (paper reporting 단위)."""

    task: str
    auroc: float
    auprc: float
    auroc_ci: tuple[float, float]
    auprc_ci: tuple[float, float]
    auroc_5fold_mean: float
    auroc_5fold_sd: float
    auprc_5fold_mean: float
    auprc_5fold_sd: float
    n_total: int
    n_positive: int
    n_patients: int
    prevalence: float
    per_fold_auroc: list[float] = field(default_factory=list)
    per_fold_auprc: list[float] = field(default_factory=list)


def aggregate_oof(
    folds: list[FoldResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fold 별 예측을 OOF 단일 set 으로 concat.

    각 환자가 정확히 1번 등장해야 한다 (CV 가정). leakage 검증으로
    여러 fold 에 같은 patient_id 가 나타나면 예외.

    Returns
    -------
    (y_true, y_score, patient_ids) — 전체 OOF.
    """
    if not folds:
        raise ValueError("no folds to aggregate")

    # leakage 검증: 환자가 여러 fold 의 test 에 등장하면 안 됨
    seen_pids: dict = {}
    for f in folds:
        for pid in np.unique(f.patient_ids):
            if pid in seen_pids:
                raise ValueError(
                    f"patient {pid!r} appears in fold {seen_pids[pid]} and "
                    f"fold {f.fold_idx} test sets — CV leakage"
                )
            seen_pids[pid] = f.fold_idx

    y_true = np.concatenate([f.y_true for f in folds])
    y_score = np.concatenate([f.y_score for f in folds])
    patient_ids = np.concatenate([f.patient_ids for f in folds])
    return y_true, y_score, patient_ids


# ── 5-fold stability ─────────────────────────────────────────────


def five_fold_stats(
    folds: list[FoldResult],
    metric_fn,
) -> tuple[float, float, list[float]]:
    """fold 별로 metric 따로 계산 → (mean, sd, per_fold_list).

    OOF pooled metric 과 직교하는 split/model variance 추정.
    """
    per_fold: list[float] = []
    for f in folds:
        try:
            v = float(metric_fn(f.y_true, f.y_score))
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        per_fold.append(v)
    arr = np.asarray(per_fold, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    # ddof=1 (sample SD); fold 1개면 0
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, sd, per_fold


# ── Top-level task 평가 ──────────────────────────────────────────


def evaluate_classification_task(
    task: str,
    folds: list[FoldResult],
    n_iter: int = 1000,
    seed: int = 42,
) -> TaskEvalResult:
    """한 task 의 fold 결과 → 완전 평가 리포트.

    Protocol:
      - OOF concat (leakage 검증 포함)
      - AUROC/AUPRC headline (전체 OOF)
      - patient-level bootstrap 95% CI
      - 5-fold mean ± SD
    """
    y_true, y_score, patient_ids = aggregate_oof(folds)

    auroc = float(compute_auroc(y_true, y_score))
    auprc = float(compute_auprc(y_true, y_score))

    auroc_ci = patient_bootstrap_ci(
        compute_auroc, y_true, y_score, patient_ids, n_iter=n_iter, seed=seed,
    )
    auprc_ci = patient_bootstrap_ci(
        compute_auprc, y_true, y_score, patient_ids, n_iter=n_iter, seed=seed,
    )

    auroc_mean, auroc_sd, pf_auroc = five_fold_stats(folds, compute_auroc)
    auprc_mean, auprc_sd, pf_auprc = five_fold_stats(folds, compute_auprc)

    n_total = int(len(y_true))
    n_positive = int((y_true > 0).sum())
    n_patients = int(len(np.unique(patient_ids)))
    prevalence = float(n_positive / n_total) if n_total else 0.0

    return TaskEvalResult(
        task=task,
        auroc=auroc,
        auprc=auprc,
        auroc_ci=auroc_ci,
        auprc_ci=auprc_ci,
        auroc_5fold_mean=auroc_mean,
        auroc_5fold_sd=auroc_sd,
        auprc_5fold_mean=auprc_mean,
        auprc_5fold_sd=auprc_sd,
        n_total=n_total,
        n_positive=n_positive,
        n_patients=n_patients,
        prevalence=prevalence,
        per_fold_auroc=pf_auroc,
        per_fold_auprc=pf_auprc,
    )


# ── Reporting ────────────────────────────────────────────────────


def format_report_table(results: list[TaskEvalResult]) -> str:
    """여러 task 결과 → paper 용 markdown 표.

    형식 (memory protocol §6):
        | Task | AUROC | 95% CI | 5-fold SD | AUPRC | 95% CI | 5-fold SD | Prev. | N(pat) |
    """
    header = (
        "| Task | AUROC | 95% CI | 5-fold SD | AUPRC | 95% CI | 5-fold SD "
        "| Prevalence | N (patients) |"
    )
    sep = "|" + "---|" * 9
    lines = [header, sep]
    for r in results:
        lines.append(
            f"| {r.task} "
            f"| {r.auroc:.3f} "
            f"| {r.auroc_ci[0]:.3f}-{r.auroc_ci[1]:.3f} "
            f"| +/-{r.auroc_5fold_sd:.3f} "
            f"| {r.auprc:.3f} "
            f"| {r.auprc_ci[0]:.3f}-{r.auprc_ci[1]:.3f} "
            f"| +/-{r.auprc_5fold_sd:.3f} "
            f"| {r.prevalence:.3f} "
            f"| {r.n_total} ({r.n_patients}) |"
        )
    return "\n".join(lines)


def result_to_dict(r: TaskEvalResult) -> dict:
    """TaskEvalResult → JSON 직렬화 가능한 dict."""
    return {
        "task": r.task,
        "auroc": r.auroc,
        "auprc": r.auprc,
        "auroc_95ci": list(r.auroc_ci),
        "auprc_95ci": list(r.auprc_ci),
        "auroc_5fold_mean": r.auroc_5fold_mean,
        "auroc_5fold_sd": r.auroc_5fold_sd,
        "auprc_5fold_mean": r.auprc_5fold_mean,
        "auprc_5fold_sd": r.auprc_5fold_sd,
        "n_total": r.n_total,
        "n_positive": r.n_positive,
        "n_patients": r.n_patients,
        "prevalence": r.prevalence,
        "per_fold_auroc": r.per_fold_auroc,
        "per_fold_auprc": r.per_fold_auprc,
    }


# ── 디스크 결과 로딩 helper ───────────────────────────────────────


def load_folds_from_jsons(
    fold_json_paths: list,
    y_true_key: str = "y_true",
    y_score_key: str = "y_score",
    patient_id_key: str = "patient_ids",
) -> list[FoldResult]:
    """fold 별 results JSON 파일 → list[FoldResult].

    각 JSON 은 최소 ``y_true``, ``y_score`` 를 담아야 하며, patient-level
    bootstrap 을 쓰려면 ``patient_ids`` 도 필요하다. 없으면 window index 를
    가짜 patient_id 로 사용 (= sample-level 로 degrade, 경고 출력).
    """
    import json
    import sys

    folds: list[FoldResult] = []
    for i, p in enumerate(fold_json_paths):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        yt = np.asarray(d[y_true_key])
        ys = np.asarray(d[y_score_key])
        if patient_id_key in d and d[patient_id_key] is not None:
            pids = np.asarray(d[patient_id_key], dtype=object)
        else:
            print(
                f"[WARN] {p}: '{patient_id_key}' 없음 — window-level 로 degrade "
                f"(patient-level CI 부정확). run.py 에서 patient_id 저장 권장.",
                file=sys.stderr,
            )
            pids = np.array([f"{i}_{j}" for j in range(len(yt))], dtype=object)
        folds.append(FoldResult(y_true=yt, y_score=ys, patient_ids=pids, fold_idx=i))
    return folds
