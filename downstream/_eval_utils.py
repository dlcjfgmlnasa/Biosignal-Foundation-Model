# -*- coding: utf-8 -*-
"""Downstream task 공용 평가 utilities.

Clinical AI 표준 protocol (memory: project_classification_eval_protocol):
1. 학습/추론 단계에서 fold 별 raw prediction 을 .npz 로 저장 (사후 자유 재계산)
2. OOF concat → patient-level bootstrap 95% CI (1000 iter)
3. 5-fold mean ± SD
4. Option A format: "0.843₍±0.008₎ [0.831-0.855]"

지표 종속 없음 — y_true/y_score/patient_id 만 저장, metric 은 metric_fn 으로 주입.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


# ---- Prediction dump format ----


@dataclass
class FoldPredictions:
    """단일 fold 의 test set raw prediction.

    y_score: binary 면 (N,) 또는 (N, 1), multi-class 면 (N, K).
    """

    y_true: np.ndarray          # (N,) int
    y_score: np.ndarray         # (N,) or (N, K) float
    patient_id: np.ndarray      # (N,) str
    fold_idx: int
    n_folds: int
    task: str
    classes: list[str] | None = None
    extra: dict | None = None   # case_id, window_id, label_value (regression target) etc.


def save_predictions(
    path: str | Path,
    preds: FoldPredictions,
) -> Path:
    """Fold prediction 을 .npz 로 저장."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "y_true": preds.y_true.astype(np.int64),
        "y_score": preds.y_score.astype(np.float32),
        "patient_id": preds.patient_id.astype(str),
        "fold_idx": np.int64(preds.fold_idx),
        "n_folds": np.int64(preds.n_folds),
        "task": str(preds.task),
        "classes": np.array(preds.classes or [], dtype=str),
    }
    if preds.extra:
        for k, v in preds.extra.items():
            payload[f"extra__{k}"] = np.asarray(v)
    np.savez_compressed(p, **payload)
    return p


def dump_fold_predictions(
    out_dir: str | Path,
    task: str,
    fold_idx: int,
    n_folds: int,
    y_true,
    y_score,
    patient_ids,
    classes: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    """run.py 공용 편의 함수: fold prediction 을 ``preds_fold{idx}.npz`` 로 저장.

    각 task run.py 가 eval 직후 y_true/y_score/patient_ids 를 그대로 넘기면 된다.
    binary 면 y_score 는 (N,) 또는 (N,1), multi-class 면 (N,K).
    """
    path = Path(out_dir) / f"preds_fold{fold_idx}.npz"
    return save_predictions(
        path,
        FoldPredictions(
            y_true=np.asarray(y_true),
            y_score=np.asarray(y_score),
            patient_id=np.asarray(patient_ids, dtype=str),
            fold_idx=int(fold_idx),
            n_folds=int(n_folds),
            task=task,
            classes=classes,
            extra=extra,
        ),
    )


def load_predictions(path: str | Path) -> FoldPredictions:
    """save_predictions 로 저장된 .npz 로드."""
    data = np.load(path, allow_pickle=False)
    extra: dict = {}
    for k in data.files:
        if k.startswith("extra__"):
            extra[k[len("extra__"):]] = data[k]
    classes_arr = data["classes"] if "classes" in data.files else None
    classes = list(classes_arr) if classes_arr is not None and len(classes_arr) > 0 else None
    return FoldPredictions(
        y_true=data["y_true"],
        y_score=data["y_score"],
        patient_id=data["patient_id"],
        fold_idx=int(data["fold_idx"]),
        n_folds=int(data["n_folds"]),
        task=str(data["task"]),
        classes=classes,
        extra=extra or None,
    )


# ---- OOF concat ----


def concat_oof(fold_paths: Sequence[str | Path]) -> FoldPredictions:
    """N fold prediction 파일을 concat → OOF prediction set.

    각 환자가 정확히 1번씩 등장 (stratified k-fold CV 의 자연 특성).
    fold_idx 는 -1 (OOF aggregate 표시).
    """
    fold_preds = [load_predictions(p) for p in fold_paths]
    if not fold_preds:
        raise ValueError("Empty fold_paths")

    # task / classes consistency
    task = fold_preds[0].task
    classes = fold_preds[0].classes
    n_folds = fold_preds[0].n_folds
    for fp in fold_preds[1:]:
        if fp.task != task:
            raise ValueError(f"Task mismatch: {task} vs {fp.task}")
        if fp.n_folds != n_folds:
            raise ValueError(f"n_folds mismatch: {n_folds} vs {fp.n_folds}")

    # CV leakage 검출: 한 환자가 2개 이상 fold 의 test 에 등장하면 안 됨
    # (stratified k-fold 는 각 환자가 정확히 1번씩 test). 위반 시 OOF 가정 깨짐.
    seen_pid_fold: dict = {}
    for fp in fold_preds:
        for pid in np.unique(fp.patient_id):
            if pid in seen_pid_fold and seen_pid_fold[pid] != fp.fold_idx:
                raise ValueError(
                    f"patient {pid!r} appears in fold {seen_pid_fold[pid]} and "
                    f"fold {fp.fold_idx} test sets — CV leakage"
                )
            seen_pid_fold[pid] = fp.fold_idx

    y_true = np.concatenate([fp.y_true for fp in fold_preds])
    y_score = np.concatenate([fp.y_score for fp in fold_preds])
    patient_id = np.concatenate([fp.patient_id for fp in fold_preds])

    # extra 도 concat (모든 fold 에 동일 key 존재 가정)
    extra: dict | None = None
    if fold_preds[0].extra:
        extra = {}
        for k in fold_preds[0].extra:
            arrs = [fp.extra[k] for fp in fold_preds if fp.extra and k in fp.extra]
            if len(arrs) == len(fold_preds):
                extra[k] = np.concatenate(arrs)

    return FoldPredictions(
        y_true=y_true,
        y_score=y_score,
        patient_id=patient_id,
        fold_idx=-1,
        n_folds=n_folds,
        task=task,
        classes=classes,
        extra=extra,
    )


# ---- Patient-level bootstrap CI ----


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_id: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iter: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Patient-level bootstrap CI.

    Resample patient_id with replacement → 그 환자들의 모든 prediction concat
    → metric_fn 계산 → percentile.

    Returns: (point_estimate, ci_low, ci_high)
    """
    point = float(metric_fn(y_true, y_score))

    # patient_id → 해당 row indices 매핑
    unique_pids = np.unique(patient_id)
    pid_to_idx: dict[str, np.ndarray] = {
        pid: np.where(patient_id == pid)[0] for pid in unique_pids
    }
    n_patients = len(unique_pids)

    rng = np.random.default_rng(seed)
    boot_metrics: list[float] = []
    for _ in range(n_iter):
        sampled = rng.choice(unique_pids, size=n_patients, replace=True)
        idx = np.concatenate([pid_to_idx[pid] for pid in sampled])
        try:
            m = float(metric_fn(y_true[idx], y_score[idx]))
            if np.isfinite(m):
                boot_metrics.append(m)
        except Exception:
            continue

    if not boot_metrics:
        return point, float("nan"), float("nan")

    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_metrics, 100.0 * alpha))
    high = float(np.percentile(boot_metrics, 100.0 * (1.0 - alpha)))
    return point, low, high


def fold_stability(
    fold_paths: Sequence[str | Path],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, float, list[float]]:
    """Per-fold metric 계산 → (mean, SD, per-fold list)."""
    vals: list[float] = []
    for p in fold_paths:
        fp = load_predictions(p)
        try:
            m = float(metric_fn(fp.y_true, fp.y_score))
            if np.isfinite(m):
                vals.append(m)
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan"), []
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), vals


# ---- 표준 metric functions ----


def _import_sklearn():
    try:
        from sklearn import metrics as _m  # type: ignore
        return _m
    except ImportError as e:
        raise ImportError("sklearn required for metric computation") from e


def auroc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary AUROC. y_score 가 (N, 2) 면 positive class 컬럼만 사용."""
    sk = _import_sklearn()
    if y_score.ndim == 2:
        if y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        elif y_score.shape[1] == 1:
            y_score = y_score[:, 0]
        else:
            raise ValueError(f"auroc_binary expects 1d or (N,2), got shape {y_score.shape}")
    return float(sk.roc_auc_score(y_true, y_score))


def auprc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary AUPRC."""
    sk = _import_sklearn()
    if y_score.ndim == 2:
        if y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        elif y_score.shape[1] == 1:
            y_score = y_score[:, 0]
        else:
            raise ValueError(f"auprc_binary expects 1d or (N,2), got shape {y_score.shape}")
    return float(sk.average_precision_score(y_true, y_score))


def auroc_macro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Multi-class macro AUROC (OvR)."""
    sk = _import_sklearn()
    return float(sk.roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


def auprc_macro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Multi-class macro AUPRC.

    sklearn average_precision_score 는 multi-class label 지원이 제한적이라
    각 class binarize 후 평균.
    """
    sk = _import_sklearn()
    classes = np.unique(y_true)
    aps: list[float] = []
    for c in classes:
        bin_true = (y_true == c).astype(int)
        if y_score.ndim == 2:
            s = y_score[:, int(c)]
        else:
            s = y_score
        aps.append(float(sk.average_precision_score(bin_true, s)))
    return float(np.mean(aps))


# ---- Option A formatter ----


_SUB_DIGITS = "₀₁₂₃₄₅₆₇₈₉"
_SUB_DOT = "."
_SUB_PLUS_MINUS = "±"


def _to_subscript(s: str) -> str:
    """Plain digit string → Unicode subscript."""
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(_SUB_DIGITS[int(ch)])
        elif ch == ".":
            out.append(_SUB_DOT)
        elif ch == "±":
            out.append(_SUB_PLUS_MINUS)
        else:
            out.append(ch)
    return "".join(out)


def format_option_a(
    point: float,
    ci_low: float,
    ci_high: float,
    sd: float,
    precision: int = 3,
) -> str:
    """Option A: 0.843₍±0.008₎ [0.831-0.855]

    Unicode subscript. LaTeX 변환은 format_option_a_latex 참조.
    """
    p = f"{point:.{precision}f}"
    s = f"±{sd:.{precision}f}"
    sub = _to_subscript(s)
    return f"{p}₍{sub}₎ [{ci_low:.{precision}f}-{ci_high:.{precision}f}]"


def format_option_a_latex(
    point: float,
    ci_low: float,
    ci_high: float,
    sd: float,
    precision: int = 3,
) -> str:
    """Option A LaTeX: $0.843_{\\pm 0.008}\\,[0.831{-}0.855]$"""
    return (
        f"${point:.{precision}f}_{{\\pm {sd:.{precision}f}}}"
        f"\\,[{ci_low:.{precision}f}{{-}}{ci_high:.{precision}f}]$"
    )


def format_plain(
    point: float,
    ci_low: float,
    ci_high: float,
    sd: float,
    precision: int = 3,
) -> str:
    """Plain text: 0.843 [95% CI: 0.831-0.855], 5-fold SD = 0.008"""
    return (
        f"{point:.{precision}f} [95% CI: {ci_low:.{precision}f}-{ci_high:.{precision}f}], "
        f"5-fold SD = ±{sd:.{precision}f}"
    )


# ---- 일괄 평가 ----


def evaluate_oof(
    fold_paths: Sequence[str | Path],
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict]:
    """모든 fold 의 prediction 으로 표준 평가 결과 dict 반환.

    Args:
        fold_paths: fold prediction .npz 경로 리스트 (N folds).
        metric_fns: {"AUROC": auroc_binary, "AUPRC": auprc_binary, ...}
        n_bootstrap: bootstrap iter (default 1000)
        seed: bootstrap RNG seed

    Returns:
        {metric_name: {
            "point": float (OOF),
            "ci_low": float, "ci_high": float,
            "fold_mean": float, "fold_sd": float, "fold_values": [...],
            "format": "0.843₍±0.008₎ [0.831-0.855]",
            "format_latex": "$0.843_{\\pm 0.008}\\,[0.831{-}0.855]$",
            "format_plain": "0.843 [95% CI: 0.831-0.855], 5-fold SD = ±0.008",
        }}
    """
    oof = concat_oof(fold_paths)
    out: dict[str, dict] = {}
    for name, fn in metric_fns.items():
        point, low, high = bootstrap_ci(
            oof.y_true, oof.y_score, oof.patient_id, fn,
            n_iter=n_bootstrap, seed=seed,
        )
        mean, sd, vals = fold_stability(fold_paths, fn)
        out[name] = {
            "point": point,
            "ci_low": low,
            "ci_high": high,
            "fold_mean": mean,
            "fold_sd": sd,
            "fold_values": vals,
            "format": format_option_a(point, low, high, sd),
            "format_latex": format_option_a_latex(point, low, high, sd),
            "format_plain": format_plain(point, low, high, sd),
        }
    return out
