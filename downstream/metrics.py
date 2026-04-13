"""평가 공통 메트릭 헬퍼.

분류 (AUROC, AUPRC, F1, Sensitivity/Specificity)와
회귀/복원 (MSE, MAE, MAPE, Pearson r, Bland-Altman) 메트릭을 제공한다.
torch.Tensor 및 np.ndarray 입력 모두 지원.
"""

from __future__ import annotations

import numpy as np
import torch


# ── 회귀/복원 메트릭 (torch) ──────────────────────────────────


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """MSE, MAE, MAPE, Pearson r 회귀 메트릭 일괄 계산.

    Parameters
    ----------
    pred, target:
        동일 shape의 1-D 텐서.

    Returns
    -------
    dict with keys: ``mse``, ``mae``, ``mape``, ``pearson_r``.
    """
    diff = pred - target

    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()

    # MAPE — |target| ≈ 0 인 위치 제외
    abs_target = target.abs()
    nonzero = abs_target > 1e-8
    mape = (
        (diff[nonzero].abs() / abs_target[nonzero]).mean().item()
        if nonzero.any()
        else 0.0
    )

    return {
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "pearson_r": pearson_r(pred, target),
    }


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation coefficient (torch)."""
    if x.numel() < 2:
        return 0.0

    xc = x - x.mean()
    yc = y - y.mean()

    num = (xc * yc).sum()
    den = (xc.pow(2).sum() * yc.pow(2).sum()).sqrt()

    if den < 1e-8:
        return 0.0

    return (num / den).item()


# ── 회귀/복원 메트릭 (numpy) ──────────────────────────────────


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient (numpy)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) < 2:
        return 0.0

    xc = y_true - y_true.mean()
    yc = y_pred - y_pred.mean()

    num = (xc * yc).sum()
    den = np.sqrt((xc**2).sum() * (yc**2).sum())

    if den < 1e-8:
        return 0.0

    return float(num / den)


def compute_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Bland-Altman 통계량을 계산한다.

    Returns
    -------
    {"bias": float, "loa_lower": float, "loa_upper": float, "std_diff": float}
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    diff = y_pred - y_true
    bias = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0

    return {
        "bias": bias,
        "loa_lower": bias - 1.96 * std_diff,
        "loa_upper": bias + 1.96 * std_diff,
        "std_diff": std_diff,
    }


# ── 분류 메트릭 ───────────────────────────────────────────────


def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float | dict[int, float]:
    """AUROC를 계산한다.

    Binary: float. Multi-class: dict[int, float] (one-vs-rest).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_score.ndim == 1:
        return _binary_auroc(y_true, y_score)

    n_classes = y_score.shape[1]
    result: dict[int, float] = {}
    for c in range(n_classes):
        binary_true = (y_true == c).astype(np.int32)
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            result[c] = 0.0
            continue
        result[c] = _binary_auroc(binary_true, y_score[:, c])
    return result


def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary AUROC (trapezoidal rule)."""
    if len(np.unique(y_true)) < 2:
        return 0.0

    desc_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_idx]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0, 0
    auc = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev, fpr_prev = tpr, fpr

    return float(auc)


def compute_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float | dict[int, float]:
    """AUPRC (Average Precision)를 계산한다.

    Binary: float. Multi-class: dict[int, float] (one-vs-rest).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_score.ndim == 1:
        return _binary_auprc(y_true, y_score)

    n_classes = y_score.shape[1]
    result: dict[int, float] = {}
    for c in range(n_classes):
        binary_true = (y_true == c).astype(np.int32)
        if binary_true.sum() == 0:
            result[c] = 0.0
            continue
        result[c] = _binary_auprc(binary_true, y_score[:, c])
    return result


def _binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary AUPRC (step-function integration)."""
    if len(np.unique(y_true)) < 2:
        return 0.0

    desc_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_idx]

    tp = 0
    ap = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
            precision = tp / (i + 1)
            ap += precision

    n_pos = y_sorted.sum()
    if n_pos == 0:
        return 0.0

    return float(ap / n_pos)


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """F1 score를 계산한다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))

    f1_per_class: list[float] = []
    support_per_class: list[int] = []

    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_per_class.append(f1)
        support_per_class.append(int((y_true == c).sum()))

    if average == "macro":
        return float(np.mean(f1_per_class))

    total = sum(support_per_class)
    if total == 0:
        return 0.0
    return float(sum(f * s for f, s in zip(f1_per_class, support_per_class)) / total)


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Binary sensitivity (recall) 와 specificity를 계산한다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
