# -*- coding:utf-8 -*-
"""Downstream 공통 평가 메트릭 및 시각화 유틸리티.

분류 (AUROC, AUPRC, F1, Sensitivity/Specificity)와
회귀/복원 (MSE, MAE, Pearson r, Bland-Altman) 메트릭을 제공한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ── 분류 메트릭 ───────────────────────────────────────────────


def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float | dict[int, float]:
    """AUROC를 계산한다.

    Parameters
    ----------
    y_true:
        정답 라벨. Binary일 경우 ``(N,)`` (0/1).
        Multi-class일 경우 ``(N,)`` (정수 클래스 인덱스).
    y_score:
        예측 확률/점수.
        Binary: ``(N,)`` — positive class 확률.
        Multi-class: ``(N, C)`` — 클래스별 확률.

    Returns
    -------
    Binary: float (단일 AUROC).
    Multi-class: dict[int, float] — 클래스별 one-vs-rest AUROC.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_score.ndim == 1:
        return _binary_auroc(y_true, y_score)

    # Multi-class: one-vs-rest per class
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
    # 양쪽 클래스가 모두 존재해야 의미 있음
    if len(np.unique(y_true)) < 2:
        return 0.0

    # score 내림차순 정렬
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
        # 트라페조이드 면적
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev, fpr_prev = tpr, fpr

    return float(auc)


def compute_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float | dict[int, float]:
    """AUPRC (Average Precision)를 계산한다.

    Parameters
    ----------
    y_true:
        Binary ``(N,)`` (0/1) 또는 multi-class ``(N,)`` (정수 인덱스).
    y_score:
        Binary ``(N,)`` 또는 multi-class ``(N, C)``.

    Returns
    -------
    Binary: float (단일 AUPRC).
    Multi-class: dict[int, float] — 클래스별 one-vs-rest AUPRC.
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
    """F1 score를 계산한다.

    Parameters
    ----------
    y_true:
        정답 라벨 ``(N,)``.
    y_pred:
        예측 라벨 ``(N,)``.
    average:
        ``"macro"`` (기본) 또는 ``"weighted"``.

    Returns
    -------
    F1 score (0~1).
    """
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

    # weighted
    total = sum(support_per_class)
    if total == 0:
        return 0.0
    return float(
        sum(f * s for f, s in zip(f1_per_class, support_per_class)) / total
    )


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Binary sensitivity (recall) 와 specificity를 계산한다.

    Parameters
    ----------
    y_true:
        정답 라벨 ``(N,)`` (0/1).
    y_pred:
        예측 라벨 ``(N,)`` (0/1).

    Returns
    -------
    {"sensitivity": float, "specificity": float}
    """
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


# ── 회귀/복원 메트릭 ────────────────────────────────────────────


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
    """Pearson correlation coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) < 2:
        return 0.0

    xc = y_true - y_true.mean()
    yc = y_pred - y_pred.mean()

    num = (xc * yc).sum()
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())

    if den < 1e-8:
        return 0.0

    return float(num / den)


def compute_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Bland-Altman 통계량을 계산한다.

    Parameters
    ----------
    y_true:
        참값 (예: 실제 BIS).
    y_pred:
        예측값 (예: 모델 BIS 예측).

    Returns
    -------
    {"bias": float, "loa_lower": float, "loa_upper": float, "std_diff": float}

    - bias: mean(y_pred - y_true).
    - loa_lower/upper: bias +/- 1.96 * std(diff) — 95% Limits of Agreement.
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


# ── 시각화 헬퍼 ─────────────────────────────────────────────


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    """ROC curve를 플롯하고 파일로 저장한다.

    Parameters
    ----------
    y_true:
        Binary 정답 라벨 ``(N,)`` (0/1).
    y_score:
        Positive class 확률 ``(N,)``.
    save_path:
        저장 경로 (.png).
    title:
        플롯 제목.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # ROC 포인트 계산
    desc_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_idx]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return

    tprs, fprs = [0.0], [0.0]
    tp, fp = 0, 0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    auroc = _binary_auroc(y_true, y_score)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fprs, tprs, linewidth=2, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    title: str = "Bland-Altman Plot",
) -> None:
    """Bland-Altman plot을 저장한다.

    X축: 두 측정의 평균 (y_true + y_pred) / 2.
    Y축: 차이 (y_pred - y_true).
    수평선: bias, +/- 1.96 * std (95% LoA).

    Parameters
    ----------
    y_true:
        참값.
    y_pred:
        예측값.
    save_path:
        저장 경로 (.png).
    title:
        플롯 제목.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mean_vals = (y_true + y_pred) / 2.0
    diff_vals = y_pred - y_true

    stats = compute_bland_altman(y_true, y_pred)
    bias = stats["bias"]
    loa_lower = stats["loa_lower"]
    loa_upper = stats["loa_upper"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.4, s=10, color="steelblue")
    ax.axhline(bias, color="red", linewidth=1.5, label=f"Bias = {bias:.2f}")
    ax.axhline(loa_upper, color="orange", linewidth=1, linestyle="--",
               label=f"+1.96 SD = {loa_upper:.2f}")
    ax.axhline(loa_lower, color="orange", linewidth=1, linestyle="--",
               label=f"-1.96 SD = {loa_lower:.2f}")
    ax.set_xlabel("Mean of True and Predicted")
    ax.set_ylabel("Difference (Predicted - True)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: str | Path,
    title: str = "Reconstruction Comparison",
    sr: float = 100.0,
) -> None:
    """원본 vs 복원 파형을 비교 플롯한다.

    Parameters
    ----------
    original:
        원본 신호 ``(n_timesteps,)`` 또는 ``(n_channels, n_timesteps)``.
    reconstructed:
        복원 신호 (같은 shape).
    save_path:
        저장 경로 (.png).
    title:
        플롯 제목.
    sr:
        sampling rate (Hz). X축을 초 단위로 표시.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if original.ndim == 1:
        original = original[np.newaxis, :]
        reconstructed = reconstructed[np.newaxis, :]

    n_channels = original.shape[0]
    n_timesteps = original.shape[1]
    time_axis = np.arange(n_timesteps) / sr

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2.5 * n_channels),
                             squeeze=False, sharex=True)

    for ch in range(n_channels):
        ax = axes[ch, 0]
        ax.plot(time_axis, original[ch], linewidth=0.8, alpha=0.8, label="Original")
        ax.plot(time_axis, reconstructed[ch], linewidth=0.8, alpha=0.8, label="Reconstructed")
        ax.set_ylabel(f"Ch {ch}")
        if ch == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
