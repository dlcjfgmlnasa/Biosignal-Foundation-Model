"""평가 시각화 헬퍼.

ROC curve, Bland-Altman plot, Reconstruction 비교 플롯을 제공한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from downstream.metrics import _binary_auroc, compute_bland_altman


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    """ROC curve를 플롯하고 파일로 저장한다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

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
    """Bland-Altman plot을 저장한다."""
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
    ax.axhline(
        loa_upper,
        color="orange",
        linewidth=1,
        linestyle="--",
        label=f"+1.96 SD = {loa_upper:.2f}",
    )
    ax.axhline(
        loa_lower,
        color="orange",
        linewidth=1,
        linestyle="--",
        label=f"-1.96 SD = {loa_lower:.2f}",
    )
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
    """원본 vs 복원 파형을 비교 플롯한다."""
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

    fig, axes = plt.subplots(
        n_channels, 1, figsize=(12, 2.5 * n_channels), squeeze=False, sharex=True
    )

    for ch in range(n_channels):
        ax = axes[ch, 0]
        ax.plot(time_axis, original[ch], linewidth=0.8, alpha=0.8, label="Original")
        ax.plot(
            time_axis,
            reconstructed[ch],
            linewidth=0.8,
            alpha=0.8,
            label="Reconstructed",
        )
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
