"""평가 공통 메트릭 헬퍼."""

from __future__ import annotations

import torch


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
    """Pearson correlation coefficient."""
    if x.numel() < 2:
        return 0.0

    xc = x - x.mean()
    yc = y - y.mean()

    num = (xc * yc).sum()
    den = (xc.pow(2).sum() * yc.pow(2).sum()).sqrt()

    if den < 1e-8:
        return 0.0

    return (num / den).item()
