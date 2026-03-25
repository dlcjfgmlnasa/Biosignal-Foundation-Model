"""ICU Forecasting 평가 모듈.

``model.generate()``로 autoregressive 다단계 예측 → ground truth 비교.
"""

from __future__ import annotations

import torch

from data.collate import PackedBatch
from eval._metrics import regression_metrics


@torch.no_grad()
def evaluate_forecasting(
    model: torch.nn.Module,
    batch: PackedBatch,
    ground_truth: torch.Tensor,  # (B, n_steps * patch_size) or (B, n_steps, patch_size)
    n_steps: int = 1,
    denormalize: bool = True,
) -> dict[str, float]:
    """Forecasting 평가.

    Parameters
    ----------
    model:
        ``generate()`` 메서드를 가진 사전학습된 모델.
    batch:
        입력 컨텍스트 ``PackedBatch`` (과거 시계열).
    ground_truth:
        미래 구간 정답 텐서.
    n_steps:
        예측할 미래 패치 수.
    denormalize:
        ``True``이면 원본 스케일로 복원 후 비교.

    Returns
    -------
    dict with keys: ``mse``, ``mae``, ``mape``, ``pearson_r``.
    """
    model.eval()

    predictions = model.generate(
        batch, n_steps=n_steps, denormalize=denormalize,
    )  # (B, n_steps, patch_size)

    pred_flat = predictions.reshape(predictions.shape[0], -1)
    gt_flat = ground_truth.reshape(ground_truth.shape[0], -1).to(pred_flat.device)

    return regression_metrics(pred_flat, gt_flat)
