"""가상 센싱 (Virtual Sensing) Imputation 평가 모듈.

``model.forward(task="masked")`` 의 재구성 출력 중 특정 variate 패치만
추출하여 원본 대비 복원 품질을 측정.
"""

from __future__ import annotations

import torch

from data.collate import PackedBatch
from eval._metrics import regression_metrics


@torch.no_grad()
def evaluate_imputation(
    model: torch.nn.Module,
    batch: PackedBatch,
    mask_variate_ids: list[int] | None = None,
) -> dict[str, float]:
    """Imputation 평가.

    원본 batch를 모델에 통과시킨 뒤, 대상 variate의 재구성 패치를
    정규화된 원본 패치와 비교.

    Parameters
    ----------
    model:
        ``forward(batch, task="masked")`` 를 지원하는 사전학습 모델.
    batch:
        마스킹 대상 variate를 포함한 ``PackedBatch``.
    mask_variate_ids:
        평가할 variate_id 목록 (1-based).
        ``None`` 이면 batch 내 마지막 unique variate를 자동 선택.

    Returns
    -------
    dict with keys: ``mse``, ``mae``, ``mape``, ``pearson_r``.
    """
    model.eval()

    original_values = batch.values.clone()
    out = model.forward(batch, task="masked")

    reconstructed = out["reconstructed"]        # (B, N, P)
    patch_mask = out["patch_mask"]              # (B, N)
    patch_variate_id = out["patch_variate_id"]  # (B, N)
    loc = out["loc"]                            # (B, L, 1)
    scale = out["scale"]                        # (B, L, 1)

    B, N, P = reconstructed.shape

    # 정규화된 원본 패치
    normalized = ((original_values.unsqueeze(-1) - loc) / scale).squeeze(-1)
    original_patches = normalized.reshape(B, N, P)

    # 대상 variate 결정
    if mask_variate_ids is None:
        unique_vids = patch_variate_id[patch_mask].unique()
        unique_vids = unique_vids[unique_vids > 0]
        if len(unique_vids) == 0:
            return _empty_metrics()
        mask_variate_ids = [unique_vids[-1].item()]

    # 대상 variate의 유효 패치만 선택
    target_mask = torch.zeros(B, N, dtype=torch.bool, device=reconstructed.device)
    for vid in mask_variate_ids:
        target_mask |= (patch_variate_id == vid) & patch_mask

    if not target_mask.any():
        return _empty_metrics()

    pred = reconstructed[target_mask].reshape(-1)
    target = original_patches[target_mask].reshape(-1)

    return regression_metrics(pred, target)


def _empty_metrics() -> dict[str, float]:
    return {"mse": 0.0, "mae": 0.0, "mape": 0.0, "pearson_r": 0.0}
