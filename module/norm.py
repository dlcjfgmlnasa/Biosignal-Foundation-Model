# -*- coding:utf-8 -*-
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Parameters
    ----------
    normalized_shape:
        정규화 대상 차원 크기.
    eps:
        수치 안정성을 위한 엡실론.
    weight:
        학습 가능한 스케일 파라미터(gamma) 사용 여부.
    dtype:
        파라미터 dtype.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 1e-5,
        weight: bool = True,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.mean_dim = tuple(range(-len(normalized_shape), 0))

        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(
        self, x: torch.Tensor,  # (*batch, *normalized_shape)
    ) -> torch.Tensor:  # (*batch, *normalized_shape)
        output = x * torch.rsqrt(
            x.pow(2).mean(dim=self.mean_dim, keepdim=True) + self.eps
        )
        if self.weight is not None:
            return output * self.weight
        return output

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"weight={self.weight is not None}"
        )
