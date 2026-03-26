# -*- coding:utf-8 -*-
from __future__ import annotations

import torch


def safe_div(
    a: torch.Tensor,  # (*any)
    b: torch.Tensor,  # (*any)
) -> torch.Tensor:  # (*any)
    """안전 나눗셈: 분모가 0인 위치는 0을 반환한다."""
    return torch.where(b == 0, torch.zeros_like(a), a / b)
