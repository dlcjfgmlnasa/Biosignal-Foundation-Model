# -*- coding:utf-8 -*-
import torch


def safe_div(
    a: torch.Tensor,  # (*any)
    b: torch.Tensor,  # (*any)
) -> torch.Tensor:  # (*any)
    """Safe division: returns 0 where divisor is 0."""
    return torch.where(b == 0, torch.zeros_like(a), a / b)
