# -*- coding:utf-8 -*-
from __future__ import annotations

"""Cross-Modal Contrastive Loss (InfoNCE).

같은 시간대(time_id)의 서로 다른 modality(variate_id) 패치를 positive pair로,
나머지를 negative로 사용하는 InfoNCE 기반 contrastive loss.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class CrossModalContrastiveLoss(nn.Module):
    """Cross-modal representation alignment을 위한 InfoNCE contrastive loss.

    같은 ``(sample_id, time_id)``를 공유하지만 서로 다른 ``variate_id``를 가진
    패치 쌍을 positive로, 같은 batch row 내 나머지 패치를 negative로 사용한다.

    Parameters
    ----------
    temperature:
        InfoNCE 초기 temperature. ``learnable_temperature=True``이면 학습 초기값.
    learnable_temperature:
        Temperature를 학습 가능한 파라미터로 설정 (log-parameterized, CLIP 방식).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(temperature))
            )
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(math.log(temperature))
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        z: torch.Tensor,                # (B, N, D) — projected embeddings
        patch_mask: torch.Tensor,        # (B, N) bool — 유효 패치
        patch_sample_id: torch.Tensor,   # (B, N) long
        patch_variate_id: torch.Tensor,  # (B, N) long
        time_id: torch.Tensor,           # (B, N) long
    ) -> torch.Tensor:  # scalar
        """InfoNCE contrastive loss를 계산한다.

        단일 variate만 존재하는 batch row는 positive pair가 없으므로 스킵한다.
        """
        b, n, d = z.shape

        # L2 normalize
        z = F.normalize(z, dim=-1)

        # Valid mask: padding 제외 유효 패치
        valid = patch_mask & (patch_variate_id > 0)  # (B, N)

        # Similarity matrix — 단일 batched matmul (for-loop 제거)
        temp = self.temperature.clamp(min=0.01, max=1.0)  # NaN 방지
        sim = torch.bmm(z, z.transpose(1, 2)) / temp  # (B, N, N)

        # Pairwise valid: query와 key 모두 유효해야 함
        valid_pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)  # (B, N, N)

        # Positive mask: same (sample_id, time_id), different variate_id
        k = time_id.max() + 1  # 0-dim 텐서 (CUDA sync 없음)
        group_key = patch_sample_id * k + time_id  # (B, N)
        same_group = group_key.unsqueeze(-1) == group_key.unsqueeze(-2)  # (B, N, N)
        diff_var = patch_variate_id.unsqueeze(-1) != patch_variate_id.unsqueeze(-2)  # (B, N, N)
        pos_mask = same_group & diff_var & valid_pair  # (B, N, N)

        # Self-mask (diagonal 제외)
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device).unsqueeze(0)  # (1, N, N)

        # has_pos: positive pair가 있는 유효 anchor
        has_pos = pos_mask.any(dim=-1) & valid  # (B, N)
        if not has_pos.any():
            return z.new_tensor(0.0)

        # Denominator: 자기 자신 제외, 유효 패치만
        denom_mask = valid_pair & ~self_mask  # (B, N, N)
        log_denom = torch.logsumexp(
            sim.masked_fill(~denom_mask, float("-inf")), dim=-1,
        )  # (B, N)

        # Numerator: positive pair만
        log_numer = torch.logsumexp(
            sim.masked_fill(~pos_mask, float("-inf")), dim=-1,
        )  # (B, N)

        # InfoNCE = -log(sum_pos / sum_all) = -(log_numer - log_denom)
        per_anchor_loss = -(log_numer - log_denom)  # (B, N)

        total_loss = per_anchor_loss[has_pos].sum()
        n_valid_anchors = has_pos.sum()

        return total_loss / n_valid_anchors
