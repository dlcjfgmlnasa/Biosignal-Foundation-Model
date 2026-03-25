# -*- coding:utf-8 -*-
"""Cross-Modal Contrastive Loss (InfoNCE).

같은 시간대(time_id)의 서로 다른 modality(variate_id) 패치를 positive pair로,
나머지를 negative로 사용하는 InfoNCE 기반 contrastive loss.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class CrossModalContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for cross-modal representation alignment.

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
        B, N, D = z.shape

        # L2 normalize
        z = F.normalize(z, dim=-1)

        total_loss = z.new_tensor(0.0)
        n_valid_anchors = 0

        for b in range(B):
            # padding 제외한 유효 패치만 선택
            valid = patch_mask[b] & (patch_variate_id[b] > 0)
            if valid.sum() < 2:
                continue

            valid_idx = valid.nonzero(as_tuple=True)[0]  # (M,)
            z_valid = z[b, valid_idx]                      # (M, D)
            sid = patch_sample_id[b, valid_idx]            # (M,)
            vid = patch_variate_id[b, valid_idx]           # (M,)
            tid = time_id[b, valid_idx]                    # (M,)
            M = z_valid.shape[0]

            # Positive mask: same (sample_id, time_id), different variate_id
            K = tid.max().item() + 1
            group_key = sid * K + tid  # (M,)
            pos_mask = (
                (group_key.unsqueeze(-1) == group_key.unsqueeze(-2))
                & (vid.unsqueeze(-1) != vid.unsqueeze(-2))
            )  # (M, M)

            # positive pair가 있는 anchor만 선택
            has_pos = pos_mask.any(dim=-1)  # (M,)
            if not has_pos.any():
                continue

            # Similarity matrix
            sim = z_valid @ z_valid.T / self.temperature  # (M, M)

            # Self-mask (diagonal 제외)
            self_mask = torch.eye(M, dtype=torch.bool, device=z.device)

            # Denominator: 자기 자신 제외한 모든 패치
            sim_all = sim.masked_fill(self_mask, float("-inf"))
            log_denom = torch.logsumexp(sim_all, dim=-1)  # (M,)

            # Numerator: positive pair만
            sim_pos = sim.masked_fill(~pos_mask, float("-inf"))
            log_numer = torch.logsumexp(sim_pos, dim=-1)  # (M,)

            # InfoNCE = -log(sum_pos / sum_all) = -(log_numer - log_denom)
            per_anchor_loss = -(log_numer - log_denom)  # (M,)

            valid_loss = per_anchor_loss[has_pos]
            total_loss = total_loss + valid_loss.sum()
            n_valid_anchors += has_pos.sum().item()

        if n_valid_anchors > 0:
            return total_loss / n_valid_anchors
        return total_loss  # 0.0
