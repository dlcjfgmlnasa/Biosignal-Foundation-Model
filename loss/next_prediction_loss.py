# -*- coding:utf-8 -*-
from __future__ import annotations

"""Next-Patch Prediction Loss.

Phase 1 (CI): 같은 variate 내 시간 예측 (dynamics).
Phase 2 (Any-variate): 같은 variate 시간 예측 + cross-modal 예측 (causality).
"""
import torch
from torch import nn


class NextPredictionLoss(nn.Module):
    """Next-Patch Prediction Loss (same-variate + cross-modal).

    Parameters
    ----------
    cross_modal_weight:
        γ 가중치. 0이면 cross-modal loss 비활성 (Phase 1 동작).
    """

    def __init__(self, cross_modal_weight: float = 0.0) -> None:
        super().__init__()
        self.cross_modal_weight = cross_modal_weight

    def forward(
        self,
        next_pred: torch.Tensor,          # (B, N, P) — same-variate 예측
        cross_pred: torch.Tensor | None,  # (B, N, P) — cross-modal 예측 (cross_head 출력)
        original_patches: torch.Tensor,   # (B, N, P)
        patch_mask: torch.Tensor,         # (B, N) bool
        patch_sample_id: torch.Tensor,    # (B, N) long
        patch_variate_id: torch.Tensor,   # (B, N) long
        time_id: torch.Tensor | None = None,  # (B, N) long — cross-modal 페어링용
        horizon: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Next-patch prediction loss 계산.

        Returns
        -------
        dict with keys:
            ``next_loss``: same-variate next-patch prediction MSE.
            ``cross_modal_loss``: cross-modal prediction MSE (0 if disabled).
        """
        # ── Same-variate next-patch loss ──
        next_loss = self._same_variate_loss(
            next_pred, original_patches, patch_mask,
            patch_sample_id, patch_variate_id, horizon,
        )

        # ── Cross-modal loss ──
        if (
            self.cross_modal_weight > 0
            and cross_pred is not None
            and time_id is not None
        ):
            cross_modal_loss = self._cross_modal_loss(
                cross_pred, original_patches, patch_mask,
                patch_sample_id, patch_variate_id, time_id,
            )
            cross_modal_loss = self.cross_modal_weight * cross_modal_loss
        else:
            cross_modal_loss = next_pred.new_tensor(0.0)

        return {
            "next_loss": next_loss,
            "cross_modal_loss": cross_modal_loss,
        }

    def _same_variate_loss(
        self,
        next_pred: torch.Tensor,         # (B, N, P)
        original_patches: torch.Tensor,  # (B, N, P)
        patch_mask: torch.Tensor,        # (B, N) bool
        patch_sample_id: torch.Tensor,   # (B, N) long
        patch_variate_id: torch.Tensor,  # (B, N) long
        horizon: int,
    ) -> torch.Tensor:
        """Same-variate next-patch prediction loss (기존 CombinedLoss 로직)."""
        target_next = original_patches[:, horizon:, :]   # (B, N-H, P)
        pred_next = next_pred[:, :-horizon, :]            # (B, N-H, P)

        valid = (
            patch_mask[:, :-horizon]
            & patch_mask[:, horizon:]
            & (patch_sample_id[:, :-horizon] == patch_sample_id[:, horizon:])
            & (patch_variate_id[:, :-horizon] == patch_variate_id[:, horizon:])
        )  # (B, N-H)

        n_valid = valid.float().sum()
        if n_valid > 0:
            horizon_weight = 1.0 / horizon
            loss = (
                (pred_next[valid] - target_next[valid]) ** 2
            ).mean() * horizon_weight
        else:
            loss = next_pred.new_tensor(0.0)
        return loss

    def _cross_modal_loss(
        self,
        cross_pred: torch.Tensor,         # (B, N, P)
        original_patches: torch.Tensor,   # (B, N, P)
        patch_mask: torch.Tensor,         # (B, N) bool
        patch_sample_id: torch.Tensor,    # (B, N) long
        patch_variate_id: torch.Tensor,   # (B, N) long
        time_id: torch.Tensor,            # (B, N) long
    ) -> torch.Tensor:
        """Cross-modal prediction loss.

        같은 (sample_id, time_id)에서 서로 다른 variate_id를 가진 패치 쌍을 매칭하고,
        cross_pred[b, i]가 original_patches[b, j]를 예측하도록 MSE를 계산한다.
        """
        # group_key: (batch, sample_id, time_id)가 같은 패치를 그룹핑
        B, N = time_id.shape
        K = time_id.max() + 1          # 0-dim 텐서 (CUDA sync 없음)
        S = patch_sample_id.max() + 1  # 0-dim 텐서 (CUDA sync 없음)
        batch_idx = torch.arange(B, device=time_id.device).unsqueeze(-1)  # (B, 1)
        group_key = batch_idx * (S * K) + patch_sample_id * K + time_id  # (B, N)

        # (B, N, N) pairwise 비교
        same_group = group_key.unsqueeze(-1) == group_key.unsqueeze(-2)
        diff_variate = patch_variate_id.unsqueeze(-1) != patch_variate_id.unsqueeze(-2)
        both_valid = patch_mask.unsqueeze(-1) & patch_mask.unsqueeze(-2)
        # 패딩 (variate_id == 0) 제외
        non_pad = (
            (patch_variate_id > 0).unsqueeze(-1)
            & (patch_variate_id > 0).unsqueeze(-2)
        )

        cross_mask = same_group & diff_variate & both_valid & non_pad  # (B, N, N)

        b_idx, i_idx, j_idx = torch.where(cross_mask)

        if len(b_idx) == 0:
            return cross_pred.new_tensor(0.0)

        pred_p = cross_pred[b_idx, i_idx]          # (K, P)
        target_p = original_patches[b_idx, j_idx]  # (K, P)
        loss = ((pred_p - target_p) ** 2).mean()

        return loss
