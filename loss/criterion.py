# -*- coding:utf-8 -*-
from __future__ import annotations

"""α * MPM + β * NextPred + γ * CrossModal + δ * Contrastive 복합 손실 함수.

각 손실 항은 독립 가중치를 가진다. β=0이면 next-pred만 비활성, γ=0이면
cross-modal만 비활성 — 이전 버전처럼 β=0이 cross-modal까지 silent 0으로
만들지 않는다.
"""
import torch
from torch import nn

from loss.contrastive_loss import CrossModalContrastiveLoss
from loss.masked_mse_loss import MaskedPatchLoss
from loss.next_prediction_loss import NextPredictionLoss


class CombinedLoss(nn.Module):
    """α * MPM + β * NextPred + γ * CrossModal + δ * Contrastive 하이브리드 손실 함수.

    각 항은 독립 가중치. β=0은 next-pred만 비활성, γ=0은 cross-modal만 비활성.

    Parameters
    ----------
    alpha:
        Masked reconstruction loss 가중치. 0이면 비활성.
    beta:
        Same-variate next-patch prediction loss 가중치. 0이면 next-patch 비활성.
    gamma:
        Cross-modal prediction loss 가중치. 0이면 cross-modal 비활성.
    delta:
        Cross-modal contrastive loss 가중치. 0이면 비활성.
    contrastive_temperature:
        InfoNCE 초기 temperature.
    learnable_temperature:
        Temperature를 학습 가능한 파라미터로 설정.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        delta: float = 0.0,
        peak_alpha: float = 0.0,
        lambda_spec: float = 0.0,
        spec_n_ffts: tuple[int, ...] = (16, 32, 64),
        contrastive_temperature: float = 0.07,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.masked_loss_fn = MaskedPatchLoss(
            peak_alpha=peak_alpha,
            lambda_spec=lambda_spec,
            spec_n_ffts=spec_n_ffts,
        )
        self.next_loss_fn = NextPredictionLoss(
            peak_alpha=peak_alpha,
            lambda_spec=lambda_spec,
            spec_n_ffts=spec_n_ffts,
        )
        if delta > 0:
            self.contrastive_loss_fn = CrossModalContrastiveLoss(
                temperature=contrastive_temperature,
                learnable_temperature=learnable_temperature,
            )

    def forward(
        self,
        reconstructed: torch.Tensor,  # (B, N, patch_size)
        next_pred: torch.Tensor | None,  # (B, N, K, patch_size) or None — Block Next Prediction
        original_patches: torch.Tensor,  # (B, N, patch_size)
        pred_mask: torch.Tensor,  # (B, N) bool — 마스킹된 패치
        patch_mask: torch.Tensor,  # (B, N) bool — 유효 패치
        patch_sample_id: torch.Tensor,  # (B, N) long — 패치별 sample_id
        patch_variate_id: torch.Tensor,  # (B, N) long — 패치별 variate_id
        cross_pred_per_type: torch.Tensor | None = None,  # (B, N, T, patch_size) — per-target-type cross-modal 예측
        time_id: torch.Tensor | None = None,  # (B, N) long — cross-modal 페어링용
        contrastive_z: torch.Tensor
        | None = None,  # (B, N, proj_dim) — contrastive 임베딩
        patch_signal_types: torch.Tensor
        | None = None,  # (B, N) long — mechanism group 필터용
    ) -> dict[str, torch.Tensor]:
        # ── Masked Reconstruction Loss (MSE + Gradient + Spectral) ──
        masked_dict = self.masked_loss_fn(reconstructed, original_patches, pred_mask)
        masked_loss = masked_dict["total"]

        # ── Block Next-Patch + Cross-Modal Prediction Loss (β / γ 독립) ──
        compute_next = self.beta > 0 and next_pred is not None
        compute_cross = (
            self.gamma > 0
            and cross_pred_per_type is not None
            and time_id is not None
        )
        if compute_next or compute_cross:
            next_dict = self.next_loss_fn(
                next_pred,
                cross_pred_per_type,
                original_patches,
                patch_mask,
                patch_sample_id,
                patch_variate_id,
                time_id=time_id,
                patch_signal_types=patch_signal_types,
                compute_next=compute_next,
                compute_cross=compute_cross,
            )
            next_loss = next_dict["next_loss"]
            next_spec = next_dict["next_spec"]
            cross_modal_loss = next_dict["cross_modal_loss"]
        else:
            next_loss = reconstructed.new_tensor(0.0)
            next_spec = reconstructed.new_tensor(0.0)
            cross_modal_loss = reconstructed.new_tensor(0.0)

        # ── Cross-Modal Contrastive Loss ──
        if self.delta > 0 and contrastive_z is not None and time_id is not None:
            contrastive_loss, contrastive_n_anchors = self.contrastive_loss_fn(
                contrastive_z,
                patch_mask,
                patch_sample_id,
                patch_variate_id,
                time_id,
            )
        else:
            contrastive_loss = reconstructed.new_tensor(0.0)
            contrastive_n_anchors = reconstructed.new_zeros((), dtype=torch.long)

        total = (
            self.alpha * masked_loss
            + self.beta * next_loss
            + self.gamma * cross_modal_loss
            + self.delta * contrastive_loss
        )

        return {
            "total": total,
            "masked_loss": masked_loss,
            "masked_mse": masked_dict["mse"],
            "masked_spec": masked_dict["spec"],
            "next_loss": next_loss,
            "next_spec": next_spec,
            "cross_modal_loss": cross_modal_loss,
            "contrastive_loss": contrastive_loss,
            # Per-batch valid-anchor count (for weighted aggregation across batches)
            "contrastive_n_anchors": contrastive_n_anchors,
        }
