# -*- coding:utf-8 -*-
from __future__ import annotations

"""α * MPM + β * NextPred (same-variate + γ * cross-modal) + δ * Contrastive 복합 손실 함수."""
import torch
from torch import nn

from loss.contrastive_loss import CrossModalContrastiveLoss
from loss.masked_mse_loss import MaskedPatchLoss
from loss.next_prediction_loss import NextPredictionLoss


class CombinedLoss(nn.Module):
    """α * MPM + β * NextPred (same-variate + γ * cross-modal) + δ * Contrastive 하이브리드 손실 함수.

    Parameters
    ----------
    alpha:
        Masked reconstruction loss 가중치.
    beta:
        Next-patch prediction loss 가중치. 0이면 next-patch 계산 스킵.
    gamma:
        Cross-modal prediction loss 가중치 (beta 내부 가중). 0이면 비활성.
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
            cross_modal_weight=gamma,
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

        # ── Block Next-Patch Prediction Loss ──
        if self.beta > 0 and next_pred is not None:
            next_dict = self.next_loss_fn(
                next_pred,
                cross_pred_per_type,
                original_patches,
                patch_mask,
                patch_sample_id,
                patch_variate_id,
                time_id=time_id,
                patch_signal_types=patch_signal_types,
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
            contrastive_loss = self.contrastive_loss_fn(
                contrastive_z,
                patch_mask,
                patch_sample_id,
                patch_variate_id,
                time_id,
            )
        else:
            contrastive_loss = reconstructed.new_tensor(0.0)

        total = (
            self.alpha * masked_loss
            + self.beta * (next_loss + cross_modal_loss)
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
        }
