# -*- coding:utf-8 -*-
from __future__ import annotations

"""α * MPM + β * NextPred + γ * CrossModal 복합 손실 함수.

각 손실 항은 독립 가중치를 가진다. β=0이면 next-pred만 비활성, γ=0이면
cross-modal만 비활성 — 이전 버전처럼 β=0이 cross-modal까지 silent 0으로
만들지 않는다.
"""
import torch  # noqa: E402
from torch import nn  # noqa: E402

from loss.masked_mse_loss import MaskedPatchLoss  # noqa: E402
from loss.next_prediction_loss import NextPredictionLoss  # noqa: E402


class CombinedLoss(nn.Module):
    """α * MPM + β * NextPred + γ * CrossModal 하이브리드 손실 함수.

    각 항은 독립 가중치. β=0은 next-pred만 비활성, γ=0은 cross-modal만 비활성.

    Parameters
    ----------
    alpha:
        Masked reconstruction loss 가중치. 0이면 비활성.
    beta:
        Same-variate next-patch prediction loss 가중치. 0이면 next-patch 비활성.
    gamma:
        Cross-modal prediction loss 가중치. 0이면 cross-modal 비활성.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        gamma_next: float = 0.0,
        peak_alpha: float = 0.0,
        lambda_spec: float = 0.0,
        spec_n_ffts: tuple[int, ...] = (16, 32, 64),
        coupling_weights: dict[tuple[int, int], float] | None = None,
        cross_masked_target_only: bool = True,
        learnable_coupling: bool = False,
        coupling_l1: float = 0.0,
        dual_coupling: bool = False,
        cross_observed_source_only: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gamma_next = gamma_next
        self.masked_loss_fn = MaskedPatchLoss(
            peak_alpha=peak_alpha,
            lambda_spec=lambda_spec,
            spec_n_ffts=spec_n_ffts,
        )
        # coupling_weights=None → directed allowlist 균일
        # 1.0(CROSS_COUPLING_WEIGHTS).
        # 경험적 가중은 폐기됨(팀 검증). dict override 시 임의 가중 주입 가능.
        self.next_loss_fn = NextPredictionLoss(
            peak_alpha=peak_alpha,
            lambda_spec=lambda_spec,
            spec_n_ffts=spec_n_ffts,
            coupling_weights=coupling_weights,
            cross_masked_target_only=cross_masked_target_only,
            learnable_coupling=learnable_coupling,
            coupling_l1=coupling_l1,
            dual_coupling=dual_coupling,
            cross_observed_source_only=cross_observed_source_only,
        )

    def forward(
        self,
        reconstructed: torch.Tensor,  # (B, N, patch_size)
        next_pred: torch.Tensor
        | None,  # (B, N, K, patch_size) or None — Block Next Prediction
        original_patches: torch.Tensor,  # (B, N, patch_size)
        pred_mask: torch.Tensor,  # (B, N) bool — 마스킹된 패치
        patch_mask: torch.Tensor,  # (B, N) bool — 유효 패치
        patch_sample_id: torch.Tensor,  # (B, N) long — 패치별 sample_id
        patch_variate_id: torch.Tensor,  # (B, N) long — 패치별 variate_id
        cross_pred_per_type: torch.Tensor
        # (B, N, T, patch_size) — per-target-type cross-modal 예측
        | None = None,
        time_id: torch.Tensor
        | None = None,  # (B, N) long — cross-modal 페어링용
        patch_signal_types: torch.Tensor
        | None = None,  # (B, N) long — mechanism group 필터용
        cross_next_pred_per_type: torch.Tensor
        # (B, N, T, patch_size) — 예측형(t→t+1) 전용 head 출력
        | None = None,
    ) -> dict[str, torch.Tensor]:
        # ── Masked Reconstruction Loss (MSE + Gradient + Spectral) ──
        masked_dict = self.masked_loss_fn(
            reconstructed, original_patches, pred_mask
        )
        masked_loss = masked_dict["total"]

        # ── Block Next-Patch + Cross-Modal Prediction Loss (β / γ 독립) ──
        compute_next = self.beta > 0 and next_pred is not None
        compute_cross = (
            self.gamma > 0
            and cross_pred_per_type is not None
            and time_id is not None
        )
        compute_cross_next = (
            self.gamma_next > 0
            and cross_pred_per_type is not None
            and time_id is not None
        )
        if compute_next or compute_cross or compute_cross_next:
            next_dict = self.next_loss_fn(
                next_pred,
                cross_pred_per_type,
                original_patches,
                patch_mask,
                patch_sample_id,
                patch_variate_id,
                time_id=time_id,
                patch_signal_types=patch_signal_types,
                pred_mask=pred_mask,
                compute_next=compute_next,
                compute_cross=compute_cross,
                compute_cross_next=compute_cross_next,
                cross_next_pred_per_type=cross_next_pred_per_type,
            )
            next_loss = next_dict["next_loss"]
            next_spec = next_dict["next_spec"]
            cross_modal_loss = next_dict["cross_modal_loss"]
            cross_next_loss = next_dict["cross_next_loss"]
        else:
            next_loss = reconstructed.new_tensor(0.0)
            next_spec = reconstructed.new_tensor(0.0)
            cross_modal_loss = reconstructed.new_tensor(0.0)
            cross_next_loss = reconstructed.new_tensor(0.0)

        total = (
            self.alpha * masked_loss
            + self.beta * next_loss
            + self.gamma * cross_modal_loss
            + self.gamma_next * cross_next_loss
        )

        # 결합 그래프 희소성. learnable_coupling 이 켜졌을 때만 0 이 아니다.
        # 이 항이 없으면 모든 쌍 가중치가 1 로 수렴해 "모델이 쌍을 고른다"는
        # 설계가 무의미해진다.
        npl = self.next_loss_fn
        if getattr(npl, "learnable_coupling", False) and npl.coupling_l1 > 0:
            coupling_l1_term = npl.coupling_l1 * npl.coupling_matrix().mean()
            if getattr(npl, "dual_coupling", False):
                # 두 그래프를 각각 희소화한다. 한쪽만 걸면 다른 쪽이 전부 1 로
                # 수렴해 "쌍을 고른다"는 설계가 그 축에서만 무너진다.
                coupling_l1_term = coupling_l1_term + (
                    npl.coupling_l1 * npl.coupling_matrix_next().mean()
                )
            total = total + coupling_l1_term
        else:
            coupling_l1_term = reconstructed.new_tensor(0.0)

        return {
            "total": total,
            "coupling_l1": coupling_l1_term,
            "masked_loss": masked_loss,
            "masked_mse": masked_dict["mse"],
            "masked_spec": masked_dict["spec"],
            "next_loss": next_loss,
            "next_spec": next_spec,
            "cross_modal_loss": cross_modal_loss,
            "cross_next_loss": cross_next_loss,
        }
