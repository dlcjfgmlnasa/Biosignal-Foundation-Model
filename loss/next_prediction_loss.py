# -*- coding:utf-8 -*-
from __future__ import annotations

"""Next-Patch Prediction Loss.

Phase 1 (CI): 같은 variate 내 시간 예측 (dynamics).
Phase 2 (Any-variate): 같은 variate 시간 예측 + cross-modal 예측 (causality).
"""
import torch
from torch import nn

from data.spatial_map import CROSS_PRED_ALLOWED_PAIRS, MECHANISM_GROUP
from loss.masked_mse_loss import compute_patch_loss


# signal_type → mechanism_group 변환용 lookup tensor (최대 signal_type + 1 크기)
_MAX_ST = max(MECHANISM_GROUP.keys()) + 1
_MECH_GROUP_LUT = torch.zeros(_MAX_ST, dtype=torch.long)
for _st, _mg in MECHANISM_GROUP.items():
    _MECH_GROUP_LUT[_st] = _mg

# Cross-pred 허용 쌍 lookup tensor (양방향)
_ALLOWED_PAIR_LUT = torch.zeros(_MAX_ST, _MAX_ST, dtype=torch.bool)
for _a, _b in CROSS_PRED_ALLOWED_PAIRS:
    _ALLOWED_PAIR_LUT[_a, _b] = True
    _ALLOWED_PAIR_LUT[_b, _a] = True


class NextPredictionLoss(nn.Module):
    """Next-Patch Prediction Loss (same-variate + cross-modal).

    Parameters
    ----------
    cross_modal_weight:
        γ 가중치. 0이면 cross-modal loss 비활성 (Phase 1 동작).
    peak_alpha:
        Peak 가중치 강도. 0이면 일반 MSE.
    lambda_spec:
        Multi-Resolution STFT loss 가중치. 0이면 비활성.
    spec_n_ffts:
        STFT window 크기들.
    """

    def __init__(
        self,
        cross_modal_weight: float = 0.0,
        peak_alpha: float = 0.0,
        lambda_spec: float = 0.0,
        spec_n_ffts: tuple[int, ...] = (16, 32, 64),
    ) -> None:
        super().__init__()
        self.cross_modal_weight = cross_modal_weight
        self.peak_alpha = peak_alpha
        self.lambda_spec = lambda_spec
        self.spec_n_ffts = spec_n_ffts

    def forward(
        self,
        next_pred: torch.Tensor,  # (B, N, P) — same-variate 예측
        cross_pred_per_type: torch.Tensor
        | None,  # (B, N, T, P) — per-target-type cross-modal 예측
        original_patches: torch.Tensor,  # (B, N, P)
        patch_mask: torch.Tensor,  # (B, N) bool
        patch_sample_id: torch.Tensor,  # (B, N) long
        patch_variate_id: torch.Tensor,  # (B, N) long
        time_id: torch.Tensor | None = None,  # (B, N) long — cross-modal 페어링용
        patch_signal_types: torch.Tensor
        | None = None,  # (B, N) long — mechanism group 필터용
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
        next_dict = self._same_variate_loss(
            next_pred,
            original_patches,
            patch_mask,
            patch_sample_id,
            patch_variate_id,
            horizon,
        )

        # ── Cross-modal loss ──
        if (
            self.cross_modal_weight > 0
            and cross_pred_per_type is not None
            and time_id is not None
        ):
            cross_dict = self._cross_modal_loss(
                cross_pred_per_type,
                original_patches,
                patch_mask,
                patch_sample_id,
                patch_variate_id,
                time_id,
                patch_signal_types,
            )
            cross_modal_loss = self.cross_modal_weight * cross_dict["total"]
        else:
            cross_modal_loss = next_pred.new_tensor(0.0)

        return {
            "next_loss": next_dict["total"],
            "next_spec": next_dict["spec"],
            "cross_modal_loss": cross_modal_loss,
        }

    def _same_variate_loss(
        self,
        next_pred: torch.Tensor,  # (B, N, P)
        original_patches: torch.Tensor,  # (B, N, P)
        patch_mask: torch.Tensor,  # (B, N) bool
        patch_sample_id: torch.Tensor,  # (B, N) long
        patch_variate_id: torch.Tensor,  # (B, N) long
        horizon: int,
    ) -> dict[str, torch.Tensor]:
        """Same-variate next-patch prediction loss.

        Returns
        -------
        dict with keys: ``mse``, ``spec``, ``total``.
        """
        target_next = original_patches[:, horizon:, :]  # (B, N-H, P)
        pred_next = next_pred[:, :-horizon, :]  # (B, N-H, P)

        valid = (
            patch_mask[:, :-horizon]
            & patch_mask[:, horizon:]
            & (patch_sample_id[:, :-horizon] == patch_sample_id[:, horizon:])
            & (patch_variate_id[:, :-horizon] == patch_variate_id[:, horizon:])
        )  # (B, N-H)

        n_valid = valid.float().sum()
        if n_valid > 0:
            horizon_weight = 1.0 / horizon
            loss_dict = compute_patch_loss(
                pred_next[valid],
                target_next[valid],
                peak_alpha=self.peak_alpha,
                lambda_spec=self.lambda_spec,
                spec_n_ffts=self.spec_n_ffts,
            )
            return {
                "mse": loss_dict["mse"] * horizon_weight,
                "spec": loss_dict["spec"] * horizon_weight,
                "total": loss_dict["total"] * horizon_weight,
            }

        zero = next_pred.new_tensor(0.0)
        return {"mse": zero, "spec": zero, "total": zero}

    def _cross_modal_loss(
        self,
        cross_pred_per_type: torch.Tensor,  # (B, N, T, P) — per-target-type prediction
        original_patches: torch.Tensor,  # (B, N, P)
        patch_mask: torch.Tensor,  # (B, N) bool
        patch_sample_id: torch.Tensor,  # (B, N) long
        patch_variate_id: torch.Tensor,  # (B, N) long
        time_id: torch.Tensor,  # (B, N) long
        patch_signal_types: torch.Tensor | None = None,  # (B, N) long
    ) -> dict[str, torch.Tensor]:
        """Cross-modal prediction loss (target-conditioned).

        같은 (sample_id, time_id)에서 서로 다른 variate_id를 가진 패치 쌍을 매칭하고,
        target의 signal type에 해당하는 cross_pred를 선택하여 MSE를 계산한다.

        ``CROSS_PRED_ALLOWED_PAIRS``에 정의된 생리학적으로 타당한 쌍만 허용.
        (ECG↔ABP, ECG↔PPG, ABP↔PPG, CO2↔AWP)
        """
        # group_key: (batch, sample_id, time_id)가 같은 패치를 그룹핑
        b, n = time_id.shape
        k = time_id.max() + 1  # 0-dim 텐서 (CUDA sync 없음)
        s = patch_sample_id.max() + 1  # 0-dim 텐서 (CUDA sync 없음)
        batch_idx = torch.arange(b, device=time_id.device).unsqueeze(-1)  # (B, 1)
        group_key = batch_idx * (s * k) + patch_sample_id * k + time_id  # (B, N)

        # (B, N, N) pairwise 비교
        same_group = group_key.unsqueeze(-1) == group_key.unsqueeze(-2)
        diff_variate = patch_variate_id.unsqueeze(-1) != patch_variate_id.unsqueeze(-2)
        both_valid = patch_mask.unsqueeze(-1) & patch_mask.unsqueeze(-2)
        # 패딩 (variate_id == 0) 제외
        non_pad = (patch_variate_id > 0).unsqueeze(-1) & (
            patch_variate_id > 0
        ).unsqueeze(-2)

        cross_mask = same_group & diff_variate & both_valid & non_pad  # (B, N, N)

        # Allowed Pair 필터: 생리학적으로 타당한 쌍만 허용
        if patch_signal_types is not None:
            allowed_lut = _ALLOWED_PAIR_LUT.to(patch_signal_types.device)
            st_i = patch_signal_types.unsqueeze(-1)  # (B, N, 1)
            st_j = patch_signal_types.unsqueeze(-2)  # (B, 1, N)
            allowed = allowed_lut[st_i, st_j]  # (B, N, N)
            cross_mask = cross_mask & allowed

        b_idx, i_idx, j_idx = torch.where(cross_mask)

        if len(b_idx) == 0:
            zero = cross_pred_per_type.new_tensor(0.0)
            return {"mse": zero, "spec": zero, "total": zero}

        # Target signal type에 맞는 prediction 선택
        target_st = patch_signal_types[b_idx, j_idx]  # (K,)
        pred_p = cross_pred_per_type[b_idx, i_idx, target_st]  # (K, P)
        target_p = original_patches[b_idx, j_idx]  # (K, P)

        return compute_patch_loss(
            pred_p,
            target_p,
            peak_alpha=self.peak_alpha,
            lambda_spec=self.lambda_spec,
            spec_n_ffts=self.spec_n_ffts,
        )
