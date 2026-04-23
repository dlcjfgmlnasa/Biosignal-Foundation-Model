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

    Same-variate / cross-modal 두 항의 raw loss를 반환한다. 가중(β, γ)은
    호출자(예: ``CombinedLoss``)에서 곱한다. 비활성화는 ``compute_next``/
    ``compute_cross`` 플래그로 제어 — 가중치를 통한 silent skip을 방지.

    Parameters
    ----------
    peak_alpha:
        Peak 가중치 강도. 0이면 일반 MSE.
    lambda_spec:
        Multi-Resolution STFT loss 가중치. 0이면 비활성.
    spec_n_ffts:
        STFT window 크기들.
    """

    def __init__(
        self,
        peak_alpha: float = 0.0,
        lambda_spec: float = 0.0,
        spec_n_ffts: tuple[int, ...] = (16, 32, 64),
    ) -> None:
        super().__init__()
        self.peak_alpha = peak_alpha
        self.lambda_spec = lambda_spec
        self.spec_n_ffts = spec_n_ffts

    def forward(
        self,
        next_pred: torch.Tensor | None,  # (B, N, K, P) — block next-patch 예측
        cross_pred_per_type: torch.Tensor
        | None,  # (B, N, T, P) — per-target-type cross-modal 예측
        original_patches: torch.Tensor,  # (B, N, P)
        patch_mask: torch.Tensor,  # (B, N) bool
        patch_sample_id: torch.Tensor,  # (B, N) long
        patch_variate_id: torch.Tensor,  # (B, N) long
        time_id: torch.Tensor | None = None,  # (B, N) long — cross-modal 페어링용
        patch_signal_types: torch.Tensor
        | None = None,  # (B, N) long — mechanism group 필터용
        compute_next: bool = True,
        compute_cross: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Block Next Prediction loss 계산.

        각 position n에서 next_pred[:, n, k-1, :]가 n+k 시점 raw patch를 예측하도록
        학습한다 (k=1..K). K step별 loss를 균등 평균한다.

        Returns
        -------
        dict with keys:
            ``next_loss``: same-variate block next-patch raw MSE (가중 없음).
            ``next_spec``: same-variate spectral loss.
            ``cross_modal_loss``: cross-modal raw MSE (가중 없음).
        """
        ref = (
            next_pred if next_pred is not None else original_patches
        )  # 텐서 device/dtype 참조용
        zero = ref.new_tensor(0.0)

        # ── Same-variate block next-patch loss ──
        if compute_next and next_pred is not None:
            next_dict = self._same_variate_loss(
                next_pred,
                original_patches,
                patch_mask,
                patch_sample_id,
                patch_variate_id,
            )
            next_loss = next_dict["total"]
            next_spec = next_dict["spec"]
        else:
            next_loss = zero
            next_spec = zero

        # ── Cross-modal loss ──
        if (
            compute_cross
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
            cross_modal_loss = cross_dict["total"]
        else:
            cross_modal_loss = zero

        return {
            "next_loss": next_loss,
            "next_spec": next_spec,
            "cross_modal_loss": cross_modal_loss,
        }

    def _same_variate_loss(
        self,
        next_pred: torch.Tensor,  # (B, N, K, P)
        original_patches: torch.Tensor,  # (B, N, P)
        patch_mask: torch.Tensor,  # (B, N) bool
        patch_sample_id: torch.Tensor,  # (B, N) long
        patch_variate_id: torch.Tensor,  # (B, N) long
    ) -> dict[str, torch.Tensor]:
        """Same-variate Block Next Prediction loss.

        각 k ∈ [1..K]에 대해:
          target = ``original_patches[:, k:N]``
          pred   = ``next_pred[:, :N-k, k-1, :]``
        같은 (sample_id, variate_id) 쌍에서만 valid. K step loss를 균등 평균.

        Returns
        -------
        dict with keys: ``mse``, ``spec``, ``total``.
        """
        b, n, k, p = next_pred.shape

        mse_sum = next_pred.new_tensor(0.0)
        spec_sum = next_pred.new_tensor(0.0)
        total_sum = next_pred.new_tensor(0.0)
        valid_steps = 0

        for step in range(1, k + 1):
            if n <= step:
                continue
            target_step = original_patches[:, step:, :]  # (B, N-step, P)
            pred_step = next_pred[:, : n - step, step - 1, :]  # (B, N-step, P)

            valid = (
                patch_mask[:, : n - step]
                & patch_mask[:, step:]
                & (patch_sample_id[:, : n - step] == patch_sample_id[:, step:])
                & (patch_variate_id[:, : n - step] == patch_variate_id[:, step:])
            )  # (B, N-step)

            if not bool(valid.any()):
                continue

            loss_dict = compute_patch_loss(
                pred_step[valid],
                target_step[valid],
                peak_alpha=self.peak_alpha,
                lambda_spec=self.lambda_spec,
                spec_n_ffts=self.spec_n_ffts,
            )
            mse_sum = mse_sum + loss_dict["mse"]
            spec_sum = spec_sum + loss_dict["spec"]
            total_sum = total_sum + loss_dict["total"]
            valid_steps += 1

        if valid_steps == 0:
            zero = next_pred.new_tensor(0.0)
            return {"mse": zero, "spec": zero, "total": zero}

        denom = float(valid_steps)
        return {
            "mse": mse_sum / denom,
            "spec": spec_sum / denom,
            "total": total_sum / denom,
        }

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

        ``CROSS_PRED_ALLOWED_PAIRS``(data/spatial_map.py)에 정의된 생리학적으로
        타당한 쌍만 허용. 현재: ECG↔ABP, ECG↔PPG, ABP↔PPG, ABP↔PAP, CVP↔PAP,
        ABP↔ICP. (CO2↔AWP, ABP↔CVP 등은 명시적으로 기각.)
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
        source_st = patch_signal_types[b_idx, i_idx]  # (K,)
        target_st = patch_signal_types[b_idx, j_idx]  # (K,)
        pred_p = cross_pred_per_type[b_idx, i_idx, target_st]  # (K, P)
        target_p = original_patches[b_idx, j_idx]  # (K, P)

        # 쌍 타입별 균등 가중 평균 (pair-balanced loss)
        # pair_key: (source_type, target_type) → 쌍 타입 식별
        pair_key = source_st * _MAX_ST + target_st  # (K,)
        unique_pairs = pair_key.unique()

        mse_sum = pred_p.new_tensor(0.0)
        spec_sum = pred_p.new_tensor(0.0)
        total_sum = pred_p.new_tensor(0.0)
        n_pairs = 0

        for pk in unique_pairs:
            mask = pair_key == pk
            pair_loss = compute_patch_loss(
                pred_p[mask],
                target_p[mask],
                peak_alpha=self.peak_alpha,
                lambda_spec=self.lambda_spec,
                spec_n_ffts=self.spec_n_ffts,
            )
            mse_sum = mse_sum + pair_loss["mse"]
            spec_sum = spec_sum + pair_loss["spec"]
            total_sum = total_sum + pair_loss["total"]
            n_pairs += 1

        return {
            "mse": mse_sum / max(n_pairs, 1),
            "spec": spec_sum / max(n_pairs, 1),
            "total": total_sum / max(n_pairs, 1),
        }
