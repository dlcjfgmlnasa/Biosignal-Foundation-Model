# -*- coding:utf-8 -*-
"""Packed scalers for normalizing packed time series batches.

Ported from Salesforce uni2ts (Apache 2.0).
Adapted: uses local safe_div helper instead of uni2ts.common.torch_util.
"""
from typing import Optional

import torch
from einops import reduce
from torch import nn

from ._util import safe_div


class PackedScaler(nn.Module):
    def forward(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: Optional[torch.Tensor] = None,  # (*batch, seq_len, #dim) bool
        sample_id: Optional[torch.Tensor] = None,  # (*batch, seq_len) long
        variate_id: Optional[torch.Tensor] = None,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )
        if variate_id is None:
            variate_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        loc, scale = self._get_loc_scale(
            target.double(), observed_mask, sample_id, variate_id
        )
        return loc.float(), scale.float()

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        raise NotImplementedError


class PackedNOPScaler(PackedScaler):
    """No-op scaler: loc=0, scale=1."""

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        loc = torch.zeros_like(target, dtype=target.dtype)
        scale = torch.ones_like(target, dtype=target.dtype)
        return loc, scale


class PackedStdScaler(PackedScaler):
    """Z-score normalization with Bessel's correction, grouped by sample_id/variate_id."""

    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        scale = torch.sqrt(var + self.minimum_scale)
        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1
        return loc, scale


class PackedAbsMeanScaler(PackedScaler):
    """Absolute mean scaling, grouped by sample_id/variate_id."""

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        scale = reduce(
            id_mask
            * reduce(target.abs() * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        scale = safe_div(scale, tobs)
        loc = torch.zeros_like(scale)

        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1
        return loc, scale
