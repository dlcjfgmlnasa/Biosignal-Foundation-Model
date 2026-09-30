# -*- coding:utf-8 -*-
from __future__ import annotations

from loss.masked_mse_loss import MaskedPatchLoss, create_patch_mask
from loss.next_prediction_loss import NextPredictionLoss
from loss.criterion import CombinedLoss

__all__ = [
    "MaskedPatchLoss",
    "create_patch_mask",
    "NextPredictionLoss",
    "CombinedLoss",
]
