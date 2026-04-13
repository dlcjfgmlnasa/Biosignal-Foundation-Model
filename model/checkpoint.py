# -*- coding:utf-8 -*-
"""Checkpoint 저장/로드 유틸리티.

모델 constructor args를 ``config``에 저장하여 재구성 가능하게 한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """모델 checkpoint를 저장한다.

    Parameters
    ----------
    path:
        저장 경로.
    model:
        저장할 모델.
    optimizer:
        옵티마이저 (선택).
    epoch:
        현재 에포크 번호.
    config:
        모델 재구성에 필요한 constructor args.
    **extra:
        추가 메타데이터.
    """
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if config is not None:
        state["config"] = config
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Checkpoint를 불러와 모델(및 옵티마이저)에 적용한다.

    Parameters
    ----------
    path:
        checkpoint 경로.
    model:
        state_dict를 로드할 모델.
    optimizer:
        state_dict를 로드할 옵티마이저 (선택).
    device:
        텐서를 로드할 디바이스.

    Returns
    -------
    dict
        checkpoint에 저장된 전체 state (epoch, config, extra 등).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(
        state["model_state_dict"],
        strict=False,
    )
    if missing:
        print(f"  [checkpoint] Missing keys (newly added): {missing}")
    if unexpected:
        print(f"  [checkpoint] Unexpected keys (removed): {unexpected}")
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state
