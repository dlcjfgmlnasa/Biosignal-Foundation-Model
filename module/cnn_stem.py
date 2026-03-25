# -*- coding:utf-8 -*-
"""Modality-specific 1D-CNN stem for biosignal patch embedding.

신호 타입(ECG, EEG, EMG 등)별 전용 1D-CNN으로 raw patch에서
저수준 특징을 추출한다. ``padding="same"`` + ``AdaptiveAvgPool1d(1)``
구조로 임의의 patch_size를 처리할 수 있어 MultiResolution에도 호환된다.
"""
import torch
from torch import nn


class Conv1dStem(nn.Module):
    """단일 modality용 1D-CNN stem.

    Raw patch ``(M, patch_size)`` → embedding ``(M, d_model)``.

    Parameters
    ----------
    d_model:
        출력 임베딩 차원.
    hidden_channels:
        중간 Conv1d 채널 수.
    num_layers:
        Conv1d 레이어 수 (최소 2).
    kernel_size:
        중간 레이어의 커널 크기. 마지막 레이어는 항상 ``1``.
    bias:
        Conv1d bias 사용 여부.
    """

    def __init__(
        self,
        d_model: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"

        layers: list[nn.Module] = []

        # First layer: 1 → hidden_channels
        layers.append(
            nn.Conv1d(1, hidden_channels, kernel_size, padding="same", bias=bias)
        )
        layers.append(nn.GELU())

        # Middle layers: hidden_channels → hidden_channels
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv1d(
                    hidden_channels, hidden_channels, kernel_size,
                    padding="same", bias=bias,
                )
            )
            layers.append(nn.GELU())

        # Last layer: hidden_channels → d_model (pointwise, no activation)
        layers.append(nn.Conv1d(hidden_channels, d_model, 1, bias=bias))

        self.convs = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:  # (M, P) → (M, d_model)
        """Forward.

        Parameters
        ----------
        patches:
            ``(M, patch_size)`` — M개의 raw patches.

        Returns
        -------
        torch.Tensor
            ``(M, d_model)`` — patch embeddings.
        """
        x = patches.unsqueeze(1)    # (M, 1, P)
        x = self.convs(x)           # (M, d_model, P)
        x = self.pool(x)            # (M, d_model, 1)
        return x.squeeze(-1)        # (M, d_model)


class ModalityCNNStem(nn.Module):
    """Modality-specific 1D-CNN stem dispatcher.

    신호 타입별 전용 ``Conv1dStem``을 보유하고, per-patch signal_type에
    따라 vectorized gather/scatter로 라우팅한다.

    Parameters
    ----------
    num_signal_types:
        신호 타입 수 (기본 6: ECG, ABP, EEG, PPG, EMG, Resp).
    d_model:
        출력 임베딩 차원.
    hidden_channels:
        각 Conv1dStem의 중간 채널 수.
    num_layers:
        각 Conv1dStem의 Conv1d 레이어 수.
    kernel_size:
        각 Conv1dStem의 커널 크기.
    bias:
        Conv1d bias 사용 여부.
    """

    def __init__(
        self,
        num_signal_types: int,
        d_model: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_signal_types = num_signal_types
        self.d_model = d_model
        self.stems = nn.ModuleList([
            Conv1dStem(d_model, hidden_channels, num_layers, kernel_size, bias)
            for _ in range(num_signal_types)
        ])

    def forward(
        self,
        patches: torch.Tensor,        # (B, N, P)
        signal_types: torch.Tensor,    # (B, N) long
    ) -> torch.Tensor:                 # (B, N, d_model)
        """Forward — per-type vectorized dispatch.

        Parameters
        ----------
        patches:
            ``(B, N, patch_size)`` — raw patches.
        signal_types:
            ``(B, N)`` — 각 패치의 신호 타입 인덱스 (0-based).

        Returns
        -------
        torch.Tensor
            ``(B, N, d_model)`` — modality-specific embeddings.
        """
        B, N, P = patches.shape
        device = patches.device
        dtype = patches.dtype

        flat_patches = patches.reshape(B * N, P)          # (B*N, P)
        flat_types = signal_types.reshape(B * N)           # (B*N,)
        flat_output = torch.zeros(
            B * N, self.d_model, device=device, dtype=dtype,
        )

        for t, stem in enumerate(self.stems):
            mask = flat_types == t                         # (B*N,)
            if mask.any():
                selected = flat_patches[mask]              # (M_t, P)
                embedded = stem(selected)                  # (M_t, d_model)
                flat_output[mask] = embedded

        return flat_output.reshape(B, N, self.d_model)    # (B, N, d_model)
