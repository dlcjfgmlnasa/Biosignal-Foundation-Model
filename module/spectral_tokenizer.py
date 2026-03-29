# -*- coding:utf-8 -*-
"""Spectral Tokenizer — EEG 패치의 주파수 특성을 target으로 변환.

raw EEG 파형 대신 STFT magnitude를 reconstruction target으로 사용하여,
phase 불변적인 주파수 특징 학습을 유도한다.

파라미터 없음 (torch.stft는 학습 불가 변환).
"""
from __future__ import annotations

import torch
from torch import nn


class SpectralTokenizer(nn.Module):
    """EEG 패치를 log-magnitude spectrogram으로 변환한다.

    Parameters
    ----------
    patch_size:
        입력 패치 크기 (time-step 수). 기본 100 (100Hz에서 1초).
    n_fft:
        FFT 윈도우 크기. 주파수 해상도 = sr / n_fft.
        64 at 100Hz → 해상도 ~1.56Hz, 주파수 빈 33개 (0~50Hz).
    hop_length:
        STFT hop 크기. 시간 프레임 수 = floor((patch_size - n_fft) / hop_length) + 1.
        16 at patch_size=100 → 3 프레임.

    출력 차원: ``(n_fft // 2 + 1) * n_frames``.
    기본값(n_fft=64, hop=16, patch=100): 33 * 3 = **99**.

    Usage::

        tokenizer = SpectralTokenizer(patch_size=100)
        patches = torch.randn(32, 100)   # (M, P)
        target = tokenizer(patches)       # (M, 99)
        print(tokenizer.output_dim)       # 99
    """

    def __init__(
        self,
        patch_size: int = 100,
        n_fft: int = 64,
        hop_length: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Hann window (non-learnable buffer)
        self.register_buffer("window", torch.hann_window(n_fft))

        # 출력 차원 계산
        n_freq_bins = n_fft // 2 + 1
        n_frames = (patch_size - n_fft) // hop_length + 1
        self.n_freq_bins = n_freq_bins
        self.n_frames = n_frames
        self.output_dim = n_freq_bins * n_frames

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """패치를 log-magnitude spectrogram으로 변환한다.

        Parameters
        ----------
        patches:
            ``(M, P)`` — M개 패치, 각 P time-steps.
            또는 ``(B, N, P)`` — batch 포함. 내부에서 flatten 후 복원.

        Returns
        -------
        ``(M, output_dim)`` 또는 ``(B, N, output_dim)`` — log-magnitude spectrogram.
        """
        orig_shape = patches.shape
        if patches.ndim == 3:
            B, N, P = patches.shape
            patches = patches.reshape(B * N, P)
        else:
            B, N = None, None

        # STFT: (M, P) → (M, n_freq, n_frames) complex
        spec = torch.stft(
            patches,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )  # (M, n_freq_bins, n_frames)

        # Magnitude + log scale (수치 안정)
        mag = spec.abs()  # (M, n_freq_bins, n_frames)
        log_mag = torch.log1p(mag)  # log(1 + |STFT|)

        # Flatten frequency × time → single vector
        output = log_mag.reshape(patches.shape[0], -1)  # (M, output_dim)

        # Per-patch 정규화: zero mean, unit variance
        # raw 패치가 Z-score 정규화되어 MSE ~1.0이므로, spectral도 동일 스케일로 맞춤
        mean = output.mean(dim=-1, keepdim=True)  # (M, 1)
        std = output.std(dim=-1, keepdim=True).clamp(min=1e-8)  # (M, 1)
        output = (output - mean) / std

        if B is not None:
            output = output.reshape(B, N, -1)  # (B, N, output_dim)

        return output
