# -*- coding:utf-8 -*-
"""Patient-level Transformer Aggregator + 윈도우 인코딩 헬퍼.

여러 10분 윈도우의 representation을 [CLS] + self-attention으로 aggregate하여
환자 단위 예측에 사용된다 (Mortality / Sepsis / Cardiac Arrest 등).

구조:
    ICU Stay / Patient
    → K개 윈도우 슬라이딩
    → Foundation Model Encoder (frozen 또는 LoRA) → h_1, ..., h_K  (d_model)
    → [CLS] + h_1..h_K → TransformerAggregator → CLS → Probe → 예측

Public API
----------
- TransformerAggregator: nn.Module — [CLS] + K windows → CLS repr
- mean_pool: (B, N, d), (B, N) → (B, d)
- encode_patient_windows: 한 환자의 K 윈도우 → (K, d)
- collate_patients: 가변 K 패딩 → (B, K_max, d), mask, labels
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from data.collate import PackCollate
from data.dataset import BiosignalSample
from data.spatial_map import get_global_spatial_id


DEFAULT_SR = 100.0

# 환자 단위 task에서 공통으로 쓰는 signal_type 매핑
SIGNAL_TYPE_INT: dict[str, int] = {
    "ecg": 0, "abp": 1, "ppg": 2, "cvp": 3,
    "co2": 4, "awp": 5, "pap": 6, "icp": 7,
}


class TransformerAggregator(nn.Module):
    """시간 순서를 반영하는 Transformer 기반 환자 표현 생성기.

    [CLS] 토큰 + K개 윈도우 representation → self-attention → CLS output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_windows: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_windows + 1, d_model) * 0.02
        )  # +1 for CLS

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        chunk_reprs: torch.Tensor,  # (B, K, d_model)
        mask: torch.Tensor | None = None,  # (B, K) bool, True=valid
    ) -> torch.Tensor:  # (B, d_model)
        b, k, _ = chunk_reprs.shape

        # [CLS] + chunk representations
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, chunk_reprs], dim=1)  # (B, K+1, D)

        # Positional embedding
        x = x + self.pos_embed[:, : k + 1, :]

        # Attention mask: CLS는 항상 valid
        if mask is not None:
            cls_mask = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, K+1)
            # TransformerEncoder의 src_key_padding_mask는 True=무시
            padding_mask = ~full_mask
        else:
            padding_mask = None

        out = self.encoder(x, src_key_padding_mask=padding_mask)
        return out[:, 0, :]  # CLS token → (B, d_model)


def mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    """Patch mask 기준 평균 pooling."""
    mask_f = patch_mask.unsqueeze(-1).float()
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


def _make_samples_for_window(
    signals: dict[str, torch.Tensor],  # {sig_type: (win_samples,)}
    idx: int,
    session_prefix: str = "patient",
) -> list[BiosignalSample]:
    """한 윈도우의 다채널 신호들을 BiosignalSample 리스트로 변환."""
    samples = []
    for ch, (sig_type, signal) in enumerate(signals.items()):
        stype_int = SIGNAL_TYPE_INT.get(sig_type, 0)
        spatial_id = get_global_spatial_id(stype_int, 0)
        samples.append(
            BiosignalSample(
                values=signal,
                length=len(signal),
                channel_idx=ch,
                recording_idx=idx,
                sampling_rate=DEFAULT_SR,
                n_channels=len(signals),
                win_start=0,
                signal_type=stype_int,
                session_id=f"{session_prefix}_{idx}",
                spatial_id=spatial_id,
            )
        )
    return samples


def encode_patient_windows(
    model,
    patient: dict,
    patch_size: int,
    max_windows: int,
    use_lora: bool = False,
    session_prefix: str = "patient",
) -> torch.Tensor:
    """한 환자의 K개 윈도우를 인코딩하여 (K, d_model) 반환.

    Parameters
    ----------
    model:
        DownstreamModelWrapper — extract_features 또는 batch_to_device + model
    patient:
        {"signals": {sig_type: (K, win_samples)}, "n_windows": K, ...}
    patch_size:
        Encoder의 patch_size (보통 100 또는 200)
    max_windows:
        K가 너무 크면 균등 샘플링으로 축소
    use_lora:
        True이면 gradient 활성 경로 (LoRA fine-tune 시). False이면 no_grad.
    session_prefix:
        BiosignalSample.session_id prefix (task별 구분용; 기본 "patient")
    """
    sig_types = list(patient["signals"].keys())
    k = patient["n_windows"]

    # max_windows 제한 (균등 샘플링)
    if k > max_windows:
        indices = np.linspace(0, k - 1, max_windows, dtype=int)
    else:
        indices = np.arange(k)

    multi = len(sig_types) > 1
    collate_mode = "any_variate" if multi else "ci"
    win_samples = patient["signals"][sig_types[0]].shape[1]

    collate = PackCollate(
        max_length=win_samples, collate_mode=collate_mode, patch_size=patch_size
    )

    grad_ctx = torch.enable_grad() if use_lora else torch.no_grad()
    chunk_reprs = []
    with grad_ctx:
        for idx in indices:
            win_signals = {st: patient["signals"][st][idx] for st in sig_types}
            samples = _make_samples_for_window(win_signals, idx, session_prefix)
            batch = collate(samples)

            if use_lora:
                batch = model.batch_to_device(batch)
                out = model.model(batch, task="masked")
                feat = mean_pool(out["encoded"], out["patch_mask"])
            else:
                feat = model.extract_features(batch, pool="mean")

            chunk_reprs.append(feat)  # (1, d_model)

    return torch.cat(chunk_reprs, dim=0)  # (K', d_model)


def collate_patients(
    patient_reprs: list[torch.Tensor],  # [(K_i, d_model), ...]
    labels: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """가변 K를 패딩하여 배치를 만든다.

    Returns
    -------
    (padded_reprs (B, K_max, d), mask (B, K_max), labels (B,))
    """
    k_max = max(r.shape[0] for r in patient_reprs)
    d_model = patient_reprs[0].shape[1]
    b = len(patient_reprs)

    padded = torch.zeros(b, k_max, d_model, device=device)
    mask = torch.zeros(b, k_max, dtype=torch.bool, device=device)

    for i, r in enumerate(patient_reprs):
        k_i = r.shape[0]
        padded[i, :k_i] = r.to(device)
        mask[i, :k_i] = True

    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    return padded, mask, labels_t
