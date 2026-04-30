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


def _time_sinusoidal_embedding(
    time_secs: torch.Tensor,  # (B, K)
    d_model: int,
    base: float = 10000.0,
) -> torch.Tensor:  # (B, K, d_model)
    """Continuous-time positional embedding (Vaswani-style on real time).

    time_secs: 각 윈도우의 시작 시각 (초). 동일 환자 내에서 상대시간이면 충분.
    """
    half = d_model // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=time_secs.dtype, device=time_secs.device) / half)
    )  # (half,)
    angles = time_secs.unsqueeze(-1) * inv_freq  # (B, K, half)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, K, 2*half)
    if emb.shape[-1] < d_model:
        # d_model이 홀수일 경우 0으로 pad
        pad = torch.zeros(
            *emb.shape[:-1], d_model - emb.shape[-1],
            dtype=emb.dtype, device=emb.device,
        )
        emb = torch.cat([emb, pad], dim=-1)
    return emb


class TransformerAggregator(nn.Module):
    """시간 순서를 반영하는 Transformer 기반 환자 표현 생성기.

    [CLS] 토큰 + K개 윈도우 representation → self-attention → CLS output.

    Positional embedding 모드 (`pos_mode`):
      - "time": time_secs 기반 sinusoidal (continuous, 갭 인지)
      - "index": K개 학습 가능한 임베딩 (legacy, backward compat)
      - "auto" (default): forward에서 time_secs가 주어지면 time, 아니면 index
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_windows: int = 128,
        pos_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pos_mode = pos_mode
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Index-based pos_embed은 fallback용 항상 보유 (CLS 포함 max_windows+1)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_windows + 1, d_model) * 0.02
        )
        # CLS 전용 학습 임베딩 (time mode에서 CLS 위치 표시)
        self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

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
        time_secs: torch.Tensor | None = None,  # (B, K) float, 윈도우 시작 시각(초)
    ) -> torch.Tensor:  # (B, d_model)
        b, k, _ = chunk_reprs.shape

        # [CLS] + chunk representations
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, chunk_reprs], dim=1)  # (B, K+1, D)

        # Positional embedding 선택
        use_time = (
            self.pos_mode == "time"
            or (self.pos_mode == "auto" and time_secs is not None)
        )
        if use_time:
            assert time_secs is not None, "time_secs required for pos_mode='time'"
            time_pe = _time_sinusoidal_embedding(time_secs, self.d_model)  # (B, K, D)
            cls_pe = self.cls_pos_embed.expand(b, -1, -1)  # (B, 1, D)
            pe = torch.cat([cls_pe, time_pe], dim=1)  # (B, K+1, D)
            x = x + pe
        else:
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
    return_time_secs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """한 환자의 K개 윈도우를 인코딩하여 (K, d_model) 반환.

    Parameters
    ----------
    model:
        DownstreamModelWrapper — extract_features 또는 batch_to_device + model
    patient:
        {"signals": {sig_type: (K, win_samples)}, "n_windows": K,
         "start_secs": (K,) optional time meta, ...}
    patch_size:
        Encoder의 patch_size (보통 100 또는 200)
    max_windows:
        K가 너무 크면 균등 샘플링으로 축소
    use_lora:
        True이면 gradient 활성 경로 (LoRA fine-tune 시). False이면 no_grad.
    session_prefix:
        BiosignalSample.session_id prefix (task별 구분용; 기본 "patient")
    return_time_secs:
        True이면 (chunk_reprs, time_secs) 튜플 반환. patient에 start_secs 없으면
        time_secs는 None.
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

    out_reprs = torch.cat(chunk_reprs, dim=0)  # (K', d_model)
    if not return_time_secs:
        return out_reprs

    start_secs_full = patient.get("start_secs")
    if start_secs_full is None:
        return out_reprs, None
    if isinstance(start_secs_full, torch.Tensor):
        start_secs_full = start_secs_full.float()
    else:
        start_secs_full = torch.as_tensor(start_secs_full, dtype=torch.float32)
    sub = start_secs_full[indices]  # (K',)
    return out_reprs, sub


def collate_patients(
    patient_reprs: list[torch.Tensor],  # [(K_i, d_model), ...]
    labels: list[int],
    device: torch.device,
    time_secs: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """가변 K를 패딩하여 배치를 만든다.

    Parameters
    ----------
    time_secs : list of (K_i,) tensor or None per patient. None이면 시간 정보 미제공.

    Returns
    -------
    (padded_reprs (B, K_max, d), mask (B, K_max), labels (B,),
     time_secs (B, K_max) or None)
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

    times_t: torch.Tensor | None = None
    if time_secs is not None and any(t is not None for t in time_secs):
        times_t = torch.zeros(b, k_max, dtype=torch.float32, device=device)
        for i, t in enumerate(time_secs):
            if t is None:
                continue
            k_i = t.shape[0]
            times_t[i, :k_i] = t.to(device)

    return padded, mask, labels_t, times_t
