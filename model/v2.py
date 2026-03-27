# -*- coding:utf-8 -*-
"""Biosignal Foundation Model V2 — EEG는 CNN stem 출력 복원, 나머지는 raw patch 복원.

V2는 V1을 상속하여 EEG (signal_type=2) 전용 reconstruction head를 추가한다.
EEG의 reconstruction target은 CNN stem 출력(d_model 차원)이며,
collapse 방지를 위해 stop-gradient(.detach())를 적용한다.
"""
from __future__ import annotations

from dataclasses import fields

import torch
from torch import nn

from data.collate import PackedBatch
from model.config import ModelConfig
from model.v1 import BiosignalFoundationModelV1

# EEG signal_type 인덱스
EEG_SIGNAL_TYPE = 2


class BiosignalFoundationModelV2(BiosignalFoundationModelV1):
    """생체신호 파운데이션 모델 V2 — EEG stem 출력 복원 + 나머지 raw patch 복원.

    V1과의 차이점:

    - **EEG (signal_type=2)**: reconstruction target이 CNN stem 출력(d_model 차원).
      Stop-gradient 적용으로 encoder가 stem을 통해 trivial solution을 학습하는 것을 방지.
    - **기타 신호**: 기존대로 raw patch reconstruction (patch_size 차원).
    - ``eeg_recon_head``: EEG 전용 reconstruction head (d_model → d_model).

    forward 출력에 추가되는 키:

    - ``eeg_reconstructed``: ``(B, N, d_model)`` — EEG head 출력 (전체 위치, 비-EEG는 0).
    - ``eeg_recon_target``: ``(B, N, d_model)`` — stem 출력 detached (전체 위치, 비-EEG는 0).
    - ``eeg_mask``: ``(B, N)`` — EEG 패치 위치 마스크 (True=EEG).

    Parameters
    ----------
    (V1의 모든 파라미터를 그대로 상속)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        patch_size: int,
        stride: int | None = None,
        num_heads: int | None = None,
        num_groups: int | None = None,
        use_glu: bool = True,
        use_moe: bool = False,
        use_rope: bool = True,
        use_var_attn_bias: bool = True,
        scaler=None,
        dropout_p: float = 0.0,
        num_signal_types: int = 7,
        num_spatial_ids: int = 12,
        use_spatial_embed: bool = True,
        max_horizon: int = 1,
        use_cnn_stem: bool = False,
        stem_hidden_channels: int = 64,
        stem_num_layers: int = 3,
        stem_kernel_size: int = 3,
        contrastive_proj_dim: int = 0,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_layers=num_layers,
            patch_size=patch_size,
            stride=stride,
            num_heads=num_heads,
            num_groups=num_groups,
            use_glu=use_glu,
            use_moe=use_moe,
            use_rope=use_rope,
            use_var_attn_bias=use_var_attn_bias,
            scaler=scaler,
            dropout_p=dropout_p,
            num_signal_types=num_signal_types,
            num_spatial_ids=num_spatial_ids,
            use_spatial_embed=use_spatial_embed,
            max_horizon=max_horizon,
            use_cnn_stem=use_cnn_stem,
            stem_hidden_channels=stem_hidden_channels,
            stem_num_layers=stem_num_layers,
            stem_kernel_size=stem_kernel_size,
            contrastive_proj_dim=contrastive_proj_dim,
        )

        # EEG 전용 Reconstruction Head: d_model → d_model
        # (EEG target이 stem 출력이므로 차원이 d_model)
        self.eeg_recon_head = nn.Linear(d_model, d_model)

    @classmethod
    def from_config(cls, config: ModelConfig) -> BiosignalFoundationModelV2:
        """ModelConfig로부터 모델 인스턴스를 생성한다."""
        kwargs = {f.name: getattr(config, f.name) for f in fields(config)}
        return cls(**kwargs)

    def forward(
        self,
        batch: PackedBatch,
        task: str = "masked",  # "masked" 또는 "next_pred"
        horizon: int = 1,
    ) -> dict[str, torch.Tensor]:
        """전체 forward 파이프라인 (V2).

        V1의 forward 출력에 EEG 전용 reconstruction 정보를 추가한다.

        Parameters
        ----------
        batch:
            PackCollate(patch_size=...)로 생성된 PackedBatch.
        task:
            ``"masked"``: 양방향 attention → reconstruction head.
            ``"next_pred"``: causal attention → next-patch prediction head.
        horizon:
            Next-patch prediction의 예측 거리 (패치 단위). 기본 1.

        Returns
        -------
        dict with keys (V1의 모든 키 + 추가 키):
            ``eeg_reconstructed``: ``(B, N, d_model)`` — EEG head 출력.
            ``eeg_recon_target``: ``(B, N, d_model)`` — stem 출력 (detached).
            ``eeg_mask``: ``(B, N)`` — EEG 패치 위치 (True=EEG).
        """
        enc = self._encode(batch, task=task)

        encoded = enc["encoded"]  # (B, N, d_model) — bidirectional (or sole encoding)
        stem_output = enc["stem_output"]  # (B, N, d_model)
        patch_signal_types = enc["patch_signal_types"]  # (B, N) or None

        out_dict: dict[str, torch.Tensor] = {
            "encoded": encoded,
            "patches": enc["patches"],
            "patch_signal_types": patch_signal_types,
            "loc": enc["loc"],
            "scale": enc["scale"],
            "patch_mask": enc["patch_mask"],
            "patch_sample_id": enc["patch_sample_id"],
            "patch_variate_id": enc["patch_variate_id"],
            "time_id": enc["time_id"],
        }

        if task in ("masked", "both"):
            out_dict["reconstructed"] = self.head(encoded)  # (B, N, patch_size)
            out_dict["cross_pred"] = self.cross_head(encoded)  # (B, N, patch_size)
            if self.contrastive_proj_dim > 0:
                out_dict["contrastive_z"] = self.contrastive_proj(encoded)  # (B, N, proj_dim)

        if task in ("next_pred", "both"):
            # task="both"이면 causal encoding 사용, 아니면 (next_pred) sole encoding 사용
            encoded_for_next = enc.get("encoded_causal", encoded)  # (B, N, d_model)
            h_emb = self.horizon_embed(
                torch.tensor(horizon - 1, device=encoded_for_next.device)
            )  # (d_model,)
            encoded_h = encoded_for_next + h_emb.unsqueeze(0).unsqueeze(0)  # (B, N, d_model)
            out_dict["next_pred"] = self.next_head(encoded_h)  # (B, N, patch_size)

        # ── EEG-specific reconstruction (masked task에서만) ─────────
        # next_pred 시에는 eeg_recon_head 불필요 (gradient 오염 방지)
        # eeg_recon은 bidirectional encoding(encoded)에서만 동작
        if patch_signal_types is not None:
            eeg_mask = patch_signal_types == EEG_SIGNAL_TYPE  # (B, N)
            out_dict["eeg_mask"] = eeg_mask

            if task in ("masked", "both"):
                out_dict["eeg_reconstructed"] = self.eeg_recon_head(encoded)  # (B, N, d_model)
                out_dict["eeg_recon_target"] = stem_output.detach()  # (B, N, d_model)

        return out_dict