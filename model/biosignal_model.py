# -*- coding:utf-8 -*-
"""Biosignal Foundation Model — 전체 모델 조립.

Scaler → PatchEmbedding → TransformerEncoder → Head 파이프라인을 하나의
nn.Module로 통합한다.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from functools import partial
from typing import Optional

import torch
from torch import nn

from data.collate import PackedBatch
from model.config import ModelConfig
from module.cnn_stem import ModalityCNNStem
from module.packed_scaler import PackedAbsMeanScaler, PackedScaler
from module.patch import PatchEmbedding
from module.position import BinaryAttentionBias, QueryKeyProjection, RotaryProjection
from module.transformer import TransformerEncoder


class BiosignalFoundationModel(nn.Module):
    """생체신호 파운데이션 모델.

    Parameters
    ----------
    d_model:
        트랜스포머 임베딩 차원.
    num_layers:
        트랜스포머 인코더 레이어 수.
    patch_size:
        패치 크기 (time-step 수).
    stride:
        패치 보폭 (overlapping 시). ``None``이면 ``patch_size``와 동일.
    num_heads:
        어텐션 헤드 수. ``None``이면 ``d_model // 64``.
    num_groups:
        GQA 그룹 수. ``None``이면 ``num_heads`` (MHA).
    use_glu:
        Gated Linear Unit FFN 사용 여부.
    use_moe:
        Mixture of Experts 사용 여부.
    use_rope:
        Rotary Position Embedding 사용 여부.
    use_var_attn_bias:
        BinaryAttentionBias (variate 간 bias) 사용 여부.
    scaler:
        입력 정규화 스케일러. ``None``이면 ``PackedAbsMeanScaler``.
    dropout_p:
        드롭아웃 확률.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        patch_size: int,
        stride: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_groups: Optional[int] = None,
        use_glu: bool = True,
        use_moe: bool = False,
        use_rope: bool = True,
        use_var_attn_bias: bool = True,
        scaler: Optional[PackedScaler] = None,
        dropout_p: float = 0.0,
        num_signal_types: int = 6,
        num_spatial_ids: int = 55,
        use_spatial_embed: bool = True,
        max_horizon: int = 1,
        use_cnn_stem: bool = False,
        stem_hidden_channels: int = 64,
        stem_num_layers: int = 3,
        stem_kernel_size: int = 3,
        contrastive_proj_dim: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # 1. Scaler (point-level)
        self.scaler = scaler or PackedAbsMeanScaler()

        # 2. Patch Embedding
        self.use_cnn_stem = use_cnn_stem
        if use_cnn_stem:
            stem = ModalityCNNStem(
                num_signal_types=num_signal_types,
                d_model=d_model,
                hidden_channels=stem_hidden_channels,
                num_layers=stem_num_layers,
                kernel_size=stem_kernel_size,
            )
        else:
            stem = None
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, d_model=d_model, stride=stride, stem=stem,
        )

        # 3. Transformer Encoder
        num_heads = num_heads or d_model // 64

        var_attn_bias_layer: Optional[Callable] = None
        if use_var_attn_bias:
            var_attn_bias_layer = partial(BinaryAttentionBias)

        time_qk_proj_layer: Optional[Callable] = None
        if use_rope:
            time_qk_proj_layer = partial(
                QueryKeyProjection,
                proj_layer=partial(RotaryProjection),
            )

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            num_groups=num_groups,
            use_glu=use_glu,
            use_moe=use_moe,
            var_attn_bias_layer=var_attn_bias_layer,
            time_qk_proj_layer=time_qk_proj_layer,
            dropout_p=dropout_p,
        )

        # 4. Spatial Positional Encoding (Dual Embedding)
        self.use_spatial_embed = use_spatial_embed
        if use_spatial_embed:
            self.signal_type_embed = nn.Embedding(num_signal_types, d_model)
            self.spatial_id_embed = nn.Embedding(num_spatial_ids, d_model)

        # 5. Reconstruction Head
        self.head = nn.Linear(d_model, patch_size)

        # 6. Next-Patch Prediction Head
        self.next_head = nn.Linear(d_model, patch_size)

        # 7. Cross-Modal Prediction Head (다른 variate 예측용)
        self.cross_head = nn.Linear(d_model, patch_size)

        # 8. Horizon Embedding (random horizon next-patch prediction)
        self.max_horizon = max_horizon
        self.horizon_embed = nn.Embedding(max_horizon, d_model)

        # 9. Contrastive Projection Head (SimCLR-style 2-layer MLP)
        self.contrastive_proj_dim = contrastive_proj_dim
        if contrastive_proj_dim > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, contrastive_proj_dim),
            )

    @classmethod
    def from_config(cls, config: ModelConfig) -> BiosignalFoundationModel:
        """ModelConfig로부터 모델 인스턴스를 생성한다."""
        kwargs = {f.name: getattr(config, f.name) for f in fields(config)}
        return cls(**kwargs)

    def forward(
        self,
        batch: PackedBatch,
        task: str = "masked",  # "masked" 또는 "next_pred"
        horizon: int = 1,
    ) -> dict[str, torch.Tensor]:
        """전체 forward 파이프라인.

        Parameters
        ----------
        batch:
            PackCollate(patch_size=...)로 생성된 PackedBatch.
        task:
            ``"masked"``: 양방향 attention → reconstruction head.
            ``"next_pred"``: causal attention → next-patch prediction head.
        horizon:
            Next-patch prediction의 예측 거리 (패치 단위). 기본 1.
            ``horizon=H``이면 i번째 위치에서 i+H번째 패치를 예측한다.

        Returns
        -------
        dict with keys:
            ``encoded``: ``(B, N, d_model)`` — 인코딩된 패치 표현.
            ``reconstructed``: ``(B, N, patch_size)`` — task="masked"일 때만 존재.
            ``cross_pred``: ``(B, N, patch_size)`` — task="masked"일 때만 존재 (cross-modal 예측).
            ``next_pred``: ``(B, N, patch_size)`` — task="next_pred"일 때만 존재.
            ``loc``: ``(B, L, 1)`` — per-variate 위치 (정규화 역변환용).
            ``scale``: ``(B, L, 1)`` — per-variate 스케일.
            ``patch_mask``: ``(B, N)`` — 유효 패치 마스크.
            ``patch_sample_id``: ``(B, N)`` — 패치별 sample_id.
            ``patch_variate_id``: ``(B, N)`` — 패치별 variate_id.
            ``time_id``: ``(B, N)`` — 패치별 시간 인덱스 (cross-modal 페어링용).
        """
        # 1. Scaler: point-level 정규화
        values = batch.values.unsqueeze(-1)  # (B, L, 1)
        loc, scale = self.scaler(
            values,
            sample_id=batch.sample_id,
            variate_id=batch.variate_id,
        )
        normalized = ((values - loc) / scale).squeeze(-1)  # (B, L)

        # 2. Patchify (projection 전 raw patches 추출)
        patches, p_sid, p_vid, time_id, patch_mask = self.patch_embed.patchify(
            normalized, batch.sample_id, batch.variate_id
        )
        # patches: (B, N, patch_size)

        B = patches.shape[0]
        device = patches.device

        # 3. global_var_idx 계산 — CNN stem과 dual embedding 모두에서 재사용
        patch_signal_types: torch.Tensor | None = None
        patch_spatial_ids: torch.Tensor | None = None

        if hasattr(batch, "spatial_ids") and batch.spatial_ids is not None:
            per_row_max_var = p_vid.max(dim=-1).values  # (B,)
            var_offsets = torch.zeros(B, dtype=torch.long, device=device)
            if B > 1:
                var_offsets[1:] = per_row_max_var[:-1].cumsum(dim=0)
            global_var_idx = var_offsets.unsqueeze(-1) + (p_vid - 1)  # (B, N)
            global_var_idx = global_var_idx.clamp(min=0)

            patch_signal_types = batch.signal_types.to(device)[global_var_idx]  # (B, N)
            patch_spatial_ids = batch.spatial_ids.to(device)[global_var_idx]  # (B, N)

        # 4. Projection (linear 또는 CNN stem)
        embedded = self.patch_embed.project(patches, patch_signal_types)
        # embedded: (B, N, d_model)

        # 5. Spatial Positional Encoding (Dual Embedding) — 패딩 보호
        if self.use_spatial_embed and patch_signal_types is not None:
            sig_emb = self.signal_type_embed(patch_signal_types)  # (B, N, d_model)
            spa_emb = self.spatial_id_embed(patch_spatial_ids)  # (B, N, d_model)

            # 패딩 토큰(p_vid==0)에는 임베딩을 더하지 않는다
            valid_mask = (p_vid > 0).unsqueeze(-1)  # (B, N, 1)
            embedded = embedded + (sig_emb + spa_emb) * valid_mask

        # 4. Base Attention Mask: 같은 sample 내에서만 attend + 유효 패치만
        base_attn_mask = (
            (p_sid.unsqueeze(-1) == p_sid.unsqueeze(-2))
            & patch_mask.unsqueeze(-2)
            & patch_mask.unsqueeze(-1)
        )  # (B, N, N)

        # 5. Task에 따른 Masking 스위치
        if task == "next_pred":
            N = base_attn_mask.shape[-1]
            causal_tri = torch.tril(
                torch.ones(N, N, dtype=torch.bool, device=embedded.device)
            )
            final_attn_mask = base_attn_mask & causal_tri.unsqueeze(0)  # (B, N, N)
        else:
            final_attn_mask = base_attn_mask

        # 6. Transformer Encoder (task에 맞는 mask 적용)
        encoded = self.encoder(
            embedded,
            attn_mask=final_attn_mask,
            var_id=p_vid,
            time_id=time_id,
        )
        # encoded: (B, N, d_model)

        # 7. Task Head 라우팅
        out_dict: dict[str, torch.Tensor] = {
            "encoded": encoded,
            "loc": loc,
            "scale": scale,
            "patch_mask": patch_mask,
            "patch_sample_id": p_sid,
            "patch_variate_id": p_vid,
            "time_id": time_id,
        }

        if task == "masked":
            out_dict["reconstructed"] = self.head(encoded)  # (B, N, patch_size)
            out_dict["cross_pred"] = self.cross_head(encoded)  # (B, N, patch_size)
            if self.contrastive_proj_dim > 0:
                out_dict["contrastive_z"] = self.contrastive_proj(encoded)  # (B, N, proj_dim)
        elif task == "next_pred":
            # Horizon conditioning: encoded에 horizon embedding 더함
            h_emb = self.horizon_embed(
                torch.tensor(horizon - 1, device=encoded.device)
            )  # (d_model,)
            encoded_h = encoded + h_emb.unsqueeze(0).unsqueeze(0)  # (B, N, d_model)
            out_dict["next_pred"] = self.next_head(encoded_h)  # (B, N, patch_size)

        return out_dict

    # ── Inference API ──────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(self, batch: PackedBatch) -> dict[str, torch.Tensor]:
        """Downstream task용 feature 추출 (양방향 attention).

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.

        Returns
        -------
        dict with keys:
            ``encoded``, ``patch_mask``, ``loc``, ``scale``,
            ``patch_sample_id``, ``patch_variate_id``.
        """
        self.eval()
        out = self.forward(batch, task="masked")
        out.pop("reconstructed", None)
        return out

    @torch.no_grad()
    def forecast(
        self,
        batch: PackedBatch,
        horizon: int = 1,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """단일-step 예측 (next-patch prediction).

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.
        horizon:
            예측 거리 (패치 단위).
        denormalize:
            ``True``이면 scaler의 loc/scale로 원본 스케일 복원.

        Returns
        -------
        torch.Tensor
            ``(B, N, patch_size)`` 전체 prediction map.
        """
        self.eval()
        out = self.forward(batch, task="next_pred", horizon=horizon)
        pred = out["next_pred"]  # (B, N, patch_size)

        if denormalize:
            loc = out["loc"]  # (B, L, 1)
            scale = out["scale"]  # (B, L, 1)
            P = self.patch_size
            # per-patch loc/scale: scaler는 variate 내 모든 point에 동일 값 부여
            # → 패치 경계에서 추출하여 (B, N, 1)로 변환
            patch_loc = loc[:, ::P, :]  # (B, N_approx, 1)
            patch_scale = scale[:, ::P, :]  # (B, N_approx, 1)
            N = pred.shape[1]
            patch_loc = patch_loc[:, :N, :]  # (B, N, 1)
            patch_scale = patch_scale[:, :N, :]  # (B, N, 1)
            pred = pred * patch_scale + patch_loc

        return pred

    @torch.no_grad()
    def generate(
        self,
        batch: PackedBatch,
        n_steps: int,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Autoregressive 다단계 생성.

        항상 ``horizon=1``로 1-step 예측 → 예측값을 입력에 append → 반복.
        ``collate_mode="ci"`` (single-variate-per-row) 전제.

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.
        n_steps:
            생성할 패치 수.
        denormalize:
            ``True``이면 최종 출력을 원본 스케일로 복원.

        Returns
        -------
        torch.Tensor
            ``(n_steps, B, patch_size)`` generated patches (normalized or denormalized).
        """
        self.eval()
        P = self.patch_size

        # 첫 forward로 loc/scale 캐시
        out = self.forward(batch, task="next_pred", horizon=1)
        loc = out["loc"]  # (B, L, 1)
        scale = out["scale"]  # (B, L, 1)
        # per-variate 단일 값 추출 (variate 내 동일)
        cached_loc = loc[:, 0:1, :]  # (B, 1, 1)
        cached_scale = scale[:, 0:1, :]  # (B, 1, 1)

        # 첫 step 예측 추출 — 각 row의 마지막 유효 위치
        generated = []
        pred = out["next_pred"]  # (B, N, patch_size)
        patch_mask = out["patch_mask"]  # (B, N)
        last_valid_idx = patch_mask.sum(dim=-1) - 1  # (B,)
        last_valid_idx = last_valid_idx.clamp(min=0)
        new_patch = pred[
            torch.arange(pred.shape[0], device=pred.device), last_valid_idx
        ]  # (B, patch_size)
        generated.append(new_patch)

        # 나머지 step
        for _ in range(n_steps - 1):
            batch = _append_patch_to_batch(batch, new_patch, P)
            out = self.forward(batch, task="next_pred", horizon=1)
            pred = out["next_pred"]
            patch_mask = out["patch_mask"]
            last_valid_idx = patch_mask.sum(dim=-1) - 1
            last_valid_idx = last_valid_idx.clamp(min=0)
            new_patch = pred[
                torch.arange(pred.shape[0], device=pred.device), last_valid_idx
            ]
            generated.append(new_patch)

        result = torch.stack(generated, dim=0)  # (n_steps, B, patch_size)

        if denormalize:
            # cached_loc/scale: (B, 1, 1) → (1, B, 1) for broadcast
            dl = cached_loc.squeeze(-1).permute(1, 0)  # (1, B)
            ds = cached_scale.squeeze(-1).permute(1, 0)  # (1, B)
            result = result * ds.unsqueeze(-1) + dl.unsqueeze(-1)

        return result


def _append_patch_to_batch(
    batch: PackedBatch,
    new_patch: torch.Tensor,  # (B, patch_size)
    patch_size: int,
) -> PackedBatch:
    """PackedBatch에 새 패치를 append한다.

    Single-variate-per-row 가정. max_length 초과 시 우측 패딩 확장.

    Parameters
    ----------
    batch:
        기존 PackedBatch.
    new_patch:
        추가할 패치. ``(B, patch_size)``.
    patch_size:
        패치 크기.

    Returns
    -------
    PackedBatch
        새 패치가 append된 PackedBatch.
    """
    B, L = batch.values.shape
    device = batch.values.device

    # 각 row의 유효 길이 (sample_id > 0인 마지막 위치 + 1)
    valid_mask = batch.sample_id > 0  # (B, L)
    # 유효한 위치가 있는 경우의 끝 위치 계산
    valid_lengths = valid_mask.sum(dim=-1)  # (B,)

    new_end = valid_lengths + patch_size  # (B,)
    max_new_end = new_end.max().item()

    if max_new_end > L:
        # 우측 패딩 확장
        pad_size = max_new_end - L
        batch = PackedBatch(
            values=torch.cat(
                [batch.values, torch.zeros(B, pad_size, device=device)], dim=-1
            ),
            sample_id=torch.cat(
                [batch.sample_id, torch.zeros(B, pad_size, dtype=torch.long, device=device)],
                dim=-1,
            ),
            variate_id=torch.cat(
                [batch.variate_id, torch.zeros(B, pad_size, dtype=torch.long, device=device)],
                dim=-1,
            ),
            lengths=batch.lengths,
            sampling_rates=batch.sampling_rates,
            signal_types=batch.signal_types,
            spatial_ids=batch.spatial_ids,
            padded_lengths=batch.padded_lengths,
        )

    # 새 패치를 각 row에 append
    for i in range(B):
        start = valid_lengths[i].item()
        end = start + patch_size
        batch.values[i, start:end] = new_patch[i]
        batch.sample_id[i, start:end] = batch.sample_id[i, start - 1] if start > 0 else 1
        batch.variate_id[i, start:end] = batch.variate_id[i, start - 1] if start > 0 else 1

    return batch
