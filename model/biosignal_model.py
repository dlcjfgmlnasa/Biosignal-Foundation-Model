# -*- coding:utf-8 -*-
"""Biosignal Foundation Model.

Scaler → PatchEmbedding → SpatialEmbedding → TransformerEncoder → Head 파이프라인.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from functools import partial

import torch
from torch import nn

from data.collate import PackedBatch
from loss.masked_mse_loss import create_patch_mask
from model._config import ModelConfig
from module.packed_scaler import PackedStdScaler, PackedScaler
from module.patch import PatchEmbedding
from module.position import BinaryAttentionBias, QueryKeyProjection, RotaryProjection
from module.transformer import TransformerEncoder


class BiosignalFoundationModel(nn.Module):
    """생체신호 파운데이션 모델 — 모든 신호를 raw patch reconstruction.

    모든 signal type에 대해 동일하게 raw patch 복원을 수행한다.
    ``_encode()``로 공통 인코딩 파이프라인(Scaler → Patchify → Project →
    SpatialEmbed → LocScale → Encoder)을 분리하여 서브클래스에서 확장 가능.

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
        입력 정규화 스케일러. ``None``이면 ``PackedStdScaler``.
    dropout_p:
        드롭아웃 확률.
    num_signal_types:
        신호 타입 수 (ecg=0, abp=1, ppg=2, cvp=3, co2=4, awp=5, pap=6, icp=7).
    num_spatial_ids:
        글로벌 spatial ID 수.
    use_spatial_embed:
        signal_type + spatial_id 이중 임베딩 사용 여부.
    max_horizon:
        Next-patch prediction 최대 예측 거리.
    contrastive_proj_dim:
        Contrastive projection head 출력 차원. 0이면 비활성.
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
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        use_rope: bool = True,
        use_var_attn_bias: bool = True,
        scaler: PackedScaler | None = None,
        dropout_p: float = 0.0,
        num_signal_types: int = 8,  # ECG(0),ABP(1),PPG(2),CVP(3),CO2(4),AWP(5),PAP(6),ICP(7)
        num_spatial_ids: int = 13,  # TOTAL_SPATIAL_IDS (8 types × 가변 spatial IDs)
        use_spatial_embed: bool = True,
        max_horizon: int = 1,
        contrastive_proj_dim: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # 1. Scaler (point-level)
        self.scaler = scaler or PackedStdScaler()

        # 2. Patch Embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            stride=stride,
        )

        # 3. Transformer Encoder
        num_heads = num_heads or d_model // 64

        var_attn_bias_layer: Callable | None = None
        if use_var_attn_bias:
            var_attn_bias_layer = partial(BinaryAttentionBias)

        time_qk_proj_layer: Callable | None = None
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
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            var_attn_bias_layer=var_attn_bias_layer,
            time_qk_proj_layer=time_qk_proj_layer,
            dropout_p=dropout_p,
        )

        # 4. Spatial Positional Encoding (Dual Embedding)
        self.use_spatial_embed = use_spatial_embed
        if use_spatial_embed:
            self.signal_type_embed = nn.Embedding(num_signal_types, d_model)
            self.spatial_id_embed = nn.Embedding(num_spatial_ids, d_model)

        # 5. Loc/Scale Injection (환자별 절대 레벨 정보 보존)
        self.loc_proj = nn.Linear(1, d_model)
        self.scale_proj = nn.Linear(1, d_model)

        # 6. Reconstruction Head (자기 variate 복원)
        self.head = nn.Linear(d_model, patch_size)

        # 7. Next-Patch Prediction Head
        self.next_head = nn.Linear(d_model, patch_size)

        # 8. Cross-Modal Prediction Head (다른 variate 예측용)
        self.cross_head = nn.Linear(d_model, patch_size)

        # 9. Horizon Embedding (random horizon next-patch prediction)
        self.max_horizon = max_horizon
        self.horizon_embed = nn.Embedding(max_horizon, d_model)

        # 10. Contrastive Projection Head (SimCLR-style 2-layer MLP)
        self.contrastive_proj_dim = contrastive_proj_dim
        if contrastive_proj_dim > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, contrastive_proj_dim),
            )

        # 11. Learnable [MASK] Token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    @classmethod
    def from_config(cls, config: ModelConfig) -> BiosignalFoundationModel:
        """ModelConfig로부터 모델 인스턴스를 생성한다."""
        import inspect

        valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        kwargs = {
            f.name: getattr(config, f.name)
            for f in fields(config)
            if f.name in valid_params
        }
        return cls(**kwargs)

    # ── Encode Pipeline ────────────────────────────────────────────

    def _encode(
        self,
        batch: PackedBatch,
        task: str = "masked",
        mask_ratio: float = 0.0,
        block_mask: bool = False,
        block_size_min: int = 3,
        block_size_max: int = 8,
        variate_mask_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """공통 인코딩 파이프라인: Scaler → Patchify → Project → SpatialEmbed → LocScale → Encoder.

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.
        task:
            ``"masked"``: 양방향 attention.
            ``"next_pred"``: causal attention.
            ``"both"``: 양방향 + causal 동시 (encoder 2회 호출, DDP single forward 호환).

        Returns
        -------
        dict with keys:
            ``encoded``: ``(B, N, d_model)`` — 양방향 인코딩된 패치 표현 (task="both"/"masked").
            ``encoded_causal``: ``(B, N, d_model)`` — causal 인코딩 (task="both"일 때만).
            ``patches``: ``(B, N, patch_size)`` — raw patches.
            ``patch_signal_types``: ``(B, N)`` — 패치별 signal type.
            ``patch_spatial_ids``: ``(B, N)`` — 패치별 spatial ID.
            ``loc``: ``(B, L, 1)`` — per-variate 위치.
            ``scale``: ``(B, L, 1)`` — per-variate 스케일.
            ``patch_mask``: ``(B, N)`` — 유효 패치 마스크.
            ``patch_sample_id``: ``(B, N)`` — 패치별 sample_id.
            ``patch_variate_id``: ``(B, N)`` — 패치별 variate_id.
            ``time_id``: ``(B, N)`` — 패치별 시간 인덱스.
        """
        # 1. Scaler: point-level 정규화
        values = batch.values.unsqueeze(-1)  # (B, L, 1)
        loc, scale = self.scaler(
            values,
            sample_id=batch.sample_id,
            variate_id=batch.variate_id,
        )
        normalized = ((values - loc) / scale.clamp(min=1e-8)).squeeze(-1)  # (B, L)

        # 2. Patchify (projection 전 raw patches 추출)
        patches, p_sid, p_vid, time_id, patch_mask = self.patch_embed.patchify(
            normalized, batch.sample_id, batch.variate_id
        )
        # patches: (B, N, patch_size)

        b = patches.shape[0]
        device = patches.device

        # 3. global_var_idx 계산 — CNN stem과 dual embedding 모두에서 재사용
        patch_signal_types: torch.Tensor | None = None
        patch_spatial_ids: torch.Tensor | None = None

        if hasattr(batch, "spatial_ids") and batch.spatial_ids is not None:
            per_row_max_var = p_vid.max(dim=-1).values  # (B,)
            var_offsets = torch.zeros(b, dtype=torch.long, device=device)
            if b > 1:
                var_offsets[1:] = per_row_max_var[:-1].cumsum(dim=0)
            global_var_idx = var_offsets.unsqueeze(-1) + (p_vid - 1)  # (B, N)
            global_var_idx = global_var_idx.clamp(min=0)

            patch_signal_types = batch.signal_types.to(device)[global_var_idx]  # (B, N)
            patch_spatial_ids = batch.spatial_ids.to(device)[global_var_idx]  # (B, N)

            # 절대 시간 기반 abs_time_id 계산 (cross-modal 매칭 전용)
            # time_id(상대적)는 RoPE용으로 유지, abs_time_id는 cross-modal loss용
            abs_time_id = time_id  # fallback: start_samples 없으면 상대적 time_id 사용
            if (
                hasattr(batch, "start_samples")
                and batch.start_samples is not None
            ):
                patch_start = batch.start_samples.to(device)[global_var_idx]  # (B, N)
                abs_time = patch_start + time_id * self.patch_size  # (B, N)
                abs_time_id = abs_time // self.patch_size  # (B, N)
                abs_time_id[~patch_mask] = 0

        # 4. Projection (linear 또는 CNN stem)
        embedded = self.patch_embed.project(patches, patch_signal_types)
        # embedded: (B, N, d_model)

        # 패딩 마스크 (p_vid==0은 패딩 토큰)
        valid_token = (p_vid > 0).unsqueeze(-1)  # (B, N, 1)

        # 5. Spatial Positional Encoding (Dual Embedding)
        if self.use_spatial_embed and patch_signal_types is not None:
            sig_emb = self.signal_type_embed(patch_signal_types)  # (B, N, d_model)
            spa_emb = self.spatial_id_embed(patch_spatial_ids)  # (B, N, d_model)
            embedded = embedded + (sig_emb + spa_emb) * valid_token

        # 6. Loc/Scale Dual Injection — 절대 레벨 정보 보존
        n = embedded.shape[1]
        stride = self.patch_embed.stride
        patch_starts = torch.arange(n, device=device) * stride  # (N,)
        patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
        patch_loc = loc[:, patch_starts, :]  # (B, N, 1)
        patch_scale = scale[:, patch_starts, :]  # (B, N, 1)
        loc_emb = self.loc_proj(patch_loc)  # (B, N, d_model)
        scale_emb = self.scale_proj(patch_scale)  # (B, N, d_model)
        embedded = embedded + (loc_emb + scale_emb) * valid_token

        # 7. [MASK] Token — 마스킹된 패치의 content를 learnable token으로 교체
        pred_mask: torch.Tensor | None = None
        if mask_ratio > 0 and task in ("masked", "both"):
            pred_mask = create_patch_mask(
                patch_mask,
                mask_ratio=mask_ratio,
                patch_variate_id=p_vid if variate_mask_prob > 0 else None,
                variate_mask_prob=variate_mask_prob,
                block_mask=block_mask,
                block_size_min=block_size_min,
                block_size_max=block_size_max,
            )

        # 8. Base Attention Mask: 같은 sample 내에서만 attend + 유효 패치만
        base_attn_mask = (
            (p_sid.unsqueeze(-1) == p_sid.unsqueeze(-2))
            & patch_mask.unsqueeze(-2)
            & patch_mask.unsqueeze(-1)
        )  # (B, N, n)

        # 9. Task에 따른 Masking 스위치 + Transformer Encoder
        result: dict[str, torch.Tensor] = {
            "patches": patches,
            "patch_signal_types": patch_signal_types,
            "patch_spatial_ids": patch_spatial_ids,
            "loc": loc,
            "scale": scale,
            "patch_mask": patch_mask,
            "patch_sample_id": p_sid,
            "patch_variate_id": p_vid,
            "time_id": time_id,          # 상대적 (RoPE용)
            "abs_time_id": abs_time_id,  # 절대적 (cross-modal 매칭용)
            "pred_mask": pred_mask,
        }

        encoder_kwargs = dict(var_id=p_vid, time_id=time_id)  # RoPE는 상대적 time_id
        use_causal = task in ("next_pred", "both")

        # causal mask (next_pred, both에서 공유)
        if use_causal:
            causal_tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
            causal_mask = base_attn_mask & causal_tri.unsqueeze(0)  # (B, N, N)

        if task == "both":
            # bidirectional: [MASK] 토큰 적용 → 마스킹된 패치 정보 차단
            embedded_bi = embedded.clone()
            if pred_mask is not None:
                mask_expanded = pred_mask.unsqueeze(-1)  # (B, N, 1)
                mask_token_expanded = self.mask_token.expand_as(embedded_bi)
                embedded_bi = torch.where(
                    mask_expanded, mask_token_expanded, embedded_bi
                )
            result["encoded"] = self.encoder(
                embedded_bi,
                attn_mask=base_attn_mask,
                **encoder_kwargs,
            )
            # causal: [MASK] 불필요 — causal attention이 미래 정보 차단
            result["encoded_causal"] = self.encoder(
                embedded.clone(),
                attn_mask=causal_mask,
                **encoder_kwargs,
            )
        elif task == "next_pred":
            result["encoded"] = self.encoder(
                embedded,
                attn_mask=causal_mask,
                **encoder_kwargs,
            )
        else:  # "masked"
            if pred_mask is not None:
                mask_expanded = pred_mask.unsqueeze(-1)
                mask_token_expanded = self.mask_token.expand_as(embedded)
                embedded = torch.where(mask_expanded, mask_token_expanded, embedded)
            result["encoded"] = self.encoder(
                embedded,
                attn_mask=base_attn_mask,
                **encoder_kwargs,
            )

        return result

    # ── Forward ────────────────────────────────────────────────────

    def forward(
        self,
        batch: PackedBatch,
        task: str = "masked",  # "masked" 또는 "next_pred"
        horizon: int = 1,
        mask_ratio: float = 0.0,
        block_mask: bool = False,
        block_size_min: int = 3,
        block_size_max: int = 8,
        variate_mask_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        enc = self._encode(
            batch,
            task=task,
            mask_ratio=mask_ratio,
            block_mask=block_mask,
            block_size_min=block_size_min,
            block_size_max=block_size_max,
            variate_mask_prob=variate_mask_prob,
        )

        encoded = enc["encoded"]  # bidirectional (or sole encoding for single-task)
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
            "time_id": enc["abs_time_id"],  # cross-modal 매칭용 (절대 시간)
            "pred_mask": enc["pred_mask"],
        }

        # ── Masked Reconstruction ──
        if task in ("masked", "both"):
            out_dict["reconstructed"] = self.head(encoded)  # (B, N, patch_size)
            out_dict["cross_pred"] = self.cross_head(encoded)  # (B, N, patch_size)
            if self.contrastive_proj_dim > 0:
                out_dict["contrastive_z"] = self.contrastive_proj(
                    encoded
                )  # (B, N, proj_dim)

        # ── Next-Patch Prediction ──
        if task in ("next_pred", "both"):
            encoded_for_next = enc.get("encoded_causal", encoded)  # (B, N, d_model)
            h_emb = self.horizon_embed(
                torch.tensor(horizon - 1, device=encoded_for_next.device)
            )  # (d_model,)
            encoded_h = encoded_for_next + h_emb.unsqueeze(0).unsqueeze(
                0
            )  # (B, N, d_model)
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
        out.pop("cross_pred", None)
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
            p = self.patch_size
            patch_loc = loc[:, ::p, :]  # (B, N_approx, 1)
            patch_scale = scale[:, ::p, :]  # (B, N_approx, 1)
            n = pred.shape[1]
            patch_loc = patch_loc[:, :n, :]  # (B, N, 1)
            patch_scale = patch_scale[:, :n, :]  # (B, N, 1)
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
            ``(n_steps, B, patch_size)`` generated patches.
        """
        self.eval()
        p = self.patch_size

        out = self.forward(batch, task="next_pred", horizon=1)
        loc = out["loc"]  # (B, L, 1)
        scale = out["scale"]  # (B, L, 1)
        cached_loc = loc[:, 0:1, :]  # (B, 1, 1)
        cached_scale = scale[:, 0:1, :]  # (B, 1, 1)

        generated = []
        pred = out["next_pred"]  # (B, N, patch_size)
        patch_mask = out["patch_mask"]  # (B, N)
        last_valid_idx = patch_mask.sum(dim=-1) - 1  # (B,)
        last_valid_idx = last_valid_idx.clamp(min=0)
        new_patch = pred[
            torch.arange(pred.shape[0], device=pred.device), last_valid_idx
        ]  # (B, patch_size)
        generated.append(new_patch)

        for _ in range(n_steps - 1):
            batch = _append_patch_to_batch(batch, new_patch, p)
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
    b, l = batch.values.shape
    device = batch.values.device

    valid_mask = batch.sample_id > 0  # (B, L)
    valid_lengths = valid_mask.sum(dim=-1)  # (B,)

    new_end = valid_lengths + patch_size  # (B,)
    max_new_end = new_end.max().item()

    if max_new_end > l:
        pad_size = max_new_end - l
        batch = PackedBatch(
            values=torch.cat(
                [batch.values, torch.zeros(b, pad_size, device=device)], dim=-1
            ),
            sample_id=torch.cat(
                [
                    batch.sample_id,
                    torch.zeros(b, pad_size, dtype=torch.long, device=device),
                ],
                dim=-1,
            ),
            variate_id=torch.cat(
                [
                    batch.variate_id,
                    torch.zeros(b, pad_size, dtype=torch.long, device=device),
                ],
                dim=-1,
            ),
            lengths=batch.lengths,
            sampling_rates=batch.sampling_rates,
            signal_types=batch.signal_types,
            spatial_ids=batch.spatial_ids,
            padded_lengths=batch.padded_lengths,
        )

    for i in range(b):
        start = valid_lengths[i].item()
        end = start + patch_size
        batch.values[i, start:end] = new_patch[i]
        batch.sample_id[i, start:end] = (
            batch.sample_id[i, start - 1] if start > 0 else 1
        )
        batch.variate_id[i, start:end] = (
            batch.variate_id[i, start - 1] if start > 0 else 1
        )

    return batch
