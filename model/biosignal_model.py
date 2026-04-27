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


class BlockNextHead(nn.Module):
    """Shared trunk + K horizon-specific heads for Block Next Prediction.

    각 position의 encoded vector를 공유 non-linear trunk로 변환한 뒤,
    K개 독립 Linear head가 horizon별 미래 patch를 예측한다.

    입력:  ``(B, N, d_model)``
    출력:  ``(B, N, K, patch_size)`` — k번째 head가 t+k 패치 예측

    Parameters
    ----------
    d_model:
        입력 차원.
    patch_size:
        출력 patch 크기 (샘플 수).
    block_size:
        K — 예측할 future patch 수.
    d_inner:
        trunk 내부 차원. ``None``이면 ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        block_size: int,
        d_inner: int | None = None,
    ) -> None:
        super().__init__()
        d_inner = d_inner if d_inner is not None else d_model
        self.block_size = block_size
        self.patch_size = patch_size

        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(d_inner, patch_size) for _ in range(block_size)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        h = self.trunk(x)  # (B, N, d_inner)
        outs = [head(h) for head in self.heads]  # list of (B, N, patch_size)
        return torch.stack(outs, dim=2)  # (B, N, K, patch_size)


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
    next_block_size:
        Block Next Prediction에서 각 position이 병렬 예측하는 future patch 수 (K).
        각 position n에서 encoded_causal[n]으로부터 n+1, n+2, ..., n+K 시점의 raw patch를
        non-autoregressive하게 동시 예측한다.
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
        next_block_size: int = 4,
        next_head_d_inner: int | None = None,
        contrastive_proj_dim: int = 0,
        use_adaln: bool = False,
        d_cond: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_signal_types = num_signal_types
        self.use_adaln = use_adaln
        # d_cond: AdaLN modulation 입력 차원. 0이면 d_model 사용.
        self.d_cond = d_cond if d_cond > 0 else d_model

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
            use_adaln=use_adaln,
            d_cond=self.d_cond if use_adaln else 0,
        )

        # 4. Spatial Positional Encoding (Dual Embedding)
        self.use_spatial_embed = use_spatial_embed
        if use_spatial_embed:
            self.signal_type_embed = nn.Embedding(num_signal_types, d_model)
            self.spatial_id_embed = nn.Embedding(num_spatial_ids, d_model)

        # 5. Loc/Scale Injection (환자별 절대 레벨 정보 보존)
        # use_adaln=True일 때는 additive embedding 대신 cond_proj가 AdaLN modulation에 사용됨.
        if use_adaln:
            # (loc, scale) 2D scalar → d_cond conditioning vector
            # MLP(2 → d_cond → d_cond) — non-linearity로 expressiveness 확보
            self.cond_proj = nn.Sequential(
                nn.Linear(2, self.d_cond),
                nn.SiLU(),
                nn.Linear(self.d_cond, self.d_cond),
            )
            self.loc_proj = None
            self.scale_proj = None
        else:
            self.cond_proj = None
            self.loc_proj = nn.Linear(1, d_model)
            self.scale_proj = nn.Linear(1, d_model)

        # 6. Reconstruction Head (자기 variate 복원)
        self.head = nn.Linear(d_model, patch_size)

        # 7. Block Next-Patch Prediction Head (공유 trunk + K개 horizon-specific head)
        # - trunk: 모든 horizon 공통 non-linear 변환 (Linear+GELU)
        # - heads: K개 독립 Linear projection (각 horizon 전용)
        # 한 Linear(d, K*P)보다 non-linearity + horizon 분업으로 장거리 예측 품질 향상.
        self.next_block_size = next_block_size
        self.next_head = BlockNextHead(
            d_model=d_model,
            patch_size=patch_size,
            block_size=next_block_size,
            d_inner=next_head_d_inner,
        )

        # 8. Cross-Modal Prediction Heads (target signal type별 독립 Linear)
        self.cross_heads = nn.ModuleDict({
            str(st): nn.Linear(d_model, patch_size)
            for st in range(num_signal_types)
        })

        # 9. Contrastive Projection Head (SimCLR-style 2-layer MLP)
        self.contrastive_proj_dim = contrastive_proj_dim
        if contrastive_proj_dim > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, contrastive_proj_dim),
            )

        # 10. Learnable [MASK] Token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    @staticmethod
    def _sample_variate_drop(
        p_sid: torch.Tensor,  # (B, N)
        p_vid: torch.Tensor,  # (B, N)
        patch_mask: torch.Tensor,  # (B, N)
        drop_prob: float,
    ) -> torch.Tensor | None:
        """Complete Variate Dropout: 행별로 하나의 variate를 attention에서 완전히 제거.

        Returns (B, N) bool mask — True = dropped from attention.
        다변량(2+ variates)인 행에서만 작동. None if no drop.
        """
        b, n = p_vid.shape
        drop_mask = torch.zeros(b, n, dtype=torch.bool, device=p_vid.device)
        any_dropped = False
        for bi in range(b):
            if torch.rand(1).item() >= drop_prob:
                continue
            valid = patch_mask[bi]
            valid_vids = p_vid[bi][valid]
            unique_vids = valid_vids[valid_vids > 0].unique()
            if len(unique_vids) < 2:
                continue  # 단일 variate → dropout 불가
            # 랜덤으로 하나 선택
            chosen = unique_vids[torch.randint(len(unique_vids), (1,))]
            drop_mask[bi] = (p_vid[bi] == chosen) & valid
            any_dropped = True
        return drop_mask if any_dropped else None

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
        variate_drop_prob: float = 0.0,
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
            #
            # 같은 sample_id 내에서 절대 시간의 최소값을 빼서
            # 버킷 내 상대 offset으로 변환 → patch_size 단위 양자화.
            # → 같은 물리적 시간대의 다른 variate 패치가 동일 abs_time_id를 가짐.
            abs_time_id = time_id  # fallback
            if (
                hasattr(batch, "start_samples")
                and batch.start_samples is not None
            ):
                patch_start = batch.start_samples.to(device)[global_var_idx]  # (B, N)
                abs_time = patch_start + time_id * self.patch_size  # (B, N)
                # patch_size 단위로 양자화 — 같은 물리적 시간의 패치가 정확히 매칭
                abs_time_id = abs_time // self.patch_size  # (B, N)
                abs_time_id[~patch_mask] = 0

        # 4. Projection (linear 또는 CNN stem) — patch content 표현만 생성
        patch_embed = self.patch_embed.project(patches, patch_signal_types)
        # patch_embed: (B, N, d_model)

        # 패딩 마스크 (p_vid==0은 패딩 토큰)
        valid_token = (p_vid > 0).unsqueeze(-1)  # (B, N, 1)

        # 5-6. Conditioning Embedding 계산 (signal_type + spatial_id + loc + scale)
        # mask_token이 patch content를 덮어써도 conditioning은 살아남도록 별도 계산.
        # 이후 mask 적용 후 합산하여 마스킹된 위치도 자기 신호 종류/레벨 정보를 유지.
        cond = torch.zeros_like(patch_embed)
        if self.use_spatial_embed and patch_signal_types is not None:
            sig_emb = self.signal_type_embed(patch_signal_types)  # (B, N, d_model)
            spa_emb = self.spatial_id_embed(patch_spatial_ids)  # (B, N, d_model)
            cond = cond + sig_emb + spa_emb

        n = patch_embed.shape[1]
        stride = self.patch_embed.stride
        patch_starts = torch.arange(n, device=device) * stride  # (N,)
        patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
        patch_loc = loc[:, patch_starts, :]  # (B, N, 1)
        patch_scale = scale[:, patch_starts, :]  # (B, N, 1)

        # AdaLN 모드: loc/scale을 cond_proj로 → encoder 모든 layer에 modulation 입력
        # additive 모드: loc/scale을 token에 더해주는 기존 방식
        ada_cond: torch.Tensor | None = None
        if self.use_adaln:
            loc_scale = torch.cat([patch_loc, patch_scale], dim=-1)  # (B, N, 2)
            ada_cond = self.cond_proj(loc_scale)  # (B, N, d_cond)
            ada_cond = ada_cond * valid_token  # 패딩 위치는 0으로
            cond = cond * valid_token  # signal_type + spatial_id만 token에 더해짐
        else:
            loc_emb = self.loc_proj(patch_loc)  # (B, N, d_model)
            scale_emb = self.scale_proj(patch_scale)  # (B, N, d_model)
            cond = (cond + loc_emb + scale_emb) * valid_token

        # 7. Pred Mask 생성 (random/block/variate-level)
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

        # 8.5. Complete Variate Dropout: attention에서 variate를 물리적으로 제거
        # → 학습 시 "해당 variate 없이 cross-pred" 시나리오를 경험
        # → zero-shot cross-modal generation의 train-inference gap 해소
        drop_mask: torch.Tensor | None = None
        if variate_drop_prob > 0 and self.training and task in ("masked", "both"):
            drop_mask = self._sample_variate_drop(
                p_sid, p_vid, patch_mask, variate_drop_prob
            )  # (B, N) bool — True = attention에서 제거
            if drop_mask is not None:
                keep = ~drop_mask  # (B, N)
                # attention에서 제거: dropped 토큰은 attend 못하고, attend 받지도 못함
                base_attn_mask = base_attn_mask & keep.unsqueeze(-1) & keep.unsqueeze(-2)

        # 9. Encoder 입력 빌드 헬퍼
        # patch content를 mask_token으로 교체(content_mask 위치) → conditioning 합산.
        # 이렇게 해야 마스킹/드롭된 위치도 signal_type·spatial·loc·scale 정보가 유지됨.
        def _make_input(content_mask: torch.Tensor | None) -> torch.Tensor:
            if content_mask is None:
                x = patch_embed
            else:
                mt = self.mask_token.expand_as(patch_embed)
                x = torch.where(content_mask.unsqueeze(-1), mt, patch_embed)
            return x + cond

        # 10. Task에 따른 Encoder 호출
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

        # MoE 라우팅에서 padded 토큰 제외용 — (B, N) bool
        token_valid = (p_vid > 0)
        encoder_kwargs = dict(
            var_id=p_vid, time_id=time_id, token_mask=token_valid, cond=ada_cond,
        )  # RoPE는 상대적 time_id; token_mask는 MoE aux_loss 산정용; cond는 AdaLN용 (None이면 무시)
        use_causal = task in ("next_pred", "both")

        # causal mask (next_pred, both에서 공유)
        if use_causal:
            causal_tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
            causal_mask = base_attn_mask & causal_tri.unsqueeze(0)  # (B, N, N)

        # bidirectional 입력: pred_mask | drop_mask 위치를 mask_token으로 교체
        bi_content_mask = drop_mask
        if pred_mask is not None:
            bi_content_mask = (
                pred_mask if bi_content_mask is None else (pred_mask | bi_content_mask)
            )

        if task == "both":
            result["encoded"] = self.encoder(
                _make_input(bi_content_mask),
                attn_mask=base_attn_mask,
                **encoder_kwargs,
            )
            # causal: drop_mask만 적용 (causal attention이 미래 정보 차단하므로
            # pred_mask는 불필요).
            result["encoded_causal"] = self.encoder(
                _make_input(drop_mask),
                attn_mask=causal_mask,
                **encoder_kwargs,
            )
        elif task == "next_pred":
            result["encoded"] = self.encoder(
                _make_input(drop_mask),
                attn_mask=causal_mask,
                **encoder_kwargs,
            )
        else:  # "masked"
            result["encoded"] = self.encoder(
                _make_input(bi_content_mask),
                attn_mask=base_attn_mask,
                **encoder_kwargs,
            )

        return result

    # ── Forward ────────────────────────────────────────────────────

    def forward(
        self,
        batch: PackedBatch,
        task: str = "masked",  # "masked" 또는 "next_pred"
        mask_ratio: float = 0.0,
        block_mask: bool = False,
        block_size_min: int = 3,
        block_size_max: int = 8,
        variate_mask_prob: float = 0.0,
        variate_drop_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        enc = self._encode(
            batch,
            task=task,
            mask_ratio=mask_ratio,
            block_mask=block_mask,
            block_size_min=block_size_min,
            block_size_max=block_size_max,
            variate_mask_prob=variate_mask_prob,
            variate_drop_prob=variate_drop_prob,
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
            # Per-target-type cross-modal prediction (separate heads)
            cross_pred_per_type = torch.stack([
                self.cross_heads[str(st)](encoded)
                for st in range(self.num_signal_types)
            ], dim=2)  # (B, N, num_signal_types, patch_size)
            out_dict["cross_pred_per_type"] = cross_pred_per_type
            if self.contrastive_proj_dim > 0:
                out_dict["contrastive_z"] = self.contrastive_proj(
                    encoded
                )  # (B, N, proj_dim)

        # ── Block Next-Patch Prediction ──
        # encoded_causal[n] → K개의 future raw patches (n+1, ..., n+K) 병렬 예측.
        # BlockNextHead (shared trunk + K heads)가 바로 (B, N, K, P) 반환.
        if task in ("next_pred", "both"):
            encoded_for_next = enc.get("encoded_causal", encoded)  # (B, N, d_model)
            out_dict["next_pred"] = self.next_head(encoded_for_next)  # (B, N, K, P)

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
        out.pop("cross_pred_per_type", None)
        return out

    @torch.no_grad()
    def generate_cross_modal(
        self,
        batch: PackedBatch,
        target_signal_type: int,
        denormalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Zero-shot cross-modal waveform generation (Virtual Token Injection).

        입력 batch의 source signal로부터 target signal type의 waveform을 생성한다.
        target variate에 [MASK] 가상 토큰을 주입하여, 학습 시 variate dropout과
        동일한 상황을 재현한다.

        Parameters
        ----------
        batch:
            Source signal만 포함된 PackedBatch.
        target_signal_type:
            생성할 target signal type (0=ECG, 1=ABP, 2=PPG, ...).
        denormalize:
            ``True``이면 source의 loc/scale로 denormalize (approximate).

        Returns
        -------
        dict with keys:
            ``waveform``: ``(B, N, patch_size)`` — 생성된 target waveform.
            ``patch_mask``: ``(B, N)`` — 유효 패치 마스크.
        """
        self.eval()

        # Forward (mask_ratio=0 → 마스킹 없이 순수 source 정보만 사용)
        out = self.forward(batch, task="masked", mask_ratio=0.0)

        cross_pred_per_type = out["cross_pred_per_type"]  # (B, N, T, P)
        target_pred = cross_pred_per_type[:, :, target_signal_type, :]  # (B, N, P)

        if denormalize:
            loc = out["loc"]  # (B, L, 1)
            scale = out["scale"]  # (B, L, 1)
            p = self.patch_size
            stride = self.patch_embed.stride
            n = target_pred.shape[1]
            patch_starts = torch.arange(n, device=loc.device) * stride
            patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
            patch_loc = loc[:, patch_starts, :]  # (B, N, 1)
            patch_scale = scale[:, patch_starts, :]  # (B, N, 1)
            target_pred = target_pred * patch_scale + patch_loc

        return {
            "waveform": target_pred,
            "patch_mask": out["patch_mask"],
        }

    @torch.no_grad()
    def forecast(
        self,
        batch: PackedBatch,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Block next-patch 예측 (non-autoregressive).

        각 position n에서 미래 K개 패치를 동시에 예측한다.

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.
        denormalize:
            ``True``이면 scaler의 loc/scale로 원본 스케일 복원.

        Returns
        -------
        torch.Tensor
            ``(B, N, K, patch_size)`` block prediction map.
        """
        self.eval()
        out = self.forward(batch, task="next_pred")
        pred = out["next_pred"]  # (B, N, K, patch_size)

        if denormalize:
            loc = out["loc"]  # (B, L, 1)
            scale = out["scale"]  # (B, L, 1)
            p = self.patch_size
            patch_loc = loc[:, ::p, :]  # (B, N_approx, 1)
            patch_scale = scale[:, ::p, :]  # (B, N_approx, 1)
            n = pred.shape[1]
            patch_loc = patch_loc[:, :n, :]  # (B, N, 1)
            patch_scale = patch_scale[:, :n, :]  # (B, N, 1)
            # Broadcast over K dimension
            pred = pred * patch_scale.unsqueeze(2) + patch_loc.unsqueeze(2)

        return pred

    @torch.no_grad()
    def generate(
        self,
        batch: PackedBatch,
        n_steps: int,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Block-autoregressive 다단계 생성.

        Block Next Prediction head가 1-shot에 K개 패치를 내놓으므로, 매 forward마다
        K개를 모두 취해 입력에 append → 다시 forward → 반복. ``collate_mode="ci"``
        (single-variate-per-row) 전제.

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
        k = self.next_block_size

        out = self.forward(batch, task="next_pred")
        loc = out["loc"]  # (B, L, 1)
        scale = out["scale"]  # (B, L, 1)
        cached_loc = loc[:, 0:1, :]  # (B, 1, 1)
        cached_scale = scale[:, 0:1, :]  # (B, 1, 1)

        generated: list[torch.Tensor] = []
        pred = out["next_pred"]  # (B, N, K, patch_size)
        patch_mask = out["patch_mask"]  # (B, N)
        b = pred.shape[0]
        last_valid_idx = patch_mask.sum(dim=-1) - 1  # (B,)
        last_valid_idx = last_valid_idx.clamp(min=0)
        arange_b = torch.arange(b, device=pred.device)
        block = pred[arange_b, last_valid_idx]  # (B, K, patch_size)

        # 한 번의 forward에서 나오는 K patches를 순서대로 append.
        for j in range(k):
            if len(generated) >= n_steps:
                break
            generated.append(block[:, j, :])  # (B, patch_size)

        while len(generated) < n_steps:
            # block의 K patches를 모두 입력에 append — 다음 forward에서 새 예측.
            for j in range(k):
                batch = _append_patch_to_batch(batch, block[:, j, :], p)

            out = self.forward(batch, task="next_pred")
            pred = out["next_pred"]  # (B, N, K, patch_size)
            patch_mask = out["patch_mask"]
            last_valid_idx = patch_mask.sum(dim=-1) - 1
            last_valid_idx = last_valid_idx.clamp(min=0)
            block = pred[arange_b, last_valid_idx]  # (B, K, patch_size)
            for j in range(k):
                if len(generated) >= n_steps:
                    break
                generated.append(block[:, j, :])

        result = torch.stack(generated[:n_steps], dim=0)  # (n_steps, B, patch_size)

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
