# -*- coding:utf-8 -*-
"""Foundation model 로딩 + encoder freeze/unfreeze + LoRA 래퍼.

다운스트림 task에서 사전학습된 모델을 로드하고, encoder를 freeze한 상태에서
feature를 추출하거나 LoRA adapter를 삽입하여 효율적 fine-tuning을 수행한다.

Usage
-----
>>> wrapper = DownstreamModelWrapper("checkpoints/best.pt", model_version="v1")
>>> features = wrapper.extract_features(batch)  # (B, d_model)
>>> probe = LinearProbe(wrapper.d_model, n_classes=3)
>>>
>>> # LoRA
>>> wrapper.inject_lora(rank=8)
>>> lora_params = wrapper.lora_parameters()
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from data.collate import PackedBatch
from model import ModelConfig
from model.biosignal_model import BiosignalFoundationModel


# ── LoRA Layer ────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    원본 Linear를 freeze하고, 학습 가능한 low-rank A, B 행렬을 추가한다.
    output = frozen_linear(x) + (x @ A @ B) * (alpha / rank)
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)

        in_features = original.in_features
        out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # 초기화: A는 Kaiming, B는 zero → 초기 출력 = 원본과 동일
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return frozen_out + lora_out * self.scaling


class DownstreamModelWrapper(nn.Module):
    """사전학습 모델 로딩 + encoder freeze + feature 추출 래퍼.

    Parameters
    ----------
    checkpoint_path:
        사전학습 checkpoint 경로 (.pt).
    model_version:
        ``"v1"`` 또는 ``"v2"``.
    device:
        모델 로드 디바이스.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_version: str = "v1",
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        # 1. Checkpoint 로드 → config 복원 → 모델 생성
        state = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        if "config" in state:
            config = ModelConfig.from_dict(state["config"])
        else:
            raise ValueError("Checkpoint에 'config' 키가 없습니다.")

        model_cls = BiosignalFoundationModel
        self.model: BiosignalFoundationModel = model_cls.from_config(config)
        self.model.to(self.device)

        # 2. State dict 로드
        missing, unexpected = self.model.load_state_dict(
            state["model_state_dict"],
            strict=False,
        )
        if missing:
            print(f"  [model_wrapper] Missing keys: {missing}")
        if unexpected:
            print(f"  [model_wrapper] Unexpected keys: {unexpected}")

        # 3. Encoder freeze + eval 모드
        self.freeze_encoder()
        self.model.eval()

        # 4. 편의 속성
        self.d_model: int = config.d_model
        self.patch_size: int = config.patch_size
        self.config = config

    def freeze_encoder(self) -> None:
        """Encoder 전체 파라미터를 freeze한다 (requires_grad=False).

        Freeze 범위: scaler, patch_embed, encoder, signal_type_embed,
        spatial_id_embed, cond_proj.
        """
        self.model.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        """Encoder 파라미터를 unfreeze한다 (fine-tuning용)."""
        self.model.requires_grad_(True)

    def inject_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_p: float = 0.0,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    ) -> int:
        """Attention layer의 target module에 LoRA adapter를 삽입한다.

        Parameters
        ----------
        rank: LoRA rank (r).
        alpha: LoRA scaling factor.
        dropout_p: LoRA dropout.
        target_modules: LoRA를 적용할 Linear 이름.

        Returns
        -------
        삽입된 LoRA 파라미터 수.
        """
        self.freeze_encoder()  # 전체 freeze 후 LoRA만 학습

        n_lora_params = 0
        for name, module in self.model.named_modules():
            for target in target_modules:
                child = getattr(module, target, None)
                if child is not None and isinstance(child, nn.Linear):
                    lora = LoRALinear(child, rank=rank, alpha=alpha, dropout_p=dropout_p)
                    lora = lora.to(self.device)
                    setattr(module, target, lora)
                    n_lora_params += rank * (child.in_features + child.out_features)

        print(f"  [LoRA] rank={rank}, alpha={alpha}, targets={target_modules}")
        print(f"  [LoRA] Trainable params: {n_lora_params:,}")
        return n_lora_params

    def lora_parameters(self) -> list[nn.Parameter]:
        """LoRA adapter의 학습 가능한 파라미터만 반환한다."""
        params = []
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                params.extend(module.lora_A.parameters())
                params.extend(module.lora_B.parameters())
        return params

    @torch.no_grad()
    def extract_features(
        self,
        batch: PackedBatch,
        pool: str = "mean",
    ) -> torch.Tensor:
        """Downstream task용 feature 추출.

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.
        pool:
            풀링 방식. ``"mean"``이면 patch_mask 기반 mean pool,
            ``"none"``이면 (B, N, d_model) 그대로 반환.

        Returns
        -------
        ``pool="mean"``: ``(B, d_model)`` — 윈도우 레벨 feature.
        ``pool="none"``: ``(B, N, d_model)`` — 패치 레벨 feature.
        """
        self.model.eval()
        batch = self.batch_to_device(batch)

        out = self.model(batch, task="masked")
        encoded = out["encoded"]  # (B, N, d_model)
        patch_mask = out["patch_mask"]  # (B, N) bool — True=유효 패치

        if pool == "none":
            return encoded

        # Mean pooling over valid patches
        mask_f = patch_mask.unsqueeze(-1).float()  # (B, N, 1)
        pooled = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(
            min=1.0
        )  # (B, d_model)
        return pooled

    @torch.no_grad()
    def forward_masked(
        self,
        batch: PackedBatch,
    ) -> dict[str, torch.Tensor]:
        """기존 forward(task="masked") 래핑.

        Returns
        -------
        dict with keys: ``reconstructed``, ``cross_pred``, ``encoded``,
        ``patch_mask``, ``loc``, ``scale``, etc.
        """
        self.model.eval()
        batch = self.batch_to_device(batch)
        return self.model(batch, task="masked")

    @torch.no_grad()
    def get_reconstruction_loss(
        self,
        batch: PackedBatch,
        mask: torch.Tensor,  # (B, N) bool — 복원 대상 패치
    ) -> torch.Tensor:
        """Masked reconstruction MSE loss (anomaly scoring용).

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.
        mask:
            ``(B, N)`` bool — ``True``인 위치의 패치에 대해 MSE 계산.

        Returns
        -------
        ``()`` — per-window 평균 MSE scalar.
        """
        self.model.eval()
        batch = self.batch_to_device(batch)

        out = self.model(batch, task="masked")
        reconstructed = out["reconstructed"]  # (B, N, patch_size)

        # 원본 패치 추출 (정규화된 값)
        normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(
            -1
        )  # (b, l)
        b, l = normalized.shape
        p = self.patch_size
        n = l // p
        original_patches = normalized.reshape(b, n, p)  # (b, n, p)

        mask = mask.to(self.device)
        if not mask.any():
            return reconstructed.new_tensor(0.0)

        loss = F.mse_loss(
            reconstructed[mask],  # (M, patch_size)
            original_patches[mask],  # (M, patch_size)
        )
        return loss

    def batch_to_device(self, batch: PackedBatch) -> PackedBatch:
        """PackedBatch 텐서를 self.device로 이동한다."""
        batch.values = batch.values.to(self.device)
        batch.sample_id = batch.sample_id.to(self.device)
        batch.variate_id = batch.variate_id.to(self.device)
        return batch


class LinearProbe(nn.Module):
    """분류/회귀 task용 lightweight linear head.

    Parameters
    ----------
    d_model:
        입력 feature 차원 (foundation model d_model).
    n_classes:
        출력 클래스 수. ``1``이면 회귀 (sigmoid 없음).
    dropout_p:
        드롭아웃 확률.
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_p),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self,
        features: torch.Tensor,  # (B, d_model)
    ) -> torch.Tensor:  # (B, n_classes)
        """Feature → logits (또는 회귀 값)."""
        return self.head(features)
