# -*- coding:utf-8 -*-
"""2-Phase 커리큘럼 학습 공유 유틸리티.

``train/1_channel_independency.py``와 ``train/2_any_variate.py``에서 공통으로 사용하는
데이터 로딩, 학습 루프, 체크포인트 함수를 정의한다.
"""
import json
import math
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from data import RecordingManifest
from loss.criterion import CombinedLoss
from loss.masked_mse_loss import create_patch_mask
from model import BiosignalFoundationModel
from model.checkpoint import save_checkpoint
from model.config import ModelConfig


# ── 설정 ────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """학습 하이퍼파라미터."""

    # 모델
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # 데이터
    processed_dir: str = "datasets/processed"
    signal_types: list[int] = field(default_factory=lambda: [2, 4, 5])
    max_subjects: int | None = None
    window_seconds: float = 30.0
    max_length: int = 50000
    cache_size: int = 16

    # 학습
    batch_size: int = 16
    lr: float = 1e-3
    n_epochs: int = 70
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.1
    mask_ratio: float = 0.15
    gradient_clip: float = 1.0
    seed: int = 42
    collate_mode: str = "ci"

    # Loss 가중치
    alpha: float = 1.0  # masked reconstruction
    beta: float = 0.0   # next-patch prediction
    gamma: float = 0.0  # cross-modal (beta 내부 가중)
    delta: float = 0.0  # cross-modal contrastive
    contrastive_temperature: float = 0.07
    learnable_temperature: bool = True

    # Masking 전략
    variate_mask_prob: float = 0.0  # Phase 2: variate-level 마스킹 확률

    # 시스템
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 0
    output_dir: str = "outputs"
    checkpoint_every: int = 10  # 에폭 간격 체크포인트 저장

    # Dry-run
    dry_run: bool = False  # True면 1 batch만 실행 후 종료

    # ── YAML 직렬화 ──────────────────────────────────────────

    def to_yaml(self, path: str | Path) -> None:
        """설정을 YAML 파일로 저장한다."""
        d = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(d, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """YAML 파일에서 TrainConfig를 복원한다.

        ``model_config`` 키는 자동으로 ``ModelConfig``로 변환한다.
        알 수 없는 키는 무시한다.
        """
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)

        # model_config를 ModelConfig 인스턴스로 변환
        if "model_config" in d and isinstance(d["model_config"], dict):
            d["model_config"] = ModelConfig.from_dict(d["model_config"])

        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_yaml_with_overrides(
        cls, path: str | Path, overrides: dict[str, Any] | None = None,
    ) -> "TrainConfig":
        """YAML 파일을 로드한 뒤, CLI 인자 등으로 오버라이드한다."""
        config = cls.from_yaml(path)
        if overrides:
            for k, v in overrides.items():
                if v is None:
                    continue
                if hasattr(config, k):
                    setattr(config, k, v)
                elif hasattr(config.model_config, k):
                    setattr(config.model_config, k, v)
        return config


# ── 데이터 로딩 ─────────────────────────────────────────────────


def load_manifest_from_processed(
    processed_dir: str | Path,
    signal_types: list[int] | None = None,
    max_subjects: int | None = None,
) -> list[RecordingManifest]:
    """processed 디렉토리에서 manifest.json을 읽어 RecordingManifest 목록을 반환한다."""
    processed_dir = Path(processed_dir)
    manifest_files = sorted(processed_dir.glob("*/manifest.json"))
    if max_subjects is not None:
        manifest_files = manifest_files[:max_subjects]

    entries: list[RecordingManifest] = []
    for mf in manifest_files:
        subject_dir = mf.parent
        with open(mf, encoding="utf-8") as f:
            meta = json.load(f)

        for session in meta["sessions"]:
            session_id = session["session_id"]
            for rec in session["recordings"]:
                if signal_types is not None and rec["signal_type"] not in signal_types:
                    continue
                entries.append(
                    RecordingManifest(
                        path=str(subject_dir / rec["file"]),
                        n_channels=rec["n_channels"],
                        n_timesteps=rec["n_timesteps"],
                        sampling_rate=rec["sampling_rate"],
                        signal_type=rec["signal_type"],
                        session_id=session_id,
                        spatial_ids=rec.get("spatial_ids"),
                    )
                )
    return entries


# ── 학습 루프 ───────────────────────────────────────────────────


def train_one_epoch(
    model: BiosignalFoundationModel,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
    phase_name: str,
) -> dict[str, float]:
    """1에폭 학습을 수행하고 평균 loss를 반환한다."""
    model.train()
    epoch_total = 0.0
    epoch_masked = 0.0
    epoch_next = 0.0
    epoch_cross = 0.0
    epoch_contrastive = 0.0
    n_batches = 0
    nan_count = 0
    max_nan_batches = 10

    enable_next = config.beta > 0

    for batch in dataloader:
        # GPU로 이동
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

        # ── Masked Reconstruction forward ──
        out = model(batch, task="masked")
        reconstructed = out["reconstructed"]  # (B, N, patch_size)
        cross_pred = out["cross_pred"]        # (B, N, patch_size)
        patch_mask = out["patch_mask"]        # (B, N) bool
        time_id = out["time_id"]              # (B, N)

        # 패치 단위 마스킹 (variate-level 마스킹 지원)
        pred_mask = create_patch_mask(
            patch_mask,
            mask_ratio=config.mask_ratio,
            patch_variate_id=out["patch_variate_id"] if config.variate_mask_prob > 0 else None,
            variate_mask_prob=config.variate_mask_prob,
        )

        # 원본 패치 추출 (정규화된 값)
        P = model.patch_size
        normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
        B, L = normalized.shape
        N = L // P
        original_patches = normalized.reshape(B, N, P)  # (B, N, P)

        # ── Next-Patch Prediction forward (별도 causal forward) ──
        H = 1
        next_pred = None
        if enable_next:
            H = random.randint(1, config.model_config.max_horizon)
            out_next = model(batch, task="next_pred", horizon=H)
            next_pred = out_next["next_pred"]  # (B, N, patch_size)

        # ── Contrastive embeddings (masked forward에서 추출) ──
        contrastive_z = out.get("contrastive_z")  # (B, N, proj_dim) or None

        # ── CombinedLoss ──
        needs_time_id = config.gamma > 0 or config.delta > 0
        losses = criterion(
            reconstructed=reconstructed,
            next_pred=next_pred,
            original_patches=original_patches,
            pred_mask=pred_mask,
            patch_mask=patch_mask,
            patch_sample_id=out["patch_sample_id"],
            patch_variate_id=out["patch_variate_id"],
            horizon=H,
            cross_pred=cross_pred if config.gamma > 0 else None,
            time_id=time_id if needs_time_id else None,
            contrastive_z=contrastive_z if config.delta > 0 else None,
        )

        # ── Backward ──
        loss = losses["total"]

        # NaN/Inf 감지
        if not torch.isfinite(loss):
            print(
                f"  [{phase_name}] WARNING: NaN/Inf loss detected at batch {n_batches + 1}, "
                f"skipping batch (masked={losses['masked_loss'].item():.4f}, "
                f"next={losses['next_loss'].item():.4f}, "
                f"cross={losses['cross_modal_loss'].item():.4f}, "
                f"contrastive={losses['contrastive_loss'].item():.4f})"
            )
            nan_count += 1
            if nan_count >= max_nan_batches:
                print(
                    f"  [{phase_name}] ERROR: {nan_count} consecutive NaN/Inf batches. "
                    f"Stopping epoch early."
                )
                break
            optimizer.zero_grad()
            continue

        nan_count = 0  # 정상 batch면 카운터 리셋
        optimizer.zero_grad()
        loss.backward()

        # Gradient NaN/Inf 감지
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.gradient_clip,
        )
        if not torch.isfinite(grad_norm):
            print(
                f"  [{phase_name}] WARNING: NaN/Inf gradient at batch {n_batches + 1}, "
                f"skipping update."
            )
            optimizer.zero_grad()
            n_batches += 1
            continue

        optimizer.step()

        # ── 로깅 ──
        epoch_total += losses["total"].item()
        epoch_masked += losses["masked_loss"].item()
        epoch_next += losses["next_loss"].item()
        epoch_cross += losses["cross_modal_loss"].item()
        epoch_contrastive += losses["contrastive_loss"].item()
        n_batches += 1

        if n_batches % 50 == 0 or config.dry_run:
            print(
                f"  [{phase_name}] batch {n_batches} | "
                f"total: {losses['total'].item():.6f} | "
                f"masked: {losses['masked_loss'].item():.6f} | "
                f"next: {losses['next_loss'].item():.6f} | "
                f"cross: {losses['cross_modal_loss'].item():.6f} | "
                f"contrastive: {losses['contrastive_loss'].item():.6f} | "
                f"grad_norm: {grad_norm:.4f}"
            )

        # Dry-run: 1 batch만 실행
        if config.dry_run:
            print(f"  [{phase_name}] dry-run: 1 batch 완료, 종료.")
            break

    denom = max(n_batches, 1)
    return {
        "total": epoch_total / denom,
        "masked_loss": epoch_masked / denom,
        "next_loss": epoch_next / denom,
        "cross_modal_loss": epoch_cross / denom,
        "contrastive_loss": epoch_contrastive / denom,
    }


# ── 유틸리티 ────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """재현 가능한 학습을 위한 시드 고정."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    """'auto' → CUDA/CPU 자동 선택."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> LambdaLR:
    """Linear warmup + Cosine decay 스케줄러를 생성한다.

    - ``[0, warmup_epochs)``: lr을 0에서 ``config.lr``까지 선형 증가.
    - ``[warmup_epochs, n_epochs)``: ``config.lr``에서 ``config.lr * min_lr_ratio``까지 cosine 감쇠.
    """
    warmup = config.warmup_epochs
    total = config.n_epochs
    min_ratio = config.min_lr_ratio

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def get_model_config(config: TrainConfig) -> dict[str, Any]:
    """checkpoint 저장용 모델 config dict."""
    return config.model_config.to_dict()


def save_training_checkpoint(
    model: BiosignalFoundationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainConfig,
    phase_name: str,
    loss: float,
    output_dir: Path,
    tag: str = "",
) -> Path:
    """학습 checkpoint를 저장하고 경로를 반환한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    filename = f"checkpoint_{phase_name}_epoch{epoch:03d}{suffix}.pt"
    path = output_dir / filename
    save_checkpoint(
        path, model, optimizer=optimizer, epoch=epoch,
        config=get_model_config(config),
        phase=phase_name, loss=loss,
    )
    return path
