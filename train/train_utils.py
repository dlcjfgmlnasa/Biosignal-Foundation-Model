# -*- coding:utf-8 -*-
from __future__ import annotations

"""2-Phase 커리큘럼 학습 공유 유틸리티.

``train/1_channel_independency.py``와 ``train/2_any_variate.py``에서 공통으로 사용하는
데이터 로딩, 학습 루프, 체크포인트 함수를 정의한다.
"""
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from data import RecordingManifest
from loss.criterion import CombinedLoss
from loss.masked_mse_loss import create_patch_mask
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
    signal_types: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    max_subjects: int | None = None
    window_seconds: float = 30.0
    max_length: int = 50000
    cache_size: int = 16
    crop_ratio_min: float = 0.0  # >0이면 random crop 활성 (min ratio)
    crop_ratio_max: float = 0.0  # >0이면 random crop 활성 (max ratio)

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
    aux_loss_weight: float = 0.01  # MoE load balancing auxiliary loss
    contrastive_temperature: float = 0.07
    learnable_temperature: bool = True

    # Masking 전략
    variate_mask_prob: float = 0.0  # Phase 2: variate-level 마스킹 확률

    # 시스템
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 4
    output_dir: str = "outputs"
    checkpoint_every: int = 10  # 에폭 간격 체크포인트 저장

    # Validation & Early Stopping
    val_ratio: float = 0.2  # subject 단위 validation 비율
    patience: int = 10  # early stopping patience (0=비활성)

    # Mixed Precision
    use_amp: bool = False  # True면 AMP (autocast + GradScaler) 활성

    # 실험 관리
    exp_name: str = ""  # 실험 이름 (비어있으면 output_dir 그대로 사용)

    # 실행 제한
    max_batches: int = 0  # >0이면 에폭당 최대 배치 수 제한
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


def split_manifest_by_subject(
    manifest: list[RecordingManifest],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[RecordingManifest], list[RecordingManifest]]:
    """Subject(디렉토리) 단위로 train/val을 분할한다.

    같은 subject의 모든 recording이 동일한 split에 들어간다.

    Returns
    -------
    (train_manifest, val_manifest)
    """
    # subject = path의 부모 디렉토리 이름
    subject_to_entries: dict[str, list[RecordingManifest]] = {}
    for entry in manifest:
        subject = str(Path(entry.path).parent)
        subject_to_entries.setdefault(subject, []).append(entry)

    subjects = sorted(subject_to_entries.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_val = max(1, int(len(subjects) * val_ratio))
    val_subjects = set(subjects[:n_val])

    train_entries: list[RecordingManifest] = []
    val_entries: list[RecordingManifest] = []
    for subj in subjects:
        if subj in val_subjects:
            val_entries.extend(subject_to_entries[subj])
        else:
            train_entries.extend(subject_to_entries[subj])

    return train_entries, val_entries


# ── 학습 루프 ───────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
    phase_name: str,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """1에폭 학습을 수행하고 평균 loss를 반환한다."""
    model.train()
    # GPU 텐서로 누적하여 배치마다 .item() CUDA sync 방지
    epoch_total = torch.zeros(1, device=device)
    epoch_masked = torch.zeros(1, device=device)
    epoch_next = torch.zeros(1, device=device)
    epoch_cross = torch.zeros(1, device=device)
    epoch_contrastive = torch.zeros(1, device=device)
    epoch_aux = torch.zeros(1, device=device)
    n_batches = 0
    nan_count = 0
    max_nan_batches = 10

    enable_next = config.beta > 0
    use_amp = scaler is not None
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for batch in dataloader:
        # GPU로 이동
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

        # ── Forward (single pass: masked + next_pred 동시) ──
        H = 1
        if enable_next:
            H = random.randint(1, config.model_config.max_horizon)

        task = "both" if enable_next else "masked"

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch, task=task, horizon=H)

            reconstructed = out["reconstructed"]  # (B, N, patch_size)
            cross_pred = out["cross_pred"]        # (B, N, patch_size)
            patch_mask = out["patch_mask"]        # (B, N) bool
            time_id = out["time_id"]              # (B, N)
            next_pred = out.get("next_pred")      # (B, N, patch_size) or None

            # 패치 단위 마스킹 (variate-level 마스킹 지원)
            pred_mask = create_patch_mask(
                patch_mask,
                mask_ratio=config.mask_ratio,
                patch_variate_id=out["patch_variate_id"] if config.variate_mask_prob > 0 else None,
                variate_mask_prob=config.variate_mask_prob,
            )

            # 원본 패치 추출 (정규화된 값)
            raw_model = model.module if isinstance(model, DDP) else model
            P = raw_model.patch_size
            normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
            B, L = normalized.shape
            N = L // P
            original_patches = normalized.reshape(B, N, P)  # (B, N, P)

            # ── Contrastive embeddings ──
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

        # ── MoE auxiliary loss 수집 ──
        aux_loss = torch.zeros(1, device=device)
        raw_model = model.module if isinstance(model, DDP) else model
        for layer in raw_model.encoder.layers:
            if hasattr(layer.ffn, "aux_loss") and layer.ffn.aux_loss is not None:
                aux_loss = aux_loss + layer.ffn.aux_loss

        # ── Backward ──
        loss = losses["total"] + config.aux_loss_weight * aux_loss

        # NaN/Inf 감지
        if not torch.isfinite(loss):
            if is_main_process():
                print(
                    f"  [{phase_name}] WARNING: NaN/Inf loss detected at batch {n_batches + 1}, "
                    f"skipping batch (masked={losses['masked_loss'].item():.4f}, "
                    f"next={losses['next_loss'].item():.4f}, "
                    f"cross={losses['cross_modal_loss'].item():.4f}, "
                    f"contrastive={losses['contrastive_loss'].item():.4f})"
                )
            nan_count += 1
            if nan_count >= max_nan_batches:
                if is_main_process():
                    print(
                        f"  [{phase_name}] ERROR: {nan_count} consecutive NaN/Inf batches. "
                        f"Stopping epoch early."
                    )
                break
            optimizer.zero_grad(set_to_none=True)
            continue

        nan_count = 0  # 정상 batch면 카운터 리셋
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Gradient NaN/Inf 감지
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.gradient_clip,
        )
        if not torch.isfinite(grad_norm):
            if is_main_process():
                print(
                    f"  [{phase_name}] WARNING: NaN/Inf gradient at batch {n_batches + 1}, "
                    f"skipping update."
                )
            optimizer.zero_grad(set_to_none=True)
            n_batches += 1
            if scaler is not None:
                scaler.update()
            continue

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # ── 로깅 (GPU 텐서로 누적, CUDA sync 없음) ──
        epoch_total += loss.detach()
        epoch_masked += losses["masked_loss"].detach()
        epoch_next += losses["next_loss"].detach()
        epoch_cross += losses["cross_modal_loss"].detach()
        epoch_contrastive += losses["contrastive_loss"].detach()
        epoch_aux += aux_loss.detach()
        n_batches += 1

        if is_main_process() and (n_batches % 50 == 0 or config.dry_run):
            print(
                f"  [{phase_name}] batch {n_batches} | "
                f"total: {loss.item():.6f} | "
                f"masked: {losses['masked_loss'].item():.6f} | "
                f"next: {losses['next_loss'].item():.6f} | "
                f"cross: {losses['cross_modal_loss'].item():.6f} | "
                f"contrastive: {losses['contrastive_loss'].item():.6f} | "
                f"aux: {aux_loss.item():.6f} | "
                f"grad_norm: {grad_norm:.4f}"
            )

        # Dry-run: 1 batch만 실행
        if config.dry_run:
            if is_main_process():
                print(f"  [{phase_name}] dry-run: 1 batch 완료, 종료.")
            break

        # max_batches 제한
        if config.max_batches > 0 and n_batches >= config.max_batches:
            if is_main_process():
                print(f"  [{phase_name}] max_batches={config.max_batches} 도달, 에폭 종료.")
            break

    denom = max(n_batches, 1)
    return {
        "total": (epoch_total / denom).item(),
        "masked_loss": (epoch_masked / denom).item(),
        "next_loss": (epoch_next / denom).item(),
        "cross_modal_loss": (epoch_cross / denom).item(),
        "contrastive_loss": (epoch_contrastive / denom).item(),
        "aux_loss": (epoch_aux / denom).item(),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: CombinedLoss,
    config: TrainConfig,
    device: torch.device,
    phase_name: str,
) -> dict[str, float]:
    """Validation 루프. train_one_epoch()과 동일한 loss 계산, backward 없이.

    DDP 환경에서는 unwrapped 모델로 forward하여 rank별 배치 수
    불일치로 인한 데드락을 방지한다.
    """
    model.eval()
    # DDP wrapper를 벗겨서 forward — validation에서는 gradient sync 불필요
    raw_model = model.module if isinstance(model, DDP) else model

    epoch_total = 0.0
    epoch_masked = 0.0
    epoch_next = 0.0
    epoch_cross = 0.0
    epoch_contrastive = 0.0
    epoch_aux = 0.0
    n_batches = 0

    enable_next = config.beta > 0
    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for batch in dataloader:
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

        H = 1
        if enable_next:
            H = random.randint(1, config.model_config.max_horizon)

        task = "both" if enable_next else "masked"

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = raw_model(batch, task=task, horizon=H)

            reconstructed = out["reconstructed"]
            cross_pred = out["cross_pred"]
            patch_mask = out["patch_mask"]
            time_id = out["time_id"]
            next_pred = out.get("next_pred")

            pred_mask = create_patch_mask(
                patch_mask,
                mask_ratio=config.mask_ratio,
                patch_variate_id=out["patch_variate_id"] if config.variate_mask_prob > 0 else None,
                variate_mask_prob=config.variate_mask_prob,
            )

            P = raw_model.patch_size
            normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
            B, L = normalized.shape
            N = L // P
            original_patches = normalized.reshape(B, N, P)

            contrastive_z = out.get("contrastive_z")

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

        # MoE auxiliary loss 수집 (로깅 전용, backward 없음)
        aux_loss = 0.0
        for layer in raw_model.encoder.layers:
            if hasattr(layer.ffn, "aux_loss") and layer.ffn.aux_loss is not None:
                aux_loss += layer.ffn.aux_loss.item()

        loss = losses["total"]
        if not torch.isfinite(loss):
            continue

        epoch_total += losses["total"].item()
        epoch_masked += losses["masked_loss"].item()
        epoch_next += losses["next_loss"].item()
        epoch_cross += losses["cross_modal_loss"].item()
        epoch_contrastive += losses["contrastive_loss"].item()
        epoch_aux += aux_loss
        n_batches += 1

        if config.max_batches > 0 and n_batches >= config.max_batches:
            break

    model.train()
    denom = max(n_batches, 1)
    return {
        "total": epoch_total / denom,
        "masked_loss": epoch_masked / denom,
        "next_loss": epoch_next / denom,
        "cross_modal_loss": epoch_cross / denom,
        "contrastive_loss": epoch_contrastive / denom,
        "aux_loss": epoch_aux / denom,
    }


# ── V2 학습 루프 (EEG stem-target reconstruction) ────────────────


def train_one_epoch_v2(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
    phase_name: str,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """1에폭 V2 학습을 수행하고 평균 loss를 반환한다.

    V1과의 차이점:
    - EEG 패치(eeg_mask)를 masked reconstruction에서 제외하고 별도 EEG loss 계산.
    - EEG loss = MSE(eeg_reconstructed, eeg_recon_target) on (pred_mask & eeg_mask).
    - 최종 total = CombinedLoss(non-EEG) + eeg_loss.
    """
    model.train()
    # GPU 텐서로 누적하여 배치마다 .item() CUDA sync 방지
    epoch_total = torch.zeros(1, device=device)
    epoch_masked = torch.zeros(1, device=device)
    epoch_next = torch.zeros(1, device=device)
    epoch_cross = torch.zeros(1, device=device)
    epoch_contrastive = torch.zeros(1, device=device)
    epoch_eeg = torch.zeros(1, device=device)
    epoch_aux = torch.zeros(1, device=device)
    n_batches = 0
    nan_count = 0
    max_nan_batches = 10

    enable_next = config.beta > 0
    use_amp = scaler is not None
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for batch in dataloader:
        # GPU로 이동
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

        # ── Forward (single pass: masked + next_pred 동시) ──
        H = 1
        if enable_next:
            H = random.randint(1, config.model_config.max_horizon)

        task = "both" if enable_next else "masked"

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch, task=task, horizon=H)

            reconstructed = out["reconstructed"]  # (B, N, patch_size)
            cross_pred = out["cross_pred"]        # (B, N, patch_size)
            patch_mask = out["patch_mask"]        # (B, N) bool
            time_id = out["time_id"]              # (B, N)
            next_pred = out.get("next_pred")      # (B, N, patch_size) or None

            # V2 전용 출력
            eeg_reconstructed = out.get("eeg_reconstructed")  # (B, N, d_model) or None
            eeg_recon_target = out.get("eeg_recon_target")    # (B, N, d_model) or None
            eeg_mask = out.get("eeg_mask")                    # (B, N) bool or None

            # 패치 단위 마스킹 (variate-level 마스킹 지원)
            pred_mask = create_patch_mask(
                patch_mask,
                mask_ratio=config.mask_ratio,
                patch_variate_id=out["patch_variate_id"] if config.variate_mask_prob > 0 else None,
                variate_mask_prob=config.variate_mask_prob,
            )

            # 원본 패치 추출 (정규화된 값)
            raw_model = model.module if isinstance(model, DDP) else model
            P = raw_model.patch_size
            normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
            B, L = normalized.shape
            N = L // P
            original_patches = normalized.reshape(B, N, P)  # (B, N, P)

            # ── EEG 패치를 제외한 non-EEG pred_mask ──
            if eeg_mask is not None:
                non_eeg_pred_mask = pred_mask & ~eeg_mask  # (B, N)
            else:
                non_eeg_pred_mask = pred_mask

            # ── Contrastive embeddings ──
            contrastive_z = out.get("contrastive_z")  # (B, N, proj_dim) or None

            # ── CombinedLoss (non-EEG 패치만) ──
            needs_time_id = config.gamma > 0 or config.delta > 0
            losses = criterion(
                reconstructed=reconstructed,
                next_pred=next_pred,
                original_patches=original_patches,
                pred_mask=non_eeg_pred_mask,
                patch_mask=patch_mask,
                patch_sample_id=out["patch_sample_id"],
                patch_variate_id=out["patch_variate_id"],
                horizon=H,
                cross_pred=cross_pred if config.gamma > 0 else None,
                time_id=time_id if needs_time_id else None,
                contrastive_z=contrastive_z if config.delta > 0 else None,
            )

            # ── EEG 전용 masked reconstruction loss ──
            eeg_loss = reconstructed.new_tensor(0.0)
            if eeg_mask is not None and eeg_reconstructed is not None:
                eeg_pred_mask = pred_mask & eeg_mask  # (B, N)
                if eeg_pred_mask.any():
                    eeg_loss = torch.nn.functional.mse_loss(
                        eeg_reconstructed[eeg_pred_mask],   # (M, d_model)
                        eeg_recon_target[eeg_pred_mask],    # (M, d_model)
                    )

            # ── 최종 total loss ──
            total_loss = losses["total"] + eeg_loss

        # ── MoE auxiliary loss 수집 ──
        aux_loss = torch.zeros(1, device=device)
        raw_model = model.module if isinstance(model, DDP) else model
        for layer in raw_model.encoder.layers:
            if hasattr(layer.ffn, "aux_loss") and layer.ffn.aux_loss is not None:
                aux_loss = aux_loss + layer.ffn.aux_loss

        # ── Backward ──
        loss = total_loss + config.aux_loss_weight * aux_loss

        # NaN/Inf 감지
        if not torch.isfinite(loss):
            if is_main_process():
                print(
                    f"  [{phase_name}] WARNING: NaN/Inf loss detected at batch {n_batches + 1}, "
                    f"skipping batch (masked={losses['masked_loss'].item():.4f}, "
                    f"next={losses['next_loss'].item():.4f}, "
                    f"cross={losses['cross_modal_loss'].item():.4f}, "
                    f"contrastive={losses['contrastive_loss'].item():.4f}, "
                    f"eeg={eeg_loss.item():.4f})"
                )
            nan_count += 1
            if nan_count >= max_nan_batches:
                if is_main_process():
                    print(
                        f"  [{phase_name}] ERROR: {nan_count} consecutive NaN/Inf batches. "
                        f"Stopping epoch early."
                    )
                break
            optimizer.zero_grad(set_to_none=True)
            continue

        nan_count = 0  # 정상 batch면 카운터 리셋
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Gradient NaN/Inf 감지
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.gradient_clip,
        )
        if not torch.isfinite(grad_norm):
            if is_main_process():
                print(
                    f"  [{phase_name}] WARNING: NaN/Inf gradient at batch {n_batches + 1}, "
                    f"skipping update."
                )
            optimizer.zero_grad(set_to_none=True)
            n_batches += 1
            if scaler is not None:
                scaler.update()
            continue

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # ── 로깅 (GPU 텐서로 누적, CUDA sync 없음) ──
        epoch_total += loss.detach()
        epoch_masked += losses["masked_loss"].detach()
        epoch_next += losses["next_loss"].detach()
        epoch_cross += losses["cross_modal_loss"].detach()
        epoch_contrastive += losses["contrastive_loss"].detach()
        epoch_eeg += eeg_loss.detach()
        epoch_aux += aux_loss.detach()
        n_batches += 1

        if is_main_process() and (n_batches % 50 == 0 or config.dry_run):
            print(
                f"  [{phase_name}] batch {n_batches} | "
                f"total: {loss.item():.6f} | "
                f"masked: {losses['masked_loss'].item():.6f} | "
                f"next: {losses['next_loss'].item():.6f} | "
                f"cross: {losses['cross_modal_loss'].item():.6f} | "
                f"contrastive: {losses['contrastive_loss'].item():.6f} | "
                f"eeg: {eeg_loss.item():.6f} | "
                f"aux: {aux_loss.item():.6f} | "
                f"grad_norm: {grad_norm:.4f}"
            )

        # Dry-run: 1 batch만 실행
        if config.dry_run:
            if is_main_process():
                print(f"  [{phase_name}] dry-run: 1 batch 완료, 종료.")
            break

        # max_batches 제한
        if config.max_batches > 0 and n_batches >= config.max_batches:
            if is_main_process():
                print(f"  [{phase_name}] max_batches={config.max_batches} 도달, 에폭 종료.")
            break

    denom = max(n_batches, 1)
    return {
        "total": (epoch_total / denom).item(),
        "masked_loss": (epoch_masked / denom).item(),
        "next_loss": (epoch_next / denom).item(),
        "cross_modal_loss": (epoch_cross / denom).item(),
        "contrastive_loss": (epoch_contrastive / denom).item(),
        "eeg_loss": (epoch_eeg / denom).item(),
        "aux_loss": (epoch_aux / denom).item(),
    }


@torch.no_grad()
def validate_v2(
    model: nn.Module,
    dataloader,
    criterion: CombinedLoss,
    config: TrainConfig,
    device: torch.device,
    phase_name: str,
) -> dict[str, float]:
    """V2 Validation 루프. train_one_epoch_v2()과 동일한 loss 계산, backward 없이.

    DDP 환경에서는 unwrapped 모델로 forward하여 rank별 배치 수
    불일치로 인한 데드락을 방지한다.
    """
    model.eval()
    # DDP wrapper를 벗겨서 forward — validation에서는 gradient sync 불필요
    raw_model = model.module if isinstance(model, DDP) else model

    epoch_total = 0.0
    epoch_masked = 0.0
    epoch_next = 0.0
    epoch_cross = 0.0
    epoch_contrastive = 0.0
    epoch_eeg = 0.0
    epoch_aux = 0.0
    n_batches = 0

    enable_next = config.beta > 0
    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for batch in dataloader:
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

        H = 1
        if enable_next:
            H = random.randint(1, config.model_config.max_horizon)

        task = "both" if enable_next else "masked"

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = raw_model(batch, task=task, horizon=H)

            reconstructed = out["reconstructed"]
            cross_pred = out["cross_pred"]
            patch_mask = out["patch_mask"]
            time_id = out["time_id"]
            next_pred = out.get("next_pred")

            # V2 전용 출력
            eeg_reconstructed = out.get("eeg_reconstructed")
            eeg_recon_target = out.get("eeg_recon_target")
            eeg_mask = out.get("eeg_mask")

            pred_mask = create_patch_mask(
                patch_mask,
                mask_ratio=config.mask_ratio,
                patch_variate_id=out["patch_variate_id"] if config.variate_mask_prob > 0 else None,
                variate_mask_prob=config.variate_mask_prob,
            )

            P = raw_model.patch_size
            normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
            B, L = normalized.shape
            N = L // P
            original_patches = normalized.reshape(B, N, P)

            # EEG 패치 제외
            if eeg_mask is not None:
                non_eeg_pred_mask = pred_mask & ~eeg_mask
            else:
                non_eeg_pred_mask = pred_mask

            contrastive_z = out.get("contrastive_z")

            needs_time_id = config.gamma > 0 or config.delta > 0
            losses = criterion(
                reconstructed=reconstructed,
                next_pred=next_pred,
                original_patches=original_patches,
                pred_mask=non_eeg_pred_mask,
                patch_mask=patch_mask,
                patch_sample_id=out["patch_sample_id"],
                patch_variate_id=out["patch_variate_id"],
                horizon=H,
                cross_pred=cross_pred if config.gamma > 0 else None,
                time_id=time_id if needs_time_id else None,
                contrastive_z=contrastive_z if config.delta > 0 else None,
            )

            # EEG 전용 loss
            eeg_loss = reconstructed.new_tensor(0.0)
            if eeg_mask is not None and eeg_reconstructed is not None:
                eeg_pred_mask = pred_mask & eeg_mask
                if eeg_pred_mask.any():
                    eeg_loss = torch.nn.functional.mse_loss(
                        eeg_reconstructed[eeg_pred_mask],
                        eeg_recon_target[eeg_pred_mask],
                    )

            total_loss = losses["total"] + eeg_loss

        # MoE auxiliary loss 수집 (로깅 전용, backward 없음)
        aux_loss = 0.0
        for layer in raw_model.encoder.layers:
            if hasattr(layer.ffn, "aux_loss") and layer.ffn.aux_loss is not None:
                aux_loss += layer.ffn.aux_loss.item()

        if not torch.isfinite(total_loss):
            continue

        epoch_total += total_loss.item()
        epoch_masked += losses["masked_loss"].item()
        epoch_next += losses["next_loss"].item()
        epoch_cross += losses["cross_modal_loss"].item()
        epoch_contrastive += losses["contrastive_loss"].item()
        epoch_eeg += eeg_loss.item()
        epoch_aux += aux_loss
        n_batches += 1

        if config.max_batches > 0 and n_batches >= config.max_batches:
            break

    model.train()
    denom = max(n_batches, 1)
    return {
        "total": epoch_total / denom,
        "masked_loss": epoch_masked / denom,
        "next_loss": epoch_next / denom,
        "cross_modal_loss": epoch_cross / denom,
        "contrastive_loss": epoch_contrastive / denom,
        "eeg_loss": epoch_eeg / denom,
        "aux_loss": epoch_aux / denom,
    }


# ── Early Stopping ────────────────────────────────────────────


class EarlyStopping:
    """Validation loss 기반 조기 종료.

    Parameters
    ----------
    patience:
        개선 없이 허용하는 에폭 수.
    min_delta:
        개선으로 인정하는 최소 감소량.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0

    def step(self, val_loss: float) -> bool:
        """val_loss를 기록하고 학습을 중단해야 하면 True를 반환한다."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── CSV 로깅 ─────────────────────────────────────────────────


class CSVLogger:
    """에폭별 학습 메트릭을 CSV 파일에 기록한다.

    Parameters
    ----------
    path:
        CSV 파일 경로. 디렉토리가 없으면 자동 생성.
    """

    COLUMNS = [
        "epoch", "phase",
        "train_total", "train_masked", "train_next", "train_cross", "train_contrastive",
        "val_total", "val_masked", "val_next", "val_cross", "val_contrastive",
        "lr", "epoch_sec",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.COLUMNS)

    def log(
        self,
        epoch: int,
        phase: str,
        train_losses: dict[str, float],
        val_losses: dict[str, float] | None,
        lr: float,
        epoch_sec: float,
    ) -> None:
        """1에폭 결과를 CSV에 추가한다."""
        row = [
            epoch, phase,
            train_losses["total"],
            train_losses["masked_loss"],
            train_losses["next_loss"],
            train_losses["cross_modal_loss"],
            train_losses["contrastive_loss"],
            val_losses["total"] if val_losses else "",
            val_losses["masked_loss"] if val_losses else "",
            val_losses["next_loss"] if val_losses else "",
            val_losses["cross_modal_loss"] if val_losses else "",
            val_losses["contrastive_loss"] if val_losses else "",
            lr,
            f"{epoch_sec:.1f}",
        ]
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


# ── 유틸리티 ────────────────────────────────────────────────────


def create_scaler(config: TrainConfig, device: torch.device) -> torch.amp.GradScaler | None:
    """AMP가 활성이고 CUDA 디바이스일 때 GradScaler를 생성한다."""
    if config.use_amp and device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


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


def setup_ddp(rank: int, world_size: int) -> None:
    """DDP 프로세스 그룹을 초기화한다."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """DDP 프로세스 그룹을 종료한다."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """현재 프로세스가 rank 0인지 (또는 단일 GPU인지) 반환한다."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


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


def resolve_output_dir(config: TrainConfig) -> Path:
    """exp_name이 설정되어 있으면 output_dir/exp_name/ 경로를 반환한다."""
    base = Path(config.output_dir)
    if config.exp_name:
        return base / config.exp_name
    return base


def save_experiment_info(config: TrainConfig, output_dir: Path, phase_name: str, extra_info: dict[str, Any] | None = None) -> None:
    """실험 정보를 experiment_info.txt 와 config.yaml로 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # config.yaml 저장
    config.to_yaml(output_dir / "config.yaml")

    # experiment_info.txt 저장
    mc = config.model_config
    lines = [
        f"# Experiment: {config.exp_name or phase_name}",
        f"# Phase: {phase_name}",
        f"",
        f"[Model]",
        f"d_model        = {mc.d_model}",
        f"num_layers     = {mc.num_layers}",
        f"patch_size     = {mc.patch_size}",
        f"num_heads      = {mc.num_heads}",
        f"num_groups     = {mc.num_groups}",
        f"use_glu        = {mc.use_glu}",
        f"use_moe        = {mc.use_moe}",
        f"use_rope       = {mc.use_rope}",
        f"use_cnn_stem   = {mc.use_cnn_stem}",
        f"max_horizon    = {mc.max_horizon}",
        f"",
        f"[Training]",
        f"batch_size     = {config.batch_size}",
        f"lr             = {config.lr}",
        f"n_epochs       = {config.n_epochs}",
        f"warmup_epochs  = {config.warmup_epochs}",
        f"mask_ratio     = {config.mask_ratio}",
        f"collate_mode   = {config.collate_mode}",
        f"seed           = {config.seed}",
        f"",
        f"[Loss]",
        f"alpha (masked)      = {config.alpha}",
        f"beta  (next-pred)   = {config.beta}",
        f"gamma (cross-modal) = {config.gamma}",
        f"delta (contrastive) = {config.delta}",
        f"variate_mask_prob   = {config.variate_mask_prob}",
        f"",
        f"[Data]",
        f"processed_dir  = {config.processed_dir}",
        f"signal_types   = {config.signal_types}",
        f"max_subjects   = {config.max_subjects}",
        f"window_seconds = {config.window_seconds}",
    ]

    if extra_info:
        lines.append("")
        lines.append("[Extra]")
        for k, v in extra_info.items():
            lines.append(f"{k:15s}= {v}")

    info_path = output_dir / "experiment_info.txt"
    info_path.write_text("\n".join(lines), encoding="utf-8")


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainConfig,
    phase_name: str,
    loss: float,
    output_dir: Path,
    tag: str = "",
) -> Path:
    """학습 checkpoint를 저장하고 경로를 반환한다."""
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    filename = f"checkpoint_{phase_name}_epoch{epoch:03d}{suffix}.pt"
    path = ckpt_dir / filename
    save_checkpoint(
        path, model, optimizer=optimizer, epoch=epoch,
        config=get_model_config(config),
        phase=phase_name, loss=loss,
    )
    return path
