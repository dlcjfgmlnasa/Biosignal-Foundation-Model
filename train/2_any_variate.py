# -*- coding:utf-8 -*-
from __future__ import annotations

"""Phase 2: Any-Variate Training (Cross-Modal).

Phase 1 checkpoint를 로드하여, 다변량 세션에서 cross-modal 학습을 수행한다.
DDP 지원, Block Masking, Horizon Curriculum 포함.

Usage (single GPU):
    python -m train.2_any_variate --resume outputs/phase1/base/checkpoints/best.pt

Usage (multi GPU -- DDP):
    torchrun --nproc_per_node=2 launch_phase2.py --resume outputs/phase1/base/checkpoints/best.pt
"""
import argparse
import gc
import os
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data import BiosignalDataset, create_dataloader
from loss.criterion import CombinedLoss
from model import BiosignalFoundationModel, ModelConfig
from model.checkpoint import load_checkpoint
from .train_utils import (
    CSVLogger,
    EarlyStopping,
    TrainConfig,
    cleanup_ddp,
    create_scaler,
    create_scheduler,
    is_main_process,
    load_manifest_from_processed,
    resolve_device,
    resolve_output_dir,
    save_experiment_info,
    save_training_checkpoint,
    set_seed,
    setup_ddp,
    split_manifest_by_subject,
    train_one_epoch,
    validate,
)
from .visualize import save_reconstruction_figure, save_next_pred_figure
from .visualize_phase2 import save_cross_modal_figure


def find_phase1_checkpoint(search_dirs: list[str] | None = None) -> Path | None:
    """Phase 1 best/final checkpoint를 자동 탐색한다."""
    if search_dirs is None:
        search_dirs = [
            "outputs/phase1",
            "outputs/phase1_ci",
            "outputs/v2_phase1_ci",
        ]

    for base_str in search_dirs:
        base = Path(base_str)
        # base 하위의 모든 서브디렉토리도 탐색
        for d in [base] + sorted(base.glob("*/checkpoints")) + sorted(base.glob("*")):
            if not d.exists() or not d.is_dir():
                continue
            best = sorted(d.glob("*_best.pt"))
            if best:
                return best[-1]
            final = sorted(d.glob("*_final.pt"))
            if final:
                return final[-1]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2: Any-Variate Training (Cross-Modal)"
    )

    # Config
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file path. CLI args override YAML values.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="1 batch만 실행 후 종료 (OOM/NaN 검증용)"
    )

    # Resume
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Phase 1 checkpoint path. Auto-search if not specified.",
    )

    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--d_model", type=int, default=64)
    g.add_argument("--num_layers", type=int, default=2)
    g.add_argument("--patch_size", type=int, default=100)
    g.add_argument("--num_heads", type=int, default=None)
    g.add_argument("--num_groups", type=int, default=None)
    g.add_argument("--use_glu", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--use_moe", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument(
        "--use_var_attn_bias", action=argparse.BooleanOptionalAction, default=True
    )
    g.add_argument("--dropout_p", type=float, default=0.0)
    g.add_argument("--next_block_size", type=int, default=4)

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--data_dir", type=str, default="datasets/processed")
    g.add_argument("--signal_types", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    g.add_argument("--max_subjects", type=int, default=None)
    g.add_argument("--window_seconds", type=float, default=30.0)
    g.add_argument("--max_length", type=int, default=50000)
    g.add_argument("--cache_size", type=int, default=16)
    g.add_argument("--crop_ratio_min", type=float, default=0.0)
    g.add_argument("--crop_ratio_max", type=float, default=0.0)

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--batch_size", type=int, default=4)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--n_epochs", type=int, default=30)
    g.add_argument("--warmup_epochs", type=int, default=3)
    g.add_argument("--min_lr_ratio", type=float, default=0.1)
    g.add_argument("--mask_ratio", type=float, default=0.15)
    g.add_argument("--gradient_clip", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=42)

    # Loss weights
    g.add_argument(
        "--alpha", type=float, default=0.7, help="Masked reconstruction weight"
    )
    g.add_argument(
        "--beta", type=float, default=0.3, help="Next-patch prediction weight"
    )
    g.add_argument("--gamma", type=float, default=1.0, help="Cross-modal loss weight")
    g.add_argument("--delta", type=float, default=0.1, help="Contrastive loss weight")
    g.add_argument("--contrastive_proj_dim", type=int, default=128)
    g.add_argument("--contrastive_temperature", type=float, default=0.07)

    # Masking
    g.add_argument(
        "--variate_mask_prob",
        type=float,
        default=0.3,
        help="Variate-level masking probability",
    )
    g.add_argument(
        "--variate_drop_prob",
        type=float,
        default=0.1,
        help="Complete variate dropout probability (zero-shot cross-modal generation용)",
    )
    g.add_argument(
        "--block_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block masking (continuous patch blocks)",
    )
    g.add_argument("--block_size_min", type=int, default=2)
    g.add_argument("--block_size_max", type=int, default=4)

    # Validation & Early Stopping
    g.add_argument("--val_ratio", type=float, default=0.2)
    g.add_argument("--patience", type=int, default=10)

    # System
    g = p.add_argument_group("System")
    g.add_argument("--use_amp", action="store_true")
    g.add_argument("--device", type=str, default="auto")
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--output_dir", type=str, default="outputs/phase2_any_variate")
    g.add_argument("--checkpoint_every", type=int, default=5)
    g.add_argument("--viz_every", type=int, default=5)
    g.add_argument("--exp_name", type=str, default="")

    return p.parse_args()


def main():
    args = parse_args()

    # ── DDP detection ──
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = local_rank >= 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if use_ddp:
        setup_ddp(local_rank, world_size)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = resolve_device(args.device)

    rank0 = is_main_process()

    # ── Config ──
    if args.config:
        config = TrainConfig.from_yaml(args.config)
        config.collate_mode = "any_variate"
        if args.dry_run:
            config.max_batches = 1
            config.n_epochs = 1
        if rank0:
            print(f"Config loaded from: {args.config}")
    else:
        model_config = ModelConfig(
            d_model=args.d_model,
            num_layers=args.num_layers,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            num_groups=args.num_groups,
            use_glu=args.use_glu,
            use_moe=args.use_moe,
            use_rope=args.use_rope,
            use_var_attn_bias=args.use_var_attn_bias,
            dropout_p=args.dropout_p,
            next_block_size=args.next_block_size,
            contrastive_proj_dim=args.contrastive_proj_dim,
        )

        config = TrainConfig(
            model_config=model_config,
            data_dir=args.data_dir,
            signal_types=args.signal_types,
            max_subjects=args.max_subjects,
            window_seconds=args.window_seconds,
            max_length=args.max_length,
            cache_size=args.cache_size,
            crop_ratio_min=args.crop_ratio_min,
            crop_ratio_max=args.crop_ratio_max,
            batch_size=args.batch_size,
            lr=args.lr,
            n_epochs=args.n_epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr_ratio=args.min_lr_ratio,
            mask_ratio=args.mask_ratio,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
            collate_mode="any_variate",
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            contrastive_temperature=args.contrastive_temperature,
            variate_mask_prob=args.variate_mask_prob,
            variate_drop_prob=args.variate_drop_prob,
            block_mask=args.block_mask,
            block_size_min=args.block_size_min,
            block_size_max=args.block_size_max,
            val_ratio=args.val_ratio,
            patience=args.patience,
            use_amp=args.use_amp,
            exp_name=args.exp_name,
            device=args.device,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            checkpoint_every=args.checkpoint_every,
        )

    set_seed(config.seed + (local_rank if use_ddp else 0))
    output_dir = resolve_output_dir(config)
    if rank0:
        output_dir.mkdir(parents=True, exist_ok=True)

    if rank0:
        print(f"{'=' * 60}")
        print("Phase 2: Any-Variate Training (Cross-Modal)")
        if config.exp_name:
            print(f"Experiment: {config.exp_name}")
        print(f"Device: {device}" + (f" (DDP: {world_size} GPUs)" if use_ddp else ""))
        print(f"{'=' * 60}")

    # ── Phase 1 Checkpoint ──
    ckpt_path = args.resume
    if ckpt_path is None:
        found = find_phase1_checkpoint()
        if found is None:
            if rank0:
                print("ERROR: Phase 1 checkpoint not found.")
                print("  Run Phase 1 first, or specify --resume path.")
            if use_ddp:
                cleanup_ddp()
            return
        ckpt_path = str(found)
    if rank0:
        print(f"Loading Phase 1 checkpoint: {ckpt_path}")

    # Model + checkpoint load
    # Phase 1 checkpoint의 config를 우선 사용하여 아키텍처 불일치 방지
    ckpt_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" in ckpt_state:
        ckpt_model_config = ModelConfig.from_dict(ckpt_state["config"])

        # contrastive_proj_dim: 덮어쓰기 안전 (Phase 1: 0 → Phase 2: 128, 새 모듈 random init)
        ckpt_model_config.contrastive_proj_dim = (
            config.model_config.contrastive_proj_dim
        )

        # next_block_size: K가 다르면 BlockNextHead.heads shape mismatch 발생 →
        # strict=False로 로드 시 일부 heads만 로드되고 나머지는 random init (silent).
        # 안전을 위해 값이 다르면 경고, 같으면 no-op(ckpt 값 유지).
        if (
            ckpt_model_config.next_block_size
            != config.model_config.next_block_size
        ):
            if rank0:
                print(
                    f"  ⚠️  next_block_size override: ckpt="
                    f"{ckpt_model_config.next_block_size} → "
                    f"config={config.model_config.next_block_size} "
                    "(일부 BlockNextHead.heads가 재초기화됩니다)"
                )
            ckpt_model_config.next_block_size = (
                config.model_config.next_block_size
            )

        config.model_config = ckpt_model_config
        if rank0:
            print(
                f"  Model config loaded from checkpoint (patch_size="
                f"{ckpt_model_config.patch_size}, "
                f"next_block_size K={ckpt_model_config.next_block_size})"
            )

    model = BiosignalFoundationModel.from_config(config.model_config)
    state = load_checkpoint(ckpt_path, model, device=device)
    model.to(device)

    phase1_epoch = state.get("epoch", "?")
    phase1_loss = state.get("loss", "?")
    if rank0:
        print(f"  Phase 1 epoch: {phase1_epoch}, loss: {phase1_loss}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {total_params:,}")

        save_experiment_info(
            config,
            output_dir,
            phase_name="Phase2_AV",
            extra_info={
                "phase1_ckpt": ckpt_path,
                "phase1_epoch": str(phase1_epoch),
                "phase1_loss": str(phase1_loss),
                "next_block_size": str(config.model_config.next_block_size),
            },
        )

    # ── Data ──
    manifest = load_manifest_from_processed(
        config.data_dir,
        signal_types=config.signal_types,
        max_subjects=config.max_subjects,
    )
    if rank0:
        print(f"Loaded {len(manifest)} recordings")

    val_dataloader = None
    sampler = None
    val_sampler = None

    if config.val_ratio > 0:
        train_manifest, val_manifest = split_manifest_by_subject(
            manifest,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
        if rank0:
            print(
                f"Train/Val split: {len(train_manifest)} train, {len(val_manifest)} val"
            )
    else:
        train_manifest = manifest

    crop_range = None
    if config.crop_ratio_min > 0 and config.crop_ratio_max > 0:
        crop_range = (config.crop_ratio_min, config.crop_ratio_max)
        if rank0:
            print(f"Random crop: {crop_range}")

    dataset = BiosignalDataset(
        train_manifest,
        window_seconds=config.window_seconds,
        cache_size=config.cache_size,
        crop_ratio_range=crop_range,
        patch_size=config.model_config.patch_size,
    )
    if rank0:
        print(f"Train dataset: {len(dataset)} windows")

    dataloader = create_dataloader(
        dataset,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_mode=config.collate_mode,
        patch_size=config.model_config.patch_size,
    )
    sampler = None  # any_variate 모드: GroupedBatchSampler가 셔플 담당
    if rank0:
        print(f"Train batches per epoch: {len(dataloader)}")

    if config.val_ratio > 0:
        val_dataset = BiosignalDataset(
            val_manifest,
            window_seconds=config.window_seconds,
            cache_size=config.cache_size,
            patch_size=config.model_config.patch_size,
        )
        val_dataloader = create_dataloader(
            val_dataset,
            max_length=config.max_length,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_mode=config.collate_mode,
            patch_size=config.model_config.patch_size,
        )
        if rank0:
            print(
                f"Val dataset: {len(val_dataset)} windows, {len(val_dataloader)} batches"
            )

    # ── DDP wrap ──
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if use_ddp else model

    # ── Optimizer, Scheduler, Criterion ──
    criterion = CombinedLoss(
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        delta=config.delta,
        peak_alpha=config.peak_alpha,
        lambda_spec=config.lambda_spec,
        spec_n_ffts=config.spec_n_ffts,
        contrastive_temperature=config.contrastive_temperature,
        learnable_temperature=config.learnable_temperature,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config.lr,
    )
    scheduler = create_scheduler(optimizer, config)
    scaler = create_scaler(config, device)
    if rank0 and scaler is not None:
        print("AMP enabled (GradScaler)")

    # ── Visualization cache ──
    viz_every = getattr(args, "viz_every", 5)
    if hasattr(config, "viz_every") and config.viz_every is not None:
        viz_every = config.viz_every
    viz_batches = None
    viz_recon_dir = None
    viz_np_dir = None
    viz_cross_dir = None
    if viz_every > 0 and val_dataloader is not None and rank0:
        viz_iter = iter(val_dataloader)
        viz_batches = []
        for _ in range(min(10, len(val_dataloader))):
            try:
                viz_batches.append(next(viz_iter))
            except StopIteration:
                break
        del viz_iter
        viz_dir = output_dir / "figures"
        viz_recon_dir = viz_dir / "recon"
        viz_np_dir = viz_dir / "next_pred"
        viz_cross_dir = viz_dir / "cross_modal"
        for d in [viz_recon_dir, viz_np_dir, viz_cross_dir]:
            d.mkdir(parents=True, exist_ok=True)
        n_types = len(
            {
                int(b.signal_types[j])
                for b in viz_batches
                for j in range(len(b.signal_types))
            }
        )
        print(
            f"Visualization every {viz_every} epochs -> {viz_dir}"
            f"  ({len(viz_batches)} batches, {n_types} signal types)"
        )

    # ── Training loop ──
    best_loss = float("inf")
    early_stopper = (
        EarlyStopping(patience=config.patience) if config.patience > 0 else None
    )
    csv_logger = CSVLogger(output_dir / "training_log.csv") if rank0 else None

    if rank0:
        print(f"\nStarting training: {config.n_epochs} epochs")
        print(
            f"  alpha={config.alpha}, beta={config.beta}, gamma={config.gamma}, delta={config.delta}"
        )
        print(
            f"  next_block_size={config.model_config.next_block_size}, mask_ratio={config.mask_ratio}"
        )
        print(f"  variate_mask_prob={config.variate_mask_prob}")
        print(
            f"  block_mask={config.block_mask}, block_size=[{config.block_size_min}, {config.block_size_max}]"
        )
        print(f"  warmup_epochs={config.warmup_epochs}")
        print(f"  collate_mode={config.collate_mode}")
        if val_dataloader is not None:
            print(f"  val_ratio={config.val_ratio}, patience={config.patience}")
        print(f"{'=' * 60}")

    for epoch in range(config.n_epochs):
        # DDP: sampler epoch sync
        if sampler is not None:
            sampler.set_epoch(epoch)
        if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)

        epoch_start = time.time()
        losses = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            config=config,
            device=device,
            epoch=epoch,
            phase_name="Phase2_AV",
            scaler=scaler,
        )
        scheduler.step()

        # Validation
        val_losses = None
        if val_dataloader is not None:
            val_losses = validate(
                model,
                val_dataloader,
                criterion,
                config=config,
                device=device,
                phase_name="Phase2_AV",
                epoch=epoch,
            )
        epoch_sec = time.time() - epoch_start

        if rank0:
            current_lr = optimizer.param_groups[0]["lr"]
            line = (
                f"Epoch {epoch:3d} | "
                f"train: {losses['total']:.6f} | "
                f"masked: {losses['masked_loss']:.6f} | "
                f"next: {losses['next_loss']:.6f} | "
                f"cross: {losses['cross_modal_loss']:.6f} | "
                f"contrastive: {losses['contrastive_loss']:.6f}"
            )
            if val_losses is not None:
                line += f" | val: {val_losses['total']:.6f}"
            line += f" | LR: {current_lr:.2e} | {epoch_sec:.0f}s"
            print(line)

            csv_logger.log(
                epoch, "Phase2_AV", losses, val_losses, current_lr, epoch_sec
            )

        # Visualization
        if (
            rank0
            and viz_batches is not None
            and (epoch % viz_every == 0 or epoch == config.n_epochs - 1)
        ):
            viz_model = raw_model
            fig_path = save_reconstruction_figure(
                viz_model,
                viz_batches,
                epoch=epoch,
                output_dir=viz_recon_dir,
                mask_ratio=config.mask_ratio,
                device=device,
                block_mask=config.block_mask,
                block_size_min=config.block_size_min,
                block_size_max=config.block_size_max,
            )
            print(f"  -> Reconstruction figure: {fig_path}")
            np_path = save_next_pred_figure(
                viz_model,
                viz_batches,
                epoch=epoch,
                output_dir=viz_np_dir,
                device=device,
            )
            print(f"  -> Next-pred figure: {np_path}")

            # Phase 2 전용 시각화
            cross_path = save_cross_modal_figure(
                viz_model,
                viz_batches,
                epoch=epoch,
                output_dir=viz_cross_dir,
                mask_ratio=config.mask_ratio,
                device=device,
                block_mask=config.block_mask,
                block_size_min=config.block_size_min,
                block_size_max=config.block_size_max,
                variate_mask_prob=config.variate_mask_prob,
            )
            if cross_path.exists():
                print(f"  -> Cross-modal figure: {cross_path}")
            else:
                print(
                    f"  -> Cross-modal figure: SKIP (no multi-variate pairs found in viz batch)"
                )

        # Best model
        if rank0:
            track_loss = (
                val_losses["total"] if val_losses is not None else losses["total"]
            )
            if track_loss < best_loss:
                best_loss = track_loss
                save_model = raw_model
                path = save_training_checkpoint(
                    save_model,
                    optimizer,
                    epoch,
                    config,
                    phase_name="phase2_av",
                    loss=best_loss,
                    output_dir=output_dir,
                    tag="best",
                )
                print(f"  -> Best model: {path}")

            # Periodic checkpoint
            if (epoch + 1) % config.checkpoint_every == 0:
                save_model = raw_model
                save_training_checkpoint(
                    save_model,
                    optimizer,
                    epoch,
                    config,
                    phase_name="phase2_av",
                    loss=losses["total"],
                    output_dir=output_dir,
                )

        # Early Stopping
        if early_stopper is not None and val_losses is not None:
            if early_stopper.step(val_losses["total"]):
                if rank0:
                    print(
                        f"\n  Early stopping at epoch {epoch} "
                        f"(patience={config.patience}, best_val={early_stopper.best_loss:.6f})"
                    )
                break

    # Final checkpoint
    if rank0:
        save_model = raw_model
        final_path = save_training_checkpoint(
            save_model,
            optimizer,
            epoch,
            config,
            phase_name="phase2_av",
            loss=losses["total"],
            output_dir=output_dir,
            tag="final",
        )
        print(f"\n{'=' * 60}")
        print(f"Phase 2 complete. Final train loss: {losses['total']:.6f}")
        if val_losses is not None:
            print(f"Final val loss: {val_losses['total']:.6f}")
        print(f"Best {'val' if val_dataloader else 'train'} loss: {best_loss:.6f}")
        print(f"Final checkpoint: {final_path}")
        print(f"{'=' * 60}")

    # Cleanup
    del dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
