# -*- coding:utf-8 -*-
from __future__ import annotations

"""Phase 1: Channel-Independent 사전학습.

collate_mode="ci"로 각 채널을 독립적으로 학습한다.
모든 신호를 raw patch MSE로 복원.

Usage (단일 GPU)
-----
    python -m train.1_channel_independency --device cuda:0

Usage (멀티 GPU -- DDP)
-----
    torchrun --nproc_per_node=2 -m train.1_channel_independency
"""
import argparse
import gc
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import BiosignalDataset, create_dataloader
from data.sampler import RecordingLocalitySampler
from loss.criterion import CombinedLoss
from model import BiosignalFoundationModel, ModelConfig
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
from model.checkpoint import load_checkpoint
from .visualize import save_reconstruction_figure, save_next_pred_figure


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: Channel-Independent Pre-training")

    # Config file + dry-run
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file path. CLI args override YAML values.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="1 batch만 실행 후 종료 (OOM/NaN 검증용)"
    )
    p.add_argument(
        "--resume", type=str, default=None, help="학습 재개할 checkpoint 경로 (.pt)"
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
    g.add_argument("--next_block_size", type=int, default=5)

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--data_dir", type=str, default="datasets/processed")
    g.add_argument(
        "--signal_types",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Signal type IDs (0=ECG, 1=ABP, 2=PPG, 3=CVP, 4=CO2, 5=AWP, 6=PAP, 7=ICP)",
    )
    g.add_argument("--max_subjects", type=int, default=None)
    g.add_argument("--window_seconds", type=float, default=30.0)
    g.add_argument("--max_length", type=int, default=50000)
    g.add_argument("--cache_size", type=int, default=16)
    g.add_argument(
        "--crop_ratio_min",
        type=float,
        default=0.0,
        help="Random crop 최소 비율 (0=비활성)",
    )
    g.add_argument(
        "--crop_ratio_max",
        type=float,
        default=0.0,
        help="Random crop 최대 비율 (0=비활성)",
    )

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--batch_size", type=int, default=16)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--n_epochs", type=int, default=70)
    g.add_argument("--warmup_epochs", type=int, default=5)
    g.add_argument("--min_lr_ratio", type=float, default=0.1)
    g.add_argument("--mask_ratio", type=float, default=0.15)
    g.add_argument("--gradient_clip", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument(
        "--alpha", type=float, default=1.0, help="Masked reconstruction weight"
    )
    g.add_argument(
        "--beta", type=float, default=1.0, help="Next-patch prediction weight"
    )
    g.add_argument(
        "--delta", type=float, default=0.0, help="Contrastive loss weight (0=disabled)"
    )
    g.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation 비율 (subject 단위, 0=비활성)",
    )
    g.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (0=비활성)"
    )

    # System
    g = p.add_argument_group("System")
    g.add_argument(
        "--use_amp", action="store_true", help="AMP (Automatic Mixed Precision) 활성"
    )
    g.add_argument("--device", type=str, default="auto")
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--output_dir", type=str, default="outputs/phase1_ci")
    g.add_argument("--checkpoint_every", type=int, default=10)
    g.add_argument(
        "--max_batches", type=int, default=0, help="에폭당 최대 배치 수 (0=무제한)"
    )
    g.add_argument(
        "--viz_every",
        type=int,
        default=5,
        help="N 에폭마다 reconstruction 시각화 저장 (0=비활성)",
    )
    g.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="실험 이름 (output_dir 하위 서브디렉토리)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # ── DDP 감지 (torchrun이 설정한 환경 변수) ──
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = local_rank >= 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if use_ddp:
        setup_ddp(local_rank, world_size)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = resolve_device(args.device)

    rank0 = is_main_process()

    # ── Phase 1 설정 ──
    if args.config:
        # YAML config 파일에서 로드
        config = TrainConfig.from_yaml(args.config)
        # Phase 1 CI 고정값
        config.collate_mode = "ci"
        config.variate_mask_prob = 0.0
        if rank0:
            print(f"Config loaded from: {args.config}")
    else:
        # CLI 인자 기반 구성 (하위 호환성)
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
            contrastive_proj_dim=0,
        )

        config = TrainConfig(
            model_config=model_config,
            # 데이터
            data_dir=args.data_dir,
            signal_types=args.signal_types,
            max_subjects=args.max_subjects,
            window_seconds=args.window_seconds,
            max_length=args.max_length,
            cache_size=args.cache_size,
            crop_ratio_min=args.crop_ratio_min,
            crop_ratio_max=args.crop_ratio_max,
            # 학습
            batch_size=args.batch_size,
            lr=args.lr,
            n_epochs=args.n_epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr_ratio=args.min_lr_ratio,
            mask_ratio=args.mask_ratio,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
            collate_mode="ci",
            # Loss
            alpha=args.alpha,
            beta=args.beta,
            gamma=0.0,
            delta=args.delta,
            # Phase 1
            variate_mask_prob=0.0,
            # Validation & Early Stopping
            val_ratio=args.val_ratio,
            patience=args.patience,
            # Mixed Precision
            use_amp=args.use_amp,
            # 실험 관리
            exp_name=args.exp_name,
            # 시스템
            device=args.device,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            checkpoint_every=args.checkpoint_every,
            max_batches=args.max_batches,
        )

    # ── Dry-run 모드 ──
    if args.dry_run:
        config.n_epochs = 1
        config.max_batches = 1
        config.dry_run = True
        if rank0:
            print("[dry-run] 1 epoch, 1 batch")

    set_seed(config.seed + (local_rank if use_ddp else 0))
    output_dir = resolve_output_dir(config)
    if rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_experiment_info(config, output_dir, phase_name="Phase1_CI")

    if rank0:
        print(f"{'=' * 60}")
        print("Phase 1: Channel-Independent Pre-training")
        if config.exp_name:
            print(f"Experiment: {config.exp_name}")
        print(f"Device: {device}" + (f" (DDP: {world_size} GPUs)" if use_ddp else ""))
        print(f"{'=' * 60}")

    # ── 데이터 로딩 ──
    manifest = load_manifest_from_processed(
        config.data_dir,
        signal_types=config.signal_types,
        max_subjects=config.max_subjects,
    )
    if rank0:
        print(f"Loaded {len(manifest)} recordings")

    # Train/Val split (subject 단위)
    val_dataloader = None
    if config.val_ratio > 0:
        train_manifest, val_manifest = split_manifest_by_subject(
            manifest,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
        if rank0:
            print(
                f"Train/Val split: {len(train_manifest)} train, {len(val_manifest)} val recordings"
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
        min_patches=config.min_patches,
        shard_index_path=config.shard_index_path,
        shard_cache_size=config.shard_cache_size,
    )
    if rank0 and config.shard_index_path:
        print(f"  Shard backend ON: {config.shard_index_path} "
              f"(shard_cache_size={config.shard_cache_size})")
    if rank0:
        print(f"Train dataset: {len(dataset)} windows")

    # Recording-locality sampler: 같은 레코딩의 윈도우를 연속 yield하여
    # LRU 캐시 히트율을 극대화한다. 네트워크 디스크 I/O 병목 해소.
    sampler = RecordingLocalitySampler(
        dataset,
        num_replicas=world_size if use_ddp else None,
        rank=local_rank if use_ddp else None,
        shuffle=True,
        seed=config.seed,
    )
    shuffle = False  # sampler가 셔플 담당

    dataloader = create_dataloader(
        dataset,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_mode=config.collate_mode,
        patch_size=config.model_config.patch_size,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        sampler=sampler,
    )
    if rank0:
        print(f"Train batches per epoch: {len(dataloader)}")

    # Validation dataloader
    if config.val_ratio > 0:
        val_dataset = BiosignalDataset(
            val_manifest,
            window_seconds=config.window_seconds,
            cache_size=config.cache_size,
            patch_size=config.model_config.patch_size,
            min_patches=config.min_patches,
            shard_index_path=config.shard_index_path,
            shard_cache_size=config.shard_cache_size,
        )
        val_sampler = RecordingLocalitySampler(
            val_dataset,
            num_replicas=world_size if use_ddp else None,
            rank=local_rank if use_ddp else None,
            shuffle=False,
            seed=config.seed,
        )
        val_dataloader = create_dataloader(
            val_dataset,
            max_length=config.max_length,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_mode=config.collate_mode,
            patch_size=config.model_config.patch_size,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True,
            sampler=val_sampler,
        )
        if rank0:
            print(
                f"Val dataset: {len(val_dataset)} windows, {len(val_dataloader)} batches"
            )

    # ── 모델 (V2) ──
    model = BiosignalFoundationModel.from_config(config.model_config)
    model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if rank0:
        raw_model = model.module if use_ddp else model
        total_params = sum(p.numel() for p in raw_model.parameters())
        print(f"Model params: {total_params:,}")

    # ── Optimizer & Scheduler ──
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

    # ── Resume from checkpoint ──
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        raw_model_for_load = model.module if use_ddp else model
        state = load_checkpoint(
            args.resume, raw_model_for_load, optimizer=optimizer, device=device
        )
        start_epoch = state.get("epoch", 0) + 1
        best_loss = state.get("loss", float("inf"))
        # scheduler를 resume epoch까지 진행 (warning 억제)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch):
                scheduler.step()
        if rank0:
            print(
                f"Resumed from {args.resume} (epoch {state.get('epoch', '?')}, loss {best_loss:.6f})"
            )
            print(
                f"  Continuing from epoch {start_epoch}, LR={optimizer.param_groups[0]['lr']:.6e}"
            )

    # ── 시각화용 배치 캐시 (rank 0만) ──
    viz_every = getattr(args, "viz_every", 5)
    # YAML config에서 viz_every가 설정되어 있으면 우선 사용
    if hasattr(config, "viz_every") and config.viz_every is not None:
        viz_every = config.viz_every
    viz_batches: list | None = None
    viz_dir = None
    if rank0 and viz_every > 0 and val_dataloader is not None:
        # 모든 signal type이 포함될 때까지 배치 수집 (최대 100 배치)
        # cold val shard 로드라 1-3분 걸릴 수 있음 → tqdm으로 진행률 표시
        viz_iter = iter(val_dataloader)
        viz_batches = []
        seen_types: set[int] = set()
        all_types = (
            set(config.signal_types) if hasattr(config, "signal_types") else set()
        )
        max_viz_batches = min(100, len(val_dataloader))
        try:
            from tqdm import tqdm
            pbar = tqdm(
                total=max_viz_batches,
                desc="viz_batches prep (cold val shard load)",
                unit="batch",
            )
        except ImportError:
            pbar = None
        for _ in range(max_viz_batches):
            try:
                b = next(viz_iter)
                viz_batches.append(b)
                for j in range(len(b.signal_types)):
                    seen_types.add(int(b.signal_types[j]))
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        types_seen=f"{len(seen_types)}/{len(all_types) if all_types else '?'}",
                    )
                if all_types and seen_types >= all_types:
                    break
            except StopIteration:
                break
        if pbar is not None:
            pbar.close()
        del viz_iter
        viz_dir = output_dir / "figures"
        viz_recon_dir = viz_dir / "recon"
        viz_np_dir = viz_dir / "next_pred"
        viz_recon_dir.mkdir(parents=True, exist_ok=True)
        viz_np_dir.mkdir(parents=True, exist_ok=True)
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

    # DDP 동기화 — rank 0이 viz_batches fetch 하는 동안 다른 rank는 학습 loop
    # 진입 → 첫 backward all_reduce에서 영원히 대기 (deadlock). barrier로
    # rank 0의 viz prep 완료 대기.
    if use_ddp:
        dist.barrier()

    # ── 학습 루프 ──
    if not args.resume:
        best_loss = float("inf")
    early_stopper = (
        EarlyStopping(patience=config.patience) if config.patience > 0 else None
    )
    csv_logger = CSVLogger(output_dir / "training_log.csv") if rank0 else None
    if rank0:
        print(f"\nStarting training: {config.n_epochs} epochs")
        print(f"  alpha={config.alpha}, beta={config.beta}, gamma={config.gamma}")
        print(
            f"  next_block_size={config.model_config.next_block_size}, mask_ratio={config.mask_ratio}"
        )
        print(f"  warmup_epochs={config.warmup_epochs}")
        if val_dataloader is not None:
            print(f"  val_ratio={config.val_ratio}, patience={config.patience}")
        print(f"{'=' * 60}")

    for epoch in range(start_epoch, config.n_epochs):
        # DDP: sampler 에폭 설정 (셔플링 동기화)
        if sampler is not None:
            sampler.set_epoch(epoch)

        # 에폭 경계 메모리 정리: val/viz 잔여 + fragmentation 해소
        if epoch > start_epoch:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_start = time.time()
        losses = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            config=config,
            device=device,
            epoch=epoch,
            phase_name="Phase1_CI",
            scaler=scaler,
        )
        scheduler.step()

        # ── Validation ──
        val_losses = None
        if val_dataloader is not None:
            val_losses = validate(
                model,
                val_dataloader,
                criterion,
                config=config,
                device=device,
                phase_name="Phase1_CI",
                epoch=epoch,
            )
        epoch_sec = time.time() - epoch_start

        if rank0:
            current_lr = optimizer.param_groups[0]["lr"]
            line = (
                f"Epoch {epoch:3d} | "
                f"train: {losses['total']:.6f} | "
                f"masked: {losses['masked_loss']:.6f} | "
                f"next: {losses['next_loss']:.6f}"
            )
            if val_losses is not None:
                line += f" | val: {val_losses['total']:.6f}"
            line += f" | LR: {current_lr:.2e} | {epoch_sec:.0f}s"
            print(line)

            # CSV 로깅
            if csv_logger is not None:
                csv_logger.log(
                    epoch, "Phase1_CI", losses, val_losses, current_lr, epoch_sec
                )

            # Reconstruction & Next-Pred 시각화
            if viz_batches is not None and (
                epoch % viz_every == 0 or epoch == config.n_epochs - 1
            ):
                viz_model = model.module if use_ddp else model
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
                print(f"  -> Reconstruction figure saved: {fig_path}")
                np_path = save_next_pred_figure(
                    viz_model,
                    viz_batches,
                    epoch=epoch,
                    output_dir=viz_np_dir,
                    device=device,
                )
                print(f"  -> Next-pred figure saved: {np_path}")

            # Best model 저장 (val_loss 기준, 없으면 train_loss)
            track_loss = (
                val_losses["total"] if val_losses is not None else losses["total"]
            )
            if track_loss < best_loss:
                best_loss = track_loss
                save_model = model.module if use_ddp else model
                path = save_training_checkpoint(
                    save_model,
                    optimizer,
                    epoch,
                    config,
                    phase_name="phase1_ci",
                    loss=best_loss,
                    output_dir=output_dir,
                    tag="best",
                )
                print(f"  -> Best model saved: {path}")

            # 주기적 체크포인트
            if (epoch + 1) % config.checkpoint_every == 0:
                save_model = model.module if use_ddp else model
                save_training_checkpoint(
                    save_model,
                    optimizer,
                    epoch,
                    config,
                    phase_name="phase1_ci",
                    loss=losses["total"],
                    output_dir=output_dir,
                )

        # rank0가 viz/checkpoint 저장하는 동안 다른 rank가 다음 epoch 진입 →
        # 첫 backward all_reduce에서 NCCL timeout. barrier로 동기화.
        if use_ddp:
            dist.barrier()

        # ── Early Stopping ──
        if early_stopper is not None and val_losses is not None:
            if early_stopper.step(val_losses["total"]):
                if rank0:
                    print(
                        f"\n  Early stopping at epoch {epoch} "
                        f"(patience={config.patience}, best_val={early_stopper.best_loss:.6f})"
                    )
                break

    # ── 최종 체크포인트 ──
    if rank0:
        save_model = model.module if use_ddp else model
        final_path = save_training_checkpoint(
            save_model,
            optimizer,
            epoch,
            config,
            phase_name="phase1_ci",
            loss=losses["total"],
            output_dir=output_dir,
            tag="final",
        )
        print(f"\n{'=' * 60}")
        print(f"Phase 1 complete. Final train loss: {losses['total']:.6f}")
        if val_losses is not None:
            print(f"Final val loss: {val_losses['total']:.6f}")
        print(f"Best {'val' if val_dataloader else 'train'} loss: {best_loss:.6f}")
        print(f"Final checkpoint: {final_path}")
        print(f"{'=' * 60}")

    # 정리
    del dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
