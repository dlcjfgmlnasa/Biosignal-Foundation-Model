# -*- coding:utf-8 -*-
"""Phase 1: Channel-Independent 사전학습.

각 채널(variate)을 독립적으로 학습하여 단일 신호의 시간적 패턴을 학습한다.
- Masked Patch Modeling (MPM): 마스킹된 패치 복원
- Next-Patch Prediction: 랜덤 horizon으로 미래 패치 예측

Usage (단일 GPU)
-----
    python -m train.1_channel_independency --device cuda:0

Usage (멀티 GPU — DDP)
-----
    torchrun --nproc_per_node=2 -m train.1_channel_independency
"""
import argparse
import gc
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data import BiosignalDataset, create_dataloader
from loss.criterion import CombinedLoss
from model import BiosignalFoundationModel, ModelConfig
from .train_utils import (
    TrainConfig,
    cleanup_ddp,
    create_scheduler,
    get_model_config,
    is_main_process,
    load_manifest_from_processed,
    resolve_device,
    resolve_output_dir,
    save_experiment_info,
    save_training_checkpoint,
    set_seed,
    setup_ddp,
    train_one_epoch,
)
from .visualize import save_reconstruction_figure


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: Channel-Independent Pre-training")

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
    g.add_argument("--use_var_attn_bias", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--dropout_p", type=float, default=0.0)
    g.add_argument("--max_horizon", type=int, default=5)
    g.add_argument("--use_cnn_stem", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--stem_hidden_channels", type=int, default=64)
    g.add_argument("--stem_num_layers", type=int, default=3)
    g.add_argument("--stem_kernel_size", type=int, default=3)

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--processed_dir", type=str, default="datasets/processed")
    g.add_argument("--signal_types", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6],
                   help="Signal type IDs (0=ECG, 1=ABP, 2=EEG, 3=PPG, 4=CVP, 5=CO2, 6=AWP)")
    g.add_argument("--max_subjects", type=int, default=None)
    g.add_argument("--window_seconds", type=float, default=30.0)
    g.add_argument("--max_length", type=int, default=50000)
    g.add_argument("--cache_size", type=int, default=16)

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
    g.add_argument("--alpha", type=float, default=1.0, help="Masked reconstruction weight")
    g.add_argument("--beta", type=float, default=1.0, help="Next-patch prediction weight")
    g.add_argument("--delta", type=float, default=0.0, help="Contrastive loss weight (0=disabled)")

    # System
    g = p.add_argument_group("System")
    g.add_argument("--device", type=str, default="auto")
    g.add_argument("--num_workers", type=int, default=0)
    g.add_argument("--output_dir", type=str, default="outputs/phase1_ci")
    g.add_argument("--checkpoint_every", type=int, default=10)
    g.add_argument("--max_batches", type=int, default=0,
                   help="에폭당 최대 배치 수 (0=무제한)")
    g.add_argument("--viz_every", type=int, default=5,
                   help="N 에폭마다 reconstruction 시각화 저장 (0=비활성)")

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
        max_horizon=args.max_horizon,
        use_cnn_stem=args.use_cnn_stem,
        stem_hidden_channels=args.stem_hidden_channels,
        stem_num_layers=args.stem_num_layers,
        stem_kernel_size=args.stem_kernel_size,
    )

    config = TrainConfig(
        model_config=model_config,

        # 데이터
        processed_dir=args.processed_dir,
        signal_types=args.signal_types,
        max_subjects=args.max_subjects,
        window_seconds=args.window_seconds,
        max_length=args.max_length,
        cache_size=args.cache_size,

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

        # Loss: MPM + Next-Pred (Phase 1은 cross-modal/contrastive 비활성)
        alpha=args.alpha,
        beta=args.beta,
        gamma=0.0,
        delta=args.delta,

        # Phase 1은 variate-level 마스킹 비활성
        variate_mask_prob=0.0,

        # 실험 관리
        exp_name=args.exp_name,

        # 시스템
        device=args.device,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        checkpoint_every=args.checkpoint_every,
        max_batches=args.max_batches,
    )

    set_seed(config.seed + (local_rank if use_ddp else 0))
    output_dir = resolve_output_dir(config)
    if rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_experiment_info(config, output_dir, phase_name="Phase1_CI")

    if rank0:
        print(f"{'='*60}")
        print(f"Phase 1: Channel-Independent Pre-training")
        if config.exp_name:
            print(f"Experiment: {config.exp_name}")
        print(f"Device: {device}" + (f" (DDP: {world_size} GPUs)" if use_ddp else ""))
        print(f"{'='*60}")

    # ── 데이터 로딩 ──
    manifest = load_manifest_from_processed(
        config.processed_dir,
        signal_types=config.signal_types,
        max_subjects=config.max_subjects,
    )
    if rank0:
        print(f"Loaded {len(manifest)} recordings")

    dataset = BiosignalDataset(
        manifest,
        window_seconds=config.window_seconds,
        cache_size=config.cache_size,
    )
    if rank0:
        print(f"Dataset size: {len(dataset)} windows")

    # DDP: DistributedSampler 사용
    sampler = None
    shuffle = True
    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
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
        sampler=sampler,
    )
    if rank0:
        print(f"Batches per epoch: {len(dataloader)}")

    # ── 모델 ──
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
        alpha=config.alpha, beta=config.beta, gamma=config.gamma,
        delta=config.delta, contrastive_temperature=config.contrastive_temperature,
        learnable_temperature=config.learnable_temperature,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=config.lr,
    )
    scheduler = create_scheduler(optimizer, config)

    # ── 시각화용 배치 캐시 (rank 0만) ──
    # 여러 배치를 수집하여 신호 타입 다양성을 확보한다.
    viz_every = args.viz_every
    viz_batches: list | None = None
    viz_dir = None
    if rank0 and viz_every > 0:
        viz_iter = iter(dataloader)
        viz_batches = []
        for _ in range(min(10, len(dataloader))):
            try:
                viz_batches.append(next(viz_iter))
            except StopIteration:
                break
        del viz_iter
        viz_dir = output_dir / "figures"
        viz_dir.mkdir(parents=True, exist_ok=True)
        n_types = len({
            int(b.signal_types[j])
            for b in viz_batches
            for j in range(len(b.signal_types))
        })
        print(f"Visualization every {viz_every} epochs → {viz_dir}"
              f"  ({len(viz_batches)} batches, {n_types} signal types)")

    # ── 학습 루프 ──
    best_loss = float("inf")
    if rank0:
        print(f"\nStarting training: {config.n_epochs} epochs")
        print(f"  alpha={config.alpha}, beta={config.beta}, gamma={config.gamma}")
        print(f"  max_horizon={config.model_config.max_horizon}, mask_ratio={config.mask_ratio}")
        print(f"  warmup_epochs={config.warmup_epochs}")
        print(f"{'='*60}")

    for epoch in range(config.n_epochs):
        # DDP: sampler 에폭 설정 (셔플링 동기화)
        if sampler is not None:
            sampler.set_epoch(epoch)

        losses = train_one_epoch(
            model, dataloader, optimizer, criterion,
            config=config,
            device=device,
            epoch=epoch,
            phase_name="Phase1_CI",
        )
        scheduler.step()

        if rank0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d} | "
                f"total: {losses['total']:.6f} | "
                f"masked: {losses['masked_loss']:.6f} | "
                f"next: {losses['next_loss']:.6f} | "
                f"LR: {current_lr:.2e}"
            )

            # Reconstruction 시각화
            if viz_batches is not None and (epoch % viz_every == 0 or epoch == config.n_epochs - 1):
                viz_model = model.module if use_ddp else model
                fig_path = save_reconstruction_figure(
                    viz_model, viz_batches, epoch=epoch,
                    output_dir=viz_dir, mask_ratio=config.mask_ratio,
                    device=device,
                )
                print(f"  → Reconstruction figure saved: {fig_path}")

            # Best model 저장
            if losses["total"] < best_loss:
                best_loss = losses["total"]
                save_model = model.module if use_ddp else model
                path = save_training_checkpoint(
                    save_model, optimizer, epoch, config,
                    phase_name="phase1_ci", loss=best_loss,
                    output_dir=output_dir, tag="best",
                )
                print(f"  → Best model saved: {path}")

            # 주기적 체크포인트
            if (epoch + 1) % config.checkpoint_every == 0:
                save_model = model.module if use_ddp else model
                save_training_checkpoint(
                    save_model, optimizer, epoch, config,
                    phase_name="phase1_ci", loss=losses["total"],
                    output_dir=output_dir,
                )

    # ── 최종 체크포인트 ──
    if rank0:
        save_model = model.module if use_ddp else model
        final_path = save_training_checkpoint(
            save_model, optimizer, config.n_epochs - 1, config,
            phase_name="phase1_ci", loss=losses["total"],
            output_dir=output_dir, tag="final",
        )
        print(f"\n{'='*60}")
        print(f"Phase 1 complete. Final loss: {losses['total']:.6f}")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Final checkpoint: {final_path}")
        print(f"{'='*60}")

    # 정리
    del dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
