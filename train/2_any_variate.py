# -*- coding:utf-8 -*-
from __future__ import annotations

"""Phase 2: Any-Variate 학습 (Cross-Modal).

Phase 1 checkpoint를 로드하여, 다변량 세션에서 cross-modal 학습을 수행한다.
- Masked Patch Modeling: variate-level 마스킹 (Virtual Sensing)
- Next-Patch Prediction: same-variate + cross-modal prediction
- Cross-Modal Loss: 같은 시간대 다른 모달리티 패치 예측

Usage
-----
    python -m train.2_any_variate
    python -m train.2_any_variate --resume outputs/phase1_ci/checkpoint_phase1_ci_epoch069_final.pt
    python -m train.2_any_variate --d_model 128 --num_layers 4 --batch_size 8
"""
import argparse
import gc
import time
from pathlib import Path

import torch
from data import BiosignalDataset, create_dataloader
from loss.criterion import CombinedLoss
from model import BiosignalFoundationModel, ModelConfig
from model.checkpoint import load_checkpoint
from .train_utils import (
    CSVLogger,
    EarlyStopping,
    TrainConfig,
    create_scaler,
    create_scheduler,
    load_manifest_from_processed,
    resolve_device,
    resolve_output_dir,
    save_experiment_info,
    save_training_checkpoint,
    set_seed,
    split_manifest_by_subject,
    train_one_epoch,
    validate,
)
from .visualize import save_reconstruction_figure, save_next_pred_figure


def find_phase1_checkpoint(output_dir: str = "outputs/phase1_ci") -> Path | None:
    """Phase 1 best 또는 final checkpoint를 자동 탐색한다."""
    base = Path(output_dir)
    # checkpoints/ 하위 디렉토리 우선, 없으면 base에서 탐색 (하위 호환)
    search_dirs = [base / "checkpoints", base]
    for d in search_dirs:
        if not d.exists():
            continue
        best_candidates = sorted(d.glob("*_best.pt"))
        if best_candidates:
            return best_candidates[-1]
        final_candidates = sorted(d.glob("*_final.pt"))
        if final_candidates:
            return final_candidates[-1]
        all_ckpts = sorted(d.glob("*.pt"))
        if all_ckpts:
            return all_ckpts[-1]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2: Any-Variate Training (Cross-Modal)")

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Phase 1 checkpoint 경로. 미지정 시 outputs/phase1_ci/에서 자동 탐색.")

    # Model (Phase 1과 동일해야 함)
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
    g.add_argument("--use_cnn_stem", action=argparse.BooleanOptionalAction, default=False)
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
    g.add_argument("--crop_ratio_min", type=float, default=0.0,
                   help="Random crop 최소 비율 (0=비활성)")
    g.add_argument("--crop_ratio_max", type=float, default=0.0,
                   help="Random crop 최대 비율 (0=비활성)")

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
    g.add_argument("--alpha", type=float, default=0.7, help="Masked reconstruction weight")
    g.add_argument("--beta", type=float, default=0.3, help="Next-patch prediction weight")
    g.add_argument("--gamma", type=float, default=1.0, help="Cross-modal loss weight")
    g.add_argument("--delta", type=float, default=0.1, help="Contrastive loss weight")
    g.add_argument("--contrastive_proj_dim", type=int, default=128,
                   help="Contrastive projection head dim (0=disabled)")
    g.add_argument("--contrastive_temperature", type=float, default=0.07)
    g.add_argument("--variate_mask_prob", type=float, default=0.3,
                   help="Variate-level masking probability")
    g.add_argument("--val_ratio", type=float, default=0.2,
                   help="Validation 비율 (subject 단위, 0=비활성)")
    g.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (0=비활성)")

    # System
    g = p.add_argument_group("System")
    g.add_argument("--use_amp", action="store_true",
                   help="AMP (Automatic Mixed Precision) 활성")
    g.add_argument("--device", type=str, default="auto")
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--output_dir", type=str, default="outputs/phase2_any_variate")
    g.add_argument("--checkpoint_every", type=int, default=5)
    g.add_argument("--viz_every", type=int, default=5,
                   help="N 에폭마다 reconstruction 시각화 저장 (0=비활성)")
    g.add_argument("--exp_name", type=str, default="",
                   help="실험 이름 (output_dir 하위 서브디렉토리)")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Phase 2 설정 ──
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
        contrastive_proj_dim=args.contrastive_proj_dim,
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
        collate_mode="any_variate",

        # Loss
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        contrastive_temperature=args.contrastive_temperature,

        # Phase 2: variate-level 마스킹
        variate_mask_prob=args.variate_mask_prob,

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
    )

    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Phase 2: Any-Variate Training (Cross-Modal)")
    if config.exp_name:
        print(f"Experiment: {config.exp_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # ── Phase 1 Checkpoint 로드 ──
    ckpt_path = args.resume
    if ckpt_path is None:
        found = find_phase1_checkpoint()
        if found is None:
            print("ERROR: Phase 1 checkpoint를 찾을 수 없습니다.")
            print("  먼저 python -m train.1_channel_independency 를 실행하세요.")
            print("  또는 --resume 옵션으로 경로를 직접 지정하세요.")
            return
        ckpt_path = str(found)
    print(f"Loading Phase 1 checkpoint: {ckpt_path}")

    # 모델 생성 + checkpoint 로드
    model = BiosignalFoundationModel.from_config(config.model_config)
    state = load_checkpoint(ckpt_path, model, device=device)
    model.to(device)

    phase1_epoch = state.get("epoch", "?")
    phase1_loss = state.get("loss", "?")
    print(f"  Phase 1 epoch: {phase1_epoch}, loss: {phase1_loss}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # 실험 정보 저장
    save_experiment_info(config, output_dir, phase_name="Phase2_AV", extra_info={
        "phase1_ckpt": ckpt_path,
        "phase1_epoch": str(phase1_epoch),
        "phase1_loss": str(phase1_loss),
        "total_params": f"{total_params:,}",
    })

    # ── 데이터 로딩 ──
    manifest = load_manifest_from_processed(
        config.processed_dir,
        signal_types=config.signal_types,
        max_subjects=config.max_subjects,
    )
    print(f"Loaded {len(manifest)} recordings")

    # Train/Val split (subject 단위)
    val_dataloader = None
    if config.val_ratio > 0:
        train_manifest, val_manifest = split_manifest_by_subject(
            manifest, val_ratio=config.val_ratio, seed=config.seed,
        )
        print(f"Train/Val split: {len(train_manifest)} train, {len(val_manifest)} val recordings")
    else:
        train_manifest = manifest

    crop_range = None
    if config.crop_ratio_min > 0 and config.crop_ratio_max > 0:
        crop_range = (config.crop_ratio_min, config.crop_ratio_max)
        print(f"Random crop: {crop_range}")

    dataset = BiosignalDataset(
        train_manifest,
        window_seconds=config.window_seconds,
        cache_size=config.cache_size,
        crop_ratio_range=crop_range,
    )
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
    print(f"Train batches per epoch: {len(dataloader)}")

    # Validation dataloader
    if config.val_ratio > 0:
        val_dataset = BiosignalDataset(
            val_manifest,
            window_seconds=config.window_seconds,
            cache_size=config.cache_size,
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
        print(f"Val dataset: {len(val_dataset)} windows, {len(val_dataloader)} batches")

    # ── Optimizer & Scheduler (Phase 2는 새 optimizer) ──
    criterion = CombinedLoss(
        alpha=config.alpha, beta=config.beta, gamma=config.gamma,
        delta=config.delta, contrastive_temperature=config.contrastive_temperature,
        learnable_temperature=config.learnable_temperature,
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=config.lr,
    )
    scheduler = create_scheduler(optimizer, config)
    scaler = create_scaler(config, device)
    if scaler is not None:
        print(f"AMP enabled (GradScaler)")

    # ── 시각화용 배치 캐시 ──
    viz_every = args.viz_every
    viz_batches: list | None = None
    viz_dir = None
    if viz_every > 0:
        viz_iter = iter(dataloader)
        viz_batches = []
        for _ in range(min(10, len(dataloader))):
            try:
                viz_batches.append(next(viz_iter))
            except StopIteration:
                break
        del viz_iter
        viz_dir = output_dir / "figures"
        viz_recon_dir = viz_dir / "recon"
        viz_np_dir = viz_dir / "next_pred"
        viz_recon_dir.mkdir(parents=True, exist_ok=True)
        viz_np_dir.mkdir(parents=True, exist_ok=True)
        n_types = len({
            int(b.signal_types[j])
            for b in viz_batches
            for j in range(len(b.signal_types))
        })
        print(f"Visualization every {viz_every} epochs → {viz_dir}"
              f"  ({len(viz_batches)} batches, {n_types} signal types)")

    # ── 학습 루프 ──
    best_loss = float("inf")
    early_stopper = EarlyStopping(patience=config.patience) if config.patience > 0 else None
    csv_logger = CSVLogger(output_dir / "training_log.csv")
    print(f"\nStarting training: {config.n_epochs} epochs")
    print(f"  alpha={config.alpha}, beta={config.beta}, gamma={config.gamma}, delta={config.delta}")
    print(f"  max_horizon={config.model_config.max_horizon}, mask_ratio={config.mask_ratio}")
    print(f"  variate_mask_prob={config.variate_mask_prob}")
    print(f"  warmup_epochs={config.warmup_epochs}")
    print(f"  collate_mode={config.collate_mode}")
    if val_dataloader is not None:
        print(f"  val_ratio={config.val_ratio}, patience={config.patience}")
    print(f"{'='*60}")

    for epoch in range(config.n_epochs):
        epoch_start = time.time()
        losses = train_one_epoch(
            model, dataloader, optimizer, criterion,
            config=config,
            device=device,
            epoch=epoch,
            phase_name="Phase2_AV",
            scaler=scaler,
        )
        scheduler.step()

        # ── Validation ──
        val_losses = None
        if val_dataloader is not None:
            val_losses = validate(
                model, val_dataloader, criterion,
                config=config, device=device, phase_name="Phase2_AV",
            )
        epoch_sec = time.time() - epoch_start

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

        # CSV 로깅
        csv_logger.log(epoch, "Phase2_AV", losses, val_losses, current_lr, epoch_sec)

        # Reconstruction & Next-Pred 시각화
        if viz_batches is not None and (epoch % viz_every == 0 or epoch == config.n_epochs - 1):
            fig_path = save_reconstruction_figure(
                model, viz_batches, epoch=epoch,
                output_dir=viz_recon_dir, mask_ratio=config.mask_ratio,
                device=device,
            )
            print(f"  → Reconstruction figure saved: {fig_path}")
            np_path = save_next_pred_figure(
                model, viz_batches, epoch=epoch,
                output_dir=viz_np_dir, horizon=1,
                device=device,
            )
            print(f"  → Next-pred figure saved: {np_path}")

        # Best model 저장 (val_loss 기준, 없으면 train_loss)
        track_loss = val_losses["total"] if val_losses is not None else losses["total"]
        if track_loss < best_loss:
            best_loss = track_loss
            path = save_training_checkpoint(
                model, optimizer, epoch, config,
                phase_name="phase2_av", loss=best_loss,
                output_dir=output_dir, tag="best",
            )
            print(f"  → Best model saved: {path}")

        # 주기적 체크포인트
        if (epoch + 1) % config.checkpoint_every == 0:
            save_training_checkpoint(
                model, optimizer, epoch, config,
                phase_name="phase2_av", loss=losses["total"],
                output_dir=output_dir,
            )

        # ── Early Stopping ──
        if early_stopper is not None and val_losses is not None:
            if early_stopper.step(val_losses["total"]):
                print(
                    f"\n  Early stopping at epoch {epoch} "
                    f"(patience={config.patience}, best_val={early_stopper.best_loss:.6f})"
                )
                break

    # ── 최종 체크포인트 ──
    final_path = save_training_checkpoint(
        model, optimizer, epoch, config,
        phase_name="phase2_av", loss=losses["total"],
        output_dir=output_dir, tag="final",
    )
    print(f"\n{'='*60}")
    print(f"Phase 2 complete. Final train loss: {losses['total']:.6f}")
    if val_losses is not None:
        print(f"Final val loss: {val_losses['total']:.6f}")
    print(f"Best {'val' if val_dataloader else 'train'} loss: {best_loss:.6f}")
    print(f"Final checkpoint: {final_path}")
    print(f"{'='*60}")

    # 메모리 정리
    del dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
