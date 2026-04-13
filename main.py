# -*- coding:utf-8 -*-
from __future__ import annotations

"""Biosignal Foundation Model — 통합 학습 엔트리포인트.

YAML config 기반으로 Phase 1/2 학습을 실행한다.

Usage
-----
    # Phase 1 학습 (GPU 서버)
    python main.py --config configs/phase1.yaml

    # Phase 2 학습 (Phase 1 checkpoint 자동 탐색)
    python main.py --config configs/phase2.yaml

    # Phase 2 + checkpoint 직접 지정
    python main.py --config configs/phase2.yaml --resume outputs/phase1_ci/checkpoint_best.pt

    # 로컬 CPU dry-run (shape 검증)
    python main.py --config configs/dry_run.yaml

    # config + CLI 오버라이드
    python main.py --config configs/phase1.yaml --dry-run --batch_size 2 --device cpu
"""
import argparse
import gc
import time
from pathlib import Path

import torch

from data import BiosignalDataset, create_dataloader
from loss.criterion import CombinedLoss
from model import BiosignalFoundationModel
from model.checkpoint import load_checkpoint
from train.train_utils import (
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Biosignal Foundation Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config 파일 경로 (e.g. configs/phase1.yaml)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Phase 1 checkpoint 경로 (Phase 2 시 사용)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="1 batch만 실행 후 종료 (shape/에러 검증용)",
    )

    # CLI 오버라이드 (yaml 값을 덮어씀)
    g = p.add_argument_group("Overrides", "YAML 값을 CLI에서 덮어쓸 수 있음")
    g.add_argument("--d_model", type=int, default=None)
    g.add_argument("--num_layers", type=int, default=None)
    g.add_argument("--patch_size", type=int, default=None)
    g.add_argument("--batch_size", type=int, default=None)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--n_epochs", type=int, default=None)
    g.add_argument("--device", type=str, default=None)
    g.add_argument("--max_subjects", type=int, default=None)
    g.add_argument("--output_dir", type=str, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument(
        "--crop_ratio_min",
        type=float,
        default=None,
        help="Random crop 최소 비율 (0=비활성)",
    )
    g.add_argument(
        "--crop_ratio_max",
        type=float,
        default=None,
        help="Random crop 최대 비율 (0=비활성)",
    )
    g.add_argument(
        "--use_amp", action="store_true", help="AMP (Automatic Mixed Precision) 활성"
    )
    g.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="Validation 비율 (subject 단위, 0=비활성)",
    )
    g.add_argument(
        "--patience", type=int, default=None, help="Early stopping patience (0=비활성)"
    )
    g.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="실험 이름 (output_dir 하위 서브디렉토리)",
    )

    return p.parse_args()


def find_phase1_checkpoint(output_dir: str = "outputs/phase1_ci") -> Path | None:
    """Phase 1 best 또는 final checkpoint를 자동 탐색한다."""
    base = Path(output_dir)
    for pattern in ["*_best.pt", "*_final.pt", "*.pt"]:
        candidates = sorted(base.glob(pattern))
        if candidates:
            return candidates[-1]
    return None


def main():
    args = parse_args()

    # ── Config 로드 ──
    overrides = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "device": args.device,
        "max_subjects": args.max_subjects,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "crop_ratio_min": args.crop_ratio_min,
        "crop_ratio_max": args.crop_ratio_max,
        "val_ratio": args.val_ratio,
        "patience": args.patience,
        "exp_name": args.exp_name,
    }
    config = TrainConfig.from_yaml_with_overrides(args.config, overrides)

    if args.use_amp:
        config.use_amp = True
    if args.dry_run:
        config.dry_run = True
        config.n_epochs = 1

    is_phase2 = config.collate_mode == "any_variate"
    phase_name = "Phase2_AV" if is_phase2 else "Phase1_CI"
    phase_tag = "phase2_av" if is_phase2 else "phase1_ci"

    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 설정 출력 ──
    print(f"{'=' * 60}")
    print(f"{phase_name} {'(dry-run)' if config.dry_run else ''}")
    if config.exp_name:
        print(f"Experiment: {config.exp_name}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    # ── 모델 생성 ──
    model = BiosignalFoundationModel.from_config(config.model_config)

    # Phase 2: Phase 1 checkpoint 로드
    if is_phase2:
        ckpt_path = args.resume
        if ckpt_path is None:
            found = find_phase1_checkpoint()
            if found is None:
                print("ERROR: Phase 1 checkpoint를 찾을 수 없습니다.")
                print("  --resume 옵션으로 경로를 직접 지정하거나,")
                print("  먼저 Phase 1을 실행하세요.")
                return
            ckpt_path = str(found)
        print(f"Loading Phase 1 checkpoint: {ckpt_path}")
        state = load_checkpoint(ckpt_path, model, device=device)
        print(
            f"  Phase 1 epoch: {state.get('epoch', '?')}, loss: {state.get('loss', '?')}"
        )

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # ── 데이터 로딩 ──
    manifest = load_manifest_from_processed(
        config.processed_dir,
        signal_types=config.signal_types,
        max_subjects=config.max_subjects,
    )
    print(f"Loaded {len(manifest)} recordings")

    # Train/Val split
    val_dataloader = None
    if config.val_ratio > 0 and not config.dry_run:
        train_manifest, val_manifest = split_manifest_by_subject(
            manifest,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
        print(
            f"Train/Val split: {len(train_manifest)} train, {len(val_manifest)} val recordings"
        )
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
        collate_mode=config.collate_mode,
        patch_size=config.model_config.patch_size,
    )
    print(f"Train batches per epoch: {len(dataloader)}")

    if config.val_ratio > 0 and not config.dry_run:
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

    # ── Optimizer & Scheduler ──
    criterion = CombinedLoss(
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        delta=config.delta,
        contrastive_temperature=config.contrastive_temperature,
        learnable_temperature=config.learnable_temperature,
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config.lr,
    )
    scheduler = create_scheduler(optimizer, config)
    scaler = create_scaler(config, device)
    if scaler is not None:
        print("AMP enabled (GradScaler)")

    # ── 실험 정보 & 설정 저장 (재현용) ──
    if not config.dry_run:
        extra = {}
        if is_phase2 and ckpt_path:
            extra["phase1_ckpt"] = ckpt_path
        save_experiment_info(
            config, output_dir, phase_name=phase_name, extra_info=extra or None
        )

    # ── 학습 루프 ──
    best_loss = float("inf")
    early_stopper = (
        EarlyStopping(patience=config.patience)
        if config.patience > 0 and not config.dry_run
        else None
    )
    csv_logger = (
        CSVLogger(output_dir / "training_log.csv") if not config.dry_run else None
    )
    print(f"\nStarting training: {config.n_epochs} epochs")
    print(
        f"  alpha={config.alpha}, beta={config.beta}, gamma={config.gamma}, delta={config.delta}"
    )
    print(
        f"  mask_ratio={config.mask_ratio}, variate_mask_prob={config.variate_mask_prob}"
    )
    if val_dataloader is not None:
        print(f"  val_ratio={config.val_ratio}, patience={config.patience}")
    if config.dry_run:
        print("  *** DRY-RUN: 1 batch만 실행합니다 ***")
    print(f"{'=' * 60}")

    for epoch in range(config.n_epochs):
        epoch_start = time.time()
        losses = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            config=config,
            device=device,
            epoch=epoch,
            phase_name=phase_name,
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
                phase_name=phase_name,
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
        if csv_logger is not None:
            csv_logger.log(epoch, phase_name, losses, val_losses, current_lr, epoch_sec)

        if config.dry_run:
            print(f"\n{'=' * 60}")
            print("Dry-run 완료. forward → loss → backward 정상 동작 확인.")
            print(f"{'=' * 60}")
            break

        # Best model 저장 (val_loss 기준, 없으면 train_loss)
        track_loss = val_losses["total"] if val_losses is not None else losses["total"]
        if track_loss < best_loss:
            best_loss = track_loss
            path = save_training_checkpoint(
                model,
                optimizer,
                epoch,
                config,
                phase_name=phase_tag,
                loss=best_loss,
                output_dir=output_dir,
                tag="best",
            )
            print(f"  → Best model saved: {path}")

        # 주기적 체크포인트
        if (epoch + 1) % config.checkpoint_every == 0:
            save_training_checkpoint(
                model,
                optimizer,
                epoch,
                config,
                phase_name=phase_tag,
                loss=losses["total"],
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

    if not config.dry_run:
        final_path = save_training_checkpoint(
            model,
            optimizer,
            epoch,
            config,
            phase_name=phase_tag,
            loss=losses["total"],
            output_dir=output_dir,
            tag="final",
        )
        print(f"\n{'=' * 60}")
        print(f"{phase_name} complete. Final train loss: {losses['total']:.6f}")
        if val_losses is not None:
            print(f"Final val loss: {val_losses['total']:.6f}")
        print(f"Best {'val' if val_dataloader else 'train'} loss: {best_loss:.6f}")
        print(f"Final checkpoint: {final_path}")
        print(f"{'=' * 60}")

    # 메모리 정리
    del dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
