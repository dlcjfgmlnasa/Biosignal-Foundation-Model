# -*- coding:utf-8 -*-
"""loc/scale embedding이 모델 학습에 실제로 기여하는지 ablation으로 측정.

Usage
-----
python -m scripts.inspect_loc_scale_usage \
    --ckpt outputs/phase2/.../checkpoint_phase2_av_epoch009.pt \
    --config configs/phase2_vital.yaml \
    --val-batches 50

측정
----
1) Weight norm: loc_proj/scale_proj vs signal_type_embed/spatial_id_embed
2) Activation norm: forward 시 loc_emb/scale_emb 출력 magnitude
3) Zero ablation: loc_proj/scale_proj 출력을 0으로 강제 → val loss 변화율
4) Shuffle ablation: 출력을 batch 차원으로 셔플 → val loss 변화율

해석
----
zero ablation Δ < 5%   → 모델이 거의 무시 (AdaLN 등 강제 mechanism 권장)
zero ablation Δ 5–20%  → 부분 사용
zero ablation Δ > 20%  → 잘 사용 중 (현재 설계 OK)
"""
from __future__ import annotations

import argparse
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from data import BiosignalDataset, create_dataloader
from loss.criterion import CombinedLoss
from model import BiosignalFoundationModel, ModelConfig
from model.checkpoint import load_checkpoint
from train.train_utils import (
    TrainConfig,
    load_manifest_from_processed,
    split_manifest_by_subject,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="loc/scale embedding ablation")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    p.add_argument("--config", type=str, required=True, help="TrainConfig YAML")
    p.add_argument("--val-batches", type=int, default=20, help="batches per measurement")
    p.add_argument("--max-subjects", type=int, default=None,
                   help="override config.max_subjects (small for sanity)")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", type=str, default="masked",
                   choices=["masked", "both"],
                   help="forward task. 'masked'는 alpha만 측정, 'both'는 alpha+beta")
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconcile_model_config(
    ckpt_state: dict, user_cfg: ModelConfig
) -> ModelConfig:
    """ckpt에 저장된 model_config를 우선 사용. 없으면 user_cfg."""
    if "config" not in ckpt_state:
        return user_cfg
    ckpt_mc = ModelConfig.from_dict(ckpt_state["config"])
    # user_cfg에서 overrides 가능한 필드만 보존 (학습용 train script와 동일한 정책)
    for f in fields(ModelConfig):
        if f.name in {"dropout_p"}:
            setattr(ckpt_mc, f.name, getattr(user_cfg, f.name))
    return ckpt_mc


def batch_to(batch, device: torch.device):
    """PackedBatch 필드를 device로 이동. 모든 tensor 필드 이동."""
    for f in fields(batch):
        v = getattr(batch, f.name)
        if isinstance(v, torch.Tensor):
            setattr(batch, f.name, v.to(device))
    return batch


def build_loaders(
    config: TrainConfig, args: argparse.Namespace
) -> tuple[torch.utils.data.DataLoader, int]:
    max_subjects = (
        args.max_subjects if args.max_subjects is not None else config.max_subjects
    )
    manifest = load_manifest_from_processed(
        config.data_dir,
        signal_types=config.signal_types,
        max_subjects=max_subjects,
    )
    print(f"Loaded {len(manifest)} recordings")
    val_ratio = args.val_ratio if args.val_ratio > 0 else 0.05
    _, val_manifest = split_manifest_by_subject(
        manifest, val_ratio=val_ratio, seed=args.seed
    )
    print(f"Val manifest: {len(val_manifest)} recordings")

    crop_range = None
    if config.crop_ratio_min > 0 and config.crop_ratio_max > 0:
        crop_range = (config.crop_ratio_min, config.crop_ratio_max)
    val_dataset = BiosignalDataset(
        val_manifest,
        window_seconds=config.window_seconds,
        cache_size=config.cache_size,
        crop_ratio_range=crop_range,
        patch_size=config.model_config.patch_size,
        min_patches=config.min_patches,
        shard_index_path=config.shard_index_path,
        shard_cache_size=config.shard_cache_size,
    )
    val_loader = create_dataloader(
        val_dataset,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # inspection만이라 단순화
        pin_memory=False,
        collate_mode=getattr(config, "collate_mode", "ci"),
        patch_size=config.model_config.patch_size,
        min_patches=config.min_patches,
    )
    return val_loader, len(val_dataset)


# ── Hook factories ──────────────────────────────────────────────────


def make_zero_hook():
    """forward output을 zeros_like로 교체."""
    def hook(_module, _inputs, output):
        return torch.zeros_like(output)
    return hook


def make_shuffle_hook(seed: int):
    """forward output을 batch 차원에서 셔플. 같은 forward 내 재현 위해 seed 고정."""
    g = torch.Generator()
    g.manual_seed(seed)

    def hook(_module, _inputs, output):
        b = output.shape[0]
        if b <= 1:
            return output
        perm = torch.randperm(b, generator=g)
        # batch 내 위치는 같지만 다른 sample의 loc/scale info → 잘못된 conditioning
        return output[perm].clone()
    return hook


# ── Measurement ─────────────────────────────────────────────────────


@torch.no_grad()
def compute_loss(
    model: BiosignalFoundationModel,
    batch,
    criterion: CombinedLoss,
    config: TrainConfig,
    task: str,
    device: torch.device,
) -> dict[str, float]:
    """단일 batch에서 loss 계산. train_utils.train_one_epoch와 동일한 호출 구조."""
    out = model(
        batch,
        task=task,
        mask_ratio=config.mask_ratio,
        block_mask=config.block_mask,
        block_size_min=config.block_size_min,
        block_size_max=config.block_size_max,
        variate_mask_prob=config.variate_mask_prob,
        variate_drop_prob=config.variate_drop_prob,
    )
    p = model.patch_size
    normalized = (
        (batch.values.unsqueeze(-1) - out["loc"]) / out["scale"].clamp(min=1e-8)
    ).squeeze(-1)
    b, l = normalized.shape
    n = l // p
    original_patches = normalized[:, : n * p].reshape(b, n, p)

    needs_time_id = config.gamma > 0 or config.delta > 0
    losses = criterion(
        reconstructed=out["reconstructed"],
        next_pred=out.get("next_pred"),
        original_patches=original_patches,
        pred_mask=out["pred_mask"],
        patch_mask=out["patch_mask"],
        patch_sample_id=out["patch_sample_id"],
        patch_variate_id=out["patch_variate_id"],
        cross_pred_per_type=out.get("cross_pred_per_type") if config.gamma > 0 else None,
        time_id=out["time_id"] if needs_time_id else None,
        contrastive_z=out.get("contrastive_z") if config.delta > 0 else None,
        patch_signal_types=out.get("patch_signal_types"),
    )
    # tensor → float
    return {k: float(v.item()) for k, v in losses.items() if torch.is_tensor(v)}


def measure(
    model: BiosignalFoundationModel,
    val_loader,
    criterion: CombinedLoss,
    config: TrainConfig,
    args: argparse.Namespace,
    device: torch.device,
    label: str,
    hook_factory=None,
    hook_targets: tuple[str, ...] = (),
) -> dict[str, float]:
    """val 셋에서 loss를 측정. hook이 있으면 등록 후 측정 → 제거."""
    handles = []
    if hook_factory is not None:
        for name in hook_targets:
            module = getattr(model, name)
            handles.append(module.register_forward_hook(hook_factory()))

    set_seed(args.seed)  # mask 패턴을 측정마다 동일하게 → fair Δ 비교
    sums: dict[str, float] = {}
    n_batches = 0
    for i, batch in enumerate(val_loader):
        if i >= args.val_batches:
            break
        try:
            batch = batch_to(batch, device)
            losses = compute_loss(model, batch, criterion, config, args.task, device)
        except Exception as e:
            print(f"  [{label}] batch {i} error: {e}")
            continue
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + v
        n_batches += 1

    for h in handles:
        h.remove()

    if n_batches == 0:
        print(f"  [{label}] no batches measured")
        return {}
    return {k: v / n_batches for k, v in sums.items()}


# ── Hook for activation magnitude probing ────────────────────────


class ActProbe:
    """forward output의 평균 L2 norm을 기록. valid token만 (norm > 0) 평균."""
    def __init__(self) -> None:
        self.sum_norm = 0.0
        self.count = 0

    def __call__(self, _module, _inputs, output):
        # output: (B, N, d_model) — token마다 L2 norm
        with torch.no_grad():
            norms = output.norm(dim=-1)  # (B, N)
            valid = norms > 1e-8
            if valid.any():
                self.sum_norm += float(norms[valid].mean().item())
                self.count += 1
        return output  # 변경 없이 통과

    @property
    def mean_norm(self) -> float:
        return self.sum_norm / max(1, self.count)


def measure_activation_norms(
    model, val_loader, criterion, config, args, device
) -> dict[str, float]:
    """4개 embedding의 activation magnitude 비교."""
    targets = ["loc_proj", "scale_proj", "signal_type_embed", "spatial_id_embed"]
    probes: dict[str, ActProbe] = {}
    handles = []
    for name in targets:
        module = getattr(model, name, None)
        if module is None:
            continue
        probe = ActProbe()
        probes[name] = probe
        handles.append(module.register_forward_hook(probe))

    set_seed(args.seed)
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= min(args.val_batches, 5):  # activation은 적게도 충분
            break
        try:
            batch = batch_to(batch, device)
            _ = compute_loss(model, batch, criterion, config, args.task, device)
        except Exception as e:
            print(f"  [act probe] batch {i} error: {e}")
            continue
        n += 1

    for h in handles:
        h.remove()

    return {name: probe.mean_norm for name, probe in probes.items()}


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # Config + ckpt
    config = TrainConfig.from_yaml(args.config)
    print(f"Config loaded: {args.config}")

    ckpt_state = torch.load(args.ckpt, map_location=device, weights_only=False)
    config.model_config = reconcile_model_config(ckpt_state, config.model_config)
    print(
        f"Model config: d_model={config.model_config.d_model}, "
        f"num_layers={config.model_config.num_layers}, "
        f"patch_size={config.model_config.patch_size}"
    )

    # Model + load weights
    model = BiosignalFoundationModel.from_config(config.model_config)
    state = load_checkpoint(args.ckpt, model, device=device)
    model.to(device).eval()
    ckpt_epoch = state.get("epoch", "?")
    ckpt_loss = state.get("loss", "?")
    ckpt_phase = state.get("phase", "?")
    print(f"Ckpt: phase={ckpt_phase}, epoch={ckpt_epoch}, loss={ckpt_loss}")

    # Data
    val_loader, n_val = build_loaders(config, args)
    print(f"Val windows: {n_val}, batches per measurement: {args.val_batches}")

    # Loss (학습 때와 동일 weights)
    criterion = CombinedLoss(
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        delta=config.delta,
        peak_alpha=getattr(config, "peak_alpha", 0.0),
        lambda_spec=getattr(config, "lambda_spec", 0.0),
        spec_n_ffts=getattr(config, "spec_n_ffts", (16, 32, 64)),
        contrastive_temperature=getattr(config, "contrastive_temperature", 0.07),
        learnable_temperature=getattr(config, "learnable_temperature", True),
    ).to(device)

    print()
    print("=" * 70)
    print("[1] Weight norms")
    print("=" * 70)
    weight_norms = {
        "loc_proj.weight":          model.loc_proj.weight.norm().item(),
        "scale_proj.weight":        model.scale_proj.weight.norm().item(),
    }
    if hasattr(model, "signal_type_embed"):
        weight_norms["signal_type_embed.weight"] = model.signal_type_embed.weight.norm().item()
    if hasattr(model, "spatial_id_embed"):
        weight_norms["spatial_id_embed.weight"] = model.spatial_id_embed.weight.norm().item()
    for k, v in weight_norms.items():
        print(f"  {k:32s} = {v:.4f}")

    print()
    print("=" * 70)
    print("[2] Activation L2 norm (per-token mean)")
    print("=" * 70)
    act_norms = measure_activation_norms(
        model, val_loader, criterion, config, args, device
    )
    for k, v in act_norms.items():
        print(f"  {k:24s} mean ||·||_2 = {v:.4f}")

    print()
    print("=" * 70)
    print(f"[3] Loss ablation (task={args.task}, batches={args.val_batches})")
    print("=" * 70)
    baseline = measure(model, val_loader, criterion, config, args, device, "baseline")
    if not baseline:
        print("ERROR: baseline measurement failed")
        return

    cases = [
        ("zero loc",        make_zero_hook,                    ("loc_proj",)),
        ("zero scale",      make_zero_hook,                    ("scale_proj",)),
        ("zero both",       make_zero_hook,                    ("loc_proj", "scale_proj")),
        ("shuffle loc",     lambda: make_shuffle_hook(args.seed), ("loc_proj",)),
        ("shuffle scale",   lambda: make_shuffle_hook(args.seed + 1), ("scale_proj",)),
        ("shuffle both",    lambda: make_shuffle_hook(args.seed + 2), ("loc_proj", "scale_proj")),
    ]

    rows = [("baseline", baseline)]
    for label, factory, targets in cases:
        result = measure(
            model, val_loader, criterion, config, args, device,
            label, hook_factory=factory, hook_targets=targets,
        )
        rows.append((label, result))

    # 출력
    print()
    print(f"  {'case':18s} | {'total':>10s} | {'masked':>10s} | {'next':>10s} | {'cross':>10s} | {'Δ total %':>10s}")
    print(f"  {'-' * 18} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}")
    base_total = baseline.get("total", baseline.get("masked_loss", 0.0))
    for label, r in rows:
        if not r:
            continue
        tot = r.get("total", r.get("masked_loss", 0.0))
        msk = r.get("masked_loss", 0.0)
        nxt = r.get("next_loss", 0.0)
        crs = r.get("cross_modal_loss", 0.0)
        delta = ((tot - base_total) / base_total * 100.0) if base_total > 0 else 0.0
        delta_str = "  ──    " if label == "baseline" else f"{delta:+8.2f}"
        print(f"  {label:18s} | {tot:10.4f} | {msk:10.4f} | {nxt:10.4f} | {crs:10.4f} | {delta_str}")

    # 의사결정
    print()
    print("=" * 70)
    print("[Verdict]")
    print("=" * 70)
    zero_both_total = next((r for l, r in rows if l == "zero both"), {}).get("total")
    if zero_both_total is None:
        zero_both_total = next((r for l, r in rows if l == "zero both"), {}).get("masked_loss", 0.0)
    if base_total > 0 and zero_both_total is not None:
        delta_zb = (zero_both_total - base_total) / base_total * 100.0
        if delta_zb < 5.0:
            verdict = "<5% : 모델이 loc/scale embedding을 거의 무시 → (c) AdaLN 도입 권장"
        elif delta_zb < 20.0:
            verdict = "5–20% : 부분 사용 → 진행 가능, paper에 ablation 표 기재"
        else:
            verdict = ">20% : 잘 사용 중 → 현재 설계 유지"
        print(f"  zero(loc+scale) Δ total = {delta_zb:+.2f}%  →  {verdict}")
    else:
        print("  (could not compute verdict — missing total loss)")


if __name__ == "__main__":
    main()
