# -*- coding:utf-8 -*-
"""Blood Pressure (SBP/DBP) Estimation — 회귀(regression) downstream 러너.

사전학습 encoder 위에 경량 회귀 head 를 얹어 입력 window(PPG, 선택적 +ECG)로부터
같은 시점의 SBP·DBP(mmHg) 를 추정한다(horizon 없음, cuffless BP). 데이터는
``prepare_data.py`` 의 chunk 산출물(``bp_{signals}_w{win}s_fold{F}_{split}_chunk*.pt``)
을 사용하며, 다른 downstream task 와 동일한 로더/평가 인프라를 공유한다.

Head (``--head``):
  - ``attn`` (기본): 학습 가능한 query 로 patch 토큰을 attention-pooling → 회귀.
    mean-pool 이 beat 형태 정보를 뭉개는 병목을 겨냥한다(BP 는 파형 morphology·PTT
    의존이 큼).
  - ``mean``: mean-pool → Linear. 표준 linear-probe 회귀 head(표현 품질 하한/비교용).

Adaptation (``--mode``):
  - ``linear_probe``: encoder frozen. frozen 토큰 feature 를 1회 캐싱 후 head 만 학습.
  - ``lora``: attention Q/V 에 LoRA 삽입 + head 를 함께 학습(스트리밍, OOM-safe).

평가: SBP·DBP 각각 MAE / RMSE / Pearson r / Bland-Altman, patient-cluster bootstrap
95% CI(1,000 iter). raw prediction(float) 을 npz 로 저장해 사후 OOF 재계산 가능.

사용법:
    # frozen linear probe (attention head, ECG+PPG, fold 0)
    python -m downstream.vital_sign.blood_pressure.run \
        --checkpoint best.pt --model-version v2 --mode linear_probe --head attn \
        --data-path .../bp_estimation/bp_ecg_ppg_w30s --fold 0 --n-folds 5 \
        --out-dir .../result/bp_estimation

    # LoRA fine-tune (batch 32 유지 — 비교성)
    python -m downstream.vital_sign.blood_pressure.run \
        --checkpoint best.pt --model-version v2 --mode lora --head attn \
        --data-path .../bp_estimation/bp_ecg_ppg_w30s --fold 0 --n-folds 5 \
        --lr 1e-4 --epochs 30 --batch-size 32 --out-dir .../result/bp_estimation
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from downstream.model_wrapper import DownstreamModelWrapper
from downstream.window_task import (
    infer_collate_mode,
    iter_window_batches,
    _multi_window_to_samples,
)
from downstream._save_utils import load_prepared_split_chunked
from downstream._eval_utils import bootstrap_ci as patient_bootstrap_ci
from downstream.metrics import compute_mae, compute_pearson_r, compute_bland_altman


TARGETS = ("sbp", "dbp")  # 회귀 출력 순서 (n_out=2)


# ── window 표현 ────────────────────────────────────────────────


@dataclass
class BPWindow:
    """단일 BP 회귀 윈도우 — ``signals``·``case_id`` 는 window_task 콜백이 사용한다."""

    signals: dict[str, np.ndarray]  # {"ppg": (win_samples,), "ecg": ...}
    sbp: float
    dbp: float
    case_id: str


def _bp_label(w: BPWindow) -> list[float]:
    """get_label 콜백 — [SBP, DBP] → iter_window_batches 가 (B, 2) 로 stack."""
    return [w.sbp, w.dbp]


def _windows_from_payload(payload: dict | None) -> list[BPWindow]:
    """chunk payload(dict) → BPWindow 리스트. 신호는 fp16 저장 → fp32 로 복원."""
    if not payload:
        return []
    sigs: dict = payload.get("signals", {})
    if not sigs:
        return []
    sig_names = list(sigs.keys())
    sbp = payload["sbp"]
    dbp = payload["dbp"]
    case_ids = payload["case_ids"]
    n = int(sbp.shape[0])
    windows: list[BPWindow] = []
    for i in range(n):
        sig_dict = {
            s: sigs[s][i].numpy().astype(np.float32) for s in sig_names
        }
        windows.append(
            BPWindow(
                signals=sig_dict,
                sbp=float(sbp[i]),
                dbp=float(dbp[i]),
                case_id=str(case_ids[i]),
            )
        )
    return windows


# ── 회귀 head ─────────────────────────────────────────────────


class BPRegressionHead(nn.Module):
    """Patch 토큰 → (SBP, DBP) 회귀 head.

    ``head="attn"`` 은 학습 가능한 query 로 유효 패치를 가중합(attention pool)하고,
    ``head="mean"`` 은 유효 패치 평균(mean pool)한다. 두 경우 모두 pooled 벡터에
    LayerNorm → Dropout → Linear(d_model → n_out) 를 적용한다(sigmoid 없음).
    """

    def __init__(
        self,
        d_model: int,
        n_out: int = 2,
        head: str = "attn",
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.head_type = head
        if head == "attn":
            # 학습 가능한 query — patch 토큰과의 내적으로 attention score 산출.
            self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout_p)
        self.out = nn.Linear(d_model, n_out)

    def _pool(
        self,
        encoded: torch.Tensor,  # (B, N, d_model)
        patch_mask: torch.Tensor,  # (B, N) — True=유효 패치
    ) -> torch.Tensor:  # (B, d_model)
        mask = patch_mask.bool()
        if self.head_type == "attn":
            scores = (encoded * self.query).sum(dim=-1)  # (B, N)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)  # (B, N)
            # 유효 패치가 0 인 행(발생 안 함이 정상)은 all -inf → nan → 0 으로 방어.
            weights = torch.nan_to_num(weights)
            return (encoded * weights.unsqueeze(-1)).sum(dim=1)
        # mean pool
        mask_f = mask.unsqueeze(-1).float()  # (B, N, 1)
        return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        encoded: torch.Tensor,  # (B, N, d_model)
        patch_mask: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:  # (B, n_out)
        pooled = self._pool(encoded, patch_mask)
        return self.out(self.drop(self.norm(pooled)))


# ── encoder forward (encoded + patch_mask) ────────────────────


def _encode(
    model: DownstreamModelWrapper,
    batch,
    train_encoder: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PackedBatch → (encoded (B,N,d_model), patch_mask (B,N)).

    ``extract_features`` 와 동일하게 ``batch_to_device`` 후 ``model(task="masked")``
    를 호출한다. frozen(``train_encoder=False``) 이면 no_grad, LoRA 면 grad 유지.
    """
    batch = model.batch_to_device(batch)
    if train_encoder:
        out = model.model(batch, task="masked")
    else:
        with torch.no_grad():
            out = model.model(batch, task="masked")
    return out["encoded"], out["patch_mask"]


def _iter_batches(windows: list[BPWindow], batch_size: int, patch_size: int, collate_mode: str):
    return iter_window_batches(
        windows,
        batch_size,
        patch_size,
        to_samples=_multi_window_to_samples,
        get_label=_bp_label,
        label_dtype=torch.float32,
        collate_mode=collate_mode,
    )


# ── frozen linear-probe 경로 (토큰 feature 1회 캐싱) ────────────


def _cache_tokens(
    model: DownstreamModelWrapper,
    windows: list[BPWindow],
    batch_size: int,
    patch_size: int,
    collate_mode: str,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """frozen encoder 로 (encoded, patch_mask, targets) 를 CPU(fp16) 로 1회 캐싱.

    frozen encoder 는 결정론적이므로 epoch 마다 재추출하지 않고 캐시를 재사용한다
    (project_downstream_oom_io_hardening: frozen feature 1회 캐싱). attention head
    는 pooling 이 학습 대상이라 mean-pool 대신 토큰 시퀀스를 통째로 보관한다.
    """
    model.model.eval()
    cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for batch, targets in _iter_batches(windows, batch_size, patch_size, collate_mode):
        encoded, patch_mask = _encode(model, batch, train_encoder=False)
        cache.append(
            (
                encoded.to("cpu", dtype=torch.float16),
                patch_mask.to("cpu"),
                targets,  # (B, 2) float32
            )
        )
    return cache


def train_head_frozen(
    head: BPRegressionHead,
    cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    head = head.to(device).train()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for encoded, patch_mask, targets in cache:
            encoded = encoded.to(device).float()
            patch_mask = patch_mask.to(device)
            pred = head(encoded, patch_mask)
            loss = criterion(pred, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / max(n, 1)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  MSE={avg:.3f}")
    return losses


@torch.no_grad()
def eval_head_frozen(
    head: BPRegressionHead,
    cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    head = head.to(device).eval()
    preds, trues = [], []
    for encoded, patch_mask, targets in cache:
        encoded = encoded.to(device).float()
        patch_mask = patch_mask.to(device)
        preds.append(head(encoded, patch_mask).cpu())
        trues.append(targets)
    return torch.cat(preds), torch.cat(trues)  # (N, 2), (N, 2)


# ── LoRA 경로 (스트리밍, encoder 매 epoch grad) ────────────────


def train_lora(
    model: DownstreamModelWrapper,
    head: BPRegressionHead,
    windows: list[BPWindow],
    epochs: int,
    lr: float,
    batch_size: int,
    patch_size: int,
    collate_mode: str,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> list[float]:
    model.model.train()
    head = head.to(device).train()
    lora_params = model.lora_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lr},
            {"params": head.parameters(), "lr": lr},
        ],
        weight_decay=0.01,
    )
    criterion = nn.MSELoss()
    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for batch, targets in _iter_batches(windows, batch_size, patch_size, collate_mode):
            encoded, patch_mask = _encode(model, batch, train_encoder=True)
            pred = head(encoded, patch_mask)
            loss = criterion(pred, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                lora_params + list(head.parameters()), gradient_clip
            )
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / max(n, 1)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  MSE={avg:.3f}")
    return losses


@torch.no_grad()
def eval_lora(
    model: DownstreamModelWrapper,
    head: BPRegressionHead,
    windows: list[BPWindow],
    batch_size: int,
    patch_size: int,
    collate_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.model.eval()
    head = head.to(device).eval()
    preds, trues = [], []
    for batch, targets in _iter_batches(windows, batch_size, patch_size, collate_mode):
        encoded, patch_mask = _encode(model, batch, train_encoder=False)
        preds.append(head(encoded, patch_mask).cpu())
        trues.append(targets)
    return torch.cat(preds), torch.cat(trues)


# ── 지표 ──────────────────────────────────────────────────────


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, np.float64) - np.asarray(y_pred, np.float64)
    return float(np.sqrt(np.mean(d * d)))


def compute_bp_metrics(
    y_true: torch.Tensor,  # (N, 2)
    y_pred: torch.Tensor,  # (N, 2)
    case_ids: list[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """SBP·DBP 각각 MAE/RMSE/Pearson r/Bland-Altman + patient-cluster bootstrap CI."""
    pid = np.asarray(case_ids, dtype=str)
    report: dict = {}
    for j, name in enumerate(TARGETS):
        yt = y_true[:, j].numpy().astype(np.float64)
        yp = y_pred[:, j].numpy().astype(np.float64)
        ba = compute_bland_altman(yt, yp)
        mae, mae_lo, mae_hi = patient_bootstrap_ci(
            yt, yp, pid, compute_mae, n_iter=n_boot, seed=seed
        )
        r, r_lo, r_hi = patient_bootstrap_ci(
            yt, yp, pid, compute_pearson_r, n_iter=n_boot, seed=seed
        )
        report[name] = {
            "mae": mae,
            "mae_ci": [mae_lo, mae_hi],
            "rmse": _rmse(yt, yp),
            "pearson_r": r,
            "pearson_r_ci": [r_lo, r_hi],
            "bias": ba["bias"],  # ME (mean error) — pred - true
            "std_diff": ba["std_diff"],
            "loa": [ba["loa_lower"], ba["loa_upper"]],
            "n": int(len(yt)),
            "n_patients": int(len(np.unique(pid))),
        }
    return report


def _save_predictions(
    out_dir: Path,
    fold: int,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    case_ids: list[str],
) -> Path:
    """raw prediction(float) 저장 — 사후 OOF 재계산용 (dump_fold_predictions 는
    y_true 를 int64 로 캐스팅해 회귀엔 부적합 → 전용 npz 사용)."""
    path = out_dir / f"bp_preds_fold{fold}.npz"
    np.savez_compressed(
        path,
        y_true=y_true.numpy().astype(np.float32),  # (N, 2)
        y_pred=y_pred.numpy().astype(np.float32),  # (N, 2)
        patient_id=np.asarray(case_ids, dtype=str),
        targets=np.asarray(TARGETS, dtype=str),
        fold_idx=np.int64(fold),
    )
    return path


# ── main ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blood Pressure (SBP/DBP) Estimation — regression downstream runner",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="사전학습 encoder checkpoint (.pt)")
    parser.add_argument("--model-version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument("--data-path", type=str, required=True,
                        help="chunk prefix (예: .../bp_ecg_ppg_w30s). "
                        "n_folds>1 이면 fold 별 chunk 를 로드한다.")
    parser.add_argument("--mode", type=str, default="linear_probe",
                        choices=["linear_probe", "lora"])
    parser.add_argument("--head", type=str, default="attn", choices=["attn", "mean"],
                        help="attn=attention-pool(기본, 권장) / mean=mean-pool linear probe")
    parser.add_argument("--fold", type=int, default=0,
                        help="현재 fold 인덱스 (OOF 집계용)")
    parser.add_argument("--n-folds", type=int, default=1,
                        help="전체 fold 수 (>1 이면 fold 별 chunk 로드)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="lora 는 32 유지 권장(비교성). frozen 은 키워도 무방.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Blood Pressure (SBP/DBP) Estimation — regression")
    print(f"  mode={args.mode}  head={args.head}  fold={args.fold}/{args.n_folds}")
    print(f"  checkpoint={args.checkpoint}  ({args.model_version})")
    print("=" * 60)

    # 1) encoder 로드 (+ LoRA)
    model = DownstreamModelWrapper(
        args.checkpoint, model_version=args.model_version, device=device
    )
    if args.mode == "lora":
        model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    patch_size = model.patch_size  # config 값(200) — window_task 기본(100) 쓰면 안 됨.

    # 2) fold 데이터 로드
    load_fold = args.fold if args.n_folds > 1 else None
    data = load_prepared_split_chunked(args.data_path, fold=load_fold)
    train_windows = _windows_from_payload(data.get("train"))
    test_windows = _windows_from_payload(data.get("test"))
    print(f"  Signals: {data.get('metadata', {}).get('input_signals', '?')}")
    print(f"  Train windows: {len(train_windows)}  Test windows: {len(test_windows)}")
    if not train_windows or not test_windows:
        print("ERROR: train/test 윈도우가 비었습니다.", file=sys.stderr)
        sys.exit(1)

    head = BPRegressionHead(
        model.d_model, n_out=len(TARGETS), head=args.head, dropout_p=args.dropout
    )

    # 3) 학습 + 평가
    train_mode = infer_collate_mode(train_windows, _multi_window_to_samples)
    test_mode = infer_collate_mode(test_windows, _multi_window_to_samples)
    if args.mode == "linear_probe":
        print("\n[Train] frozen encoder — caching tokens...")
        train_cache = _cache_tokens(
            model, train_windows, args.batch_size, patch_size, train_mode
        )
        train_head_frozen(head, train_cache, args.epochs, args.lr, device)
        del train_cache
        print("\n[Eval] test...")
        test_cache = _cache_tokens(
            model, test_windows, args.batch_size, patch_size, test_mode
        )
        y_pred, y_true = eval_head_frozen(head, test_cache, device)
    else:
        print("\n[Train] LoRA fine-tune...")
        train_lora(
            model, head, train_windows, args.epochs, args.lr,
            args.batch_size, patch_size, train_mode, device,
        )
        print("\n[Eval] test...")
        y_pred, y_true = eval_lora(
            model, head, test_windows, args.batch_size, patch_size, test_mode, device
        )

    case_ids = [w.case_id for w in test_windows]

    # 4) 지표 + 저장
    report = compute_bp_metrics(
        y_true, y_pred, case_ids, n_boot=args.n_boot, seed=args.seed
    )
    for name in TARGETS:
        m = report[name]
        print(
            f"\n  {name.upper()}: MAE={m['mae']:.2f} "
            f"({m['mae_ci'][0]:.2f}–{m['mae_ci'][1]:.2f})  "
            f"RMSE={m['rmse']:.2f}  r={m['pearson_r']:.3f} "
            f"({m['pearson_r_ci'][0]:.3f}–{m['pearson_r_ci'][1]:.3f})  "
            f"ME={m['bias']:.2f}±{m['std_diff']:.2f}"
        )

    results = {
        "task": "bp_estimation",
        "mode": args.mode,
        "head": args.head,
        "fold": args.fold,
        "n_folds": args.n_folds,
        "model_version": args.model_version,
        "input_signals": data.get("metadata", {}).get("input_signals"),
        "window_sec": data.get("metadata", {}).get("window_sec"),
        "metrics": report,
    }
    fold_suffix = f"_fold{args.fold}" if int(args.n_folds) > 1 else ""
    results_path = out_dir / f"bp_estimation_results_{args.mode}_{args.head}{fold_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    preds_path = _save_predictions(out_dir, args.fold, y_true, y_pred, case_ids)

    print(f"\n  Results: {results_path}")
    print(f"  Predictions: {preds_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
