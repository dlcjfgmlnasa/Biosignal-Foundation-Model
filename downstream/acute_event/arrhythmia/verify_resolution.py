# -*- coding:utf-8 -*-
"""SVA(상심실) 저성능이 '해상도/토큰화' 탓인지 통제 실험으로 검증 (진단 전용).

가설
----
우리 foundation model 은 100Hz · patch=200(=2초/token) 이라, SVA 판별에 필요한
미세 P파·beat-to-beat 변화를 token 표현이 버려서 SVA recall 이 낮다.

방법
----
foundation model 과 **무관한** 작은 1D-CNN(토큰화 없음, per-sample conv)을 **동일
arrhythmia fold 데이터의 ECG** 로 여러 sample rate(기본 100/50/25Hz)에서 from-scratch
학습·평가하여 class 별(특히 SVA) recall / one-vs-rest AUPRC / confusion 을 비교한다.
imbalance 혼동을 제거하려고 **class-weighted CE**(inverse-freq)로 학습한다 — 즉
"신호가 SVA 를 분리할 수 있는가(separability)" 만 본다.

해석 (결정 트리)
----------------
1. CNN@100 의 SVA recall >> (당신의 FM SVA recall)
   → 정보는 100Hz 에 있는데 FM 의 2초-token 표현이 버린 것 = **해상도/토큰화 문제(우리
     모델 측)**. 처방: 더 미세한 patch / 높은 Hz 사전학습, 또는 beat-aware head.
2. CNN@100 SVA 도 낮고, 50→25Hz 로 갈수록 **더 하락**
   → 표본 해상도(sampling) 한계 가능성. raw 고-Hz 재-prep(예: 250/500Hz) 후 재검 필요.
3. CNN@100 SVA 낮고 100/50/25 가 **평탄**
   → 해상도 아님. 이질성(SVTA+atrial ectopy+WAP/MAT)·라벨·본질적 난이도 → 수용 + caveat.

사용
----
    python -m downstream.acute_event.arrhythmia.verify_resolution \
        --data-path /.../downstream/arrhythmia/arrhythmia_ecg_ppg \
        --fold 0 --device cuda --sample-rates 100 50 25 --epochs 25

※ 단일 .pt 또는 prefix(+--fold) 모두 지원(run.py 와 동일 규칙). ECG 채널만 사용한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, recall_score
from torch import nn

DEFAULT_SR = 100.0


# ── 데이터 로딩 (run.py._load_data 와 동일 규칙) ─────────────────


def _resolve_path(data_path: str, fold: int) -> Path:
    p = Path(data_path)
    if not p.exists():
        cand = Path(f"{data_path}_fold{int(fold)}.pt")
        if cand.exists():
            p = cand
    if not p.exists():
        print(f"ERROR: data not found: {data_path} (or _fold{fold}.pt)", file=sys.stderr)
        sys.exit(1)
    return p


def _split_xy(split: dict) -> tuple[np.ndarray, np.ndarray]:
    """split → (X ecg (N,T) float32, y (N,) int)."""
    sig = split["signals"]
    if "ecg" not in sig:
        print(f"ERROR: 'ecg' 채널이 없습니다 (있는 키: {list(sig.keys())}).", file=sys.stderr)
        sys.exit(1)
    x = sig["ecg"].float().numpy()  # (N, T)
    y = split["labels"].long().numpy()  # (N,)
    return x, y


# ── 모델 ────────────────────────────────────────────────────────


class TinyECGCNN(nn.Module):
    """토큰화 없는 per-sample 1D-CNN. AdaptiveAvgPool 로 입력 길이(=해상도) 무관."""

    def __init__(self, n_classes: int, k: int = 7) -> None:
        super().__init__()

        def block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(ci, co, k, stride=2, padding=k // 2),
                nn.BatchNorm1d(co),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 128), block(128, 128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, T)
        h = self.net(x).squeeze(-1)  # (B, 128)
        return self.head(h)


# ── 전처리 ──────────────────────────────────────────────────────


def _resample(x: torch.Tensor, src_hz: float, dst_hz: float) -> torch.Tensor:
    """(N,T) @src_hz → @dst_hz (linear interpolate; downsample 시 저역통과 효과)."""
    if abs(dst_hz - src_hz) < 1e-6:
        return x
    new_t = max(1, round(x.shape[1] * dst_hz / src_hz))
    return F.interpolate(
        x.unsqueeze(1), size=new_t, mode="linear", align_corners=False
    ).squeeze(1)


def _znorm(x: torch.Tensor) -> torch.Tensor:
    """per-window z-score (NaN/flat 안전)."""
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp(min=1e-6)
    out = (x - mu) / sd
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ── 학습/평가 ───────────────────────────────────────────────────


def _train_eval_at_sr(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray,
    sr: float, n_classes: int, epochs: int, batch: int, lr: float,
    device: torch.device, seed: int,
) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    Xtr = _znorm(_resample(torch.from_numpy(xtr), DEFAULT_SR, sr))
    Xte = _znorm(_resample(torch.from_numpy(xte), DEFAULT_SR, sr))
    ytr_t = torch.from_numpy(ytr).long()
    yte_t = torch.from_numpy(yte).long()

    # class-weighted CE (inverse-freq) — imbalance 혼동 제거.
    counts = torch.bincount(ytr_t, minlength=n_classes).float().clamp(min=1.0)
    weight = (counts.sum() / (n_classes * counts)).to(device)

    model = TinyECGCNN(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weight)

    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i : i + batch]
            xb = Xtr[idx].unsqueeze(1).to(device)  # (B,1,T)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    # eval
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], batch):
            xb = Xte[i : i + batch].unsqueeze(1).to(device)
            probs.append(torch.softmax(model(xb), dim=1).cpu())
    p = torch.cat(probs, dim=0).numpy()  # (Nte, C)
    y_pred = p.argmax(axis=1)

    rec = recall_score(yte, y_pred, labels=list(range(n_classes)),
                       average=None, zero_division=0)
    # one-vs-rest AUPRC (class 가 test 에 없으면 nan)
    auprc = []
    for c in range(n_classes):
        yc = (yte == c).astype(int)
        auprc.append(average_precision_score(yc, p[:, c]) if yc.sum() > 0 else float("nan"))
    cm = confusion_matrix(yte, y_pred, labels=list(range(n_classes)))
    return {"recall": rec, "auprc": np.array(auprc), "cm": cm,
            "in_len": Xtr.shape[1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", required=True, help="단일 .pt 또는 prefix")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--sample-rates", type=float, nargs="+", default=[100, 50, 25])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    path = _resolve_path(args.data_path, args.fold)
    print(f"Loading: {path}")
    data = torch.load(path, weights_only=False)
    meta = data.get("metadata", {})
    class_names = list(meta.get("class_names") or meta.get("classes")
                       or [f"c{i}" for i in range(5)])
    n_classes = len(class_names)
    xtr, ytr = _split_xy(data["train"])
    xte, yte = _split_xy(data["test"])
    print(f"  classes={class_names}  train={len(ytr)}  test={len(yte)}")
    tr_dist = np.bincount(ytr, minlength=n_classes)
    te_dist = np.bincount(yte, minlength=n_classes)
    print("  class               train / test")
    for c in range(n_classes):
        print(f"    {class_names[c]:<18} {tr_dist[c]:>6} / {te_dist[c]}")

    results: dict[float, dict] = {}
    for sr in args.sample_rates:
        print(f"\n=== sample rate {sr:g} Hz (toy 1D-CNN, no tokenization) ===")
        r = _train_eval_at_sr(xtr, ytr, xte, yte, sr, n_classes,
                              args.epochs, args.batch_size, args.lr, device, args.seed)
        results[sr] = r
        print(f"  input length: {r['in_len']} samples")
        print("  class               recall   AUPRC")
        for c in range(n_classes):
            print(f"    {class_names[c]:<18} {r['recall'][c]:.3f}    {r['auprc'][c]:.3f}")
        print(f"  macro recall={np.nanmean(r['recall']):.3f}  "
              f"macro AUPRC={np.nanmean(r['auprc']):.3f}")

    # ── 요약: SVA(있으면) recall 의 해상도별 추이 ──
    print("\n" + "=" * 60)
    print("  RESOLUTION SWEEP SUMMARY (per-class recall)")
    header = "  class           " + "".join(f"{sr:>8g}Hz" for sr in args.sample_rates)
    print(header)
    for c in range(n_classes):
        row = f"  {class_names[c]:<14}" + "".join(
            f"{results[sr]['recall'][c]:>10.3f}" for sr in args.sample_rates
        )
        print(row)
    print("=" * 60)
    print("해석: 위 docstring 의 결정 트리 참고. CNN@100 SVA 를 당신의 FM SVA recall 과")
    print("비교하세요 (CNN@100 >> FM → 토큰화/해상도 문제, 평탄+낮음 → 본질적 난이도).")
    # 가장 큰 sr 의 confusion matrix 출력 (SVA 가 무엇과 혼동되는지)
    sr_max = max(args.sample_rates)
    print(f"\nConfusion matrix @ {sr_max:g}Hz (행=true, 열=pred):")
    print("        " + " ".join(f"{cn[:6]:>7}" for cn in class_names))
    cm = results[sr_max]["cm"]
    for c in range(n_classes):
        print(f"  {class_names[c][:6]:>6} " + " ".join(f"{cm[c, j]:>7d}" for j in range(n_classes)))


if __name__ == "__main__":
    main()
