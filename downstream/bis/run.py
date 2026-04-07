# -*- coding:utf-8 -*-
"""Task 7: 마취 심도 추정 (BIS Prediction).

EEG waveform 입력 -> BIS (Bispectral Index) 값 regression.
Encoder frozen + LinearProbe(d_model, 1) 학습.

BIS 범위: 0~100 (0=isoelectric, 40~60=적정 마취, 60~100=각성).

Usage
-----
# 실제 모델:
python -m downstream.bis.run --checkpoint outputs/phase1_v1/best.pt --n-cases 50

# 더미 테스트:
python -m downstream.bis.run --dummy
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data.collate import PackedBatch
from data.spatial_map import get_global_spatial_id

# ── 상수 ──────────────────────────────────────────────────────

EEG_SIGNAL_TYPE = 2
TARGET_SR = 100.0
EEG_NATIVE_SR = 128.0


# ── 데이터 구조 ───────────────────────────────────────────────


@dataclass
class BISSample:
    """EEG waveform + BIS label 윈도우."""
    eeg: np.ndarray      # (win_samples,) at 100Hz
    bis_value: float      # 윈도우 평균 BIS (0~100)
    case_id: int


# ── VitalDB 데이터 로딩 ──────────────────────────────────────


def find_bis_eeg_cases(n_cases: int = 50) -> list[int]:
    """BIS + EEG 동시 존재 케이스 ID를 반환한다."""
    import vitaldb
    cases = vitaldb.find_cases(["Solar8000/BIS", "BIS/EEG1_WAV"])
    cases = sorted(cases)
    # 뒤쪽 n_cases (pilot pool)
    return cases[-n_cases:]


def load_bis_eeg_windows(
    case_ids: list[int],
    window_sec: float = 30.0,
    stride_sec: float = 10.0,
    min_bis: float = 1.0,
    max_bis_std: float = 20.0,
) -> list[BISSample]:
    """VitalDB에서 EEG waveform + BIS numeric을 로드하고 윈도우를 생성한다.

    Parameters
    ----------
    case_ids:
        로드할 VitalDB case ID 목록.
    window_sec:
        윈도우 길이 (초).
    stride_sec:
        슬라이드 보폭 (초).
    min_bis:
        BIS가 이 미만인 윈도우는 제외 (센서 탈착).
    max_bis_std:
        윈도우 내 BIS std가 이 초과면 제외 (급변 구간).
    """
    import vitaldb
    from data.parser._common import resample_to_target

    win_samples_eeg = int(window_sec * TARGET_SR)  # 100Hz 기준
    stride_samples = int(stride_sec * TARGET_SR)
    results: list[BISSample] = []

    for case_id in case_ids:
        try:
            # EEG waveform (128Hz)
            eeg_raw = vitaldb.load_case(case_id, ["BIS/EEG1_WAV"], interval=1.0 / EEG_NATIVE_SR)
            if eeg_raw is None or len(eeg_raw) == 0:
                continue
            eeg_data = eeg_raw[:, 0].flatten()

            # BIS numeric (1초 간격)
            bis_raw = vitaldb.load_case(case_id, ["Solar8000/BIS"], interval=1.0)
            if bis_raw is None or len(bis_raw) == 0:
                continue
            bis_data = bis_raw[:, 0].flatten()
        except Exception:
            continue

        # EEG를 100Hz로 리샘플링
        valid_eeg = np.where(np.isnan(eeg_data), 0.0, eeg_data).astype(np.float32)
        eeg_100hz = resample_to_target(valid_eeg, orig_sr=EEG_NATIVE_SR, target_sr=TARGET_SR)

        # 시간 정렬: 둘 다 초 단위로 맞춤
        eeg_duration_s = len(eeg_100hz) / TARGET_SR
        bis_duration_s = len(bis_data)  # 1초 간격
        overlap_s = min(eeg_duration_s, bis_duration_s)

        if overlap_s < window_sec:
            continue

        # 슬라이딩 윈도우
        win_sec_int = int(window_sec)
        for start_s in range(0, int(overlap_s) - win_sec_int + 1, int(stride_sec)):
            end_s = start_s + win_sec_int

            # EEG 윈도우 (100Hz)
            eeg_start = int(start_s * TARGET_SR)
            eeg_end = int(end_s * TARGET_SR)
            if eeg_end > len(eeg_100hz):
                break
            eeg_win = eeg_100hz[eeg_start:eeg_end]

            # EEG NaN 비율 체크
            nan_ratio = np.isnan(eeg_win).sum() / len(eeg_win)
            if nan_ratio > 0.1:
                continue
            eeg_win = np.nan_to_num(eeg_win, nan=0.0)

            # BIS 윈도우 (1초 간격)
            bis_win = bis_data[start_s:end_s]
            valid_bis = bis_win[~np.isnan(bis_win)]
            if len(valid_bis) < win_sec_int * 0.5:
                continue

            bis_mean = float(np.mean(valid_bis))
            bis_std = float(np.std(valid_bis))

            # 품질 필터링
            if bis_mean < min_bis or bis_mean > 100:
                continue
            if bis_std > max_bis_std:
                continue

            results.append(BISSample(
                eeg=eeg_win.astype(np.float32),
                bis_value=bis_mean,
                case_id=case_id,
            ))

    return results


def create_dummy_samples(
    n_samples: int = 100,
    win_samples: int = 3000,
) -> list[BISSample]:
    """더미 EEG + BIS 데이터 생성."""
    rng = np.random.default_rng(42)
    samples = []

    for i in range(n_samples):
        # BIS value: 균등 분포 (0~100 중 유효 범위)
        bis = rng.uniform(10, 95)

        # 단순 EEG 합성: BIS 높을수록 고주파 성분 많음
        t = np.arange(win_samples) / TARGET_SR
        # Delta (1-4Hz) — 깊은 마취에서 우세
        delta_amp = max(0, (80 - bis) / 80)
        # Beta (13-30Hz) — 각성에서 우세
        beta_amp = max(0, (bis - 30) / 70)

        eeg = (
            delta_amp * np.sin(2 * np.pi * 2.0 * t)
            + beta_amp * 0.3 * np.sin(2 * np.pi * 20.0 * t)
            + rng.normal(0, 0.1, win_samples)
        ).astype(np.float32)

        samples.append(BISSample(eeg=eeg, bis_value=bis, case_id=7000 + i))

    return samples


# ── PackedBatch 구성 (CI mode) ────────────────────────────────


def build_eeg_batch(
    sample: BISSample,
    patch_size: int = 100,
) -> PackedBatch:
    """단일 EEG 윈도우에서 CI mode PackedBatch를 구성한다."""
    sig = torch.tensor(sample.eeg, dtype=torch.float32)

    rem = len(sig) % patch_size
    if rem > 0:
        sig = torch.cat([sig, torch.zeros(patch_size - rem)])

    L = len(sig)
    values = sig.unsqueeze(0)  # (1, L)
    sample_id = torch.ones(1, L, dtype=torch.long)
    variate_id = torch.ones(1, L, dtype=torch.long)

    signal_types = torch.tensor([EEG_SIGNAL_TYPE], dtype=torch.long)
    spatial_id = get_global_spatial_id(EEG_SIGNAL_TYPE, 0)  # Unknown
    spatial_ids = torch.tensor([spatial_id], dtype=torch.long)

    return PackedBatch(
        values=values,
        sample_id=sample_id,
        variate_id=variate_id,
        lengths=torch.tensor([len(sample.eeg)], dtype=torch.long),
        sampling_rates=torch.tensor([TARGET_SR]),
        signal_types=signal_types,
        spatial_ids=spatial_ids,
        padded_lengths=torch.tensor([L], dtype=torch.long),
    )


# ── 학습 & 평가 ──────────────────────────────────────────────


def train_and_evaluate(
    wrapper,
    train_samples: list[BISSample],
    test_samples: list[BISSample],
    patch_size: int = 100,
    lr: float = 1e-3,
    n_epochs: int = 20,
    device: str = "cpu",
) -> dict:
    """LinearProbe를 학습하고 테스트 세트에서 평가한다.

    Returns
    -------
    dict with keys: mae, mse, rmse, pearson_r, bland_altman, bin_accuracy, n_train, n_test.
    """
    from eval._metrics import (
        compute_bland_altman,
        compute_mae,
        compute_mse,
        compute_pearson_r,
    )
    from downstream.common.model_wrapper import LinearProbe

    device_t = torch.device(device)
    d_model = wrapper.d_model

    # ── Feature 추출 (frozen encoder) ──
    print("  Extracting train features...")
    train_features, train_labels = _extract_all_features(
        wrapper, train_samples, patch_size, device_t,
    )
    print("  Extracting test features...")
    test_features, test_labels = _extract_all_features(
        wrapper, test_samples, patch_size, device_t,
    )

    # ── LinearProbe 학습 ──
    probe = LinearProbe(d_model=d_model, n_classes=1, dropout_p=0.1).to(device_t)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_features = train_features.to(device_t)
    train_targets = train_labels.to(device_t)

    probe.train()
    for epoch in range(n_epochs):
        # Mini-batch (전체가 메모리에 올라감 — 수백~수천 샘플)
        preds = probe(train_features).squeeze(-1)  # (N,)
        loss = F.mse_loss(preds, train_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} | Loss: {loss.item():.4f}")

    # ── 평가 ──
    probe.eval()
    with torch.no_grad():
        test_preds = probe(test_features.to(device_t)).squeeze(-1)
        test_preds = test_preds.clamp(0, 100)  # BIS 범위 제한

    y_pred = test_preds.cpu().numpy()
    y_true = test_labels.numpy()

    mae = compute_mae(y_true, y_pred)
    mse = compute_mse(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    pearson_r = compute_pearson_r(y_true, y_pred)
    bland_altman = compute_bland_altman(y_true, y_pred)

    # 3-bin classification: Deep(0-40), Adequate(40-60), Light(60-100)
    bin_true = _bis_to_bin(y_true)
    bin_pred = _bis_to_bin(y_pred)
    bin_accuracy = float((bin_true == bin_pred).mean())

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "bland_altman": bland_altman,
        "bin_accuracy": bin_accuracy,
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "y_pred": y_pred,
        "y_true": y_true,
    }


def _extract_all_features(
    wrapper,
    samples: list[BISSample],
    patch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """모든 샘플의 feature를 추출한다.

    Returns
    -------
    (features, labels) — features: (N, d_model), labels: (N,)
    """
    features_list: list[torch.Tensor] = []
    labels_list: list[float] = []

    for sample in samples:
        batch = build_eeg_batch(sample, patch_size=patch_size)
        feat = wrapper.extract_features(batch)  # (1, d_model)
        features_list.append(feat.cpu())
        labels_list.append(sample.bis_value)

    features = torch.cat(features_list, dim=0)  # (N, d_model)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    return features, labels


def _bis_to_bin(values: np.ndarray) -> np.ndarray:
    """BIS 값을 3-bin으로 변환: 0=Deep(0-40), 1=Adequate(40-60), 2=Light(60-100)."""
    bins = np.zeros_like(values, dtype=int)
    bins[values >= 40] = 1
    bins[values >= 60] = 2
    return bins


# ── 더미 모델 ────────────────────────────────────────────────


class DummyWrapper:
    """Checkpoint 없이 feature 추출을 시뮬레이션한다.

    EEG의 주파수 대역 파워를 feature로 반환하여
    BIS와 약한 상관관계를 보이도록 한다 (파이프라인 검증용).
    """

    def __init__(self, d_model: int = 128, patch_size: int = 100) -> None:
        self.d_model = d_model
        self.patch_size = patch_size

    def extract_features(self, batch: PackedBatch) -> torch.Tensor:
        """(1, d_model) feature 반환. 첫 몇 차원에 주파수 정보 삽입."""
        values = batch.values[0].numpy()  # (L,)
        n = len(values)

        # FFT 기반 간단한 feature
        fft_vals = np.fft.rfft(values)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / TARGET_SR)

        total_power = power.sum() + 1e-10

        # 대역별 파워 비율
        delta = power[(freqs >= 1) & (freqs <= 4)].sum() / total_power
        theta = power[(freqs >= 4) & (freqs <= 8)].sum() / total_power
        alpha = power[(freqs >= 8) & (freqs <= 13)].sum() / total_power
        beta = power[(freqs >= 13) & (freqs <= 30)].sum() / total_power

        feat = torch.zeros(1, self.d_model)
        feat[0, 0] = float(delta)
        feat[0, 1] = float(theta)
        feat[0, 2] = float(alpha)
        feat[0, 3] = float(beta)
        # 나머지는 약간의 노이즈
        feat[0, 4:] = torch.randn(self.d_model - 4) * 0.01

        return feat

    def forward_masked(self, batch: PackedBatch) -> dict[str, torch.Tensor]:
        B, L = batch.values.shape
        P = self.patch_size
        N = L // P
        loc = batch.values.mean(dim=-1, keepdim=True).unsqueeze(-1).expand(B, L, 1)
        scale = batch.values.std(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-8).expand(B, L, 1)
        return {
            "reconstructed": torch.randn(B, N, P) * 0.1,
            "cross_pred": torch.randn(B, N, P) * 0.1,
            "loc": loc, "scale": scale,
            "patch_mask": torch.ones(B, N, dtype=torch.bool),
            "patch_sample_id": torch.ones(B, N, dtype=torch.long),
            "patch_variate_id": torch.ones(B, N, dtype=torch.long),
            "time_id": torch.arange(N).unsqueeze(0).expand(B, -1),
        }


# ── CLI 진입점 ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 7: BIS Prediction")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="사전학습 checkpoint 경로 (.pt)")
    parser.add_argument("--model-version", type=str, default="v1",
                        choices=["v1", "v2"])
    parser.add_argument("--n-cases", type=int, default=50,
                        help="VitalDB 케이스 수")
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=20,
                        help="LinearProbe 학습 에폭")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="outputs/task7_bis")
    parser.add_argument("--dummy", action="store_true",
                        help="더미 모델로 파이프라인 테스트")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 준비 ──
    if args.dummy:
        print("[Task 7] Dummy mode: 합성 EEG+BIS 데이터")
        all_samples = create_dummy_samples(
            n_samples=100,
            win_samples=int(args.window_sec * TARGET_SR),
        )
    else:
        print(f"[Task 7] VitalDB BIS+EEG 케이스 {args.n_cases}개 탐색 중...")
        case_ids = find_bis_eeg_cases(n_cases=args.n_cases)
        print(f"  BIS+EEG 케이스: {len(case_ids)}개")

        all_samples = load_bis_eeg_windows(
            case_ids, window_sec=args.window_sec, stride_sec=10.0,
        )
        print(f"  윈도우 추출: {len(all_samples)}개")

    if len(all_samples) < 10:
        print("[Task 7] ERROR: 샘플이 너무 적습니다 (최소 10개 필요).")
        return

    # Train/Test split (case 단위)
    case_ids_unique = sorted(set(s.case_id for s in all_samples))
    rng = np.random.default_rng(42)
    rng.shuffle(case_ids_unique)
    n_train_cases = max(1, int(len(case_ids_unique) * args.train_ratio))
    train_case_set = set(case_ids_unique[:n_train_cases])

    train_samples = [s for s in all_samples if s.case_id in train_case_set]
    test_samples = [s for s in all_samples if s.case_id not in train_case_set]

    print(f"  Train: {len(train_samples)} windows ({n_train_cases} cases)")
    print(f"  Test:  {len(test_samples)} windows ({len(case_ids_unique) - n_train_cases} cases)")

    if not train_samples or not test_samples:
        print("[Task 7] ERROR: Train 또는 Test 세트가 비어있습니다.")
        return

    # BIS 분포 출력
    all_bis = [s.bis_value for s in all_samples]
    print(f"  BIS range: [{min(all_bis):.1f}, {max(all_bis):.1f}], mean={np.mean(all_bis):.1f}")

    # ── 모델 로드 ──
    if args.dummy or args.checkpoint is None:
        d_model = 128
        patch_size = 100
        wrapper = DummyWrapper(d_model=d_model, patch_size=patch_size)
        print(f"[Task 7] DummyWrapper (d_model={d_model}, patch_size={patch_size})")
    else:
        from downstream.common.model_wrapper import DownstreamModelWrapper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        wrapper = DownstreamModelWrapper(
            args.checkpoint, model_version=args.model_version, device=device,
        )
        d_model = wrapper.d_model
        patch_size = wrapper.patch_size
        print(f"[Task 7] 모델 로드: d_model={d_model}, patch_size={patch_size}")

    # ── 학습 & 평가 ──
    device_str = "cpu" if args.dummy or args.checkpoint is None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[Task 7] LinearProbe 학습 시작 (epochs={args.epochs}, lr={args.lr})...")

    results = train_and_evaluate(
        wrapper, train_samples, test_samples,
        patch_size=patch_size,
        lr=args.lr,
        n_epochs=args.epochs,
        device=device_str,
    )

    # ── 시각화 ──
    from eval._viz import plot_bland_altman

    plot_bland_altman(
        results["y_true"], results["y_pred"],
        save_path=output_dir / "bland_altman.png",
        title="BIS Prediction - Bland-Altman",
    )

    # Scatter plot: predicted vs true
    _plot_scatter(
        results["y_true"], results["y_pred"],
        save_path=output_dir / "scatter.png",
    )

    # ── 결과 출력 ──
    ba = results["bland_altman"]
    print()
    print("=" * 60)
    print("  Task 7: BIS Prediction Results")
    print("=" * 60)
    print(f"  Train / Test       : {results['n_train']} / {results['n_test']}")
    print(f"  MAE                : {results['mae']:.2f} BIS units")
    print(f"  RMSE               : {results['rmse']:.2f}")
    print(f"  Pearson r          : {results['pearson_r']:.4f}")
    print(f"  Bland-Altman bias  : {ba['bias']:.2f}")
    print(f"  95% LoA            : [{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}]")
    print(f"  3-bin accuracy     : {results['bin_accuracy']:.4f}")
    print("=" * 60)

    # JSON 저장 (numpy 제거)
    save_metrics = {k: v for k, v in results.items() if k not in ("y_pred", "y_true")}
    results_path = output_dir / "task7_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


def _plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
) -> None:
    """Predicted vs True BIS scatter plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")
    lims = [0, 100]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_xlabel("True BIS")
    ax.set_ylabel("Predicted BIS")
    ax.set_title("BIS Prediction: True vs Predicted")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
