# -*- coding:utf-8 -*-
"""Task 3: 채널 단위 Imputation -특정 채널 전체 마스킹 → 나머지 채널로 복원.

Cross-modal 표현력 검증: 예를 들어 ABP 채널을 전체 마스킹하고
ECG+PPG+CVP로 ABP를 복원할 수 있는지 평가한다.

Usage
-----
# 실제 checkpoint 사용:
python -m downstream.task3_imputation.run --checkpoint path/to/best.pt --target-signal abp

# 더미 테스트 (checkpoint 없이):
python -m downstream.task3_imputation.run --dummy
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample

# signal_type_key → signal_type_id 매핑
SIGNAL_TYPE_IDS: dict[str, int] = {
    "ecg": 0, "abp": 1, "eeg": 2, "ppg": 3, "cvp": 4, "co2": 5, "awp": 6,
}


# ── 결과 구조 ────────────────────────────────────────────────


@dataclass
class ImputationResult:
    """단일 채널 조합의 imputation 결과."""
    target_signal: str
    context_signals: list[str]
    mse: float
    mae: float
    pearson_r: float
    n_patches: int


# ── 다변량 PackedBatch 생성 ──────────────────────────────────


def build_multivariate_batch(
    signals: dict[str, np.ndarray],
    patch_size: int = 100,
    session_id: str = "eval_0",
    sr: float = 100.0,
) -> PackedBatch:
    """Signal type별 numpy 배열 → 다변량 PackedBatch.

    Parameters
    ----------
    signals:
        ``{signal_type_key: (n_timesteps,)}`` 딕셔너리.
        모든 신호는 동일한 길이, 동일한 sampling rate여야 한다.
    patch_size:
        패치 크기. 신호 길이가 patch_size의 배수가 되도록 잘린다.
    session_id:
        세션 ID (그루핑용).
    sr:
        sampling rate (Hz).

    Returns
    -------
    PackedBatch
    """
    samples: list[BiosignalSample] = []

    for ch_idx, (stype_key, signal) in enumerate(signals.items()):
        signal_type = SIGNAL_TYPE_IDS[stype_key]
        # patch_size 배수로 자름
        n_usable = (len(signal) // patch_size) * patch_size
        if n_usable == 0:
            continue
        signal = signal[:n_usable]

        samples.append(BiosignalSample(
            values=torch.tensor(signal, dtype=torch.float32),
            length=n_usable,
            channel_idx=ch_idx,
            recording_idx=0,
            sampling_rate=sr,
            n_channels=len(signals),
            win_start=0,
            signal_type=signal_type,
            session_id=session_id,
            spatial_id=0,  # 로컬 spatial_id (Unknown); collate가 global로 변환
        ))

    if not samples:
        raise ValueError("No valid signals for batch construction")

    # any_variate 모드로 다변량 패킹
    max_length = sum(s.length for s in samples)
    collate = PackCollate(
        max_length=max_length,
        collate_mode="any_variate",
        patch_size=patch_size,
    )
    batch = collate(samples)
    return batch


# ── Imputation 평가 ──────────────────────────────────────────


def evaluate_imputation(
    model: torch.nn.Module,
    batch: PackedBatch,
    target_signal_type: int,
    patch_size: int,
    device: torch.device,
) -> ImputationResult | None:
    """단일 배치에 대해 target 채널 imputation을 수행하고 메트릭을 계산한다.

    Parameters
    ----------
    model:
        BiosignalFoundationModelV1 or V2 (eval 모드).
    batch:
        다변량 PackedBatch.
    target_signal_type:
        마스킹 대상 signal type ID.
    patch_size:
        모델의 patch_size.
    device:
        연산 디바이스.

    Returns
    -------
    ImputationResult or None (target 채널이 배치에 없으면 None).
    """
    model.eval()
    batch.values = batch.values.to(device)
    batch.sample_id = batch.sample_id.to(device)
    batch.variate_id = batch.variate_id.to(device)

    with torch.no_grad():
        out = model(batch, task="masked")

    reconstructed = out["reconstructed"]       # (B, N, patch_size)
    patch_signal_types = out["patch_signal_types"]  # (B, N) or None
    patch_mask = out["patch_mask"]             # (B, N) bool
    patch_variate_id = out["patch_variate_id"]  # (B, N)

    if patch_signal_types is None:
        print("  [imputation] patch_signal_types is None -단일 variate?")
        return None

    # 원본 패치 추출 (정규화 스케일)
    normalized = (
        (batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]
    ).squeeze(-1)  # (B, L)
    B, L = normalized.shape
    N = L // patch_size
    original_patches = normalized.reshape(B, N, patch_size)  # (B, N, P)

    # target 채널 패치 식별
    target_mask = (patch_signal_types == target_signal_type) & patch_mask  # (B, N)

    if not target_mask.any():
        return None

    # target 패치의 복원값 vs 원본 비교
    pred = reconstructed[target_mask].cpu()    # (M, P)
    orig = original_patches[target_mask].cpu()  # (M, P)

    mse = F.mse_loss(pred, orig).item()
    mae = F.l1_loss(pred, orig).item()

    # Pearson r: 패치를 flatten한 전체 파형 기준
    pred_flat = pred.reshape(-1)
    orig_flat = orig.reshape(-1)
    pearson_r = _pearson_r(pred_flat, orig_flat)

    # context signals 추출
    all_types_in_batch = patch_signal_types[patch_mask].unique().tolist()
    id_to_name = {v: k for k, v in SIGNAL_TYPE_IDS.items()}
    context = [id_to_name.get(int(t), f"type_{t}") for t in all_types_in_batch if int(t) != target_signal_type]

    return ImputationResult(
        target_signal=id_to_name.get(target_signal_type, f"type_{target_signal_type}"),
        context_signals=context,
        mse=round(mse, 6),
        mae=round(mae, 6),
        pearson_r=round(pearson_r, 4),
        n_patches=int(target_mask.sum().item()),
    )


def _pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """1D 텐서 간 Pearson correlation coefficient."""
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = (x.norm() * y.norm()).clamp(min=1e-8)
    return (num / den).item()


# ── 더미 테스트 ──────────────────────────────────────────────


def run_dummy_test() -> list[ImputationResult]:
    """Checkpoint 없이 랜덤 모델로 더미 imputation 테스트를 수행한다."""
    from model.config import ModelConfig
    from model.v1 import BiosignalFoundationModelV1

    print("=" * 60)
    print("Task 3: Channel Imputation -Dummy Test (random model)")
    print("=" * 60)

    config = ModelConfig(d_model=64, num_layers=2, patch_size=100)
    model = BiosignalFoundationModelV1.from_config(config)
    model.eval()
    device = torch.device("cpu")

    # 테스트 시나리오
    scenarios = [
        {"target": "abp", "signals": ["ecg", "abp", "ppg"]},
        {"target": "ppg", "signals": ["ecg", "abp", "ppg"]},
        {"target": "ecg", "signals": ["ecg", "abp", "cvp"]},
    ]

    results: list[ImputationResult] = []

    for scenario in scenarios:
        target_key = scenario["target"]
        # 합성 신호 생성 (30초 = 3000 samples at 100Hz)
        n_samples = 3000
        signals = {}
        for stype in scenario["signals"]:
            t = np.arange(n_samples) / 100.0
            freq = {"ecg": 1.2, "abp": 1.1, "ppg": 1.0, "cvp": 0.9}.get(stype, 1.0)
            signals[stype] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(n_samples)

        batch = build_multivariate_batch(signals, patch_size=config.patch_size)
        target_type_id = SIGNAL_TYPE_IDS[target_key]

        result = evaluate_imputation(
            model, batch, target_type_id, config.patch_size, device,
        )
        if result is not None:
            results.append(result)

    _print_results_table(results)
    return results


# ── 실제 평가 실행 ───────────────────────────────────────────


def run_checkpoint_eval(
    checkpoint_path: str,
    target_signal: str,
    model_version: str = "v1",
    n_cases: int = 20,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
) -> list[ImputationResult]:
    """Checkpoint로 실제 imputation 평가를 수행한다."""
    from downstream.common.model_wrapper import DownstreamModelWrapper
    from downstream.common.data_utils import load_pilot_cases, extract_windows, apply_pipeline

    print("=" * 60)
    print(f"Task 3: Channel Imputation -{target_signal.upper()} reconstruction")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  model_version: {model_version}")
    print("=" * 60)

    # 1. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = DownstreamModelWrapper(checkpoint_path, model_version, device)
    model = wrapper.model
    patch_size = wrapper.patch_size

    # 2. 데이터 로드 -target + context 신호가 모두 있는 케이스
    all_types = list(SIGNAL_TYPE_IDS.keys())
    needed_types = [t for t in ["ecg", "abp", "ppg", "cvp"] if t != target_signal]
    needed_types = [target_signal] + needed_types
    cases = load_pilot_cases(n_cases=n_cases, signal_types=needed_types)

    # 3. 멀티채널 윈도우 구성 + 평가
    target_type_id = SIGNAL_TYPE_IDS[target_signal]
    results: list[ImputationResult] = []

    for case in cases:
        # target + 최소 1개 context 채널이 있는 케이스만
        if target_signal not in case.tracks:
            continue
        context_types = [t for t in case.tracks if t != target_signal]
        if not context_types:
            continue

        # 공통 길이로 자름
        min_len = min(len(case.tracks[t]) for t in case.tracks)
        win_samples = int(window_sec * 100.0)
        stride_samples = int(stride_sec * 100.0)

        for start in range(0, min_len - win_samples + 1, stride_samples):
            signals = {}
            for stype_key, arr in case.tracks.items():
                signals[stype_key] = arr[start:start + win_samples]

            try:
                batch = build_multivariate_batch(signals, patch_size=patch_size)
                result = evaluate_imputation(
                    model, batch, target_type_id, patch_size, device,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"  Warning: case {case.case_id} window @{start}: {e}")
                continue

    if results:
        _print_results_table(results)
        _print_aggregate(results)
    else:
        print("  No valid imputation results.")

    return results


# ── 결과 출력 ────────────────────────────────────────────────


def _print_results_table(results: list[ImputationResult]) -> None:
    """결과 테이블 출력."""
    print(f"\n{'Target':<8} {'Context':<25} {'MSE':<12} {'MAE':<12} {'Pearson r':<12} {'#Patches':<10}")
    print("-" * 79)
    for r in results:
        ctx = "+".join(r.context_signals)
        print(f"{r.target_signal:<8} {ctx:<25} {r.mse:<12.6f} {r.mae:<12.6f} {r.pearson_r:<12.4f} {r.n_patches:<10}")


def _print_aggregate(results: list[ImputationResult]) -> None:
    """집계 통계 출력."""
    if not results:
        return
    mse_vals = [r.mse for r in results]
    mae_vals = [r.mae for r in results]
    r_vals = [r.pearson_r for r in results]

    print(f"\n--- Aggregate ({len(results)} windows) ---")
    print(f"  MSE:       mean={np.mean(mse_vals):.6f}  std={np.std(mse_vals):.6f}")
    print(f"  MAE:       mean={np.mean(mae_vals):.6f}  std={np.std(mae_vals):.6f}")
    print(f"  Pearson r: mean={np.mean(r_vals):.4f}  std={np.std(r_vals):.4f}")


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3: Channel Imputation evaluation",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained checkpoint path")
    parser.add_argument("--target-signal", type=str, default="abp", help="Target signal to mask/reconstruct")
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--n-cases", type=int, default=20, help="Number of pilot cases to evaluate")
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--dummy", action="store_true", help="Run dummy test (no checkpoint needed)")
    args = parser.parse_args()

    if args.dummy:
        run_dummy_test()
    elif args.checkpoint:
        run_checkpoint_eval(
            checkpoint_path=args.checkpoint,
            target_signal=args.target_signal,
            model_version=args.model_version,
            n_cases=args.n_cases,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
        )
    else:
        print("Error: provide --checkpoint or --dummy")
        sys.exit(1)


if __name__ == "__main__":
    main()
