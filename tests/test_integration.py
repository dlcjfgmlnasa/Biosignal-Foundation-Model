# -*- coding:utf-8 -*-
"""통합 테스트: data 파이프라인 → TransformerEncoder end-to-end.

가짜 생체신호(ECG, EEG, EMG 등)를 생성하고, BiosignalDataset → PackCollate →
간단한 Linear projection → TransformerEncoder로 forward pass를 수행한다.
"""
import tempfile
from functools import partial
from pathlib import Path

import pytest
import torch
from torch import nn

from data.collate import PackCollate, PackedBatch
from data.dataloader import create_dataloader
from data.dataset import BiosignalDataset, RecordingManifest
from module.packed_scaler import PackedStdScaler, PackedNOPScaler, PackedAbsMeanScaler
from module.position import BinaryAttentionBias, QueryKeyProjection, RotaryProjection
from module.transformer import TransformerEncoder
from model.biosignal_model import BiosignalFoundationModel
from model.checkpoint import save_checkpoint, load_checkpoint
from loss.criterion import CombinedLoss
from loss.masked_mse_loss import MaskedPatchLoss, create_patch_mask
from loss.next_prediction_loss import NextPredictionLoss


# ── 가짜 생체신호 생성 헬퍼 ──────────────────────────────────────


def make_fake_ecg(
    n_channels: int = 3,
    n_timesteps: int = 2500,
    sampling_rate: float = 500.0,
) -> torch.Tensor:
    """가짜 ECG: 심박 주기 ~1초, 진폭 ~1mV."""
    t = torch.linspace(0, n_timesteps / sampling_rate, n_timesteps)
    # R-peak 유사 파형: 주기적 sharp spike + 저주파 baseline
    ecg_base = torch.sin(2 * torch.pi * 1.2 * t)  # ~72 bpm
    r_peak = torch.exp(-((t % (1 / 1.2) - 0.05) ** 2) / 0.0005)
    signal = ecg_base * 0.3 + r_peak * 1.5
    # 채널별 약간의 변형
    channels = []
    for ch in range(n_channels):
        noise = torch.randn(n_timesteps) * 0.05
        phase_shift = ch * 0.1
        channels.append(signal * (1 + phase_shift) + noise)
    return torch.stack(channels)  # (n_channels, n_timesteps)


def make_fake_eeg(
    n_channels: int = 16,
    n_timesteps: int = 2560,
    sampling_rate: float = 256.0,
) -> torch.Tensor:
    """가짜 EEG: alpha(10Hz) + beta(20Hz) + noise."""
    t = torch.linspace(0, n_timesteps / sampling_rate, n_timesteps)
    channels = []
    for ch in range(n_channels):
        alpha = torch.sin(2 * torch.pi * (10 + ch * 0.2) * t) * 20  # μV
        beta = torch.sin(2 * torch.pi * (20 + ch * 0.3) * t) * 5
        noise = torch.randn(n_timesteps) * 3
        channels.append(alpha + beta + noise)
    return torch.stack(channels)


def make_fake_emg(
    n_channels: int = 2,
    n_timesteps: int = 1000,
    sampling_rate: float = 1000.0,
) -> torch.Tensor:
    """가짜 EMG: 고주파 burst + noise."""
    t = torch.linspace(0, n_timesteps / sampling_rate, n_timesteps)
    channels = []
    for ch in range(n_channels):
        burst = torch.sin(2 * torch.pi * 150 * t) * torch.exp(
            -(((t - 0.5) / 0.1) ** 2)
        )
        noise = torch.randn(n_timesteps) * 0.1
        channels.append(burst + noise)
    return torch.stack(channels)


def save_recordings(
    recordings: list[torch.Tensor],
    tmpdir: Path,
    sampling_rate: float,
    signal_type: int,
    session_id: str = "",
) -> list[RecordingManifest]:
    tmpdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, rec in enumerate(recordings):
        pt_path = tmpdir / f"rec_{signal_type}_{i:04d}.pt"
        torch.save(rec, pt_path)
        manifest.append(
            RecordingManifest(
                path=str(pt_path),
                n_channels=rec.shape[0],
                n_timesteps=rec.shape[1],
                sampling_rate=sampling_rate,
                signal_type=signal_type,
                session_id=session_id,
            )
        )
    return manifest


# ── 테스트 ────────────────────────────────────────────────────────


class TestSingleModalECGPipeline:
    """ECG 단일 모달: Dataset → PackCollate → Projection → Encoder."""

    def test_ecg_forward_pass(self):
        d_model = 64
        max_length = 256
        encoder = TransformerEncoder(
            d_model=d_model, num_layers=2,
            var_attn_bias_layer=BinaryAttentionBias,
        )
        encoder.eval()
        input_proj = nn.Linear(1, d_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=3, n_timesteps=1000, sampling_rate=500.0)
            manifest = save_recordings([ecg], Path(tmpdir), 500.0, signal_type=0)
            ds = BiosignalDataset(manifest, window_seconds=0.5, stride_seconds=0.5)

            assert len(ds) > 0

            collate = PackCollate(max_length=max_length)
            samples = [ds[i] for i in range(min(len(ds), 6))]
            batch = collate(samples)

            assert isinstance(batch, PackedBatch)
            assert batch.values.shape[1] == max_length

            # (batch, max_length) → (batch, max_length, 1) → (batch, max_length, d_model)
            x = input_proj(batch.values.unsqueeze(-1))
            out = encoder(x, var_id=batch.variate_id, time_id=None)

            assert out.shape == (batch.values.shape[0], max_length, d_model)
            assert torch.isfinite(out).all()


class TestSingleModalEEGPipeline:
    """EEG 16채널: 고채널 수 + BinaryAttentionBias + RoPE."""

    def test_eeg_with_rope(self):
        d_model = 128
        max_length = 512
        encoder = TransformerEncoder(
            d_model=d_model, num_layers=2, num_heads=4, num_groups=2,
            var_attn_bias_layer=BinaryAttentionBias,
            time_qk_proj_layer=partial(
                QueryKeyProjection, proj_layer=RotaryProjection,
            ),
        )
        encoder.eval()
        input_proj = nn.Linear(1, d_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            eeg = make_fake_eeg(n_channels=16, n_timesteps=2560, sampling_rate=256.0)
            manifest = save_recordings([eeg], Path(tmpdir), 256.0, signal_type=1)
            ds = BiosignalDataset(manifest, window_seconds=1.0, stride_seconds=1.0)

            collate = PackCollate(max_length=max_length)
            samples = [ds[i] for i in range(min(len(ds), 8))]
            batch = collate(samples)

            x = input_proj(batch.values.unsqueeze(-1))
            time_id = torch.arange(max_length).unsqueeze(0).expand(x.shape[0], -1)
            out = encoder(x, var_id=batch.variate_id, time_id=time_id)

            assert out.shape == (x.shape[0], max_length, d_model)
            assert torch.isfinite(out).all()


class TestMultiModalPipeline:
    """ECG + EEG + EMG 동시 세션: cross-modal any-variate 패킹 → Encoder."""

    def test_cross_modal_session(self):
        d_model = 64
        # 짧은 신호로 max_length 안에 다 들어오도록 설정
        max_length = 2048
        encoder = TransformerEncoder(
            d_model=d_model, num_layers=2,
            var_attn_bias_layer=BinaryAttentionBias,
        )
        encoder.eval()
        input_proj = nn.Linear(1, d_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # 같은 세션, 짧은 신호 (모두 1초 분량)
            ecg = make_fake_ecg(n_channels=1, n_timesteps=100, sampling_rate=100.0)
            eeg = make_fake_eeg(n_channels=2, n_timesteps=100, sampling_rate=100.0)
            emg = make_fake_emg(n_channels=1, n_timesteps=100, sampling_rate=100.0)

            manifest_ecg = save_recordings([ecg], tmpdir, 100.0, 0, session_id="S001")
            manifest_eeg = save_recordings([eeg], tmpdir, 100.0, 1, session_id="S001")
            manifest_emg = save_recordings([emg], tmpdir, 100.0, 4, session_id="S001")

            ds = BiosignalDataset(
                manifest_ecg + manifest_eeg + manifest_emg,
                window_seconds=1.0, stride_seconds=1.0,
            )
            assert len(ds) > 0

            collate = PackCollate(max_length=max_length)
            samples = [ds[i] for i in range(len(ds))]
            batch = collate(samples)

            # 다양한 signal_type이 존재
            unique_types = batch.signal_types.unique().tolist()
            assert len(unique_types) >= 2, f"signal_types={unique_types}"

            x = input_proj(batch.values.unsqueeze(-1))
            out = encoder(x, var_id=batch.variate_id)

            assert out.shape == (x.shape[0], max_length, d_model)
            assert torch.isfinite(out).all()


class TestScalerThenEncoder:
    """PackedScaler로 정규화 후 Encoder에 입력."""

    @pytest.mark.parametrize("ScalerCls", [PackedStdScaler, PackedAbsMeanScaler, PackedNOPScaler])
    def test_scaler_before_encoder(self, ScalerCls):
        d_model = 64
        max_length = 128
        encoder = TransformerEncoder(d_model=d_model, num_layers=1)
        encoder.eval()
        input_proj = nn.Linear(1, d_model)
        scaler = ScalerCls() if ScalerCls is not PackedStdScaler else ScalerCls(correction=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=2, n_timesteps=500, sampling_rate=250.0)
            manifest = save_recordings([ecg], Path(tmpdir), 250.0, signal_type=0)
            ds = BiosignalDataset(manifest, window_seconds=0.5, stride_seconds=0.5)

            collate = PackCollate(max_length=max_length)
            samples = [ds[i] for i in range(min(len(ds), 4))]
            batch = collate(samples)

            # Scaler 적용
            target = batch.values.unsqueeze(-1)  # (batch, max_length, 1)
            loc, scale = scaler(
                target,
                observed_mask=(batch.sample_id > 0).unsqueeze(-1),
                sample_id=batch.sample_id,
                variate_id=batch.variate_id,
            )
            normalized = (target - loc) / scale

            x = input_proj(normalized)
            out = encoder(x, var_id=batch.variate_id)

            assert out.shape == (x.shape[0], max_length, d_model)
            assert torch.isfinite(out).all()


class TestDataLoaderToEncoder:
    """DataLoader 루프 → Encoder: 실제 학습 루프와 동일한 흐름."""

    def test_full_training_loop_shape(self):
        d_model = 64
        max_length = 256
        encoder = TransformerEncoder(
            d_model=d_model, num_layers=2,
            var_attn_bias_layer=BinaryAttentionBias,
            time_qk_proj_layer=partial(
                QueryKeyProjection, proj_layer=RotaryProjection,
            ),
        )
        encoder.eval()
        input_proj = nn.Linear(1, d_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            all_manifests = []
            for i in range(5):
                ecg = make_fake_ecg(
                    n_channels=2,
                    n_timesteps=500 + i * 100,
                    sampling_rate=250.0,
                )
                subdir = tmpdir / str(i)
                m = save_recordings([ecg], subdir, 250.0, signal_type=0)
                all_manifests.extend(m)

            ds = BiosignalDataset(all_manifests, window_seconds=1.0, stride_seconds=1.0)
            loader = create_dataloader(ds, max_length=max_length, batch_size=8, shuffle=False)

            total_batches = 0
            for batch in loader:
                assert isinstance(batch, PackedBatch)

                x = input_proj(batch.values.unsqueeze(-1))
                time_id = torch.arange(max_length).unsqueeze(0).expand(x.shape[0], -1)
                out = encoder(x, var_id=batch.variate_id, time_id=time_id)

                assert out.shape[1] == max_length
                assert out.shape[2] == d_model
                assert torch.isfinite(out).all()
                total_batches += 1

            assert total_batches > 0


class TestGradientEndToEnd:
    """Encoder 출력에서 loss → backward → gradient가 input_proj와 encoder에 흐르는지."""

    def test_gradient_through_pipeline(self):
        d_model = 64
        max_length = 64
        encoder = TransformerEncoder(d_model=d_model, num_layers=1)
        input_proj = nn.Linear(1, d_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=1, n_timesteps=200, sampling_rate=250.0)
            manifest = save_recordings([ecg], Path(tmpdir), 250.0, signal_type=0)
            ds = BiosignalDataset(manifest, max_length=max_length)

            collate = PackCollate(max_length=max_length)
            batch = collate([ds[0]])

            x = input_proj(batch.values.unsqueeze(-1))
            out = encoder(x)

            # 간단한 MSE loss
            target = torch.zeros_like(out)
            loss = ((out - target) ** 2).mean()
            loss.backward()

            # encoder, input_proj 파라미터에 gradient가 있어야 함
            for name, p in encoder.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, f"No grad for encoder.{name}"

            for name, p in input_proj.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, f"No grad for input_proj.{name}"


class TestMoEWithBiosignal:
    """MoE Encoder에 생체신호 입력."""

    def test_moe_forward(self):
        d_model = 64
        max_length = 128
        encoder = TransformerEncoder(
            d_model=d_model, num_layers=2, use_moe=True,
        )
        encoder.eval()
        torch.nn.init.normal_(encoder.centroid)
        input_proj = nn.Linear(1, d_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=2, n_timesteps=500, sampling_rate=250.0)
            manifest = save_recordings([ecg], Path(tmpdir), 250.0, signal_type=0)
            ds = BiosignalDataset(manifest, window_seconds=0.5, stride_seconds=0.5)

            collate = PackCollate(max_length=max_length)
            samples = [ds[i] for i in range(min(len(ds), 4))]
            batch = collate(samples)

            x = input_proj(batch.values.unsqueeze(-1))
            out = encoder(x)

            assert out.shape == (x.shape[0], max_length, d_model)
            assert torch.isfinite(out).all()


class TestPackedBatchMetadataPreserved:
    """data pipeline 메타데이터가 encoder까지 올바르게 전달되는지."""

    def test_metadata_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # 짧은 신호로 padding이 발생하도록 설정
            ecg = make_fake_ecg(n_channels=1, n_timesteps=100, sampling_rate=500.0)
            eeg = make_fake_eeg(n_channels=1, n_timesteps=50, sampling_rate=256.0)

            manifest_ecg = save_recordings([ecg], tmpdir, 500.0, 0)
            manifest_eeg = save_recordings([eeg], tmpdir, 256.0, 1)

            ds = BiosignalDataset(manifest_ecg + manifest_eeg)
            # max_length를 총 신호 길이보다 크게 → 패딩 발생
            collate = PackCollate(max_length=256)
            samples = [ds[i] for i in range(len(ds))]
            batch = collate(samples)

            # sample_id: padding은 0, 나머지는 >0
            assert (batch.sample_id >= 0).all()
            padding_mask = batch.sample_id == 0
            assert padding_mask.any(), "패딩이 있어야 함"

            # variate_id: padding은 0
            assert (batch.variate_id[padding_mask] == 0).all()

            # signal_types에 ECG(0)과 EEG(1) 모두 존재
            types = batch.signal_types.unique().tolist()
            assert 0 in types and 1 in types

            # sampling_rates에 500.0과 256.0 존재
            rates = batch.sampling_rates.unique().tolist()
            assert 500.0 in rates and 256.0 in rates

            # lengths 합 = non-padding timestep 수
            total_filled = (batch.sample_id > 0).sum().item()
            assert batch.lengths.sum().item() == total_filled


# ── Phase 4: BiosignalFoundationModel 통합 테스트 ────────────────


class TestBiosignalFoundationModel:
    def test_basic_forward(self):
        """기본 forward: PackedBatch → model → output dict."""
        d_model, num_layers, patch_size = 64, 2, 16

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=2, n_timesteps=500, sampling_rate=250.0)
            manifest = save_recordings([ecg], Path(tmpdir), 250.0, 0)

            ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.4)
            collate = PackCollate(max_length=512, patch_size=patch_size)
            samples = [ds[i] for i in range(min(8, len(ds)))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=num_layers,
                patch_size=patch_size,
            )
            out = model(batch)

            assert "encoded" in out
            assert "reconstructed" in out
            assert "patch_mask" in out

            B = batch.values.shape[0]
            N = batch.values.shape[1] // patch_size
            assert out["encoded"].shape == (B, N, d_model)
            assert out["reconstructed"].shape == (B, N, patch_size)
            assert out["patch_mask"].shape == (B, N)

    def test_gradient_flow(self):
        """모델 전체를 통해 gradient가 흐른다."""
        d_model, num_layers, patch_size = 64, 1, 8

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=1, n_timesteps=200, sampling_rate=100.0)
            manifest = save_recordings([ecg], Path(tmpdir), 100.0, 0)

            ds = BiosignalDataset(manifest, window_seconds=0.5, stride_seconds=0.5)
            collate = PackCollate(max_length=128, patch_size=patch_size)
            samples = [ds[i] for i in range(min(4, len(ds)))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=num_layers,
                patch_size=patch_size,
                use_rope=False,
                use_var_attn_bias=False,
            )
            out = model(batch)

            mask = out["patch_mask"]
            # reconstructed + cross_pred 모두 사용하여 gradient 확인
            loss = (
                (out["reconstructed"][mask] ** 2).mean()
                + (out["cross_pred"][mask] ** 2).mean()
            )
            loss.backward()

            # 모든 파라미터에 gradient 존재
            # task="masked"(기본값)일 때 next_head, horizon_embed는 사용되지 않음
            skip_prefixes = ("next_head.", "horizon_embed.")
            for name, param in model.named_parameters():
                if param.requires_grad and not name.startswith(skip_prefixes):
                    assert param.grad is not None, f"{name}에 gradient가 없습니다"

    def test_with_rope_and_bias(self):
        """RoPE + BinaryAttentionBias 활성화 시 forward 성공."""
        d_model, num_layers, patch_size = 64, 2, 16

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=2, n_timesteps=400, sampling_rate=250.0)
            manifest = save_recordings([ecg], Path(tmpdir), 250.0, 0)

            ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.4)
            collate = PackCollate(max_length=256, patch_size=patch_size)
            samples = [ds[i] for i in range(min(6, len(ds)))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=num_layers,
                patch_size=patch_size,
                use_rope=True,
                use_var_attn_bias=True,
            )
            out = model(batch)

            assert out["encoded"].shape[2] == d_model
            assert out["patch_mask"].any()

    def test_with_overlapping_stride(self):
        """overlapping stride로 모델 forward."""
        d_model, num_layers, P, S = 64, 1, 16, 8

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=1, n_timesteps=200, sampling_rate=100.0)
            manifest = save_recordings([ecg], Path(tmpdir), 100.0, 0)

            ds = BiosignalDataset(manifest, window_seconds=0.5, stride_seconds=0.5)
            collate = PackCollate(max_length=128, patch_size=P, stride=S)
            samples = [ds[i] for i in range(min(4, len(ds)))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=num_layers,
                patch_size=P,
                stride=S,
                use_rope=False,
                use_var_attn_bias=False,
            )
            out = model(batch)

            L = batch.values.shape[1]
            N = (L - P) // S + 1
            assert out["encoded"].shape[1] == N

    def test_reconstruction_shape(self):
        """reconstruction head 출력이 원본 패치와 같은 크기."""
        d_model, num_layers, patch_size = 64, 1, 10

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=1, n_timesteps=100, sampling_rate=100.0)
            manifest = save_recordings([ecg], Path(tmpdir), 100.0, 0)

            ds = BiosignalDataset(manifest)
            collate = PackCollate(max_length=200, patch_size=patch_size)
            samples = [ds[i] for i in range(len(ds))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=num_layers,
                patch_size=patch_size,
                use_rope=False,
                use_var_attn_bias=False,
            )
            out = model(batch)

            assert out["reconstructed"].shape[-1] == patch_size

    def test_horizon1_backward_compatible(self):
        """horizon=1은 기존 next_pred 동작과 동일한 출력."""
        d_model, num_layers, patch_size = 64, 1, 16

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=2, n_timesteps=500, sampling_rate=250.0)
            manifest = save_recordings([ecg], Path(tmpdir), 250.0, 0)

            ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.4)
            collate = PackCollate(max_length=512, patch_size=patch_size)
            samples = [ds[i] for i in range(min(4, len(ds)))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=num_layers,
                patch_size=patch_size,
                use_rope=False,
                use_var_attn_bias=False,
                max_horizon=5,
            )
            model.eval()

            with torch.no_grad():
                out_default = model(batch, task="next_pred")
                out_h1 = model(batch, task="next_pred", horizon=1)

            assert torch.allclose(out_default["next_pred"], out_h1["next_pred"])

    def test_horizon_output_shape(self):
        """horizon>1일 때 next_pred 출력 shape이 (B, N, patch_size)."""
        d_model, num_layers, patch_size = 64, 1, 16

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=2, n_timesteps=500, sampling_rate=250.0)
            manifest = save_recordings([ecg], Path(tmpdir), 250.0, 0)

            ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.4)
            collate = PackCollate(max_length=512, patch_size=patch_size)
            samples = [ds[i] for i in range(min(4, len(ds)))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=num_layers,
                patch_size=patch_size,
                use_rope=False,
                use_var_attn_bias=False,
                max_horizon=5,
            )
            model.eval()

            B = batch.values.shape[0]
            N = batch.values.shape[1] // patch_size

            for H in [1, 3, 5]:
                with torch.no_grad():
                    out = model(batch, task="next_pred", horizon=H)
                assert out["next_pred"].shape == (B, N, patch_size)

    def test_horizon_gradient_flow(self):
        """horizon_embed을 포함한 모든 파라미터에 gradient가 흐른다."""
        d_model, num_layers, patch_size = 64, 1, 8

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=1, n_timesteps=200, sampling_rate=100.0)
            manifest = save_recordings([ecg], Path(tmpdir), 100.0, 0)

            ds = BiosignalDataset(manifest, window_seconds=0.5, stride_seconds=0.5)
            collate = PackCollate(max_length=128, patch_size=patch_size)
            samples = [ds[i] for i in range(min(4, len(ds)))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=num_layers,
                patch_size=patch_size,
                use_rope=False,
                use_var_attn_bias=False,
                max_horizon=5,
            )
            out = model(batch, task="next_pred", horizon=3)

            mask = out["patch_mask"]
            loss = (out["next_pred"][mask] ** 2).mean()
            loss.backward()

            # task="next_pred" → head(reconstruction), cross_head는 미사용
            skip_prefixes = ("head.", "cross_head.")
            for name, param in model.named_parameters():
                if param.requires_grad and not name.startswith(skip_prefixes):
                    assert param.grad is not None, f"{name}에 gradient가 없습니다"

            # horizon_embed에 반드시 gradient 존재
            assert model.horizon_embed.weight.grad is not None

    def test_horizon_different_outputs(self):
        """다른 horizon 값은 다른 출력을 생성한다."""
        d_model, num_layers, patch_size = 64, 1, 16

        with tempfile.TemporaryDirectory() as tmpdir:
            ecg = make_fake_ecg(n_channels=1, n_timesteps=300, sampling_rate=250.0)
            manifest = save_recordings([ecg], Path(tmpdir), 250.0, 0)

            ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.4)
            collate = PackCollate(max_length=256, patch_size=patch_size)
            samples = [ds[i] for i in range(min(4, len(ds)))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=num_layers,
                patch_size=patch_size,
                use_rope=False,
                use_var_attn_bias=False,
                max_horizon=5,
            )
            model.eval()

            with torch.no_grad():
                out_h1 = model(batch, task="next_pred", horizon=1)
                out_h3 = model(batch, task="next_pred", horizon=3)

            # horizon embedding이 다르므로 출력이 달라야 함
            assert not torch.allclose(out_h1["next_pred"], out_h3["next_pred"])

    def test_combined_loss_horizon_shift(self):
        """CombinedLoss가 horizon에 맞게 shift된 target을 사용한다."""
        B, N, P = 2, 10, 8
        H = 3

        original = torch.randn(B, N, P)
        next_pred = torch.randn(B, N, P)
        reconstructed = torch.randn(B, N, P)
        pred_mask = torch.zeros(B, N, dtype=torch.bool)
        patch_mask = torch.ones(B, N, dtype=torch.bool)
        sample_id = torch.ones(B, N, dtype=torch.long)
        variate_id = torch.ones(B, N, dtype=torch.long)

        criterion = CombinedLoss(alpha=0.0, beta=1.0)

        # horizon=1
        losses_h1 = criterion(
            reconstructed, next_pred, original,
            pred_mask, patch_mask, sample_id, variate_id,
            horizon=1,
        )
        # horizon=H
        losses_hH = criterion(
            reconstructed, next_pred, original,
            pred_mask, patch_mask, sample_id, variate_id,
            horizon=H,
        )

        # 서로 다른 shift → 다른 loss 값
        assert not torch.allclose(losses_h1["next_loss"], losses_hH["next_loss"])

        # horizon weight: H=3이면 1/3 스케일
        # 수동 계산으로 검증
        target_h3 = original[:, H:, :]
        pred_h3 = next_pred[:, :-H, :]
        expected = ((pred_h3 - target_h3) ** 2).mean() * (1.0 / H)
        assert torch.allclose(losses_hH["next_loss"], expected, atol=1e-6)

    def test_combined_loss_horizon_short_sequence(self):
        """시퀀스 길이 < horizon이면 valid=False로 loss=0."""
        B, N, P = 2, 3, 8
        H = 5  # N < H

        original = torch.randn(B, N, P)
        next_pred = torch.randn(B, N, P)
        reconstructed = torch.randn(B, N, P)
        pred_mask = torch.zeros(B, N, dtype=torch.bool)
        patch_mask = torch.ones(B, N, dtype=torch.bool)
        # 각 패치가 다른 sample → valid pair 없음
        sample_id = torch.arange(N).unsqueeze(0).expand(B, -1) + 1
        variate_id = torch.ones(B, N, dtype=torch.long)

        criterion = CombinedLoss(alpha=0.0, beta=1.0)
        losses = criterion(
            reconstructed, next_pred, original,
            pred_mask, patch_mask, sample_id, variate_id,
            horizon=H,
        )
        # N=3, H=5 → slicing [:, 5:] is empty → loss = 0
        assert losses["next_loss"].item() == 0.0


# ── Phase 5: Inference API 테스트 ──────────────────────────────


def _make_model_and_batch(
    d_model: int = 64,
    num_layers: int = 1,
    patch_size: int = 16,
    max_horizon: int = 5,
    max_length: int = 512,
    n_channels: int = 2,
    n_timesteps: int = 500,
    sampling_rate: float = 250.0,
) -> tuple:
    """테스트용 모델과 batch를 생성하는 헬퍼."""
    tmpdir = tempfile.mkdtemp()
    ecg = make_fake_ecg(n_channels=n_channels, n_timesteps=n_timesteps, sampling_rate=sampling_rate)
    manifest = save_recordings([ecg], Path(tmpdir), sampling_rate, 0)
    ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.4)
    collate = PackCollate(max_length=max_length, patch_size=patch_size)
    samples = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate(samples)
    model = BiosignalFoundationModel(
        d_model=d_model,
        num_layers=num_layers,
        patch_size=patch_size,
        use_rope=False,
        use_var_attn_bias=False,
        max_horizon=max_horizon,
    )
    return model, batch, tmpdir


class TestExtractFeatures:
    def test_extract_features(self):
        """shape, patch_mask 존재, reconstructed key 없음 확인."""
        model, batch, _ = _make_model_and_batch()

        out = model.extract_features(batch)

        assert "encoded" in out
        assert "patch_mask" in out
        assert "loc" in out
        assert "scale" in out
        assert "reconstructed" not in out

        B = batch.values.shape[0]
        P = model.patch_size
        N = batch.values.shape[1] // P
        assert out["encoded"].shape == (B, N, model.d_model)
        assert out["patch_mask"].shape == (B, N)


class TestForecast:
    def test_forecast_h1(self):
        """H=1 예측: shape, finite, denormalize 전후 값 차이."""
        model, batch, _ = _make_model_and_batch()

        pred_denorm = model.forecast(batch, horizon=1, denormalize=True)
        pred_norm = model.forecast(batch, horizon=1, denormalize=False)

        B = batch.values.shape[0]
        P = model.patch_size
        N = batch.values.shape[1] // P

        assert pred_denorm.shape == (B, N, P)
        assert pred_norm.shape == (B, N, P)
        assert torch.isfinite(pred_denorm).all()
        assert torch.isfinite(pred_norm).all()
        # denormalize하면 값이 달라져야 함
        assert not torch.allclose(pred_denorm, pred_norm)

    def test_forecast_multi_horizon(self):
        """H=1,3,5로 각각 예측, 서로 다른 결과."""
        model, batch, _ = _make_model_and_batch()

        preds = {}
        for h in [1, 3, 5]:
            preds[h] = model.forecast(batch, horizon=h, denormalize=False)

        assert not torch.allclose(preds[1], preds[3])
        assert not torch.allclose(preds[1], preds[5])
        assert not torch.allclose(preds[3], preds[5])


class TestGenerate:
    def test_generate_single_step(self):
        """n_steps=1 생성: shape=(1, B, P)."""
        model, batch, _ = _make_model_and_batch(n_channels=1)

        result = model.generate(batch, n_steps=1, denormalize=False)

        B = batch.values.shape[0]
        P = model.patch_size
        assert result.shape == (1, B, P)
        assert torch.isfinite(result).all()

    def test_generate_multi_step(self):
        """n_steps=3 생성: shape=(3, B, P), finite."""
        model, batch, _ = _make_model_and_batch(n_channels=1)

        result = model.generate(batch, n_steps=3, denormalize=False)

        B = batch.values.shape[0]
        P = model.patch_size
        assert result.shape == (3, B, P)
        assert torch.isfinite(result).all()


class TestCheckpointRoundtrip:
    def test_checkpoint_roundtrip(self):
        """save → load → 동일 output 확인."""
        model, batch, tmpdir = _make_model_and_batch()
        model.eval()

        # 원본 출력
        with torch.no_grad():
            out_before = model(batch, task="masked")

        # save
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        config = {"d_model": 64, "num_layers": 1, "patch_size": 16}
        save_checkpoint(ckpt_path, model, epoch=5, config=config)

        # load into fresh model
        model2 = BiosignalFoundationModel(
            d_model=64, num_layers=1, patch_size=16,
            use_rope=False, use_var_attn_bias=False, max_horizon=5,
        )
        state = load_checkpoint(ckpt_path, model2)
        model2.eval()

        assert state["epoch"] == 5
        assert state["config"] == config

        with torch.no_grad():
            out_after = model2(batch, task="masked")

        assert torch.allclose(out_before["encoded"], out_after["encoded"])
        assert torch.allclose(out_before["reconstructed"], out_after["reconstructed"])


# ── Cross-Modal Loss 테스트 ─────────────────────────────────────


class TestCrossModalLoss:
    """Cross-modal prediction loss 테스트."""

    def test_single_variate_cross_modal_zero(self):
        """단일 variate → cross_modal_loss == 0."""
        B, N, P = 2, 10, 8
        original = torch.randn(B, N, P)
        next_pred = torch.randn(B, N, P)
        cross_pred = torch.randn(B, N, P)
        patch_mask = torch.ones(B, N, dtype=torch.bool)
        sample_id = torch.ones(B, N, dtype=torch.long)
        variate_id = torch.ones(B, N, dtype=torch.long)  # 단일 variate
        time_id = torch.arange(N).unsqueeze(0).expand(B, -1)

        loss_fn = NextPredictionLoss(cross_modal_weight=1.0)
        result = loss_fn(
            next_pred, cross_pred, original,
            patch_mask, sample_id, variate_id,
            time_id=time_id, horizon=1,
        )

        assert result["cross_modal_loss"].item() == 0.0

    def test_multi_variate_cross_modal_nonzero(self):
        """다중 variate + 같은 time_id → cross_modal_loss > 0."""
        B, N, P = 1, 6, 4
        original = torch.randn(B, N, P)
        next_pred = torch.randn(B, N, P)
        cross_pred = torch.randn(B, N, P)
        patch_mask = torch.ones(B, N, dtype=torch.bool)
        sample_id = torch.ones(B, N, dtype=torch.long)
        # 3 patches variate=1, 3 patches variate=2, 같은 time_id
        variate_id = torch.tensor([[1, 1, 1, 2, 2, 2]])
        time_id = torch.tensor([[0, 1, 2, 0, 1, 2]])

        loss_fn = NextPredictionLoss(cross_modal_weight=1.0)
        result = loss_fn(
            next_pred, cross_pred, original,
            patch_mask, sample_id, variate_id,
            time_id=time_id, horizon=1,
        )

        assert result["cross_modal_loss"].item() > 0.0

    def test_cross_modal_weight_zero_disables(self):
        """gamma=0 → cross_modal_loss 항상 0."""
        B, N, P = 1, 6, 4
        original = torch.randn(B, N, P)
        next_pred = torch.randn(B, N, P)
        cross_pred = torch.randn(B, N, P)
        patch_mask = torch.ones(B, N, dtype=torch.bool)
        sample_id = torch.ones(B, N, dtype=torch.long)
        variate_id = torch.tensor([[1, 1, 1, 2, 2, 2]])
        time_id = torch.tensor([[0, 1, 2, 0, 1, 2]])

        loss_fn = NextPredictionLoss(cross_modal_weight=0.0)
        result = loss_fn(
            next_pred, cross_pred, original,
            patch_mask, sample_id, variate_id,
            time_id=time_id, horizon=1,
        )

        assert result["cross_modal_loss"].item() == 0.0

    def test_combined_loss_with_cross_modal(self):
        """CombinedLoss(gamma>0)에서 cross_modal_loss key 존재 + 값 > 0."""
        B, N, P = 1, 6, 4
        original = torch.randn(B, N, P)
        next_pred = torch.randn(B, N, P)
        reconstructed = torch.randn(B, N, P)
        cross_pred = torch.randn(B, N, P)
        pred_mask = torch.zeros(B, N, dtype=torch.bool)
        pred_mask[0, 0] = True
        patch_mask = torch.ones(B, N, dtype=torch.bool)
        sample_id = torch.ones(B, N, dtype=torch.long)
        variate_id = torch.tensor([[1, 1, 1, 2, 2, 2]])
        time_id = torch.tensor([[0, 1, 2, 0, 1, 2]])

        criterion = CombinedLoss(alpha=1.0, beta=1.0, gamma=1.0)
        losses = criterion(
            reconstructed, next_pred, original,
            pred_mask, patch_mask, sample_id, variate_id,
            horizon=1, cross_pred=cross_pred, time_id=time_id,
        )

        assert "cross_modal_loss" in losses
        assert losses["cross_modal_loss"].item() > 0.0
        assert losses["total"].item() > 0.0

    def test_combined_loss_backward_compatible(self):
        """gamma=0, cross_pred=None → 기존 CombinedLoss와 동일 동작."""
        B, N, P = 2, 10, 8
        original = torch.randn(B, N, P)
        next_pred = torch.randn(B, N, P)
        reconstructed = torch.randn(B, N, P)
        pred_mask = torch.zeros(B, N, dtype=torch.bool)
        pred_mask[:, :3] = True
        patch_mask = torch.ones(B, N, dtype=torch.bool)
        sample_id = torch.ones(B, N, dtype=torch.long)
        variate_id = torch.ones(B, N, dtype=torch.long)

        criterion = CombinedLoss(alpha=1.0, beta=1.0, gamma=0.0)
        losses = criterion(
            reconstructed, next_pred, original,
            pred_mask, patch_mask, sample_id, variate_id,
            horizon=1,
        )

        assert losses["cross_modal_loss"].item() == 0.0
        assert losses["masked_loss"].item() > 0.0
        assert losses["next_loss"].item() > 0.0


class TestMaskedPatchLoss:
    """MaskedPatchLoss 단위 테스트."""

    def test_basic(self):
        B, N, P = 2, 5, 4
        pred = torch.randn(B, N, P)
        target = torch.randn(B, N, P)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, :2] = True

        loss_fn = MaskedPatchLoss()
        loss = loss_fn(pred, target, mask)

        expected = ((pred[:, :2] - target[:, :2]) ** 2).mean()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_no_mask(self):
        B, N, P = 2, 5, 4
        pred = torch.randn(B, N, P)
        target = torch.randn(B, N, P)
        mask = torch.zeros(B, N, dtype=torch.bool)

        loss_fn = MaskedPatchLoss()
        loss = loss_fn(pred, target, mask)

        assert loss.item() == 0.0


class TestCreatePatchMask:
    """create_patch_mask 단위 테스트."""

    def test_basic_mask_ratio(self):
        B, N = 2, 20
        patch_mask = torch.ones(B, N, dtype=torch.bool)

        pred_mask = create_patch_mask(patch_mask, mask_ratio=0.5)

        assert pred_mask.shape == (B, N)
        assert pred_mask.dtype == torch.bool
        # 50% 마스킹 → 약 10개, 최소 1개
        for b in range(B):
            assert pred_mask[b].sum() >= 1

    def test_variate_level_masking(self):
        """variate_mask_prob=1.0 → 반드시 전체 variate 마스킹."""
        B, N = 1, 10
        patch_mask = torch.ones(B, N, dtype=torch.bool)
        variate_id = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]])

        pred_mask = create_patch_mask(
            patch_mask, mask_ratio=0.15,
            patch_variate_id=variate_id,
            variate_mask_prob=1.0,
        )

        # 하나의 variate 전체가 마스킹되어야 함
        var1_masked = pred_mask[0, :5].all().item()
        var2_masked = pred_mask[0, 5:].all().item()
        assert var1_masked or var2_masked


class TestModelCrossHead:
    """BiosignalFoundationModel cross_head 출력 테스트."""

    def test_cross_pred_in_masked_output(self):
        """task='masked'일 때 cross_pred key 존재."""
        model, batch, _ = _make_model_and_batch()
        model.eval()

        with torch.no_grad():
            out = model(batch, task="masked")

        assert "cross_pred" in out
        assert "time_id" in out
        assert out["cross_pred"].shape == out["reconstructed"].shape

    def test_cross_pred_not_in_next_pred_output(self):
        """task='next_pred'일 때 cross_pred key 없음."""
        model, batch, _ = _make_model_and_batch()
        model.eval()

        with torch.no_grad():
            out = model(batch, task="next_pred")

        assert "cross_pred" not in out
        assert "time_id" in out

    def test_cross_head_gradient_flow(self):
        """cross_head에 gradient가 흐른다 (task='masked')."""
        model, batch, _ = _make_model_and_batch(
            d_model=64, num_layers=1, patch_size=16,
        )

        out = model(batch, task="masked")
        mask = out["patch_mask"]
        loss = (out["cross_pred"][mask] ** 2).mean()
        loss.backward()

        assert model.cross_head.weight.grad is not None
