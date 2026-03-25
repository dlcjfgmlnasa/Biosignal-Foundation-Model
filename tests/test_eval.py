# -*- coding:utf-8 -*-
"""다운스트림 평가 파이프라인 테스트.

- PrototypicalClassifier: 합성 임베딩으로 단위 테스트
- compute_classification_metrics: 메트릭 정확성
- evaluate_forecasting / evaluate_imputation: 메트릭 dict 반환 확인
"""
import tempfile
from pathlib import Path

import pytest
import torch

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalDataset, RecordingManifest
from model.biosignal_model import BiosignalFoundationModel
from eval.fewshot import PrototypicalClassifier, compute_classification_metrics
from eval.forecasting import evaluate_forecasting
from eval.imputation import evaluate_imputation


# ── 헬퍼 ──────────────────────────────────────────────────────────


def make_fake_signal(
    n_channels: int = 1,
    n_timesteps: int = 512,
    sampling_rate: float = 256.0,
) -> torch.Tensor:
    """테스트용 가짜 생체신호 생성."""
    return torch.randn(n_channels, n_timesteps)


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


def make_model_and_batches(
    tmpdir: Path,
    n_samples: int = 4,
    d_model: int = 64,
    num_layers: int = 1,
    patch_size: int = 16,
    max_length: int = 256,
    n_timesteps: int = 256,
) -> tuple[BiosignalFoundationModel, list[PackedBatch]]:
    """모델 + 단일 샘플 배치 리스트 생성."""
    signals = [make_fake_signal(n_timesteps=n_timesteps) for _ in range(n_samples)]
    manifests = save_recordings(signals, tmpdir, 256.0, signal_type=2)
    ds = BiosignalDataset(manifests, window_seconds=n_timesteps / 256.0, stride_seconds=n_timesteps / 256.0)

    collate = PackCollate(max_length=max_length, patch_size=patch_size)

    batches = []
    for i in range(min(n_samples, len(ds))):
        batch = collate([ds[i]])
        batches.append(batch)

    model = BiosignalFoundationModel(
        d_model=d_model,
        num_layers=num_layers,
        patch_size=patch_size,
    )
    model.eval()
    return model, batches


# ── compute_classification_metrics 단위 테스트 ────────────────────


class TestComputeClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]
        m = compute_classification_metrics(y_pred, y_true)

        assert m["accuracy"] == 1.0
        assert m["balanced_accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["cohens_kappa"] == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        m = compute_classification_metrics(y_pred, y_true)

        assert m["accuracy"] == 0.0
        assert m["macro_f1"] == 0.0

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 2, 2, 1, 1]
        m = compute_classification_metrics(y_pred, y_true, class_labels=[0, 1, 2])

        cm = m["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

        # row sums should equal class counts in y_true
        assert sum(cm[0]) == 2  # class 0 appears 2 times
        assert sum(cm[1]) == 2  # class 1 appears 2 times
        assert sum(cm[2]) == 1  # class 2 appears 1 time

    def test_cohens_kappa_range(self):
        y_true = [0, 1, 0, 1, 0, 1, 2, 2]
        y_pred = [0, 1, 1, 0, 0, 1, 2, 0]
        m = compute_classification_metrics(y_pred, y_true)

        assert -1.0 <= m["cohens_kappa"] <= 1.0

    def test_empty(self):
        m = compute_classification_metrics([], [], class_labels=[0, 1])
        assert m["accuracy"] == 0.0


# ── PrototypicalClassifier 단위 테스트 ────────────────────────────


class TestPrototypicalClassifier:
    def test_fit_and_predict(self):
        """합성 임베딩으로 fit → predict 동작 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, batches = make_model_and_batches(Path(tmpdir), n_samples=6)

            clf = PrototypicalClassifier()
            # support: 클래스 0 → 처음 3개, 클래스 1 → 나머지 3개
            support_batches = batches[:6]
            labels = [0, 0, 0, 1, 1, 1]
            clf.fit(model, support_batches, labels)

            assert clf.prototypes is not None
            assert clf.prototypes.shape[0] == 2  # 2 classes
            assert clf.class_labels == [0, 1]

    def test_predict_returns_valid_labels(self):
        """predict가 fit에서 본 클래스만 반환."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, batches = make_model_and_batches(Path(tmpdir), n_samples=4)

            clf = PrototypicalClassifier()
            clf.fit(model, batches[:2], [0, 1])

            preds = clf.predict(model, batches[2])
            for p in preds:
                assert p in [0, 1]

    def test_evaluate_returns_metrics(self):
        """evaluate가 올바른 메트릭 키를 반환."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, batches = make_model_and_batches(Path(tmpdir), n_samples=4)

            clf = PrototypicalClassifier()
            clf.fit(model, batches[:2], [0, 1])

            metrics = clf.evaluate(model, batches[2:4], [0, 1])
            assert "accuracy" in metrics
            assert "balanced_accuracy" in metrics
            assert "macro_f1" in metrics
            assert "cohens_kappa" in metrics
            assert "confusion_matrix" in metrics


# ── evaluate_imputation 테스트 ────────────────────────────────────


class TestEvaluateImputation:
    def test_returns_metrics(self):
        """imputation 평가가 메트릭 dict를 반환."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d_model, patch_size = 64, 16
            max_length = 256

            signals = [make_fake_signal(n_timesteps=256) for _ in range(2)]
            manifests = save_recordings(signals, Path(tmpdir), 256.0, signal_type=2)
            ds = BiosignalDataset(manifests, window_seconds=1.0, stride_seconds=1.0)
            collate = PackCollate(max_length=max_length, patch_size=patch_size)

            samples = [ds[i] for i in range(min(2, len(ds)))]
            batch = collate(samples)

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=1,
                patch_size=patch_size,
            )

            metrics = evaluate_imputation(model, batch)

            assert "mse" in metrics
            assert "mae" in metrics
            assert "mape" in metrics
            assert "pearson_r" in metrics
            assert metrics["mse"] >= 0.0
            assert metrics["mae"] >= 0.0
            assert -1.0 <= metrics["pearson_r"] <= 1.0


# ── evaluate_forecasting 테스트 ───────────────────────────────────


class TestEvaluateForecasting:
    def test_returns_metrics(self):
        """forecasting 평가가 메트릭 dict를 반환."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d_model, patch_size = 64, 16
            max_length = 256

            signals = [make_fake_signal(n_timesteps=256)]
            manifests = save_recordings(signals, Path(tmpdir), 256.0, signal_type=2)
            ds = BiosignalDataset(manifests, window_seconds=1.0, stride_seconds=1.0)
            collate = PackCollate(max_length=max_length, patch_size=patch_size)

            batch = collate([ds[0]])

            model = BiosignalFoundationModel(
                d_model=d_model,
                num_layers=1,
                patch_size=patch_size,
            )

            # ground truth: 1 step of patch_size
            gt = torch.randn(batch.values.shape[0], 1, patch_size)

            metrics = evaluate_forecasting(
                model, batch, ground_truth=gt, n_steps=1, denormalize=False,
            )

            assert "mse" in metrics
            assert "mae" in metrics
            assert "mape" in metrics
            assert "pearson_r" in metrics
            assert metrics["mse"] >= 0.0
            assert metrics["mae"] >= 0.0
