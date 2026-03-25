# -*- coding:utf-8 -*-
"""module/packed_scaler.py 테스트: PackedNOPScaler, PackedStdScaler, PackedAbsMeanScaler."""
import pytest
import torch

from module.packed_scaler import (
    PackedAbsMeanScaler,
    PackedNOPScaler,
    PackedScaler,
    PackedStdScaler,
)
from module._util import safe_div


# ── safe_div ──────────────────────────────────────────────────────


class TestSafeDiv:
    def test_normal_division(self):
        a = torch.tensor([6.0, 10.0])
        b = torch.tensor([2.0, 5.0])
        assert torch.allclose(safe_div(a, b), torch.tensor([3.0, 2.0]))

    def test_zero_divisor(self):
        a = torch.tensor([5.0, 3.0])
        b = torch.tensor([0.0, 0.0])
        assert torch.allclose(safe_div(a, b), torch.tensor([0.0, 0.0]))

    def test_mixed(self):
        a = torch.tensor([6.0, 3.0, 10.0])
        b = torch.tensor([2.0, 0.0, 5.0])
        assert torch.allclose(safe_div(a, b), torch.tensor([3.0, 0.0, 2.0]))


# ── PackedNOPScaler ───────────────────────────────────────────────


class TestPackedNOPScaler:
    def test_loc_zero_scale_one(self):
        scaler = PackedNOPScaler()
        target = torch.randn(2, 10, 3)
        loc, scale = scaler(target)
        assert torch.allclose(loc, torch.zeros_like(target))
        assert torch.allclose(scale, torch.ones_like(target))

    def test_output_shape(self):
        scaler = PackedNOPScaler()
        target = torch.randn(4, 20, 1)
        loc, scale = scaler(target)
        assert loc.shape == target.shape
        assert scale.shape == target.shape


# ── PackedStdScaler ───────────────────────────────────────────────


class TestPackedStdScaler:
    def test_single_variate_zscore(self):
        """단일 variate에 대해 z-score 정규화 결과 검증."""
        scaler = PackedStdScaler(correction=1)
        # 간단한 데이터: [1, 2, 3, 4, 5]
        target = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])  # (1, 5, 1)
        sample_id = torch.tensor([[1, 1, 1, 1, 1]])
        variate_id = torch.tensor([[1, 1, 1, 1, 1]])
        observed_mask = torch.ones_like(target, dtype=torch.bool)

        loc, scale = scaler(target, observed_mask, sample_id, variate_id)

        # mean = 3.0, std = sqrt(2.5 + 1e-5) ≈ 1.5811
        expected_loc = 3.0
        assert torch.allclose(loc[0, :, 0], torch.tensor([expected_loc] * 5), atol=1e-4)
        assert (scale > 0).all()

    def test_padding_ignored(self):
        """sample_id=0인 위치는 loc=0, scale=1."""
        scaler = PackedStdScaler()
        target = torch.tensor([[[10.0], [20.0], [0.0], [0.0]]])
        sample_id = torch.tensor([[1, 1, 0, 0]])
        variate_id = torch.tensor([[1, 1, 0, 0]])
        observed_mask = torch.ones_like(target, dtype=torch.bool)

        loc, scale = scaler(target, observed_mask, sample_id, variate_id)
        # padding 위치(sample_id=0)는 loc=0, scale=1
        assert loc[0, 2, 0].item() == 0.0
        assert loc[0, 3, 0].item() == 0.0
        assert scale[0, 2, 0].item() == 1.0
        assert scale[0, 3, 0].item() == 1.0

    def test_two_variates(self):
        """두 개의 서로 다른 variate가 독립적으로 스케일링."""
        scaler = PackedStdScaler(correction=0)
        target = torch.tensor([[[10.0], [20.0], [100.0], [200.0]]])
        sample_id = torch.tensor([[1, 1, 2, 2]])
        variate_id = torch.tensor([[1, 1, 1, 1]])
        observed_mask = torch.ones_like(target, dtype=torch.bool)

        loc, scale = scaler(target, observed_mask, sample_id, variate_id)
        # sample 1: mean=15, sample 2: mean=150
        assert torch.allclose(loc[0, 0, 0], torch.tensor(15.0), atol=1e-3)
        assert torch.allclose(loc[0, 2, 0], torch.tensor(150.0), atol=1e-3)

    def test_default_ids(self):
        """sample_id/variate_id가 None이면 기본값 사용."""
        scaler = PackedStdScaler()
        target = torch.randn(1, 8, 2)
        loc, scale = scaler(target)
        # sample_id=0 → loc=0, scale=1
        assert torch.allclose(loc, torch.zeros_like(loc))
        assert torch.allclose(scale, torch.ones_like(scale))


# ── PackedAbsMeanScaler ───────────────────────────────────────────


class TestPackedAbsMeanScaler:
    def test_loc_is_zero(self):
        """AbsMeanScaler는 항상 loc=0."""
        scaler = PackedAbsMeanScaler()
        target = torch.tensor([[[5.0], [-3.0], [7.0]]])
        sample_id = torch.tensor([[1, 1, 1]])
        variate_id = torch.tensor([[1, 1, 1]])
        observed_mask = torch.ones_like(target, dtype=torch.bool)

        loc, scale = scaler(target, observed_mask, sample_id, variate_id)
        assert torch.allclose(loc[0, :, 0], torch.tensor([0.0, 0.0, 0.0]))

    def test_scale_is_abs_mean(self):
        """scale = mean(|x|)."""
        scaler = PackedAbsMeanScaler()
        target = torch.tensor([[[4.0], [-6.0]]])
        sample_id = torch.tensor([[1, 1]])
        variate_id = torch.tensor([[1, 1]])
        observed_mask = torch.ones_like(target, dtype=torch.bool)

        loc, scale = scaler(target, observed_mask, sample_id, variate_id)
        # abs mean = (4 + 6) / 2 = 5.0
        assert torch.allclose(scale[0, 0, 0], torch.tensor(5.0), atol=1e-4)

    def test_padding_ignored(self):
        """sample_id=0인 위치는 loc=0, scale=1."""
        scaler = PackedAbsMeanScaler()
        target = torch.tensor([[[10.0], [0.0]]])
        sample_id = torch.tensor([[1, 0]])
        variate_id = torch.tensor([[1, 0]])
        observed_mask = torch.ones_like(target, dtype=torch.bool)

        loc, scale = scaler(target, observed_mask, sample_id, variate_id)
        assert scale[0, 1, 0].item() == 1.0


# ── PackedScaler ABC ──────────────────────────────────────────────


class TestPackedScalerABC:
    def test_base_not_implemented(self):
        """PackedScaler._get_loc_scale는 NotImplementedError."""
        scaler = PackedScaler()
        target = torch.randn(1, 5, 1)
        with pytest.raises(NotImplementedError):
            scaler(target)
