# -*- coding:utf-8 -*-
"""PatchEmbedding 및 PackCollate patch_size 정렬 테스트."""
import pytest
import torch

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample
from module.patch import PatchEmbedding


# ── 헬퍼 ──────────────────────────────────────────────────────────

def _make_sample(
    length: int,
    channel_idx: int = 0,
    recording_idx: int = 0,
    sampling_rate: float = 100.0,
    signal_type: int = 0,
    session_id: str = "",
    win_start: int = 0,
) -> BiosignalSample:
    return BiosignalSample(
        values=torch.randn(length),
        length=length,
        channel_idx=channel_idx,
        recording_idx=recording_idx,
        sampling_rate=sampling_rate,
        n_channels=1,
        signal_type=signal_type,
        session_id=session_id,
        win_start=win_start,
    )


# ── PackCollate patch_size 정렬 테스트 ────────────────────────────


class TestPackCollatePatchAlignment:
    def test_max_length_rounded_up(self):
        """patch_size 설정 시 max_length가 배수로 올림된다."""
        collate = PackCollate(max_length=105, patch_size=10)
        assert collate.max_length == 110

        collate_exact = PackCollate(max_length=100, patch_size=10)
        assert collate_exact.max_length == 100

    def test_output_shape_divisible_by_patch_size(self):
        """출력 values의 max_length가 patch_size의 배수이다."""
        samples = [_make_sample(length=47, recording_idx=i) for i in range(3)]
        collate = PackCollate(max_length=200, patch_size=16)
        batch = collate(samples)

        assert batch.values.shape[1] % 16 == 0

    def test_padded_lengths_present(self):
        """patch_size 설정 시 padded_lengths가 존재하고 patch_size 배수이다."""
        samples = [_make_sample(length=25, recording_idx=i) for i in range(2)]
        collate = PackCollate(max_length=100, patch_size=10)
        batch = collate(samples)

        assert batch.padded_lengths is not None
        for pl in batch.padded_lengths:
            assert pl.item() % 10 == 0

    def test_padded_lengths_none_without_patch_size(self):
        """patch_size 미설정 시 padded_lengths는 None."""
        samples = [_make_sample(length=25, recording_idx=i) for i in range(2)]
        collate = PackCollate(max_length=100)
        batch = collate(samples)

        assert batch.padded_lengths is None

    def test_padded_lengths_geq_lengths(self):
        """padded_lengths >= lengths (원래 길이 이상)."""
        samples = [_make_sample(length=33, recording_idx=i) for i in range(3)]
        collate = PackCollate(max_length=200, patch_size=10)
        batch = collate(samples)

        assert (batch.padded_lengths >= batch.lengths).all()

    def test_variate_id_covers_padding(self):
        """variate 내부 패딩 영역에도 variate_id가 설정된다."""
        # 길이 25인 샘플, patch_size=10 → padded to 30
        # sample_id/variate_id가 25~29 위치에도 설정되어야 함
        samples = [_make_sample(length=25, recording_idx=0)]
        collate = PackCollate(max_length=100, patch_size=10)
        batch = collate(samples)

        row = batch.variate_id[0]
        padded_len = batch.padded_lengths[0].item()
        # 패딩 포함 variate 영역 전체에 variate_id가 1이어야 함
        assert (row[:padded_len] == 1).all()
        # 나머지는 0 (행 패딩)
        assert (row[padded_len:] == 0).all()

    def test_sample_id_covers_padding(self):
        """variate 내부 패딩 영역에도 sample_id가 설정된다."""
        samples = [_make_sample(length=25, recording_idx=0)]
        collate = PackCollate(max_length=100, patch_size=10)
        batch = collate(samples)

        row = batch.sample_id[0]
        padded_len = batch.padded_lengths[0].item()
        assert (row[:padded_len] == 1).all()
        assert (row[padded_len:] == 0).all()

    def test_values_zero_in_variate_padding(self):
        """variate 내부 패딩 영역의 값은 0이다."""
        samples = [_make_sample(length=25, recording_idx=0)]
        collate = PackCollate(max_length=100, patch_size=10)
        batch = collate(samples)

        original_len = batch.lengths[0].item()
        padded_len = batch.padded_lengths[0].item()
        # 원래 데이터 뒤 ~ padded 경계까지 0
        assert (batch.values[0, original_len:padded_len] == 0).all()

    def test_multi_variate_patch_alignment(self):
        """같은 그룹의 여러 variate가 각각 patch_size 정렬된다."""
        s1 = _make_sample(length=23, channel_idx=0, recording_idx=0, win_start=0)
        s2 = _make_sample(length=17, channel_idx=1, recording_idx=0, win_start=0)
        collate = PackCollate(max_length=200, patch_size=10)
        batch = collate([s1, s2])

        # 23 → 30, 17 → 20 → padded_lengths = [30, 20]
        assert batch.padded_lengths[0].item() == 30
        assert batch.padded_lengths[1].item() == 20

    def test_backward_compat_no_patch_size(self):
        """patch_size 없이 기존 동작과 동일."""
        samples = [_make_sample(length=25, recording_idx=i) for i in range(3)]
        collate = PackCollate(max_length=100)
        batch = collate(samples)

        assert batch.padded_lengths is None
        assert batch.values.shape[1] == 100


# ── PatchEmbedding 테스트 ─────────────────────────────────────────


class TestPatchEmbedding:
    def test_basic_shapes(self):
        """기본 출력 shape 확인."""
        P, d = 10, 64
        embed = PatchEmbedding(patch_size=P, d_model=d)

        B, L = 2, 100  # 100 / 10 = 10 patches
        values = torch.randn(B, L)
        sample_id = torch.ones(B, L, dtype=torch.long)
        variate_id = torch.ones(B, L, dtype=torch.long)

        out, p_sid, p_vid, time_id, mask = embed(values, sample_id, variate_id)

        N = L // P
        assert out.shape == (B, N, d)
        assert p_sid.shape == (B, N)
        assert p_vid.shape == (B, N)
        assert time_id.shape == (B, N)
        assert mask.shape == (B, N)

    def test_assert_not_divisible(self):
        """max_length가 patch_size의 배수가 아니면 에러."""
        embed = PatchEmbedding(patch_size=10, d_model=64)
        values = torch.randn(1, 15)
        sample_id = torch.ones(1, 15, dtype=torch.long)
        variate_id = torch.ones(1, 15, dtype=torch.long)

        with pytest.raises(AssertionError, match="patch_size"):
            embed(values, sample_id, variate_id)

    def test_metadata_downsampling(self):
        """sample_id, variate_id가 올바르게 다운샘플된다."""
        P = 4
        embed = PatchEmbedding(patch_size=P, d_model=32)

        # 2개 variate: variate_1(8ts) + variate_2(4ts) + padding(4ts) = 16ts
        sample_id = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
        variate_id = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0]])
        values = torch.randn(1, 16)

        _, p_sid, p_vid, _, _ = embed(values, sample_id, variate_id)

        # 16 / 4 = 4 patches
        assert p_sid.shape == (1, 4)
        assert p_vid.tolist() == [[1, 1, 2, 0]]
        assert p_sid.tolist() == [[1, 1, 1, 0]]

    def test_time_id_sequential(self):
        """같은 variate 내에서 time_id가 0부터 순차 증가한다."""
        P = 4
        embed = PatchEmbedding(patch_size=P, d_model=32)

        # variate_1: 3 patches, variate_2: 2 patches, padding: 1 patch
        sample_id = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
        variate_id = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]])
        values = torch.randn(1, 24)

        _, _, _, time_id, _ = embed(values, sample_id, variate_id)

        # variate_1: patches 0,1,2 → time_id [0,1,2]
        # variate_2: patches 3,4 → time_id [0,1]
        # padding: patch 5 → time_id [0]
        assert time_id.tolist() == [[0, 1, 2, 0, 1, 0]]

    def test_patch_mask(self):
        """sample_id=0인 패치의 mask가 False."""
        P = 4
        embed = PatchEmbedding(patch_size=P, d_model=32)

        sample_id = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        variate_id = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        values = torch.randn(1, 8)

        _, _, _, _, mask = embed(values, sample_id, variate_id)

        assert mask.tolist() == [[True, False]]

    def test_gradient_flow(self):
        """패치 임베딩을 통해 gradient가 흐른다."""
        P, d = 10, 32
        embed = PatchEmbedding(patch_size=P, d_model=d)

        values = torch.randn(2, 40, requires_grad=True)
        sample_id = torch.ones(2, 40, dtype=torch.long)
        variate_id = torch.ones(2, 40, dtype=torch.long)

        out, _, _, _, _ = embed(values, sample_id, variate_id)
        loss = out.sum()
        loss.backward()

        assert values.grad is not None
        assert values.grad.shape == values.shape

    def test_multi_sample_in_row(self):
        """한 행에 여러 sample이 있을 때 time_id가 각각 리셋된다."""
        P = 4
        embed = PatchEmbedding(patch_size=P, d_model=32)

        # sample_1(var1: 2 patches), sample_2(var1: 2 patches), padding(2 patches)
        sample_id = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]])
        variate_id = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        values = torch.randn(1, 24)

        _, _, _, time_id, _ = embed(values, sample_id, variate_id)

        # sample_1/var1: [0,1], sample_2/var1: [0,1], padding: [0,0]
        assert time_id.tolist() == [[0, 1, 0, 1, 0, 0]]

    def test_batch_dimension(self):
        """배치 차원에서 독립적으로 처리된다."""
        P, d = 4, 16
        embed = PatchEmbedding(patch_size=P, d_model=d)

        # batch 0: 2 variate patches + 1 padding
        # batch 1: 1 variate patch + 2 padding
        sample_id = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        variate_id = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        values = torch.randn(2, 12)

        _, _, _, time_id, mask = embed(values, sample_id, variate_id)

        assert time_id[0].tolist() == [0, 1, 0]
        assert time_id[1].tolist() == [0, 0, 0]
        assert mask[0].tolist() == [True, True, False]
        assert mask[1].tolist() == [True, False, False]


# ── End-to-end: PackCollate + PatchEmbedding ─────────────────────


class TestCollateAndPatchIntegration:
    def test_collate_then_patch(self):
        """PackCollate(patch_size) → PatchEmbedding 전체 파이프라인."""
        P, d = 10, 32
        samples = [
            _make_sample(length=47, recording_idx=0),
            _make_sample(length=33, recording_idx=1),
            _make_sample(length=21, recording_idx=2),
        ]
        collate = PackCollate(max_length=200, patch_size=P)
        batch = collate(samples)

        embed = PatchEmbedding(patch_size=P, d_model=d)
        out, p_sid, p_vid, time_id, mask = embed(
            batch.values, batch.sample_id, batch.variate_id
        )

        # max_length는 P의 배수
        assert batch.values.shape[1] % P == 0
        N = batch.values.shape[1] // P
        assert out.shape == (batch.values.shape[0], N, d)

        # 유효 패치 수 확인
        valid_patches = mask.sum().item()
        expected_patches = sum(
            -(-l.item() // P) for l in batch.lengths
        )
        assert valid_patches == expected_patches

    def test_multi_variate_collate_then_patch(self):
        """다채널 그룹이 patch 경계를 유지한다."""
        P, d = 8, 32
        s1 = _make_sample(length=20, channel_idx=0, recording_idx=0, win_start=0)
        s2 = _make_sample(length=15, channel_idx=1, recording_idx=0, win_start=0)
        collate = PackCollate(max_length=200, patch_size=P)
        batch = collate([s1, s2])

        embed = PatchEmbedding(patch_size=P, d_model=d)
        _, p_sid, p_vid, time_id, mask = embed(
            batch.values, batch.sample_id, batch.variate_id
        )

        # variate_1: 20 → padded 24 → 3 patches
        # variate_2: 15 → padded 16 → 2 patches
        valid_mask = mask[0]
        valid_vids = p_vid[0, valid_mask]
        assert (valid_vids[:3] == 1).all()  # variate 1의 패치 3개
        assert (valid_vids[3:] == 2).all()  # variate 2의 패치 2개

        # time_id: variate별로 리셋
        valid_tids = time_id[0, valid_mask]
        assert valid_tids[:3].tolist() == [0, 1, 2]
        assert valid_tids[3:].tolist() == [0, 1]


# ── Phase 2: Overlapping Patch 테스트 ─────────────────────────────


class TestOverlappingPatch:
    def test_stride_parameter(self):
        """stride < patch_size로 overlapping 패치 생성."""
        P, S, d = 16, 8, 32
        embed = PatchEmbedding(patch_size=P, d_model=d, stride=S)

        assert embed.stride == S
        assert embed.patch_size == P

    def test_stride_assert_divisibility(self):
        """patch_size가 stride의 배수가 아니면 에러."""
        with pytest.raises(AssertionError, match="stride"):
            PatchEmbedding(patch_size=16, d_model=32, stride=7)

    def test_overlapping_basic_shape(self):
        """overlapping 패치의 출력 shape 확인."""
        P, S, d = 16, 8, 32
        embed = PatchEmbedding(patch_size=P, d_model=d, stride=S)

        # L=48: N = (48 - 16) / 8 + 1 = 5 patches
        B, L = 2, 48
        values = torch.randn(B, L)
        sample_id = torch.ones(B, L, dtype=torch.long)
        variate_id = torch.ones(B, L, dtype=torch.long)

        out, p_sid, p_vid, time_id, mask = embed(values, sample_id, variate_id)

        N = (L - P) // S + 1
        assert N == 5
        assert out.shape == (B, N, d)
        assert p_sid.shape == (B, N)
        assert mask.all()  # 전부 유효

    def test_overlapping_more_patches_than_non_overlapping(self):
        """overlapping이 non-overlapping보다 더 많은 패치를 생성한다."""
        P, d = 16, 32
        B, L = 1, 48

        embed_no = PatchEmbedding(patch_size=P, d_model=d, stride=P)
        embed_ov = PatchEmbedding(patch_size=P, d_model=d, stride=8)

        values = torch.randn(B, L)
        sid = torch.ones(B, L, dtype=torch.long)
        vid = torch.ones(B, L, dtype=torch.long)

        out_no, _, _, _, _ = embed_no(values, sid, vid)
        out_ov, _, _, _, _ = embed_ov(values, sid, vid)

        assert out_ov.shape[1] > out_no.shape[1]  # 5 > 3

    def test_overlapping_time_id(self):
        """overlapping에서 time_id가 variate 내에서 순차 증가."""
        P, S, d = 8, 4, 16
        embed = PatchEmbedding(patch_size=P, d_model=d, stride=S)

        # 단일 variate, L=24: N = (24-8)/4 + 1 = 5
        B, L = 1, 24
        values = torch.randn(B, L)
        sid = torch.ones(B, L, dtype=torch.long)
        vid = torch.ones(B, L, dtype=torch.long)

        _, _, _, time_id, mask = embed(values, sid, vid)

        assert mask.all()
        assert time_id[0].tolist() == [0, 1, 2, 3, 4]

    def test_overlapping_boundary_mask(self):
        """variate 경계를 넘는 패치가 mask=False로 표시된다."""
        P, S, d = 8, 4, 16
        embed = PatchEmbedding(patch_size=P, d_model=d, stride=S)

        # var1: 16ts, var2: 8ts, padding: 없음 → L=24
        # unfold: positions 0,4,8,12,16 → patches cover [0:8],[4:12],[8:16],[12:20],[16:24]
        # patch [12:20] crosses var1→var2 boundary at 16 → invalid
        sid = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        vid = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]])
        values = torch.randn(1, 24)

        _, _, _, _, mask = embed(values, sid, vid)

        # N = (24-8)/4 + 1 = 5
        assert mask.shape == (1, 5)
        # patch 0: [0:8] all vid=1 ✓
        # patch 1: [4:12] all vid=1 ✓
        # patch 2: [8:16] all vid=1 ✓
        # patch 3: [12:20] vid=1(12-15) + vid=2(16-19) → invalid ✗
        # patch 4: [16:24] all vid=2 ✓
        assert mask[0].tolist() == [True, True, True, False, True]

    def test_overlapping_gradient_flow(self):
        """overlapping 패치를 통해 gradient가 흐른다."""
        P, S, d = 8, 4, 16
        embed = PatchEmbedding(patch_size=P, d_model=d, stride=S)

        values = torch.randn(2, 24, requires_grad=True)
        sid = torch.ones(2, 24, dtype=torch.long)
        vid = torch.ones(2, 24, dtype=torch.long)

        out, _, _, _, _ = embed(values, sid, vid)
        loss = out.sum()
        loss.backward()

        assert values.grad is not None

    def test_overlapping_padding_region(self):
        """padding 영역의 패치가 mask=False."""
        P, S, d = 8, 4, 16
        embed = PatchEmbedding(patch_size=P, d_model=d, stride=S)

        # 8ts data + 16ts padding = 24ts total
        sid = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        vid = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        values = torch.randn(1, 24)

        _, _, _, _, mask = embed(values, sid, vid)

        # N=5, only patch 0 [0:8] is valid
        assert mask[0, 0].item() is True
        # patch 1 [4:12] crosses data→padding boundary
        assert mask[0, 1].item() is False
        # patches 2-4 are all padding
        assert not mask[0, 2:].any()


class TestOverlappingCollateIntegration:
    def test_collate_stride(self):
        """PackCollate(stride) → PatchEmbedding(stride) 파이프라인."""
        P, S, d = 16, 8, 32
        samples = [
            _make_sample(length=50, recording_idx=0),
            _make_sample(length=30, recording_idx=1),
        ]
        collate = PackCollate(max_length=200, patch_size=P, stride=S)
        batch = collate(samples)

        embed = PatchEmbedding(patch_size=P, d_model=d, stride=S)
        out, _, _, _, mask = embed(
            batch.values, batch.sample_id, batch.variate_id
        )

        L = batch.values.shape[1]
        N = (L - P) // S + 1
        assert out.shape[1] == N

        # 유효 패치가 존재하는지 확인
        assert mask.any()

    def test_collate_stride_padded_lengths(self):
        """stride 사용 시 padded_lengths가 올바르게 계산된다."""
        P, S = 16, 8
        samples = [_make_sample(length=25, recording_idx=0)]
        collate = PackCollate(max_length=200, patch_size=P, stride=S)
        batch = collate(samples)

        # 25ts: excess = 25 - 16 = 9, ceil(9/8)*8 = 16, padded = 16 + 16 = 32
        assert batch.padded_lengths[0].item() == 32
        assert batch.lengths[0].item() == 25

    def test_collate_stride_max_length_alignment(self):
        """stride 사용 시 max_length가 stride의 배수로 올림된다."""
        collate = PackCollate(max_length=105, patch_size=16, stride=8)
        assert collate.max_length % 8 == 0
        assert collate.max_length >= 16

    def test_collate_stride_assert(self):
        """patch_size가 stride의 배수가 아니면 에러."""
        with pytest.raises(AssertionError):
            PackCollate(max_length=200, patch_size=16, stride=7)

    def test_collate_multi_variate_stride(self):
        """다채널 + stride에서 variate 경계가 유지된다."""
        P, S, d = 8, 4, 16
        s1 = _make_sample(length=20, channel_idx=0, recording_idx=0, win_start=0)
        s2 = _make_sample(length=12, channel_idx=1, recording_idx=0, win_start=0)
        collate = PackCollate(max_length=200, patch_size=P, stride=S)
        batch = collate([s1, s2])

        embed = PatchEmbedding(patch_size=P, d_model=d, stride=S)
        _, p_sid, p_vid, time_id, mask = embed(
            batch.values, batch.sample_id, batch.variate_id
        )

        # 유효 패치의 variate_id가 올바른지 확인
        valid_vids = p_vid[0, mask[0]]
        # var1 patches should come before var2 patches
        vid_changes = (valid_vids[1:] != valid_vids[:-1]).sum().item()
        assert vid_changes == 1  # exactly one transition from var1 to var2
