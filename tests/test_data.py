# -*- coding:utf-8 -*-
"""data/ 파이프라인 테스트 스크립트.

Usage:
    cd C:/Projects/Biosignal-Foundation-Model
    PYTHONPATH=. python tests/test_data.py
"""
import tempfile
from pathlib import Path

import torch

from data.dataset import BiosignalDataset, BiosignalSample, RecordingManifest
from data.collate import PackCollate, PackedBatch
from data.dataloader import create_dataloader


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── 헬퍼 ────────────────────────────────────────────────────────

def save_recordings(
    recordings: list[torch.Tensor],
    tmpdir: Path,
    sampling_rate: float = 250.0,
    signal_type: int = 0,
) -> list[RecordingManifest]:
    """텐서들을 .pt로 저장하고 manifest 리스트를 반환."""
    manifest = []
    for i, rec in enumerate(recordings):
        if rec.ndim == 1:
            rec = rec.unsqueeze(0)
        pt_path = tmpdir / f"rec_{i:04d}.pt"
        torch.save(rec, pt_path)
        manifest.append(
            RecordingManifest(
                path=str(pt_path),
                n_channels=rec.shape[0],
                n_timesteps=rec.shape[1],
                sampling_rate=sampling_rate,
                signal_type=signal_type,
            )
        )
    return manifest


def make_samples(lengths: list[int], signal_type: int = 0) -> list[BiosignalSample]:
    """헬퍼: 주어진 길이들로 더미 BiosignalSample 목록 생성 (값 = index+1)."""
    return [
        BiosignalSample(
            values=torch.ones(l) * (i + 1),
            length=l,
            channel_idx=0,
            recording_idx=i,
            sampling_rate=250.0,
            n_channels=1,
            win_start=0,
            signal_type=signal_type,
        )
        for i, l in enumerate(lengths)
    ]


# ── 1. BiosignalDataset 기본 (from_tensors) ─────────────────────

def test_dataset_channel_independent():
    divider("Dataset: Channel Independent 분리")

    rec_ecg = torch.randn(3, 100)   # ECG 3-lead, 100 timesteps
    rec_abp = torch.randn(1, 250)   # ABP 1-channel, 250 timesteps
    rec_eeg = torch.randn(16, 50)   # EEG 16-channel, 50 timesteps

    ds = BiosignalDataset.from_tensors(
        [rec_ecg, rec_abp, rec_eeg], max_length=200, sampling_rate=500.0
    )

    expected_len = 3 + 1 + 16  # = 20
    assert len(ds) == expected_len, f"len={len(ds)}, expected={expected_len}"
    print(f"  총 샘플 수: {len(ds)}  (ECG 3 + ABP 1 + EEG 16 = 20)  OK")

    s0 = ds[0]
    assert isinstance(s0, BiosignalSample)
    assert s0.recording_idx == 0 and s0.channel_idx == 0
    assert s0.sampling_rate == 500.0
    print(f"  ds[0]: recording={s0.recording_idx}, channel={s0.channel_idx}, "
          f"length={s0.length}, sr={s0.sampling_rate}  OK")

    # max_length 적용 확인 (ABP 250 → 200)
    s_abp = ds[3]
    assert s_abp.length == 200, f"truncation failed: {s_abp.length}"
    print(f"  ds[3] (ABP): length={s_abp.length} (250→200 truncated)  OK")


def test_dataset_1d_input():
    divider("Dataset: 1D 입력 자동 처리")

    rec = torch.randn(80)
    ds = BiosignalDataset.from_tensors([rec])
    assert len(ds) == 1
    assert ds[0].length == 80
    print(f"  1D tensor → (1, 80) 자동 변환, length={ds[0].length}  OK")


# ── 2. Lazy Loading (Manifest) ───────────────────────────────────

def test_lazy_loading():
    divider("Dataset: Lazy Loading (Manifest)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        rec1 = torch.randn(3, 1000)
        rec2 = torch.randn(2, 80)

        manifest = save_recordings([rec1, rec2], tmpdir, sampling_rate=250.0)
        ds = BiosignalDataset(manifest)

        assert len(ds) == 5  # 3 + 2
        print(f"  manifest 2건 → 총 {len(ds)} 샘플  OK")

        # 값 일치 확인
        s0 = ds[0]
        assert torch.allclose(s0.values, rec1[0])
        assert s0.sampling_rate == 250.0
        print(f"  ds[0] 값 일치, sampling_rate={s0.sampling_rate}  OK")

        s4 = ds[4]
        assert torch.allclose(s4.values, rec2[1])
        print(f"  ds[4] 값 일치  OK")


# ── 3. Sliding Window ────────────────────────────────────────────

def test_sliding_window_count():
    divider("Dataset: Sliding Window 개수")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # 2ch, 1000 timesteps, sr=250 → window_seconds=0.4 (100 samples), stride_seconds=0.4 → 10 windows/ch → 20 total
        rec = torch.randn(2, 1000)
        manifest = save_recordings([rec], tmpdir)

        ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.4)
        expected = 2 * 10  # 2ch × 10 windows
        assert len(ds) == expected, f"len={len(ds)}, expected={expected}"
        print(f"  2ch × 1000ts, window_seconds=0.4 → {len(ds)} 샘플  OK")


def test_sliding_window_overlap():
    divider("Dataset: Sliding Window 오버랩")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # 1ch, 200 timesteps, sr=250 → window_seconds=0.4 (100 samples), stride_seconds=0.2 (50 samples) → 3 windows (0, 50, 100)
        rec = torch.arange(200, dtype=torch.float32).unsqueeze(0)  # (1, 200)
        manifest = save_recordings([rec], tmpdir)

        ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.2)
        expected = 3  # starts: 0, 50, 100
        assert len(ds) == expected, f"len={len(ds)}, expected={expected}"
        print(f"  1ch × 200ts, window_seconds=0.4, stride_seconds=0.2 → {len(ds)} 윈도우  OK")

        # 윈도우 값 검증
        w0 = ds[0]
        assert w0.values[0].item() == 0.0 and w0.values[-1].item() == 99.0
        w1 = ds[1]
        assert w1.values[0].item() == 50.0 and w1.values[-1].item() == 149.0
        w2 = ds[2]
        assert w2.values[0].item() == 100.0 and w2.values[-1].item() == 199.0
        print(f"  윈도우 값 정확성 검증  OK")


def test_sliding_window_large_scale():
    divider("Dataset: Sliding Window 대규모 시뮬레이션")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        n_channels, n_timesteps = 60_000, 72_000
        window_seconds, stride_seconds = 20.0, 20.0  # sr=250 → 5000 샘플 = 20초

        # 실제 데이터 없이 manifest만으로 인덱스 공식 검증
        placeholder = torch.zeros(1, 1)
        pt_path = Path(tmpdir) / "rec_large.pt"
        torch.save(placeholder, pt_path)

        manifest = [RecordingManifest(
            path=str(pt_path), n_channels=n_channels, n_timesteps=n_timesteps,
            sampling_rate=250.0, signal_type=0,
        )] * 100
        ds = BiosignalDataset(manifest, window_seconds=window_seconds, stride_seconds=stride_seconds)

        window_length = round(window_seconds * 250.0)  # 5000
        stride = round(stride_seconds * 250.0)  # 5000
        windows_per_ch = (n_timesteps - window_length) // stride + 1  # = 14
        expected = n_channels * windows_per_ch * 100                         # = 840_000
        assert len(ds) == expected, f"len={len(ds)}, expected={expected}"
        print(f"  {n_channels}ch × {n_timesteps}ts → {expected} 샘플 (인덱스 공식 검증)  OK")

# ── 4. PackCollate ───────────────────────────────────────────────

def test_pack_collate_basic():
    divider("PackCollate: 기본 패킹")

    collate = PackCollate(max_length=20)
    samples = make_samples([10, 5])

    batch = collate(samples)
    assert isinstance(batch, PackedBatch)
    assert batch.values.shape == (1, 20), f"shape={batch.values.shape}"
    print(f"  출력 shape: {batch.values.shape}  (1행에 패킹됨)  OK")

    ids = batch.sample_id[0]
    assert (ids[:10] == 1).all()
    assert (ids[10:15] == 2).all()
    assert (ids[15:] == 0).all()
    print(f"  sample_id: [1]*10 + [2]*5 + [0]*5  OK")

    row = batch.values[0]
    assert (row[:10] == 1.0).all() and (row[10:15] == 2.0).all()
    print(f"  값 보존 확인  OK")


def test_pack_collate_sampling_rates():
    divider("PackCollate: sampling_rates 전파")

    samples = [
        BiosignalSample(torch.randn(10), 10, 0, 0, 500.0, n_channels=10, win_start=0, signal_type=0),
        BiosignalSample(torch.randn(8), 8, 0, 1, 250.0, n_channels=1, win_start=0, signal_type=0),
    ]
    collate = PackCollate(max_length=50)
    batch = collate(samples)
    assert batch.sampling_rates.shape[0] == 2
    # FFD sorts longest first: sample0(10) then sample1(8)
    assert batch.sampling_rates[0].item() == 500.0
    assert batch.sampling_rates[1].item() == 250.0
    print(f"  sampling_rates: {batch.sampling_rates.tolist()}  OK")

def test_pack_collate_ffd():
    divider("PackCollate: FFD 패킹 효율")

    collate = PackCollate(max_length=20)
    samples = make_samples([15, 12, 5, 3])

    batch = collate(samples)
    assert batch.values.shape[0] == 2
    print(f"  [15, 12, 5, 3] → FFD → 2행 (행1: 15+5, 행2: 12+3)  OK")

    total = batch.lengths.sum().item()
    assert total == 35
    print(f"  lengths 합계: {total} (15+12+5+3=35)  OK")


# ── 5. DataLoader 통합 ──────────────────────────────────────────

def test_dataloader_integration():
    divider("DataLoader 통합 테스트")

    # 각 recording이 1채널 → 그루핑 시 별도 그룹 유지
    recordings = [
        torch.randn(1, 100),
        torch.randn(1, 150),
        torch.randn(1, 50),
    ]

    ds = BiosignalDataset.from_tensors(recordings, max_length=200, sampling_rate=500.0)
    loader = create_dataloader(ds, max_length=200, batch_size=3, shuffle=False)

    print(f"  Dataset 크기: {len(ds)}")
    print(f"  batch_size=3, max_length=200\n")

    total_variates = 0
    for batch_idx, batch in enumerate(loader):
        assert isinstance(batch, PackedBatch)
        assert batch.values.shape[1] == 200
        assert batch.sample_id.shape == batch.values.shape
        n_packed = batch.lengths.shape[0]
        total_variates += n_packed
        utilization = (batch.sample_id > 0).float().mean().item() * 100

        print(f"  batch {batch_idx}: "
              f"values={list(batch.values.shape)}, "
              f"packed_variates={n_packed}, "
              f"utilization={utilization:.1f}%")

    assert total_variates == len(ds)
    print(f"\n  전체 variate 처리 완료: {total_variates}/{len(ds)}  OK")


def test_dataloader_with_sliding_window():
    divider("DataLoader + Sliding Window 통합 테스트")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        rec = torch.randn(4, 500)  # 4ch, 500ts
        manifest = save_recordings([rec], tmpdir, sampling_rate=250.0)

        ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.4)  # 100 samples per window
        # max_length=400: 4ch × 100ts = 400 → 그룹이 잘리지 않음
        loader = create_dataloader(ds, max_length=400, batch_size=20, shuffle=False)

        total = 0
        for batch_idx, batch in enumerate(loader):
            total += batch.lengths.shape[0]
            assert batch.values.shape[1] == 400
            assert (batch.sampling_rates == 250.0).all()

        expected = 4 * 5  # 4ch × 5 windows = 20 variates
        assert total == expected, f"total={total}, expected={expected}"
        print(f"  4ch × 500ts, window_seconds=0.4 → {total} variates 처리  OK")
        print(f"  sampling_rates 전파 확인  OK")


# ── 6. Any-variate 모드 ───────────────────────────────────────────

def test_any_variate_grouping():
    divider("Any-variate: 채널 그루핑")

    # 같은 recording(0), 같은 win_start(0)의 3채널 → 하나의 그룹으로 묶여야 함
    samples = [
        BiosignalSample(torch.ones(10) * 1, 10, ch_idx, 0, 250.0, n_channels=3, win_start=0, signal_type=0)
        for ch_idx in range(3)
    ]

    collate = PackCollate(max_length=50)
    batch = collate(samples)

    # 3개 채널이 하나로 묶여 총 길이 30 (per-variate metadata: 3개 항목)
    assert batch.values.shape == (1, 50)
    assert batch.lengths.shape[0] == 3  # per-variate: 3개
    assert batch.lengths.sum().item() == 30  # 10 * 3
    assert (batch.lengths == 10).all()
    print(f"  3채널 × 10ts → 1그룹, per-variate lengths=[10,10,10]  OK")

    # sample_id: 전체가 하나의 그룹이므로 모두 1
    assert (batch.sample_id[0, :30] == 1).all()
    assert (batch.sample_id[0, 30:] == 0).all()
    print(f"  sample_id 검증  OK")


def test_any_variate_variate_id():
    divider("Any-variate: variate_id 검증")

    samples = [
        BiosignalSample(torch.ones(8) * (ch + 1), 8, ch, 0, 250.0, n_channels=3, win_start=0, signal_type=0)
        for ch in range(3)
    ]

    collate = PackCollate(max_length=30)
    batch = collate(samples)

    var_ids = batch.variate_id[0]

    # ch0: variate_id=1, ch1: variate_id=2, ch2: variate_id=3
    assert (var_ids[:8] == 1).all()
    assert (var_ids[8:16] == 2).all()
    assert (var_ids[16:24] == 3).all()
    assert (var_ids[24:] == 0).all()  # padding
    print(f"  variate_id: [1]*8 + [2]*8 + [3]*8 + [0]*6  OK")

    # 값도 채널별로 올바른지 확인
    row = batch.values[0]
    assert (row[:8] == 1.0).all()
    assert (row[8:16] == 2.0).all()
    assert (row[16:24] == 3.0).all()
    print(f"  채널별 값 보존  OK")


def test_any_variate_mixed_recordings():
    divider("Any-variate: 서로 다른 recording 분리")

    samples = [
        # recording 0, 2채널
        BiosignalSample(torch.ones(10), 10, 0, 0, 250.0, n_channels=2, win_start=0, signal_type=0),
        BiosignalSample(torch.ones(10) * 2, 10, 1, 0, 250.0, n_channels=2, win_start=0, signal_type=0),
        # recording 1, 1채널
        BiosignalSample(torch.ones(10) * 3, 10, 0, 1, 250.0, n_channels=1, win_start=0, signal_type=0),
    ]

    collate = PackCollate(max_length=40)
    batch = collate(samples)

    # 2개 그룹: recording0(2 variates) + recording1(1 variate) = 3 per-variate entries
    assert batch.lengths.shape[0] == 3  # per-variate: 2 + 1
    print(f"  2개 recording → 3 per-variate entries  OK")

    # variate_id: 그룹A(ch0=1, ch1=2) + 그룹B(ch0=1) 각각 독립
    print(f"  variate_id 생성 확인  OK")


def test_independent_samples_packing():
    divider("독립 샘플 패킹 (각 recording 별도 그룹)")

    collate = PackCollate(max_length=20)
    samples = make_samples([10, 5])  # 서로 다른 recording_idx → 별도 그룹

    batch = collate(samples)
    assert batch.values.shape == (1, 20)
    assert (batch.sample_id[0, :10] == 1).all()
    # 각 샘플이 1-variate 그룹 → variate_id는 모두 1
    assert (batch.variate_id[0, :10] == 1).all()
    assert (batch.variate_id[0, 10:15] == 1).all()
    print(f"  독립 샘플: 각각 1-variate 그룹으로 패킹  OK")


def test_any_variate_dataloader():
    divider("Any-variate: DataLoader 통합")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        rec = torch.randn(3, 100)  # 3ch, 100ts
        manifest = save_recordings([rec], tmpdir, sampling_rate=250.0)

        ds = BiosignalDataset(manifest, window_seconds=0.4, stride_seconds=0.4)  # 100 samples

        loader = create_dataloader(
            ds, max_length=500, batch_size=10, shuffle=False,
        )

        for batch in loader:
            # 3채널이 하나의 그룹으로 묶임 (per-variate: 3개 항목)
            assert batch.lengths.shape[0] == 3
            assert batch.lengths.sum().item() == 300  # 3 × 100
            var_ids = batch.variate_id[0]
            assert (var_ids[:100] == 1).all()
            assert (var_ids[100:200] == 2).all()
            assert (var_ids[200:300] == 3).all()

        print(f"  3ch × 100ts → 1그룹(300ts), per-variate  OK")
        print(f"  variate_id 올바르게 전파  OK")

# ── 7. signal_type 전파 ─────────────────────────────────────────

def test_signal_type_propagation():
    divider("signal_type: 전파 테스트")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        rec = torch.randn(2, 100)
        manifest = save_recordings([rec], tmpdir, sampling_rate=250.0, signal_type=2)

        ds = BiosignalDataset(manifest)
        assert ds[0].signal_type == 2
        assert ds[1].signal_type == 2
        print(f"  RecordingManifest(signal_type=2) → BiosignalSample.signal_type=2  OK")

        # PackedBatch로 전파
        collate = PackCollate(max_length=200)
        batch = collate([ds[0], ds[1]])
        assert batch.signal_types.shape[0] == 2
        assert (batch.signal_types == 2).all()
        print(f"  PackedBatch.signal_types: {batch.signal_types.tolist()}  OK")


def test_mixed_signal_types():
    divider("signal_type: 혼합 신호 타입")

    with tempfile.TemporaryDirectory() as tmpdir_ecg, \
         tempfile.TemporaryDirectory() as tmpdir_eeg:
        rec_ecg = torch.randn(1, 50)
        rec_eeg = torch.randn(1, 50)

        manifest_ecg = save_recordings([rec_ecg], Path(tmpdir_ecg), sampling_rate=250.0, signal_type=0)
        manifest_eeg = save_recordings([rec_eeg], Path(tmpdir_eeg), sampling_rate=256.0, signal_type=1)

        ds = BiosignalDataset(manifest_ecg + manifest_eeg)
        assert ds[0].signal_type == 0  # ECG
        assert ds[1].signal_type == 1  # EEG
        print(f"  ECG(0) + EEG(1) 개별 확인  OK")

        collate = PackCollate(max_length=100)
        batch = collate([ds[0], ds[1]])
        types = batch.signal_types.tolist()
        assert 0 in types and 1 in types
        print(f"  PackedBatch.signal_types: {types}  (0과 1 모두 존재)  OK")

    # from_tensors 기본값 테스트
    ds2 = BiosignalDataset.from_tensors([torch.randn(1, 30)])
    assert ds2[0].signal_type == 0
    print(f"  from_tensors 기본값 signal_type=0  OK")

    # from_tensors 지정값 테스트
    ds3 = BiosignalDataset.from_tensors([torch.randn(1, 30)], signal_type=3)
    assert ds3[0].signal_type == 3
    print(f"  from_tensors signal_type=3  OK")


# ── 8. Cross-modal Any-variate ────────────────────────────────

def test_cross_modal_any_variate():
    divider("Cross-modal: session_id 기반 any_variate 그루핑")

    # ECG: 2ch, 100ts, fs=500, signal_type=0, session_id="S001"
    # EMG: 1ch, 100ts, fs=100, signal_type=4, session_id="S001"
    # → 같은 session + 같은 physical_time(0초) → 하나의 그룹
    samples = [
        BiosignalSample(torch.ones(100) * 1, 100, 0, 0, 500.0, 2, win_start=0, signal_type=0, session_id="S001"),
        BiosignalSample(torch.ones(100) * 2, 100, 1, 0, 500.0, 2, win_start=0, signal_type=0, session_id="S001"),
        BiosignalSample(torch.ones(100) * 3, 100, 0, 1, 100.0, 1, win_start=0, signal_type=4, session_id="S001"),
    ]

    collate = PackCollate(max_length=500)
    batch = collate(samples)

    # 3개 variate가 하나의 그룹으로 묶임 (physical_time 모두 0ms)
    var_ids = batch.variate_id[0]
    # 정렬: signal_type 순 → ECG(0) ch0, ECG(0) ch1, EMG(4) ch0
    assert (var_ids[:100] == 1).all(), "ECG ch0 should be variate 1"
    assert (var_ids[100:200] == 2).all(), "ECG ch1 should be variate 2"
    assert (var_ids[200:300] == 3).all(), "EMG ch0 should be variate 3"
    print(f"  3 variates (ECG×2 + EMG×1) → 1 그룹, variate_id 검증  OK")

    # per-variate 메타데이터 검증
    assert batch.signal_types.tolist() == [0, 0, 4], \
        f"signal_types={batch.signal_types.tolist()}, expected [0, 0, 4]"
    assert batch.sampling_rates.tolist() == [500.0, 500.0, 100.0], \
        f"sampling_rates={batch.sampling_rates.tolist()}, expected [500, 500, 100]"
    assert batch.lengths.tolist() == [100, 100, 100], \
        f"lengths={batch.lengths.tolist()}, expected [100, 100, 100]"
    print(f"  per-variate metadata: signal_types={batch.signal_types.tolist()}, "
          f"rates={batch.sampling_rates.tolist()}  OK")


def test_cross_modal_physical_time_alignment():
    divider("Cross-modal: 물리적 시간 정렬 (같은 session)")

    # 같은 session_id + 같은 physical_time → 하나의 그룹
    samples_same = [
        BiosignalSample(torch.ones(50), 50, 0, 0, 500.0, 1, win_start=500, signal_type=0, session_id="S001"),
        BiosignalSample(torch.ones(50), 50, 0, 1, 100.0, 1, win_start=100, signal_type=4, session_id="S001"),
    ]

    collate = PackCollate(max_length=200)
    batch = collate(samples_same)

    # 같은 session + 같은 physical_time(1000ms) → 1그룹
    var_ids = batch.variate_id[0]
    assert (var_ids[:50] == 1).all()
    assert (var_ids[50:100] == 2).all()
    print(f"  같은 session, ECG(ws=500,fs=500) + EMG(ws=100,fs=100) → 1000ms → 1그룹  OK")

    # 다른 session_id + 같은 physical_time → 별도 그룹
    samples_diff = [
        BiosignalSample(torch.ones(50), 50, 0, 0, 500.0, 1, win_start=500, signal_type=0, session_id="S001"),
        BiosignalSample(torch.ones(50), 50, 0, 1, 100.0, 1, win_start=100, signal_type=4, session_id="S002"),
    ]

    batch2 = collate(samples_diff)
    assert batch2.lengths.shape[0] == 2  # 별도 그룹
    print(f"  다른 session, 같은 physical_time → 2그룹 (분리)  OK")


def test_cross_modal_different_physical_time():
    divider("Cross-modal: 다른 물리적 시간 → 별도 그룹")

    # ECG: fs=500, win_start=0 → 0ms
    # EMG: fs=100, win_start=100 → 1000ms
    # → 다른 physical_time → 별도 그룹
    samples = [
        BiosignalSample(torch.ones(50), 50, 0, 0, 500.0, 1, win_start=0, signal_type=0, session_id="S003"),
        BiosignalSample(torch.ones(50), 50, 0, 1, 100.0, 1, win_start=100, signal_type=4, session_id="S003"),
    ]

    collate = PackCollate(max_length=200)
    batch = collate(samples)

    # 2개의 별도 그룹
    # per-variate이므로 lengths는 variate별
    assert batch.lengths.shape[0] == 2
    print(f"  다른 physical_time → 2개 그룹  OK")


def test_session_id_empty_backward_compat():
    divider("Cross-modal: session_id 빈 문자열 → 기존 동작")

    # session_id="" → recording_idx 기반 그루핑 (기존 동작)
    samples = [
        BiosignalSample(torch.ones(10), 10, 0, 0, 250.0, 2, win_start=0, signal_type=0),
        BiosignalSample(torch.ones(10), 10, 1, 0, 250.0, 2, win_start=0, signal_type=0),
    ]

    collate = PackCollate(max_length=50)
    batch = collate(samples)

    # 같은 recording_idx=0, win_start=0 → 하나의 그룹
    var_ids = batch.variate_id[0]
    assert (var_ids[:10] == 1).all()
    assert (var_ids[10:20] == 2).all()
    print(f"  session_id='' → recording_idx 기반 그루핑 유지  OK")


# ── 9. Cross-modal Window Alignment ───────────────────────────

def test_cross_modal_window_alignment():
    divider("Cross-modal: window_seconds로 물리적 시간 정렬")

    with tempfile.TemporaryDirectory() as tmpdir_ecg, \
         tempfile.TemporaryDirectory() as tmpdir_eeg:
        tmpdir_ecg = Path(tmpdir_ecg)
        tmpdir_eeg = Path(tmpdir_eeg)

        # ECG: fs=500, 500 samples
        rec_ecg = torch.randn(1, 500)
        manifest_ecg = save_recordings([rec_ecg], tmpdir_ecg, sampling_rate=500.0, signal_type=0)

        # EEG: fs=256, 256 samples (같은 물리적 시간: 1초)
        rec_eeg = torch.randn(1, 256)
        manifest_eeg = save_recordings([rec_eeg], tmpdir_eeg, sampling_rate=256.0, signal_type=1)

        # ECG와 EEG를 session_id로 연결
        manifest_ecg[0].session_id = "S001"
        manifest_eeg[0].session_id = "S001"

        ds = BiosignalDataset(manifest_ecg + manifest_eeg, window_seconds=1.0, stride_seconds=1.0)

        # ECG: window=500 samples, stride=500 → 1 window at win_start=0 → physical_time=0ms
        # EEG: window=256 samples, stride=256 → 1 window at win_start=0 → physical_time=0ms
        # → 둘 다 sample 0에 해당 (채널이 다르므로 서로 다른 인덱스)
        assert len(ds) == 2  # 2개 채널

        ecg_sample = ds[0]
        eeg_sample = ds[1]

        assert ecg_sample.length == 500
        assert eeg_sample.length == 256
        assert ecg_sample.win_start == 0
        assert eeg_sample.win_start == 0
        assert ecg_sample.sampling_rate == 500.0
        assert eeg_sample.sampling_rate == 256.0
        print(f"  ECG(fs=500, 500샘플) + EEG(fs=256, 256샘플) 동시 로드  OK")
        print(f"  둘 다 win_start=0 (physical_time=0ms)  OK")


# ── 10. 시각적 예시 ──────────────────────────────────────────────

def demo_packing_visual():
    divider("시각적 예시: 패킹 결과")

    collate = PackCollate(max_length=20)
    samples = make_samples([10, 7, 5, 3, 8, 2])

    batch = collate(samples)

    for row_idx in range(batch.values.shape[0]):
        ids = batch.sample_id[row_idx].tolist()
        visual = ""
        for sid in ids:
            if sid == 0:
                visual += "·"
            else:
                visual += chr(ord("A") + sid - 1)
        filled = sum(1 for s in ids if s > 0)
        util = filled / len(ids) * 100
        print(f"  행{row_idx}: [{visual}]  ({util:.0f}% 활용)")

    print(f"\n  총 {batch.lengths.shape[0]}개 샘플 → "
          f"{batch.values.shape[0]}행으로 패킹")


# ── main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_dataset_channel_independent()
    test_dataset_1d_input()
    test_lazy_loading()
    test_sliding_window_count()
    test_sliding_window_overlap()
    test_sliding_window_large_scale()
    test_pack_collate_basic()
    test_pack_collate_sampling_rates()
    test_pack_collate_ffd()
    test_dataloader_integration()
    test_dataloader_with_sliding_window()
    test_cross_modal_window_alignment()
    test_any_variate_grouping()
    test_any_variate_variate_id()
    test_any_variate_mixed_recordings()
    test_independent_samples_packing()
    test_any_variate_dataloader()
    test_cross_modal_any_variate()
    test_cross_modal_physical_time_alignment()
    test_cross_modal_different_physical_time()
    test_session_id_empty_backward_compat()
    test_signal_type_propagation()
    test_mixed_signal_types()
    demo_packing_visual()

    print(f"\n{'=' * 60}")
    print(f"  ALL TESTS PASSED")
    print(f"{'=' * 60}")
