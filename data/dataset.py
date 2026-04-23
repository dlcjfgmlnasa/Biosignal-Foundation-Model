# -*- coding:utf-8 -*-
from __future__ import annotations

import bisect
import random
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Value
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class BiosignalSample:
    """Channel-independent 단일 채널 시계열 세그먼트.

    멀티채널 레코딩에서 각 채널을 분리(CI)하여 독립적인 샘플로 취급한다.
    """

    values: torch.Tensor  # (time,)
    length: int
    channel_idx: int
    recording_idx: int
    sampling_rate: float
    n_channels: int
    win_start: int
    signal_type: int
    session_id: str = ""
    spatial_id: int = 0  # 로컬 spatial_id (signal_type 내 채널 위치, 0=Unknown)
    start_sample: int = 0  # 세션 내 절대 시작 sample (recording 단위)


@dataclass
class RecordingManifest:
    """디스크에 저장된 레코딩 하나의 메타데이터.

    실제 텐서 데이터는 ``path``의 ``.pt`` 파일에 ``(channels, time)``
    형태로 저장되어 있으며, 로딩은 ``BiosignalDataset``이 on-demand로 수행한다.
    """

    path: str
    n_channels: int
    n_timesteps: int
    sampling_rate: float
    signal_type: int = 0
    session_id: str = ""
    spatial_ids: list[int] | None = (
        None  # per-channel 로컬 spatial_id, len == n_channels
    )
    start_sample: int = 0  # 세션 내 절대 시작 sample (TARGET_SR 기준)


class BiosignalDataset(Dataset[BiosignalSample]):
    """Channel-Independent 생체신호 데이터셋 (Lazy-loading + Sliding window).

    각 레코딩은 디스크의 ``.pt`` 파일로 존재하며, ``RecordingManifest``로
    메타데이터만 관리한다. ``__getitem__`` 호출 시 on-demand로 로딩하고
    LRU 캐시로 반복 로드를 방지한다.

    CI 패러다임에 따라 모든 채널을 개별 샘플로 풀어헤치며,
    ``window_seconds`` 설정 시 긴 레코딩을 sliding window로 분할한다.
    각 레코딩의 sampling_rate에 따라 샘플 수로 자동 변환된다.

    Parameters
    ----------
    manifest:
        ``RecordingManifest`` 시퀀스. 레코딩마다 채널 수, 시간 길이,
        샘플링 레이트가 다를 수 있다.
    max_length:
        ``window_seconds`` 미사용 시, 이보다 긴 세그먼트를 잘라낸다.
        ``window_seconds`` 사용 시 무시된다.
    window_seconds:
        설정 시 각 채널을 이 시간 길이의 윈도우로 분할한다 (초 단위).
    stride_seconds:
        윈도우 간 간격 (초 단위). 기본값은 ``window_seconds`` (비중첩).
    cache_size:
        LRU 캐시에 유지할 최대 레코딩 수.
    """

    def __init__(
        self,
        manifest: Sequence[RecordingManifest],
        max_length: int | None = None,
        window_seconds: float | None = None,
        stride_seconds: float | None = None,
        cache_size: int = 8,
        use_mmap: bool = False,
        crop_ratio_range: tuple[float, float] | None = None,
        patch_size: int | None = None,
        min_patches: int = 5,  # random crop 최소 patch 수 (임상: 10s floor @ patch_size=200,100Hz)
        preload: bool = False,  # deprecated, 무시됨
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.crop_ratio_range = crop_ratio_range  # e.g. (0.5, 1.0)
        self._patch_size = patch_size  # crop 시 patch 배수 정렬용
        self._min_patches = min_patches
        self._manifest = list(manifest)
        self._use_mmap = use_mmap
        self._cache_size = cache_size

        # 공유 epoch counter — DataLoader worker(자식 프로세스)에서도 보임
        # train loop의 set_epoch(epoch)으로 갱신 → __getitem__의 crop seed에
        # 섞여 동일 (rec, win) 샘플도 epoch마다 다른 crop을 받게 한다.
        # mp.Value는 shared memory라 persistent_workers와도 호환.
        self._epoch_value: Value | None = (
            Value("q", 0) if crop_ratio_range is not None else None
        )

        # LRU 캐시를 인스턴스별로 생성 (lru_cache는 함수 레벨이므로 래핑)
        self._load_recording = lru_cache(maxsize=cache_size)(self._load_recording_impl)

        # 레코딩별 window/stride (샘플 단위로 변환)
        self._window_lengths_per_rec: list[int | None] = []
        self._strides_per_rec: list[int] = []

        # Lazy 인덱스: 레코딩별 누적 샘플 수만 저장 → O(M) 메모리
        self._rec_offsets: list[int] = [0]
        self._n_windows_per_rec: list[int] = []
        for entry in self._manifest:
            if window_seconds is not None:
                wl = round(window_seconds * entry.sampling_rate)
                st = round(
                    (stride_seconds if stride_seconds is not None else window_seconds)
                    * entry.sampling_rate
                )
                if entry.n_timesteps >= wl:
                    # 긴 recording: sliding window + random crop 적용 (augmentation)
                    n_win = (entry.n_timesteps - wl) // st + 1
                else:
                    # 짧은 recording: 전체를 1 window로 그대로 사용 (NO crop, 정보 보존)
                    # patch_size 배수로 floor (collate silent drop 방지)
                    ps = self._patch_size if self._patch_size is not None else 1
                    wl = (entry.n_timesteps // ps) * ps
                    if wl < ps:
                        # patch 1개도 못 만들 정도로 짧음 → skip
                        n_win = 0
                        wl = 0
                        st = 0
                    else:
                        st = wl
                        n_win = 1
            else:
                wl = None
                st = 0
                n_win = 1
            self._window_lengths_per_rec.append(wl)
            self._strides_per_rec.append(st)
            self._n_windows_per_rec.append(n_win)
            self._rec_offsets.append(self._rec_offsets[-1] + entry.n_channels * n_win)

    def _load_recording_impl(self, rec_idx: int) -> torch.Tensor:  # (channels, time)
        path = self._manifest[rec_idx].path

        # Transient I/O 에러 방어: 최대 3회 retry (exponential backoff)
        # 지속 실패 시 다른 recording으로 fallback (pretrain에선 수천-수만 recording 중
        # 1개 누락은 무시 가능 수준)
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                if "#" in path:
                    import h5py

                    h5_path, ds_name = path.split("#", 1)
                    with h5py.File(h5_path, "r") as hf:
                        return torch.from_numpy(hf[ds_name][:]).float()
                if path.endswith(".zarr"):
                    import zarr

                    arr = zarr.open(path, mode="r")
                    return torch.from_numpy(arr[:]).float()
                return torch.load(path, weights_only=True, mmap=self._use_mmap)
            except (OSError, RuntimeError) as e:
                last_err = e
                if attempt < 2:
                    import time

                    time.sleep(0.1 * (2 ** attempt))  # 0.1s, 0.2s, 0.4s
                    continue

        # 3회 실패 → fallback: 다른 random recording
        import sys

        print(
            f"WARN: persistent I/O error on '{path}' after 3 retries ({last_err}). "
            f"Falling back to next recording.",
            file=sys.stderr,
        )
        fallback_idx = (rec_idx + 1) % len(self._manifest)
        # 무한 재귀 방지: fallback도 실패하면 예외 발생
        if fallback_idx == rec_idx:
            raise last_err  # type: ignore[misc]
        return self._load_recording_impl(fallback_idx)

    def __getstate__(self) -> dict:
        """Pickle 직렬화: lru_cache wrapper는 pickle 불가이므로 제외."""
        state = self.__dict__.copy()
        del state["_load_recording"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Pickle 역직렬화: Worker에서 캐시를 복원."""
        self.__dict__.update(state)
        self._load_recording = lru_cache(maxsize=self._cache_size)(
            self._load_recording_impl
        )

    def __len__(self) -> int:
        return self._rec_offsets[-1]

    def set_epoch(self, epoch: int) -> None:
        """Random crop seed에 epoch을 섞어 epoch마다 다른 crop이 나오게 한다.

        DataLoader worker가 fork되기 전에 호출해야 효과 있음 (mp.Value는
        shared memory라 fork 후에도 보이지만, 보다 안전한 호출 시점은 epoch
        시작 전이다). crop_ratio_range가 None이면 no-op.
        """
        if self._epoch_value is not None:
            self._epoch_value.value = int(epoch)

    def _current_epoch(self) -> int:
        """현재 epoch 값 (worker에서도 안전하게 읽힘)."""
        if self._epoch_value is None:
            return 0
        return int(self._epoch_value.value)

    def __getitem__(self, idx: int) -> BiosignalSample:
        rec_idx = (
            bisect.bisect_right(self._rec_offsets, idx, hi=len(self._rec_offsets) - 1)
            - 1
        )
        local = idx - self._rec_offsets[rec_idx]
        n_win = self._n_windows_per_rec[rec_idx]
        ch_idx = local // n_win
        stride = self._strides_per_rec[rec_idx]
        win_start = (local % n_win) * stride
        entry = self._manifest[rec_idx]

        recording = self._load_recording(rec_idx)
        channel: torch.Tensor = recording[ch_idx]  # (time,)

        win_length = self._window_lengths_per_rec[rec_idx]
        if win_length is not None:
            values = channel[win_start : win_start + win_length]
        else:
            values = channel
            if self.max_length is not None:
                values = values[: self.max_length]

        # Random crop: 긴 recording(full window_seconds)에만 적용 (augmentation)
        # 짧은 recording은 이미 제한된 정보 → crop 생략하여 전체 정보 보존
        # min_patches 보장 — 너무 짧은 crop 방지 (임상 최소 context 확보)
        full_window_len: int | None = None
        if self.window_seconds is not None:
            full_window_len = round(self.window_seconds * entry.sampling_rate)
        is_long_recording = (
            full_window_len is not None and win_length == full_window_len
        )

        if (
            is_long_recording
            and self.crop_ratio_range is not None
            and len(values) > 0
        ):
            # 결정적 seed: 같은 (recording, window, epoch)의 모든 채널이 동일한
            # crop을 받도록 한다. any_variate 모드에서 sibling 채널마다 독립 crop
            # 이면 같은 recording에서 온 variate들이 시간적으로 어긋나 cross-modal
            # pair가 실제로는 다른 시간 구간을 비교하게 되는 버그 방지.
            # epoch을 섞어 augmentation diversity 유지 — train loop의
            # dataset.set_epoch(epoch) 호출로 갱신.
            # bit-packed: rec_idx는 high 32 bits, win_start는 low 32 bits.
            # epoch은 별도 prime으로 XOR (collision 회피).
            base_seed = rec_idx * 2**32 + int(win_start)
            epoch_salt = self._current_epoch() * 0x9E3779B97F4A7C15  # golden ratio prime
            crop_rng = random.Random((base_seed ^ epoch_salt) & ((1 << 63) - 1))
            lo, hi = self.crop_ratio_range
            ratio = crop_rng.uniform(lo, hi)
            crop_len = max(1, int(len(values) * ratio))
            if hasattr(self, "_patch_size") and self._patch_size is not None:
                # 최소 crop 길이: min_patches × patch_size
                min_crop_len = min(
                    self._min_patches * self._patch_size, len(values)
                )
                # patch_size 배수 정렬 + 최소 보장
                crop_len = max(
                    min_crop_len,
                    (crop_len // self._patch_size) * self._patch_size,
                )
            if crop_len < len(values):
                start = crop_rng.randint(0, len(values) - crop_len)
                values = values[start : start + crop_len]
                win_start = win_start + start

        spatial_id = 0
        if entry.spatial_ids is not None:
            spatial_id = entry.spatial_ids[ch_idx]

        return BiosignalSample(
            values=values,
            length=values.shape[0],
            channel_idx=ch_idx,
            recording_idx=rec_idx,
            sampling_rate=entry.sampling_rate,
            n_channels=entry.n_channels,
            win_start=win_start,
            signal_type=entry.signal_type,
            session_id=entry.session_id,
            spatial_id=spatial_id,
            start_sample=entry.start_sample,
        )

    @classmethod
    def from_tensors(
        cls,
        recordings: Sequence[torch.Tensor],  # list of (channels, time)
        max_length: int | None = None,
        window_seconds: float | None = None,
        stride_seconds: float | None = None,
        sampling_rate: float = 1.0,
        signal_type: int = 0,
        cache_dir: str | None = None,
        cache_size: int = 8,
    ) -> BiosignalDataset:
        """인메모리 텐서로부터 데이터셋 생성 (테스트/프로토타이핑용).

        각 텐서를 임시 ``.pt`` 파일로 저장한 뒤 manifest를 구성한다.
        """
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="biosignal_")
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        manifest: list[RecordingManifest] = []
        for i, rec in enumerate(recordings):
            if rec.ndim == 1:
                rec = rec.unsqueeze(0)
            pt_path = cache_path / f"rec_{i:06d}.pt"
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

        return cls(
            manifest,
            max_length=max_length,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            cache_size=cache_size,
        )
