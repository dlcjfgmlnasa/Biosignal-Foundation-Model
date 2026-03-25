# -*- coding:utf-8 -*-
import bisect
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

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
    spatial_ids: list[int] | None = None  # per-channel 로컬 spatial_id, len == n_channels


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
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self._manifest = list(manifest)
        self._use_mmap = use_mmap
        self._cache_size = cache_size

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
                st = round((stride_seconds if stride_seconds is not None else window_seconds)
                           * entry.sampling_rate)
                n_win = max(0, (entry.n_timesteps - wl) // st + 1)
            else:
                wl = None
                st = 0
                n_win = 1
            self._window_lengths_per_rec.append(wl)
            self._strides_per_rec.append(st)
            self._n_windows_per_rec.append(n_win)
            self._rec_offsets.append(self._rec_offsets[-1] + entry.n_channels * n_win)

    def _load_recording_impl(
        self, rec_idx: int
    ) -> torch.Tensor:  # (channels, time)
        return torch.load(self._manifest[rec_idx].path, weights_only=True, mmap=self._use_mmap)

    def __getstate__(self) -> dict:
        """Pickle 직렬화: lru_cache wrapper는 pickle 불가이므로 제외."""
        state = self.__dict__.copy()
        del state["_load_recording"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Pickle 역직렬화: Worker에서 새 lru_cache를 생성."""
        self.__dict__.update(state)
        self._load_recording = lru_cache(maxsize=self._cache_size)(
            self._load_recording_impl
        )

    def __len__(self) -> int:
        return self._rec_offsets[-1]

    def __getitem__(self, idx: int) -> BiosignalSample:
        rec_idx = bisect.bisect_right(self._rec_offsets, idx, hi=len(self._rec_offsets) - 1) - 1
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
    ) -> "BiosignalDataset":
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
