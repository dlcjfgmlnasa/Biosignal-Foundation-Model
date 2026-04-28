# -*- coding:utf-8 -*-
"""Length-aware batch sampler — FFD packing memory variance 완화.

PackCollate (FFD bin-packing) 의 row 수는 input batch 의 길이 분산에 비례한다.
긴 윈도우와 짧은 윈도우가 섞이면 packing efficiency 가 떨어져 row 수가 증가하고,
어쩌다 긴 것만 몰린 batch 는 메모리 spike → OOM 위험.

본 sampler 는 wrapping 한 baseline sampler (e.g. RecordingLocalitySampler) 가
yield 한 인덱스를 ``batch_size × overpack`` 만큼 buffer 에 모은 뒤 길이로 정렬해
``batch_size`` 단위로 잘라 yield 한다. 매 batch 의 길이 분산이 작아져
FFD packing efficiency 가 일정해지고 memory peak 가 deterministic 해진다.

Locality 영향:
  baseline sampler 가 recording-locality 를 보장해도, 본 sampler 가 buffer
  내부에서 길이로 정렬하면 같은 recording 내 인덱스가 batch 간 흩어질 수 있다.
  단 buffer 가 같은 recording 의 windows 들을 한꺼번에 받는 경우 (모두 같은
  길이) 정렬 후에도 인접해 같은 batch 에 들어간다 — locality 보존은 부분적.

Drop-in 사용:
    base_sampler = RecordingLocalitySampler(dataset, ...)
    batch_sampler = LengthAwareBatchSampler(
        base_sampler, dataset, batch_size=128, overpack=8
    )
    DataLoader(dataset, batch_sampler=batch_sampler, ...)
"""
from __future__ import annotations

from collections.abc import Iterator

from torch.utils.data import Sampler


class LengthAwareBatchSampler(Sampler[list[int]]):
    """Batch sampler — length-aware mini-bucketing on top of a base sampler.

    Parameters
    ----------
    base_sampler:
        인덱스 단위 sampler (e.g. ``RecordingLocalitySampler``).
    dataset:
        ``length_at(idx) -> int`` 메서드를 제공하는 dataset (e.g. ``BiosignalDataset``).
    batch_size:
        한 batch 가 yield 하는 인덱스 수.
    overpack:
        ``batch_size × overpack`` 만큼 buffer 에 모은 뒤 길이로 정렬.
        클수록 batch 내 길이 분산이 작아지지만 (= 더 균일) locality 영향이 커진다.
        4~16 권장, 기본 8.
    drop_last:
        flush 시 batch_size 미만 잔여를 버릴지 여부.
    """

    def __init__(
        self,
        base_sampler: Sampler[int],
        dataset,
        batch_size: int,
        overpack: int = 8,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if overpack < 1:
            raise ValueError(f"overpack must be >= 1, got {overpack}")
        if not hasattr(dataset, "length_at"):
            raise AttributeError(
                "dataset must expose `length_at(idx) -> int`. "
                "Use BiosignalDataset which provides this method."
            )
        self.base_sampler = base_sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.overpack = overpack
        self.drop_last = drop_last
        self._buffer_size = batch_size * overpack

    def __iter__(self) -> Iterator[list[int]]:
        buffer: list[tuple[int, int]] = []  # (idx, length)
        for idx in self.base_sampler:
            buffer.append((idx, self.dataset.length_at(idx)))
            if len(buffer) >= self._buffer_size:
                yield from self._flush(buffer, full=True)
                buffer = []
        if buffer:
            yield from self._flush(buffer, full=False)

    def _flush(
        self, buffer: list[tuple[int, int]], full: bool
    ) -> Iterator[list[int]]:
        # 길이로 정렬 후 batch_size 단위로 슬라이스
        buffer.sort(key=lambda x: x[1])
        n = len(buffer)
        for i in range(0, n, self.batch_size):
            chunk = buffer[i : i + self.batch_size]
            if len(chunk) < self.batch_size and self.drop_last:
                continue
            yield [idx for idx, _ in chunk]

    def __len__(self) -> int:
        try:
            n = len(self.base_sampler)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            raise TypeError(
                "base_sampler does not implement __len__; cannot compute "
                "LengthAwareBatchSampler length."
            )
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """DistributedSampler 호환을 위해 base_sampler 로 epoch 전달."""
        if hasattr(self.base_sampler, "set_epoch"):
            self.base_sampler.set_epoch(epoch)
