# -*- coding:utf-8 -*-
from __future__ import annotations

"""GroupedBatchSampler: any_variate 모드를 위한 배치 샘플러.

같은 (session_id, physical_time_ms)의 채널들을 항상 같은 배치에 넣어서
PackCollate의 any_variate 그루핑이 제대로 동작하도록 보장한다.
"""
import bisect
from collections import defaultdict
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from data.dataset import BiosignalDataset


class GroupedBatchSampler(Sampler[list[int]]):
    """any_variate 모드용 배치 샘플러.

    같은 (session_id, physical_time_ms) 그룹의 인덱스들을
    항상 같은 배치에 포함시킨다. 이는 PackCollate의 any_variate
    그루핑이 같은 subject + time의 채널들을 제대로 하나의 PackUnit으로
    묶을 수 있도록 보장한다.

    Parameters
    ----------
    dataset:
        BiosignalDataset 인스턴스.
    batch_size:
        배치당 최대 샘플 수. 그룹은 분리되지 않으므로 그룹 크기가
        batch_size를 초과하면 그 배치는 batch_size보다 클 수 있다.
    shuffle:
        그룹 순서를 섞을지 여부.
    drop_last:
        마지막 불완전한 배치를 버릴지 여부. 마지막 그룹(들)이
        batch_size 미만이면 버린다. 단, 단일 그룹이 batch_size를
        초과하는 경우는 버리지 않는다.
    generator:
        재현 가능한 셔플링을 위한 torch.Generator. ``None``이면 전역 RNG 사용.
    """

    def __init__(
        self,
        dataset: BiosignalDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        # 그룹 빌딩: (session_id, physical_time_ms) → [flat_idx_list]
        # 키 공식은 collate.py lines 82-86과 동일해야 함
        groups: dict[tuple, list[int]] = defaultdict(list)

        for idx in range(len(dataset)):
            # 인덱스 디코딩 (dataset.__getitem__과 동일)
            rec_idx = bisect.bisect_right(
                dataset._rec_offsets, idx, hi=len(dataset._rec_offsets) - 1
            ) - 1
            local = idx - dataset._rec_offsets[rec_idx]
            n_win = dataset._n_windows_per_rec[rec_idx]
            win_idx = local % n_win
            stride = dataset._strides_per_rec[rec_idx]
            win_start = win_idx * stride

            entry = dataset._manifest[rec_idx]

            # 키 생성 (collate.py lines 82-86과 정확히 동일)
            if entry.session_id:
                physical_time_ms = round(win_start / entry.sampling_rate * 1000)
                key = (entry.session_id, physical_time_ms)
            else:
                key = (rec_idx, win_start)

            groups[key].append(idx)

        # 딕셔너리 → 리스트 (반복 시 순서 안정성)
        self._groups: list[list[int]] = list(groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        """배치를 yield한다. 각 배치는 완전한 그룹(들)을 포함한다."""
        # 그룹 순서 결정
        group_indices = list(range(len(self._groups)))
        if self.shuffle:
            perm = torch.randperm(len(self._groups), generator=self.generator).tolist()
            group_indices = perm

        batch: list[int] = []
        for g_idx in group_indices:
            group = self._groups[g_idx]

            # 그룹을 추가하면 batch_size 초과 && 배치가 비어있지 않음
            # → 먼저 현재 배치를 yield한 뒤 새 배치 시작
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []

            # 그룹 추가
            batch.extend(group)

            # 배치가 batch_size 이상이면 바로 yield
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        # 마지막 배치 처리
        if batch:
            if not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """배치 개수 (근사치).

        정확한 개수는 그룹 크기에 따라 달라지므로,
        이는 근사치이다. DataLoader의 tqdm 진행률 표시용.
        """
        total_samples = sum(len(g) for g in self._groups)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size
