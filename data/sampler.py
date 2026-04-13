# -*- coding:utf-8 -*-
from __future__ import annotations

"""Sampler 모듈.

- RecordingLocalitySampler: 레코딩 단위 locality를 보장하는 샘플러.
  같은 레코딩의 윈도우/채널 인덱스를 연속으로 yield하여 LRU 캐시 히트율을
  극대화한다. 네트워크 디스크 환경에서 I/O 병목을 해소하는 핵심 컴포넌트.
  DDP 환경에서는 레코딩을 rank별로 분배한다.

- GroupedBatchSampler: any_variate 모드를 위한 배치 샘플러.
  같은 (session_id, physical_time_ms)의 채널들을 항상 같은 배치에 넣어서
  PackCollate의 any_variate 그루핑이 제대로 동작하도록 보장한다.
"""
import bisect
import math
from collections import defaultdict
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from data.dataset import BiosignalDataset


class RecordingLocalitySampler(Sampler[int]):
    """레코딩 단위 locality 샘플러 (DDP 호환).

    레코딩 순서를 셔플한 뒤, 각 레코딩 내 (channel × window) 인덱스를
    연속으로 yield한다. 한 번 로드한 .pt 파일의 모든 윈도우를 소진한 뒤
    다음 레코딩으로 넘어가므로 LRU 캐시 히트율이 ~100%에 근접한다.

    DDP 사용 시 레코딩 단위로 rank에 분배하여, 같은 레코딩이 여러 rank에
    걸쳐 분할되지 않도록 한다.

    Parameters
    ----------
    dataset:
        BiosignalDataset 인스턴스.
    num_replicas:
        DDP world size. ``None``이면 단일 GPU로 간주.
    rank:
        현재 프로세스 rank. ``None``이면 단일 GPU로 간주.
    shuffle:
        에폭마다 레코딩 순서를 셔플할지 여부.
    seed:
        재현 가능한 셔플링을 위한 시드.
    """

    def __init__(
        self,
        dataset: BiosignalDataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # 레코딩별 flat 인덱스 범위 구축
        n_recs = len(dataset._manifest)
        self._rec_ranges: list[tuple[int, int]] = []
        for r in range(n_recs):
            start = dataset._rec_offsets[r]
            end = dataset._rec_offsets[r + 1]
            if end > start:
                self._rec_ranges.append((start, end))

        # DDP: 레코딩 수를 num_replicas 배수로 패딩 (균등 분배)
        self._n_recs = len(self._rec_ranges)
        self._n_recs_padded = (
            math.ceil(self._n_recs / self.num_replicas) * self.num_replicas
        )

    def set_epoch(self, epoch: int) -> None:
        """에폭별 셔플링 시드 설정 (DDP 동기화용)."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        # 레코딩 순서 결정 (모든 rank에서 동일한 순서)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            rec_order = torch.randperm(self._n_recs, generator=g).tolist()
        else:
            rec_order = list(range(self._n_recs))

        # 패딩: 부족분은 앞에서 반복 (DistributedSampler와 동일 전략)
        if self._n_recs_padded > self._n_recs:
            rec_order += rec_order[: self._n_recs_padded - self._n_recs]

        # 현재 rank에 할당된 레코딩만 추출
        per_rank = self._n_recs_padded // self.num_replicas
        my_recs = rec_order[self.rank * per_rank : (self.rank + 1) * per_rank]

        # 각 레코딩 내 인덱스를 셔플하여 yield
        indices: list[int] = []
        for rec_i in my_recs:
            if rec_i >= self._n_recs:
                # 패딩된 가상 레코딩 — 스킵
                continue
            start, end = self._rec_ranges[rec_i]
            local_indices = list(range(start, end))
            if self.shuffle:
                # 레코딩 내 윈도우 순서도 셔플 (재현성 위해 별도 seed)
                lg = torch.Generator()
                lg.manual_seed(self.seed + self.epoch * 10000 + rec_i)
                perm = torch.randperm(len(local_indices), generator=lg).tolist()
                local_indices = [local_indices[p] for p in perm]
            indices.extend(local_indices)

        return iter(indices)

    def __len__(self) -> int:
        # 현재 rank에 할당된 총 샘플 수 (근사치)
        per_rank = self._n_recs_padded // self.num_replicas
        total = 0
        # 정확한 계산은 비용이 크므로, 전체를 rank 수로 나눈 근사치 사용
        total = math.ceil(len(self.dataset) / self.num_replicas)
        return total


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

        group_to_rec: dict[tuple, int] = {}  # 그룹 → 레코딩 인덱스 (locality용)

        for idx in range(len(dataset)):
            # 인덱스 디코딩 (dataset.__getitem__과 동일)
            rec_idx = (
                bisect.bisect_right(
                    dataset._rec_offsets, idx, hi=len(dataset._rec_offsets) - 1
                )
                - 1
            )
            local = idx - dataset._rec_offsets[rec_idx]
            n_win = dataset._n_windows_per_rec[rec_idx]
            win_idx = local % n_win
            stride = dataset._strides_per_rec[rec_idx]
            win_start = win_idx * stride

            entry = dataset._manifest[rec_idx]

            # 키 생성 (collate.py와 정확히 동일 — 윈도우 크기 단위 버킷팅)
            if entry.session_id:
                abs_sample = entry.start_sample + win_start
                bucket_size = dataset.max_length or 60000
                bucket = abs_sample // bucket_size
                key = (entry.session_id, bucket)
            else:
                key = (rec_idx, win_start)

            groups[key].append(idx)
            group_to_rec[key] = rec_idx

        # 딕셔너리 → 리스트 (반복 시 순서 안정성)
        self._groups: list[list[int]] = list(groups.values())
        # 각 그룹이 속한 레코딩 인덱스 (recording-locality 셔플용)
        keys = list(groups.keys())
        self._group_rec_ids: list[int] = [group_to_rec[k] for k in keys]

    def __iter__(self) -> Iterator[list[int]]:
        """배치를 yield한다. 각 배치는 완전한 그룹(들)을 포함한다."""
        # 그룹 순서 결정: recording-locality 보장
        # 레코딩 순서를 셔플한 뒤, 같은 레코딩의 그룹들을 연속 배치
        group_indices = list(range(len(self._groups)))
        if self.shuffle:
            # 1) 레코딩 순서 셔플
            unique_recs = sorted(set(self._group_rec_ids))
            rec_perm = torch.randperm(
                len(unique_recs), generator=self.generator
            ).tolist()
            rec_order = [unique_recs[i] for i in rec_perm]

            # 2) 레코딩별 그룹 인덱스 모으기
            rec_to_groups: dict[int, list[int]] = defaultdict(list)
            for g_idx, r_id in enumerate(self._group_rec_ids):
                rec_to_groups[r_id].append(g_idx)

            # 3) 레코딩 내 그룹 순서 셔플 후 연결
            group_indices = []
            for r_id in rec_order:
                g_list = rec_to_groups[r_id]
                perm = torch.randperm(len(g_list), generator=self.generator).tolist()
                group_indices.extend(g_list[p] for p in perm)

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
