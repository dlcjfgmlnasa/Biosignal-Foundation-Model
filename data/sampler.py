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
from dataclasses import dataclass

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
        rank: int = 0,
        world_size: int = 1,
        max_length: int = 60000,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.max_length = max_length
        self.epoch = 0
        # DDP: 모든 rank가 동일한 셔플 순서를 사용하도록 고정 시드 generator
        self.generator = generator or torch.Generator()
        self.generator.manual_seed(42)

        # 그룹 빌딩: 실제 시간 overlap 기반
        # 같은 session_id 내에서 시간이 겹치는 윈도우들을 하나의 그룹으로 묶는다.
        # → cross-modal pair가 보장되는 그룹만 생성

        # 1단계: 모든 윈도우의 절대 시간 범위 수집
        @dataclass
        class _WindowInfo:
            idx: int
            rec_idx: int
            session_id: str
            signal_type: int
            abs_start: int
            abs_end: int

        windows_by_session: dict[str, list[_WindowInfo]] = defaultdict(list)

        for idx in range(len(dataset)):
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
            if not entry.session_id:
                # session_id 없으면 독립 그룹
                windows_by_session[f"__norec_{idx}"].append(
                    _WindowInfo(idx, rec_idx, "", entry.signal_type, 0, 0)
                )
                continue

            abs_start = entry.start_sample + win_start
            win_len = dataset._window_lengths_per_rec[rec_idx]
            if win_len is None:
                win_len = entry.n_timesteps
            abs_end = abs_start + min(win_len, entry.n_timesteps - win_start)

            windows_by_session[entry.session_id].append(
                _WindowInfo(idx, rec_idx, entry.session_id, entry.signal_type, abs_start, abs_end)
            )

        # 2단계: 세션별로 시간 overlap 그룹 생성
        groups: dict[int, list[int]] = {}  # group_id → [flat_idx_list]
        group_to_rec: dict[int, int] = {}
        group_counter = 0

        for session_id, wins in windows_by_session.items():
            if not wins[0].session_id:
                # session_id 없는 독립 윈도우
                for w in wins:
                    groups[group_counter] = [w.idx]
                    group_to_rec[group_counter] = w.rec_idx
                    group_counter += 1
                continue

            # abs_start 기준 정렬
            wins.sort(key=lambda w: w.abs_start)

            # sweep line: 시간이 겹치는 윈도우를 그룹으로 묶기
            # group_max_end = 그룹 내 가장 늦게 끝나는 윈도우의 abs_end
            # → 어떤 윈도우라도 새 윈도우와 겹치면 그룹 유지
            current_group: list[_WindowInfo] = []
            group_max_end = 0

            for w in wins:
                if current_group and w.abs_start >= group_max_end:
                    # 어떤 기존 윈도우와도 겹치지 않음 → 그룹 확정
                    groups[group_counter] = [cw.idx for cw in current_group]
                    group_to_rec[group_counter] = current_group[0].rec_idx
                    group_counter += 1
                    current_group = []
                    group_max_end = 0

                current_group.append(w)
                group_max_end = max(group_max_end, w.abs_end)

            if current_group:
                groups[group_counter] = [cw.idx for cw in current_group]
                group_to_rec[group_counter] = current_group[0].rec_idx
                group_counter += 1

        self._groups: list[list[int]] = list(groups.values())
        self._group_rec_ids: list[int] = [group_to_rec[k] for k in groups.keys()]

    def set_epoch(self, epoch: int) -> None:
        """에폭마다 셔플 시드를 변경한다. DDP에서 모든 rank가 동일 호출 필수."""
        self.epoch = epoch

    def _shuffle_groups(self) -> list[int]:
        """Recording-locality를 보존하면서 그룹 순서를 셔플한다.

        모든 rank가 동일한 시드 → 동일한 순서를 생성한다.
        """
        group_indices = list(range(len(self._groups)))
        if not self.shuffle:
            return group_indices

        unique_recs = sorted(set(self._group_rec_ids))
        rec_perm = torch.randperm(
            len(unique_recs), generator=self.generator
        ).tolist()
        rec_order = [unique_recs[i] for i in rec_perm]

        rec_to_groups: dict[int, list[int]] = defaultdict(list)
        for g_idx, r_id in enumerate(self._group_rec_ids):
            rec_to_groups[r_id].append(g_idx)

        group_indices = []
        for r_id in rec_order:
            g_list = rec_to_groups[r_id]
            perm = torch.randperm(len(g_list), generator=self.generator).tolist()
            group_indices.extend(g_list[p] for p in perm)

        return group_indices

    def _groups_to_batches(self, group_indices: list[int]) -> list[list[int]]:
        """그룹 인덱스 리스트로부터 배치를 구성한다."""
        batches: list[list[int]] = []
        batch: list[int] = []
        for g_idx in group_indices:
            group = self._groups[g_idx]

            if batch and len(batch) + len(group) > self.batch_size:
                batches.append(batch)
                batch = []

            batch.extend(group)

            if len(batch) >= self.batch_size:
                batches.append(batch)
                batch = []

        if batch:
            if not self.drop_last:
                batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        """배치를 yield한다. 각 배치는 완전한 그룹(들)을 포함한다.

        DDP: 그룹 단위로 rank에 분배한다.
        - 그룹은 분리되지 않으므로 cross-modal pair 100% 보존
        - 각 rank가 자기 그룹으로 독립적으로 배치 구성
        - 배치 수를 deterministic하게 맞춰 deadlock 방지
        """
        self.generator.manual_seed(42 + self.epoch)
        group_indices = self._shuffle_groups()

        if self.world_size <= 1:
            # 단일 GPU: 전체 그룹 사용
            yield from self._groups_to_batches(group_indices)
            return

        # ── DDP: 그룹 단위 분배 ──

        # 1. 그룹 수를 world_size 배수로 패딩
        n_groups = len(group_indices)
        per_rank = math.ceil(n_groups / self.world_size)
        padded = group_indices.copy()
        while len(padded) < per_rank * self.world_size:
            padded.append(group_indices[len(padded) % n_groups])

        # 2. 연속 청크로 rank에 분배 (recording locality 보존)
        my_groups = padded[self.rank * per_rank : (self.rank + 1) * per_rank]

        # 3. 각 rank가 자기 그룹으로 배치 구성
        my_batches = self._groups_to_batches(my_groups)

        # 4. 모든 rank의 배치 수를 동일하게 맞춤 (통신 없이 deterministic 계산)
        all_batch_counts = []
        for r in range(self.world_size):
            r_groups = padded[r * per_rank : (r + 1) * per_rank]
            # 배치 수만 카운트 (실제 배치 구성 불필요)
            count, cur_size = 0, 0
            for g_idx in r_groups:
                g_size = len(self._groups[g_idx])
                if cur_size > 0 and cur_size + g_size > self.batch_size:
                    count += 1
                    cur_size = 0
                cur_size += g_size
                if cur_size >= self.batch_size:
                    count += 1
                    cur_size = 0
            if cur_size > 0 and not self.drop_last:
                count += 1
            all_batch_counts.append(count)

        max_batches = max(all_batch_counts) if all_batch_counts else 0

        # 부족분은 기존 배치 반복으로 패딩
        if my_batches and len(my_batches) < max_batches:
            orig_len = len(my_batches)
            while len(my_batches) < max_batches:
                my_batches.append(my_batches[len(my_batches) % orig_len])

        yield from my_batches

    def __len__(self) -> int:
        """배치 개수 (근사치)."""
        total_samples = sum(len(g) for g in self._groups)
        per_rank_samples = total_samples // max(self.world_size, 1)
        if self.drop_last:
            return per_rank_samples // self.batch_size
        else:
            return (per_rank_samples + self.batch_size - 1) // self.batch_size
