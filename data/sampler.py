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
            # Shard-aware shuffling: dataset이 shard backend를 쓰면 같은 shard
            # recordings를 묶어서 셔플 → shard LRU cache hit율 ~100% 보장.
            # 미사용 시 (file backend) 기존 random 셔플로 fallback.
            rec_to_shard = getattr(self.dataset, "_rec_to_shard", None)
            if rec_to_shard is not None:
                rec_order = self._shard_aware_shuffle(rec_to_shard, g)
            else:
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

    def _shard_aware_shuffle(
        self,
        rec_to_shard: dict[str, int],
        generator: torch.Generator,
    ) -> list[int]:
        """Shard 단위로 묶어서 셔플 — shard cache hit율 ~100% 보장.

        1. (rec_idx → shard_id) 매핑으로 recording을 shard별로 그루핑
        2. shard 순서 셔플 (외부)
        3. 각 shard 내 recording 순서 셔플 (내부)

        결과: 같은 shard recording들이 연속으로 yield → 한 번 로드한 shard에서
        모든 recording 처리 후 다음 shard로 → cache eviction storm 방지.
        """
        from collections import defaultdict

        shard_to_recs: dict[int, list[int]] = defaultdict(list)
        for rec_i in range(self._n_recs):
            sid = rec_to_shard.get(str(rec_i), 0)
            shard_to_recs[sid].append(rec_i)

        shard_ids = list(shard_to_recs.keys())
        shard_perm = torch.randperm(len(shard_ids), generator=generator).tolist()
        rec_order: list[int] = []
        for sp in shard_perm:
            sid = shard_ids[sp]
            recs = shard_to_recs[sid]
            inner_perm = torch.randperm(len(recs), generator=generator).tolist()
            rec_order.extend(recs[i] for i in inner_perm)
        return rec_order

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
        slot_size: int = 60000,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.slot_size = slot_size
        self.epoch = 0
        # DDP: 모든 rank가 동일한 셔플 순서를 사용하도록 고정 시드 generator
        self.generator = generator or torch.Generator()
        self.generator.manual_seed(42)

        # 그룹 빌딩: (session_id, time_slot) 기반
        # collate와 동일한 키 → 같은 그룹이 같은 배치 + 같은 sample_id
        # recording 단위로 순회하여 O(M×W) — bisect 없이 직접 인덱스 계산
        groups: dict[tuple, list[int]] = defaultdict(list)
        group_to_rec: dict[tuple, int] = {}

        for rec_idx, entry in enumerate(dataset._manifest):
            n_ch = entry.n_channels
            n_win = dataset._n_windows_per_rec[rec_idx]
            if n_win == 0:
                continue
            stride = dataset._strides_per_rec[rec_idx]
            base = dataset._rec_offsets[rec_idx]
            has_session = bool(entry.session_id)

            for w in range(n_win):
                win_start = w * stride

                if has_session:
                    slot = (entry.start_sample + win_start) // slot_size
                    key = (entry.session_id, slot)
                else:
                    key = (rec_idx, win_start)

                # flat index = base + ch * n_win + w (dataset 인덱싱과 동일)
                groups[key].extend(base + ch * n_win + w for ch in range(n_ch))
                group_to_rec[key] = rec_idx

        self._groups: list[list[int]] = list(groups.values())
        keys = list(groups.keys())
        self._group_rec_ids: list[int] = [group_to_rec[k] for k in keys]
        # Shard backend 지원 — recording → shard 매핑 보존 (있을 때만)
        rec_to_shard = getattr(dataset, "_rec_to_shard", None)
        self._rec_to_shard: dict[int, int] | None = (
            {int(k): v for k, v in rec_to_shard.items()}
            if rec_to_shard is not None
            else None
        )

    def set_epoch(self, epoch: int) -> None:
        """에폭마다 셔플 시드를 변경한다. DDP에서 모든 rank가 동일 호출 필수."""
        self.epoch = epoch

    def _shuffle_groups(self) -> list[int]:
        """Shard 우선 → recording → group 순서로 셔플한다.

        Shard-aware (dataset이 shard backend 사용 시):
            shard 셔플 → 각 shard 내 recording 셔플 → 각 recording 내 group 셔플
            → 같은 shard recordings가 연속 yield → cache hit ~100%
        Shard 미사용 시:
            recording 셔플 → recording 내 group 셔플 (기존 동작)

        모든 rank가 동일한 시드 → 동일한 순서.
        """
        group_indices = list(range(len(self._groups)))
        if not self.shuffle:
            return group_indices

        rec_to_groups: dict[int, list[int]] = defaultdict(list)
        for g_idx, r_id in enumerate(self._group_rec_ids):
            rec_to_groups[r_id].append(g_idx)

        unique_recs = sorted(set(self._group_rec_ids))

        # ── Shard-aware 경로 ──
        if self._rec_to_shard is not None:
            shard_to_recs: dict[int, list[int]] = defaultdict(list)
            for r_id in unique_recs:
                sid = self._rec_to_shard.get(r_id, 0)
                shard_to_recs[sid].append(r_id)

            shard_ids = list(shard_to_recs.keys())
            shard_perm = torch.randperm(
                len(shard_ids), generator=self.generator
            ).tolist()
            rec_order: list[int] = []
            for sp in shard_perm:
                sid = shard_ids[sp]
                recs = shard_to_recs[sid]
                inner = torch.randperm(
                    len(recs), generator=self.generator
                ).tolist()
                rec_order.extend(recs[i] for i in inner)
        else:
            # ── 기존 random recording 셔플 ──
            rec_perm = torch.randperm(
                len(unique_recs), generator=self.generator
            ).tolist()
            rec_order = [unique_recs[i] for i in rec_perm]

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
