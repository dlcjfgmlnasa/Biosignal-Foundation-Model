# -*- coding:utf-8 -*-
from __future__ import annotations

from torch.utils.data import DataLoader, Sampler

from data.collate import PackCollate
from data.dataset import BiosignalDataset
from data.sampler import GroupedBatchSampler


def create_dataloader(
    dataset: BiosignalDataset,
    max_length: int,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int | None = None,
    collate_mode: str = "any_variate",
    patch_size: int | None = None,
    stride: int | None = None,
    sampler: Sampler | None = None,
) -> DataLoader:
    """PackCollate가 적용된 DataLoader를 생성한다."""
    collate_fn = PackCollate(
        max_length=max_length,
        collate_mode=collate_mode,
        patch_size=patch_size,
        stride=stride,
    )

    # any_variate 모드: GroupedBatchSampler로 같은 (session, time) 채널들을 같은 배치에 넣기
    if collate_mode == "any_variate":
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        batch_sampler = GroupedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            rank=rank,
            world_size=world_size,
            max_length=max_length,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        )
    else:
        # CI 모드: 개별 샘플 랜덤 샘플링 (DDP 시 sampler 사용)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        )
