# -*- coding:utf-8 -*-
from __future__ import annotations

from torch.utils.data import DataLoader, DistributedSampler, Sampler

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
    """PackCollate가 적용된 DataLoader를 생성한다.

    Parameters
    ----------
    dataset:
        BiosignalDataset.
    max_length:
        PackCollate의 행 너비. 패킹된 텐서의 시간축 길이.
    batch_size:
        DataLoader가 한 번에 꺼내는 샘플 수.
        PackCollate가 이 샘플들을 bin-packing하므로
        실제 출력 행 수는 batch_size 이하가 된다.
    shuffle:
        에폭마다 샘플 순서를 섞을지 여부.
    num_workers:
        데이터 로딩 워커 수.
    drop_last:
        마지막 불완전 배치를 버릴지 여부.
    pin_memory:
        GPU 전송 속도 향상을 위해 pinned memory 사용 여부.
    persistent_workers:
        워커 프로세스를 에폭 간에 유지할지 여부 (num_workers>0일 때만 유효).
    prefetch_factor:
        워커당 미리 로딩할 배치 수 (num_workers>0일 때만 유효).
    collate_mode:
        PackCollate의 그루핑 모드. ``"ci"``이면 채널 독립,
        ``"any_variate"``이면 세션/레코딩 기반 그루핑.
    patch_size:
        패치 크기. 설정 시 PackCollate가 variate 길이를 patch_size 배수로 정렬.
    stride:
        패치 보폭 (overlapping 시). ``None``이면 ``patch_size``와 동일.
    sampler:
        외부 sampler (DDP DistributedSampler 등). 전달 시 shuffle 무시.
    """
    collate_fn = PackCollate(
        max_length=max_length,
        collate_mode=collate_mode,
        patch_size=patch_size,
        stride=stride,
    )

    import torch.distributed as dist

    use_ddp = dist.is_initialized()

    if collate_mode == "any_variate" and not use_ddp:
        # 단일 GPU: GroupedBatchSampler로 cross-modal 그루핑 보장
        batch_sampler = GroupedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
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

    # DDP 또는 CI 모드: DistributedSampler (DDP) 또는 기본 sampler
    # any_variate + DDP에서는 DistributedSampler로 샘플을 분배하고,
    # collate가 같은 버킷의 샘플을 자연스럽게 그루핑한다.
    # 일부 배치에서 cross-modal pair가 줄어들 수 있지만 deadlock 없음.
    if use_ddp and sampler is None:
        sampler = DistributedSampler(
            dataset, shuffle=shuffle, drop_last=drop_last,
        )

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
