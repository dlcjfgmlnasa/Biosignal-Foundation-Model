# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader, Sampler

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalDataset
from data.sampler import GroupedBatchSampler


def create_dataloader(
    dataset: BiosignalDataset,
    max_length: int,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
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
    """
    collate_fn = PackCollate(
        max_length=max_length,
        collate_mode=collate_mode,
        patch_size=patch_size,
        stride=stride,
    )

    # any_variate 모드: GroupedBatchSampler로 같은 (session, time) 채널들을 같은 배치에 넣기
    if collate_mode == "any_variate":
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
