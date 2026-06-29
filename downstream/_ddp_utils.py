# -*- coding:utf-8 -*-
"""Downstream LoRA용 DDP 헬퍼 (B안: torchrun 단일-run 데이터 병렬).

``torchrun --nproc_per_node=N`` 으로 실행되면(WORLD_SIZE>1) 각 rank 가 **동일 fold**
의 train window 를 rank 별로 분할(shard)해 forward/backward 하고, trainable
gradient(LoRA + probe)만 NCCL all-reduce 한다. val/test 평가·best-ckpt·결과 저장은
rank0 만 수행한다. torchrun 이 아니면(WORLD_SIZE 미설정/1) **모든 헬퍼가 단일 GPU
동작으로 폴백**하여 기존 경로가 그대로 유지된다(결과 불변).

설계 핵심
---------
- ``model.model._encode(...)`` 를 직접 호출하면 DDP 의 forward 훅을 타지 않아 grad
  all-reduce 가 **안 된다**. 그래서 encode→pool→probe 를 한 ``forward`` 로 묶은
  :class:`LoRATrainModule` 을 DDP 로 감싸, 학습 루프가 그 ``forward`` 를 호출하게 한다.
- DDP 는 rank 간 forward/backward 횟수가 같아야 한다(다르면 all-reduce 에서 hang).
  sharding 으로 rank 별 window 수가 달라질 수 있으므로 :func:`equalize_shard` 로
  전 rank 최소 길이에 맞춰 tail 을 잘라 step 수를 동기화한다.
- ``find_unused_parameters=True`` 필수: frozen encoder + LoRA 혼재 구조
  (memory: feedback_ddp_find_unused_true).
- pretrain ``train/train_utils.py`` 의 setup_ddp 와 동일 NCCL 관례(30분 timeout).
"""
from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn


# ── env 조회 ──────────────────────────────────────────────────


def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def ddp_rank() -> int:
    return int(os.environ.get("RANK", 0))


def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def ddp_enabled() -> bool:
    """torchrun(WORLD_SIZE>1) 으로 실행됐는지."""
    return ddp_world_size() > 1


def is_main() -> bool:
    """rank0(또는 비-DDP) 인지 — 평가/저장/출력 게이트용."""
    return ddp_rank() == 0


# ── 초기화 / 종료 ─────────────────────────────────────────────


def maybe_init_ddp() -> torch.device | None:
    """torchrun 이면 NCCL init + cuda device 설정 후 그 device 반환, 아니면 None.

    호출측은 반환이 None 이면 자기 ``args.device`` 를 그대로 쓴다. DDP 면 device 를
    ``cuda:LOCAL_RANK`` 로 강제한다(프로세스당 GPU 1장 핀).
    """
    if not ddp_enabled():
        return None
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=30),  # pretrain 관례와 동일(rank straggler 견딤)
    )
    local = ddp_local_rank()
    torch.cuda.set_device(local)
    return torch.device(f"cuda:{local}")


def cleanup_ddp() -> None:
    """프로세스 그룹 종료(있으면). 종료 전 barrier 로 rank 동기화."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ── 데이터 분할 ───────────────────────────────────────────────


def shard_for_rank(items: list) -> list:
    """list 를 rank 별로 분할(``items[rank::world]``). 비-DDP 면 원본 그대로.

    stride 분할이라 인접 fold·label 분포가 rank 간 고르게 섞인다(연속 슬라이스보다
    class imbalance skew 가 작다).
    """
    if not ddp_enabled():
        return items
    return items[ddp_rank() :: ddp_world_size()]


def equalize_shard(items: list) -> list:
    """전 rank 의 list 길이를 **최소값**에 맞춰 tail 을 자른다(비-DDP 면 그대로).

    DDP 는 rank 간 step 수가 같아야 backward all-reduce 에서 hang 하지 않는다.
    train window 수가 rank 별로 ±1 다를 수 있으므로 최소 길이에 맞춰 동기화한다.
    (잘리는 건 학습 window 일부뿐 — 평가는 rank0 이 full set 으로 수행하므로 metric
    에 영향 없음.)
    """
    if not ddp_enabled():
        return items
    local_n = torch.tensor([len(items)], device=torch.cuda.current_device())
    dist.all_reduce(local_n, op=dist.ReduceOp.MIN)
    n_min = int(local_n.item())
    return items[:n_min]


# ── DDP forward 래퍼 ─────────────────────────────────────────


def _mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    mask_f = patch_mask.unsqueeze(-1).float()
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


class LoRATrainModule(nn.Module):
    """encoder(``_encode``) → mean-pool → probe 를 한 forward 로 묶은 DDP 래퍼.

    DDP grad all-reduce 는 **wrap 한 모듈의 forward 호출**에서 backward 훅이 준비된다.
    기존 run.py 는 ``model.model._encode`` 를 직접 호출해 forward 를 우회하므로 DDP
    하에선 grad 가 동기화되지 않는다. 이 모듈을 DDP 로 감싸 학습 루프가 ``forward``
    를 타게 하면, 단일 GPU 경로(``_encode`` 직접 호출)와 **수치적으로 동일한** logits
    를 내면서 grad 만 추가로 all-reduce 된다.

    Parameters
    ----------
    encoder: ``DownstreamModelWrapper.model`` (frozen encoder + 주입된 LoRA adapter).
    probe:   LinearProbe (또는 동일 시그니처 head).
    """

    def __init__(self, encoder: nn.Module, probe: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.probe = probe

    def forward(self, batch) -> torch.Tensor:  # (B, n_classes)
        out = self.encoder._encode(batch, task="masked")
        feats = _mean_pool(out["encoded"], out["patch_mask"])
        return self.probe(feats)


def wrap_lora_ddp(encoder: nn.Module, probe: nn.Module):
    """(encoder, probe) → DDP 로 감싼 :class:`LoRATrainModule` (비-DDP 면 raw 모듈).

    반환된 모듈은 ``module(batch)`` 호출로 logits 를 낸다. 단일 GPU 면 DDP 없이
    LoRATrainModule 자체를 반환(동일 forward, 동기화 없음).
    """
    mod = LoRATrainModule(encoder, probe)
    if not ddp_enabled():
        return mod
    local = ddp_local_rank()
    return nn.parallel.DistributedDataParallel(
        mod,
        device_ids=[local],
        output_device=local,
        find_unused_parameters=True,  # frozen encoder + LoRA (memory feedback)
    )


# ── Patient-level DDP forward 래퍼 ───────────────────────────
#
# window-level :class:`LoRATrainModule` 은 (batch → logits) 라 patient-level
# aggregator task(mortality/sepsis/aki/vent_need/extubation)에 맞지 않는다.
# 이 task 들의 학습 단위는 **환자**다: ``encode_patient_windows`` 가 한 환자의 K
# 윈도우를 내부 루프로 인코딩(``model.model(batch, task="masked")``)한 뒤
# aggregator + probe 로 환자 표현을 만든다. 아래 모듈은 encode→aggregate→probe 를
# 한 ``forward`` 로 묶어 DDP grad all-reduce 가 정상 등록되게 한다.


class _PatientBatchPayload:
    """DDP scatter 가 내부 텐서를 건드리지 않도록 forward 인자를 감싸는 opaque holder.

    DDP 는 ``device_ids`` 가 지정되면 forward 전에 ``_to_kwargs`` →
    ``_recursive_to`` 로 입력을 순회하며 **list/tuple/dict/namedtuple 안의 Tensor
    를 자동으로 GPU 로 이동**시킨다. 환자 배치를 ``list[dict]`` 로 그대로 넘기면 그
    안의 모든 window 신호 텐서(K×채널)가 한꺼번에 GPU 로 올라가 ① per-window
    스트리밍 인코딩의 의미가 사라지고 ② 메모리가 폭증한다(OOM 위험).

    일반 파이썬 객체는 ``_recursive_to`` 가 재귀하지 않는 leaf 라(window-level
    :class:`LoRATrainModule` 이 ``PackedBatch`` dataclass 를 그대로 통과시키는 것과
    동일 원리), 이 holder 로 감싸면 디바이스 이동/스트리밍을
    ``encode_patient_windows`` 내부(model.batch_to_device)가 전담하게 된다.
    """

    __slots__ = (
        "model", "patients", "patch_size", "max_windows",
        "session_prefix", "use_time_secs",
    )

    def __init__(
        self, model, patients, patch_size, max_windows,
        session_prefix, use_time_secs,
    ) -> None:
        self.model = model
        self.patients = patients
        self.patch_size = patch_size
        self.max_windows = max_windows
        self.session_prefix = session_prefix
        self.use_time_secs = use_time_secs


class AggregatorTrainModule(nn.Module):
    """환자-단위 encode → aggregate → probe 를 한 forward 로 묶은 DDP 래퍼.

    DDP grad all-reduce 는 **wrap 한 모듈의 forward 호출**에서 backward 훅이
    준비된다. patient-level run.py 는 ``encode_patient_windows`` → ``aggregator``
    → ``probe`` 를 따로 호출하는데, ``encode_patient_windows`` 안의
    ``model.model(...)`` 직접 호출은 DDP forward 훅을 우회하므로 grad 가
    동기화되지 않는다. 이 모듈의 forward 를 학습 루프가 호출하게 하면, 단일 GPU
    경로와 **수치적으로 동일한** logits 를 내면서 encoder(LoRA)·aggregator·probe
    의 grad 만 추가로 all-reduce 된다.

    핵심: ``self.encoder`` 는 ``model.model`` (frozen encoder + 주입된 LoRA
    adapter) 과 **동일 객체**다. forward 안에서 ``encode_patient_windows(model,
    ...)`` 가 ``model.model``(=self.encoder)을 호출하므로 인코더 forward 가 실제로
    이 모듈을 통과한다 → autograd 그래프가 self.encoder/self.aggregator/self.probe
    의 파라미터를 모두 포함 → DDP reducer 가 (requires_grad=True 인) LoRA·aggregator
    ·probe 파라미터를 backward 에서 all-reduce 한다. frozen encoder 파라미터는
    requires_grad=False 라 DDP 가 무시한다.

    forward 인자는 :class:`_PatientBatchPayload` 단일 객체로 받는다(DDP scatter 가
    환자 텐서를 건드리지 않게 하기 위함 — 위 holder docstring 참조). 호출은
    :func:`run_aggregator_forward` 헬퍼를 통해 한다.

    Parameters
    ----------
    encoder:    ``DownstreamModelWrapper.model`` (frozen encoder + LoRA adapter).
    aggregator: TransformerAggregator (학습 가능).
    probe:      LinearProbe (또는 동일 시그니처 head).
    """

    def __init__(
        self, encoder: nn.Module, aggregator: nn.Module, probe: nn.Module
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.probe = probe

    def forward(self, payload: "_PatientBatchPayload") -> torch.Tensor:  # (B, n_classes)
        # 지연 import: aggregator 모듈은 data.* 무거운 의존을 끌어오므로 _ddp_utils
        # import 시점이 아니라 forward 시점에 로드한다(순환참조 회피·import 비용 절감).
        from downstream.aggregator import (
            collate_patients,
            encode_patient_windows,
        )

        model = payload.model
        patients = payload.patients
        patch_size = payload.patch_size
        max_windows = payload.max_windows
        session_prefix = payload.session_prefix
        use_time_secs = payload.use_time_secs

        device = next(self.probe.parameters()).device
        patient_reprs: list[torch.Tensor] = []
        patient_times: list[torch.Tensor | None] = []
        for p in patients:
            res = encode_patient_windows(
                model, p, patch_size, max_windows,
                use_lora=True,  # grad 활성 경로(enable_grad) → self.encoder 통과
                session_prefix=session_prefix,
                return_time_secs=use_time_secs,
            )
            if use_time_secs:
                reprs, times = res
            else:
                reprs, times = res, None
            patient_reprs.append(reprs)
            patient_times.append(times)

        # labels 는 호출측(run.py)이 loss 계산에 쓰므로 여기선 placeholder(0).
        # collate_patients 는 reprs/mask/time_secs 만 사용한다.
        padded, mask, _labels, time_secs = collate_patients(
            patient_reprs,
            [0] * len(patients),
            device,
            time_secs=patient_times if use_time_secs else None,
        )
        if use_time_secs:
            patient_repr = self.aggregator(padded, mask, time_secs=time_secs)
        else:
            patient_repr = self.aggregator(padded, mask)
        return self.probe(patient_repr)


def wrap_aggregator_ddp(
    encoder: nn.Module, aggregator: nn.Module, probe: nn.Module
):
    """(encoder, aggregator, probe) → DDP 로 감싼 :class:`AggregatorTrainModule`.

    비-DDP(단일 GPU)면 raw :class:`AggregatorTrainModule` 을 반환하지만, run.py 의
    단일 GPU 경로는 이 래퍼를 **사용하지 않고**(use_ddp 가 False 면 ddp_module=None)
    기존 encode→aggregate→probe 직접 호출을 그대로 탄다 → 결과 불변. DDP 면
    ``find_unused_parameters=True`` (frozen encoder + LoRA 혼재, memory feedback).

    주의: 호출 전 aggregator/probe/encoder 가 모두 동일 cuda device(LOCAL_RANK)에
    올라가 있어야 한다(DDP 가 construction 시 rank0 파라미터를 broadcast).
    """
    mod = AggregatorTrainModule(encoder, aggregator, probe)
    if not ddp_enabled():
        return mod
    local = ddp_local_rank()
    return nn.parallel.DistributedDataParallel(
        mod,
        device_ids=[local],
        output_device=local,
        find_unused_parameters=True,  # frozen encoder + LoRA (memory feedback)
    )


def run_aggregator_forward(
    agg_module,  # wrap_aggregator_ddp() 반환 (DDP 또는 raw AggregatorTrainModule)
    model,  # DownstreamModelWrapper (model.model 은 wrap 한 encoder 와 동일 객체)
    patients: list,  # [patient dict, ...]  (한 patient-batch)
    patch_size: int,
    max_windows: int,
    session_prefix: str = "patient",
    use_time_secs: bool = False,
) -> torch.Tensor:  # (B, n_classes)
    """환자 배치를 :class:`_PatientBatchPayload` 로 감싸 ``agg_module`` forward 호출.

    payload 로 감싸는 이유는 DDP scatter 가 환자 신호 텐서를 GPU 로 자동 이동시키지
    않게 하기 위함(:class:`_PatientBatchPayload` docstring 참조). DDP 면
    ``agg_module(payload)`` 가 DDP.forward 를 타 grad all-reduce 가 등록된다.
    """
    payload = _PatientBatchPayload(
        model, patients, patch_size, max_windows, session_prefix, use_time_secs,
    )
    return agg_module(payload)


# ── linear_probe feature 추출 병렬화 (gradient sync 없음) ──────
#
# linear_probe 는 encoder 를 frozen 으로 두고 feature 를 1회 no_grad 추출해 캐싱한
# 뒤 작은 probe 만 학습한다(encoder backprop 없음 → DDP 무의미). 비싼 부분은 feature
# 추출 1패스다. torchrun 에서는 각 rank 가 데이터의 **shard 만** 추출(≈world_size 배
# 가속)한 뒤 :func:`gather_concat` 으로 전 rank 결과를 모아 rank0 가 probe 학습·평가·
# 저장을 전담한다(rank0-only 저장 → 결과파일 race 없음).
#
# 사용 패턴 (각 split):
#   items_shard = shard_for_rank(windows)            # rank 별 분할 (equalize 불필요)
#   local = extract_features(items_shard)            # [(feat_cpu, label[, case_id]), ...]
#   full  = gather_concat(local)                     # 전 rank → rank-major flat concat
#   if not is_main(): cleanup → return               # non-rank0 는 추출만 돕고 종료
#   ... rank0 가 full 로 probe 학습/평가/저장 ...
#
# ⚠ 정렬(co-index) 보존: feature·label·case_id 는 **같은 리스트의 한 원소(튜플)** 로
# 묶어 한 번에 gather 해야 한다. 따로 gather 하면 rank 경계에서 어긋날 수 있다.
# gather_concat 은 rank0,1,2,… 순으로 이어붙이므로, 같은 sharding 순서에서 나온
# 리스트들끼리는 동일 rank-major 순서를 가져 상호 정렬이 유지된다.


def gather_concat(local_list: list) -> list:
    """list 를 전 rank 에서 all_gather 하여 rank-major 로 평탄 concat (비-DDP 면 원본).

    ``all_gather_object`` 라 picklable 이면 무엇이든 가능. **GPU 텐서는 호출 전에
    ``.cpu()`` 로 내려** 직렬화 device 이슈를 피하고, rank0 가 사용 직전 ``.to(device)``
    로 올리는 것을 권장한다. 반환 길이 = Σ(각 rank local_list 길이).
    """
    if not ddp_enabled():
        return local_list
    gathered: list = [None] * ddp_world_size()
    dist.all_gather_object(gathered, local_list)
    out: list = []
    for part in gathered:
        if part:
            out.extend(part)
    return out
