# -*- coding:utf-8 -*-
from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import torch

from data.dataset import BiosignalSample
from data.spatial_map import get_global_spatial_id


@dataclass
class PackedBatch:
    """PackCollate의 출력.

    Attributes
    ----------
    values:
        패킹된 신호 값. ``(batch, max_length)``.
        남는 위치는 0으로 채워진다.
    sample_id:
        각 time-step이 어떤 원본 샘플에 속하는지 나타내는 ID.
        행(row) 내에서 1부터 시작하며, 0은 패딩을 의미한다.
        ``(batch, max_length)``.
    variate_id:
        각 time-step이 속한 variate ID (1-based, 0=padding).
        ``(batch, max_length)``.
    lengths:
        per-variate 원래 길이. ``(total_variates,)``.
    sampling_rates:
        per-variate 샘플링 레이트 (Hz). ``(total_variates,)``.
    signal_types:
        per-variate 신호 타입. ``(total_variates,)``.
    padded_lengths:
        per-variate patch 정렬된 길이. ``(total_variates,)``.
        패치가 설정된 경우에만 존재하며, 각 variate의 길이가
        ``patch_size``의 배수로 올림 패딩된 값이다. ``None``이면 미사용.
    """

    values: torch.Tensor  # (batch, max_length)
    sample_id: torch.Tensor  # (batch, max_length) long
    variate_id: torch.Tensor  # (batch, max_length) long
    lengths: torch.Tensor  # (total_variates,) long
    sampling_rates: torch.Tensor  # (total_variates,) float
    signal_types: torch.Tensor  # (total_variates,) long
    spatial_ids: torch.Tensor  # (total_variates,) long — 전역 spatial_id
    padded_lengths: torch.Tensor | None = None  # (total_variates,) long
    start_samples: torch.Tensor | None = None  # (total_variates,) long — 절대 시작 sample


# ── 내부 데이터 구조 ────────────────────────────────────────────


@dataclass
class _PackUnit:
    """FFD 패킹의 단위. 같은 그룹의 variate들을 이어 붙인 시퀀스."""

    values: torch.Tensor  # (time,)
    total_length: int
    channel_spans: list[tuple[int, int, int]]  # [(ch_idx, start, end), ...]
    variate_rates: list[float]
    variate_types: list[int]
    variate_spatial_ids: list[int]  # per-variate 전역 spatial_id
    variate_lengths: list[int]
    padded_variate_lengths: list[int]
    variate_start_samples: list[int]


class PackCollate:
    """Bin-packing collate: 가변 길이 시계열을 빈틈없이 채워 넣는다.

    같은 ``(recording_idx, win_start)`` 또는 ``(session_id, physical_time)``의
    채널들을 하나의 그룹으로 묶고, FFD 알고리즘으로 행에 패킹한다.

    Parameters
    ----------
    max_length:
        출력 텐서의 고정 행 너비. 이보다 긴 샘플은 잘린다.
    collate_mode:
        그루핑 모드. ``"any_variate"`` (기본) 또는 ``"ci"`` (채널 독립).
    patch_size:
        고정 패치 크기.
    stride:
        패치 간 보폭 (overlapping 지원). ``patch_size``와 함께 사용.

    패킹 전략: First-Fit Decreasing (FFD)
    """

    def __init__(
        self,
        max_length: int,
        collate_mode: str = "any_variate",
        patch_size: int | None = None,
        stride: int | None = None,
        slot_size: int = 60000,
        min_patches: int = 5,
    ) -> None:
        self.patch_size = patch_size
        # cross-modal 그루핑 슬롯 크기 (같은 슬롯 = 같은 sample_id)
        self._slot_size = slot_size
        # any_variate 모드에서 cross-modal 매칭을 위한 variate 최소 길이 (patch 단위)
        # 임상 기준 10s (patch_size=200 × 5 / 100Hz)
        self._min_patches = min_patches

        if patch_size is not None:
            self.stride = stride if stride is not None else patch_size
            assert patch_size % self.stride == 0, (
                f"patch_size({patch_size})는 stride({self.stride})의 배수여야 합니다."
            )
            max_length = max(patch_size, -(-max_length // self.stride) * self.stride)
        else:
            self.stride = None

        self.max_length = max_length
        self.collate_mode = collate_mode

    def __call__(self, samples: list[BiosignalSample]) -> PackedBatch:
        # 1. 그루핑: collate_mode에 따라 키 결정
        groups: dict[tuple, list[BiosignalSample]] = defaultdict(list)
        for i, s in enumerate(samples):
            if self.collate_mode == "ci":
                key = (i,)  # 고유 키 → 채널 간 그루핑 없음
            elif s.session_id:
                # session_id + 시간 슬롯으로 그루핑
                # 같은 슬롯 내 다른 signal type → 같은 sample_id → cross-modal pair
                abs_sample = s.start_sample + s.win_start
                slot = abs_sample // self._slot_size
                key = (s.session_id, slot)
            else:
                key = (s.recording_idx, s.win_start)
            groups[key].append(s)

        # 2. 각 그룹을 정렬 후 이어 붙여 하나의 PackUnit으로
        units: list[_PackUnit] = []
        for _key, group_samples in groups.items():
            group_samples.sort(key=lambda s: (s.signal_type, s.channel_idx))

            # Any-Variate 모드: Multi-tier length truncate
            # - 10s 미만 variate는 임상적 무의미 → 그룹에서 제거
            # - 남은 variate 중 valid tier(≥2 variate 유지) 랜덤 선택
            # - 선택된 tier 이상 길이의 variate만 유지하고 tier 길이로 truncate
            # - 결과: row 내 모든 variate가 같은 길이 → cross-modal pair 완벽 매칭
            # CI 모드는 영향 없음 (각 그룹이 1 variate라 조건에 안 걸림)
            group_limit: int | None = None
            if (
                self.collate_mode == "any_variate"
                and self.patch_size is not None
                and len(group_samples) >= 2
            ):
                ps = self.patch_size
                min_required = self._min_patches * ps

                # 각 variate의 post-trim effective 길이 계산
                sample_effs: list[tuple[BiosignalSample, int]] = []
                for s in group_samples:
                    abs_start = s.start_sample + s.win_start
                    remainder = abs_start % ps
                    trim = (ps - remainder) if remainder > 0 else 0
                    eff_len = s.values.shape[0] - trim
                    if eff_len >= min_required:
                        sample_effs.append((s, eff_len))

                if len(sample_effs) >= 2:
                    # patch 배수로 정렬된 unique 길이들
                    candidate_lengths = sorted(
                        {(eff // ps) * ps for _, eff in sample_effs}
                    )
                    # Valid tier: 해당 길이 이상 variate가 ≥2개인 tier
                    valid_tiers = [
                        L
                        for L in candidate_lengths
                        if sum(1 for _, eff in sample_effs if eff >= L) >= 2
                    ]
                    if valid_tiers:
                        # Sqrt-length-weighted 선택 — 긴 tier 약간 선호
                        # 실제 VitalDB 데이터 검증 결과 (crop ON 기준):
                        # sqrt는 평균 ~5분, median 5분, 300-600s 구간 48% 집중
                        # → 극단적 short/long 없이 clinical context 중심 분포
                        chosen = random.choices(
                            valid_tiers,
                            weights=[math.sqrt(L) for L in valid_tiers],
                            k=1,
                        )[0]
                        group_samples = [
                            s for s, eff in sample_effs if eff >= chosen
                        ]
                        group_limit = chosen
                    else:
                        group_samples = [s for s, _ in sample_effs]
                elif len(sample_effs) == 1:
                    # 1개만 남으면 cross-modal 불가, 그대로 단일 variate packing
                    group_samples = [s for s, _ in sample_effs]
                else:
                    # 모두 10s 미만 → 그룹 완전 제외
                    continue

            channel_values: list[torch.Tensor] = []  # each (time,)
            channel_spans: list[tuple[int, int, int]] = []
            variate_rates: list[float] = []
            variate_types: list[int] = []
            variate_spatial_ids: list[int] = []
            variate_lengths: list[int] = []
            padded_variate_lengths: list[int] = []
            variate_start_samples: list[int] = []
            offset = 0

            # [최적화] Concat 전에 미리 길이 확인하여 초과분 제거
            for s in group_samples:
                remaining = self.max_length - offset
                if remaining <= 0:
                    break  # 이미 max_length 도달

                # 절대 시작 sample
                abs_start = s.start_sample + s.win_start

                # 공통 시간 그리드 정렬: abs_start를 patch_size 배수로 올림
                # 앞부분을 잘라내어 모든 variate의 패치 경계가 동일 절대 시간에 정렬
                trim = 0
                if self.patch_size is not None:
                    remainder = abs_start % self.patch_size
                    if remainder > 0:
                        trim = self.patch_size - remainder
                        abs_start += trim  # patch_size 배수로 올림

                values = s.values[trim:]  # 앞부분 잘라냄
                seg_len = min(values.shape[0], remaining)
                # Multi-tier truncate: any_variate 모드에서 tier 길이로 제한
                if group_limit is not None:
                    seg_len = min(seg_len, group_limit)

                if seg_len <= 0:
                    continue

                # per-variate patch 파라미터 결정
                var_p: int | None = None
                var_s: int | None = None
                if self.patch_size is not None:
                    var_p = self.patch_size
                    var_s = self.stride

                if var_p is not None:
                    # 1 patch도 못 들어가면 이 variate 포기
                    if remaining < var_p:
                        break
                    if seg_len < var_p:
                        continue

                    # seg_len을 valid patch 길이(var_p + k*var_s)로 FLOOR
                    # — partial patch의 zero-padding 방지 (시각화/학습 아티팩트 제거)
                    seg_len = var_p + ((seg_len - var_p) // var_s) * var_s
                    # remaining에 의한 상한도 floor로 (remaining은 이미 stride-aligned)
                    remaining_valid = var_p + ((remaining - var_p) // var_s) * var_s
                    seg_len = min(seg_len, remaining_valid)

                    padded_seg_len = seg_len       # 더 이상 zero-padding 없음
                    v = values[:seg_len]           # 순수 real signal
                    effective_len = padded_seg_len
                else:
                    v = values[:seg_len]
                    effective_len = seg_len
                    padded_seg_len = seg_len

                channel_spans.append((s.channel_idx, offset, offset + effective_len))
                channel_values.append(v)
                variate_rates.append(s.sampling_rate)
                variate_types.append(s.signal_type)
                variate_spatial_ids.append(
                    get_global_spatial_id(s.signal_type, s.spatial_id)
                )
                variate_lengths.append(seg_len)
                padded_variate_lengths.append(padded_seg_len)
                variate_start_samples.append(abs_start)
                offset += effective_len

            # Concat은 이미 max_length 이하이므로 재정리 불필요
            if channel_values:  # 빈 그룹 체크
                concat = torch.cat(channel_values)

                units.append(
                    _PackUnit(
                        values=concat,
                        total_length=concat.shape[0],
                        channel_spans=channel_spans,
                        variate_rates=variate_rates,
                        variate_types=variate_types,
                        variate_spatial_ids=variate_spatial_ids,
                        variate_lengths=variate_lengths,
                        padded_variate_lengths=padded_variate_lengths,
                        variate_start_samples=variate_start_samples,
                    )
                )

        # 3. FFD 패킹
        bins = self._ffd_pack(units)

        # 4. 텐서 생성 — 최종 크기로 바로 할당 (중간 텐서 없음)
        n_rows = len(bins)

        padded_values = torch.zeros(n_rows, self.max_length)
        padded_ids = torch.zeros(n_rows, self.max_length, dtype=torch.long)
        padded_var_ids = torch.zeros(n_rows, self.max_length, dtype=torch.long)

        all_lengths: list[int] = []
        all_padded_lengths: list[int] = []
        all_rates: list[float] = []
        all_types: list[int] = []
        all_spatial_ids: list[int] = []
        all_start_samples: list[int] = []

        for row_idx, contents in enumerate(bins):
            row_len = min(
                sum(u.total_length for u in contents),
                self.max_length,
            )
            row_offset = 0
            for local_id, unit in enumerate(contents, start=1):
                seg_len = unit.total_length
                end_offset = min(row_offset + seg_len, row_len)
                actual_seg_len = end_offset - row_offset

                if actual_seg_len > 0:
                    padded_values[row_idx, row_offset:end_offset] = unit.values[
                        :actual_seg_len
                    ]
                    padded_ids[row_idx, row_offset:end_offset] = local_id

                    # variate_id 할당
                    for var_id, (_ch_idx, start, end) in enumerate(
                        unit.channel_spans, start=1
                    ):
                        var_start = max(row_offset, row_offset + start)
                        var_end = min(end_offset, row_offset + end)
                        if var_end > var_start:
                            padded_var_ids[row_idx, var_start:var_end] = var_id

                # metadata 수집
                if actual_seg_len == seg_len:  # 전체 unit이 포함됨
                    all_lengths.extend(unit.variate_lengths)
                    all_padded_lengths.extend(unit.padded_variate_lengths)
                    all_rates.extend(unit.variate_rates)
                    all_types.extend(unit.variate_types)
                    all_spatial_ids.extend(unit.variate_spatial_ids)
                    all_start_samples.extend(unit.variate_start_samples)
                else:
                    # 잘린 unit: 포함된 variate만 수집.
                    # span (start, end)는 unit-relative 좌표 (0-based).
                    # 절단 컷오프는 unit-relative로 actual_seg_len.
                    # included = min(end, actual_seg_len) - start  (unit-relative).
                    # NOTE: row_offset을 더하지 않고 unit-relative로 비교해야
                    # row_offset > 0인 unit이 잘릴 때 included가 잘못 계산되지 않음.
                    for var_id, (_ch_idx, start, end) in enumerate(unit.channel_spans):
                        seg_end_in_unit = min(end, actual_seg_len)
                        if seg_end_in_unit > start:
                            included = seg_end_in_unit - start
                            all_lengths.append(
                                min(included, unit.variate_lengths[var_id])
                            )
                            all_padded_lengths.append(included)
                            all_rates.append(unit.variate_rates[var_id])
                            all_types.append(unit.variate_types[var_id])
                            all_spatial_ids.append(unit.variate_spatial_ids[var_id])
                            all_start_samples.append(unit.variate_start_samples[var_id])

                row_offset += actual_seg_len
                if row_offset >= row_len:
                    break

        padded_lengths_tensor = (
            torch.tensor(all_padded_lengths, dtype=torch.long)
            if self.patch_size is not None
            else None
        )

        return PackedBatch(
            values=padded_values,
            sample_id=padded_ids,
            variate_id=padded_var_ids,
            lengths=torch.tensor(all_lengths, dtype=torch.long),
            sampling_rates=torch.tensor(all_rates, dtype=torch.float32),
            signal_types=torch.tensor(all_types, dtype=torch.long),
            spatial_ids=torch.tensor(all_spatial_ids, dtype=torch.long),
            padded_lengths=padded_lengths_tensor,
            start_samples=torch.tensor(all_start_samples, dtype=torch.long),
        )

    # ── FFD 패킹 ─────────────────────────────────────────────────

    def _ffd_pack(self, units: list[_PackUnit]) -> list[list[_PackUnit]]:
        """First-Fit Decreasing bin-packing (최적화). 긴 unit부터 배치한다."""
        sorted_units = sorted(units, key=lambda u: u.total_length, reverse=True)

        # [최적화] 버전 번호를 사용하여 오래된 항목 제거
        heap: list[tuple[int, int, int]] = []  # (-remaining, bin_idx, version)
        bin_remaining: list[int] = []
        bin_version: list[int] = []  # 각 bin의 현재 버전
        bin_contents: list[list[_PackUnit]] = []

        for unit in sorted_units:
            placed = False

            # [최적화] 오래된 항목을 한 번에 정리
            while heap and heap[0][2] != bin_version[heap[0][1]]:
                heapq.heappop(heap)

            if heap:
                neg_rem, bin_idx, ver = heap[0]
                remaining = -neg_rem
                if remaining >= unit.total_length:
                    heapq.heappop(heap)
                    bin_contents[bin_idx].append(unit)
                    new_remaining = remaining - unit.total_length
                    bin_remaining[bin_idx] = new_remaining
                    bin_version[bin_idx] += 1
                    if new_remaining > 0:
                        heapq.heappush(
                            heap, (-new_remaining, bin_idx, bin_version[bin_idx])
                        )
                    placed = True

            if not placed:
                bi = len(bin_contents)
                rem = self.max_length - unit.total_length
                bin_contents.append([unit])
                bin_remaining.append(rem)
                bin_version.append(1)
                if rem > 0:
                    heapq.heappush(heap, (-rem, bi, 1))

        return bin_contents
