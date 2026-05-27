# -*- coding:utf-8 -*-
"""K-MIMIC ablation 25% subset subject ID 생성.

본 ablation 14 variant 를 풀 K-MIMIC 으로 돌리는 것은 비현실적이므로
25% subject-stratified subset 으로 진행한다 (FM 논문 표준 관행).

Stratification 방침:
  - 각 subject 의 modality 집합(remap_v2 후 signal_type set) 계산
  - 희소 modality 보유 subject 를 비율대로 보존 (sampler 만으로 풀 수 없는
    공급량 자체를 미리 확보)
  - 흔한 modality(PPG/ABP/ECG) 만 가진 subject 는 단순 random sample

Usage:
  python -m scripts.make_ablation_subset \\
      --manifest /home/coder/.../k-mimic/manifest_full.jsonl \\
      --out configs/ablation/subset_25pct_subjects.txt \\
      --ratio 0.25 \\
      --seed 42

Output 형식 (한 줄에 subject_id 하나):
  subject_id_A
  subject_id_B
  ...
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from data.spatial_map import SIGNAL_TYPE_NAMES, remap_record_v2

# 희소 우선순위 — 작은 signal_type 일수록 흔한 modality.
# (집계로 직접 결정하지만, tie-breaker 와 인덱스 명확성을 위해 명시.)
SPARSE_PRIORITY = [7, 8, 9, 4, 5, 3, 0, 1, 2]  # ICP/RESP_Imp/RESP_Flow/CO2/AWP/CVP/ECG/ABP/PPG


def collect_subject_modalities(
    manifest_path: Path,
) -> dict[str, frozenset[int]]:
    """manifest_full.jsonl → {subject_id: frozenset(signal_types after remap)}."""
    out: dict[str, frozenset[int]] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            subject_id = meta.get("subject_id", "")
            if not subject_id:
                continue
            sts: set[int] = set()
            for session in meta.get("sessions", []):
                for rec in session.get("recordings", []):
                    remapped = remap_record_v2(rec["signal_type"], rec.get("spatial_ids"))
                    if remapped is None:
                        continue  # PAP drop
                    sts.add(remapped[0])
            out[subject_id] = frozenset(sts)
    return out


def stratified_sample(
    subject_modalities: dict[str, frozenset[int]],
    ratio: float,
    seed: int,
) -> list[str]:
    """Modality 분포 보존한 25% subject sampling.

    전략: 가장 희소한 modality 부터 역순으로 target ratio 달성.
      1) modality 별 보유 subject 집합 계산
      2) 희소한 modality 순으로 target 수 만큼 우선 채택
      3) 잔여 quota 는 random sample
    """
    rng = random.Random(seed)
    n_total = len(subject_modalities)
    n_target = max(1, int(round(n_total * ratio)))

    # modality 별 subject 집합 (sorted for determinism)
    modality_subjects: dict[int, list[str]] = defaultdict(list)
    for sid, mods in subject_modalities.items():
        for m in mods:
            modality_subjects[m].append(sid)
    for m in modality_subjects:
        modality_subjects[m].sort()

    # 희소 우선순위: 보유 subject 수 적은 modality 부터
    order = sorted(
        modality_subjects.keys(),
        key=lambda m: (len(modality_subjects[m]), SPARSE_PRIORITY.index(m) if m in SPARSE_PRIORITY else 99),
    )

    selected: set[str] = set()
    for m in order:
        pool = [s for s in modality_subjects[m] if s not in selected]
        n_have = len([s for s in modality_subjects[m] if s in selected])
        n_total_m = len(modality_subjects[m])
        n_want_m = max(1, int(round(n_total_m * ratio)))
        n_need = max(0, n_want_m - n_have)
        if n_need == 0 or not pool:
            continue
        rng.shuffle(pool)
        for s in pool[:n_need]:
            selected.add(s)
            if len(selected) >= n_target:
                break
        if len(selected) >= n_target:
            break

    # 잔여 quota: 나머지 subject 에서 random sample
    if len(selected) < n_target:
        remaining = sorted(set(subject_modalities.keys()) - selected)
        rng.shuffle(remaining)
        for s in remaining[: n_target - len(selected)]:
            selected.add(s)

    # 초과 시 trim (희소 우선이라 거의 발생 안 함)
    if len(selected) > n_target:
        common = [s for s in selected if all(
            len(modality_subjects[m]) > 100 for m in subject_modalities[s]
        )]
        rng.shuffle(common)
        for s in common[: len(selected) - n_target]:
            selected.discard(s)

    return sorted(selected)


def report_coverage(
    all_modalities: dict[str, frozenset[int]],
    selected: list[str],
) -> None:
    """선택된 subset 의 modality 분포 보고."""
    all_counter: Counter[int] = Counter()
    sub_counter: Counter[int] = Counter()
    for sid, mods in all_modalities.items():
        for m in mods:
            all_counter[m] += 1
    for sid in selected:
        for m in all_modalities[sid]:
            sub_counter[m] += 1

    print()
    print(f"{'Modality':<22} {'Full':>8} {'Subset':>8} {'Subset/Full':>12}")
    print("-" * 54)
    for st in sorted(all_counter.keys()):
        name = SIGNAL_TYPE_NAMES.get(st, f"st{st}")
        full = all_counter[st]
        sub = sub_counter.get(st, 0)
        ratio = sub / full if full else 0.0
        print(f"{name:<22} {full:>8} {sub:>8} {ratio:>11.1%}")
    print("-" * 54)
    print(f"{'Subjects total':<22} {len(all_modalities):>8} {len(selected):>8} {len(selected)/len(all_modalities):>11.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="K-MIMIC manifest_full.jsonl 경로")
    ap.add_argument("--out", type=Path, required=True,
                    help="subject ID 목록 출력 경로 (한 줄에 subject_id 하나)")
    ap.add_argument("--ratio", type=float, default=0.25,
                    help="subject sampling 비율 (default 0.25)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[1/3] Reading manifest: {args.manifest}")
    subject_mods = collect_subject_modalities(args.manifest)
    print(f"  Total subjects: {len(subject_mods)}")

    print(f"[2/3] Stratified sampling (ratio={args.ratio}, seed={args.seed})")
    selected = stratified_sample(subject_mods, args.ratio, args.seed)

    print(f"[3/3] Writing {len(selected)} subject IDs → {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(selected) + "\n", encoding="utf-8")

    report_coverage(subject_mods, selected)


if __name__ == "__main__":
    main()
