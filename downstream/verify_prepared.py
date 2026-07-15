# -*- coding:utf-8 -*-
"""Chunked downstream prepared 출력 숫자 검증.

prepare_data.py 가 OOM 회피로 fold/split/chunk 단위 .pt 로 저장한 출력을
읽어 환자 단위 Patients/Events/Prevalence 를 확인한다.

각 chunk 의 metadata 에 n_patients/n_pos/fold_idx/split 이 있어,
무거운 'data'(신호) 는 건드리지 않고 metadata 만 합산한다 (빠름).

사용:
    python -m downstream.verify_prepared /home/coder/workspace/k-mimic-/bio_fm/data/downstream/sepsis
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="prepared 출력 디렉토리")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.out_dir}/*_chunk*.pt"))
    if not files:
        print(f"chunk 파일 없음: {args.out_dir}  (아직 생성 전이거나 경로 오류)")
        return

    by = defaultdict(lambda: [0, 0])   # (fold, split) -> [patients, pos]
    task = source = None
    for f in files:
        meta = torch.load(f, map_location="cpu", weights_only=False)["metadata"]
        task, source = meta.get("task"), meta.get("source")
        key = (meta["fold_idx"], meta["split"])
        by[key][0] += int(meta["n_patients"])
        by[key][1] += int(meta["n_pos"])

    print(f"task={task} | source={source} | chunk 파일={len(files)}개\n")
    print("fold | split | patients | events | prevalence")
    print("-" * 48)
    fold = defaultdict(lambda: [0, 0])
    for (fi, sp), (n, p) in sorted(by.items()):
        prev = f"{p / n:.1%}" if n else "—"
        print(f"  {fi}  | {sp:5s} | {n:7,d} | {p:6,d} | {prev}")
        fold[fi][0] += n
        fold[fi][1] += p

    print("\nfold TOTAL (train+val+test = 전체 prepared cohort, fold마다 동일해야 정상):")
    for fi, (n, p) in sorted(fold.items()):
        prev = f"{p / n:.1%}" if n else "—"
        print(f"  fold{fi}: patients={n:,}, events={p:,}, prevalence={prev}")


if __name__ == "__main__":
    main()
