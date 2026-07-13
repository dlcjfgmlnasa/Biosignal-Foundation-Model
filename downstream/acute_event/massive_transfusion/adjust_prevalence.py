# -*- coding:utf-8 -*-
"""완료된 per_case MT 데이터셋의 prevalence(양성율) 조정 — 재파싱 없이 음성 서브샘플.

양성 window 는 파형 품질상 상한이 있어(≈140/416), prevalence 는 음성 수로 조절한다.
이 스크립트는 이미 생성된 fold/split/chunk `.pt` 를 읽어 **양성 전량 유지 + 음성은
subject_id 해시로 keep** 하여 target prevalence 로 재저장한다(재파싱 불필요).

핵심: 음성 keep 은 `hash(subject_id)` 기반 **global·patient-consistent** — 같은 음성
case 는 전 horizon·fold 에서 동일하게 keep/drop → 5-fold CV 정합(한 case 가 어떤 fold
에선 포함, 다른 fold 에선 제외되는 오염) 을 방지한다.

사용법:
    python -m downstream.acute_event.massive_transfusion.adjust_prevalence \
        --data-dir F:/Massive_Transfusion_Downstream \
        --target-prevalence 0.023 --seed 42
    # 특정 keep-frac 직접 지정(count pass 생략, 빠름):
    #   ... --keep-frac 0.78
    # 원본 보존하려면 --out-dir 지정(기본=제자리 덮어쓰기).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import sys
from pathlib import Path

import torch


def _keep_control(pid: str, seed: int, keep_frac: float) -> bool:
    """subject_id 해시 → [0,1) < keep_frac 이면 keep (deterministic·patient-consistent)."""
    h = int(hashlib.sha1(f"{seed}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < keep_frac


def _iter_files(data_dir: str, prefix: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(f"{data_dir}/{prefix}*_fold*_*.pt"))


def _count_pos_neg(files: list[Path]) -> tuple[int, int]:
    """전 파일 라벨 합산 → (양성 window 수, 음성 window 수). label 만 읽고 즉시 해제."""
    n_pos = n_neg = 0
    for f in files:
        d = torch.load(f, weights_only=False)
        labs = d["data"]["labels"]
        p = int(labs.sum().item())
        n_pos += p
        n_neg += int(labs.numel()) - p
        del d
    return n_pos, n_neg


def _filter_file(
    f: Path, seed: int, keep_frac: float, out_dir: Path | None,
) -> tuple[int, int]:
    """파일의 양성 전량 + 해시-keep 음성만 남겨 재저장. (kept_pos, kept_neg) 반환."""
    d = torch.load(f, weights_only=False)
    data = d["data"]
    labels = data["labels"]
    subs = data["subject_ids"]
    n = int(labels.numel())

    keep_idx = [
        i for i in range(n)
        if int(labels[i].item()) == 1 or _keep_control(str(subs[i]), seed, keep_frac)
    ]
    idx_t = torch.tensor(keep_idx, dtype=torch.long)

    data["labels"] = labels[idx_t]
    data["label_values"] = data["label_values"][idx_t]
    data["signals"] = {k: v[idx_t] for k, v in data["signals"].items()}
    data["gap_masks"] = {k: v[idx_t] for k, v in data["gap_masks"].items()}
    data["case_ids"] = [data["case_ids"][i] for i in keep_idx]
    data["subject_ids"] = [subs[i] for i in keep_idx]

    md = d.get("metadata", {})
    md["n_samples"] = len(keep_idx)
    md["prevalence_adjusted"] = True
    md["keep_frac_neg"] = keep_frac

    out_path = (out_dir / f.name) if out_dir is not None else f
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(d, out_path)
    kept_pos = int(data["labels"].sum().item())
    return kept_pos, len(keep_idx) - kept_pos


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="MT per_case prevalence 조정 (음성 서브샘플)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prefix", default="massive_transfusion_abp_ppg_w600s")
    ap.add_argument("--target-prevalence", type=float, default=0.023,
                    help="목표 양성율. --keep-frac 미지정 시 count pass 로 keep-frac 산출.")
    ap.add_argument("--keep-frac", type=float, default=None,
                    help="음성 keep 비율 직접 지정(0~1). 지정 시 count pass 생략(빠름).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None, help="미지정 시 제자리 덮어쓰기.")
    args = ap.parse_args()

    files = _iter_files(args.data_dir, args.prefix)
    if not files:
        print(f"ERROR: no files under {args.data_dir}/{args.prefix}*", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(files)} files matched.")

    keep_frac = args.keep_frac
    if keep_frac is None:
        print("  [count pass] 전 파일 라벨 합산 중...")
        n_pos, n_neg = _count_pos_neg(files)
        p = args.target_prevalence
        # 전체 prevalence = n_pos / (n_pos + keep_frac*n_neg) = p
        target_neg = n_pos * (1.0 - p) / p
        keep_frac = max(0.0, min(1.0, target_neg / max(1, n_neg)))
        print(f"    현재 pos={n_pos} neg={n_neg} (prev {100*n_pos/(n_pos+n_neg):.2f}%) "
              f"→ target {100*p:.1f}% → keep_frac_neg={keep_frac:.3f}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    tot_pos = tot_neg = 0
    for f in files:
        kp, kn = _filter_file(f, args.seed, keep_frac, out_dir)
        tot_pos += kp
        tot_neg += kn
    dest = args.out_dir or args.data_dir
    prev = 100 * tot_pos / max(1, tot_pos + tot_neg)
    print(f"  Done → {dest}")
    print(f"  kept: pos={tot_pos} neg={tot_neg} | overall prevalence≈{prev:.2f}% "
          f"(파일 합산, window 단위)")


if __name__ == "__main__":
    main()
