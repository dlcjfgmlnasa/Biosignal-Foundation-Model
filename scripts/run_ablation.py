# -*- coding:utf-8 -*-
"""Ablation sweep runner.

variants.yaml 의 각 variant 에 대해:
  1) Phase 1: _phase1_base.yaml + variant.phase1.overrides → temp yaml → torchrun
  2) Phase 2: _phase2_base.yaml + variant.phase2.overrides → temp yaml → torchrun
순차 실행한다.

지원 기능:
  * --variant <name>      : 특정 variant 만 실행 (반복 가능)
  * --group <group>       : ablation_group 단위 실행
  * --phase1-only         : Phase 1 만 실행
  * --phase2-only         : Phase 2 만 실행 (Phase 1 ckpt 존재 가정)
  * --dry-run             : 실제 train 실행 없이 merge 된 config 출력만
  * --nproc <N>           : torchrun --nproc_per_node (default 2)
  * --skip-existing       : Phase 1/2 best.pt 가 이미 존재하면 skip

phase1.reuse: <variant_name> 이면 해당 variant 의 Phase 1 ckpt 를 Phase 2
resume 에 사용 (Phase 1 baseline 공유).

Usage 예시:
  # 전체 sweep
  python -m scripts.run_ablation --nproc 2

  # 특정 ablation 그룹
  python -m scripts.run_ablation --group loss_component --nproc 2

  # 단일 variant dry-run
  python -m scripts.run_ablation --variant 02_no_lscnorm --dry-run
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ABL_DIR = REPO_ROOT / "configs" / "ablation"
PHASE1_BASE = ABL_DIR / "_phase1_base.yaml"
PHASE2_BASE = ABL_DIR / "_phase2_base.yaml"
VARIANTS_FILE = ABL_DIR / "variants.yaml"
# OUTPUT_ROOT 는 env override 가능 — yaml 의 output_dir 와 일치해야 함.
OUTPUT_ROOT = Path(
    os.environ.get(
        "ABLATION_OUTPUT_ROOT",
        "/home/coder/workspace/k-mimic-/bio_fm/outputs/ablation",
    )
)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override 가 우선. base 는 mutate 하지 않음."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_temp_yaml(config: dict, prefix: str) -> Path:
    """Merge 된 config 를 임시 yaml 파일로 저장하고 경로 반환."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{prefix}.yaml", delete=False, encoding="utf-8",
    )
    yaml.dump(config, tmp, sort_keys=False, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


def resolve_variants(
    all_variants: list[dict],
    filter_names: list[str] | None,
    filter_groups: list[str] | None,
) -> list[dict]:
    """--variant / --group 필터 적용."""
    if not filter_names and not filter_groups:
        return all_variants
    name_set = set(filter_names or [])
    group_set = set(filter_groups or [])
    out = []
    for v in all_variants:
        if v["name"] in name_set or v.get("ablation_group") in group_set:
            out.append(v)
    return out


def ckpt_path_for_exp(exp_name: str) -> Path:
    """exp_name 으로부터 가장 최신 *_best.pt 경로 반환.

    train_utils.save_training_checkpoint 은
        checkpoint_{phase_name}_epoch{NNN}_best.pt
    형식으로 저장하므로 단일 best.pt 가 아닌 glob 매칭이 필요.

    파일이 아직 없으면 (dry-run / 학습 미완료) placeholder 경로 반환 —
    caller (run_variant) 가 reuse 시점에 .exists() 로 검증한다.
    """
    ckpt_dir = OUTPUT_ROOT / exp_name / "checkpoints"
    if ckpt_dir.is_dir():
        candidates = sorted(ckpt_dir.glob("*_best.pt"))
        if candidates:
            return candidates[-1]
    # 아직 없음 — caller 에서 .exists() False 로 처리됨
    return ckpt_dir / "best.pt"  # placeholder


def build_torchrun_cmd(
    script_module: str,
    config_path: Path,
    nproc: int,
) -> list[str]:
    """torchrun command 구성."""
    if nproc <= 1:
        return [sys.executable, "-m", script_module, "--config", str(config_path)]
    return [
        "torchrun", f"--nproc_per_node={nproc}",
        "-m", script_module,
        "--config", str(config_path),
    ]


def run_phase1(
    variant: dict,
    nproc: int,
    dry_run: bool,
    skip_existing: bool,
) -> Path:
    """variant 의 Phase 1 학습 실행. Phase 1 ckpt 경로 반환.

    phase1.reuse: <other_variant_name> 이면 학습 없이 다른 variant 의 ckpt 사용.
    """
    p1 = variant.get("phase1", {})
    reuse_name = p1.get("reuse")
    if reuse_name:
        # 재사용 — 다른 variant 의 exp_name 으로부터 ckpt 경로 추론
        # variants 전역 조회 필요하므로 caller 가 따로 처리. 여기선 가정.
        # caller 는 reuse_name 의 phase1.overrides.exp_name 을 알아야 함.
        # 단순화: caller 가 phase1_ckpt 를 직접 명시하도록 위임 (run_variant 가 lookup).
        raise RuntimeError(
            "reuse 경로는 run_variant 에서 처리합니다 (run_phase1 직접 호출 금지)."
        )

    overrides = p1.get("overrides", {})
    base = load_yaml(PHASE1_BASE)
    cfg = deep_merge(base, overrides)
    exp_name = cfg.get("exp_name", "ablation_unknown_p1")

    ckpt = ckpt_path_for_exp(exp_name)
    if skip_existing and ckpt.exists():
        print(f"  [skip] Phase 1 ckpt exists: {ckpt}")
        return ckpt

    tmp_yaml = write_temp_yaml(cfg, f"{variant['name']}_p1")
    cmd = build_torchrun_cmd("train.1_channel_independency", tmp_yaml, nproc)

    print(f"  [Phase 1] variant={variant['name']} exp={exp_name}")
    print(f"    config: {tmp_yaml}")
    print(f"    cmd: {' '.join(cmd)}")
    if not dry_run:
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 1 failed (variant={variant['name']}, rc={result.returncode})")
    return ckpt


def run_phase2(
    variant: dict,
    phase1_ckpt: Path,
    nproc: int,
    dry_run: bool,
    skip_existing: bool,
) -> Path | None:
    """variant 의 Phase 2 학습 실행. run_phase2: false 면 None 반환."""
    if variant.get("run_phase2", True) is False:
        print(f"  [Phase 2] skipped (run_phase2: false, variant={variant['name']})")
        return None
    p2 = variant.get("phase2", {})
    overrides = copy.deepcopy(p2.get("overrides", {}))
    # resume 경로 자동 주입 — overrides 에 명시되어 있으면 우선
    overrides.setdefault("resume", str(phase1_ckpt))

    base = load_yaml(PHASE2_BASE)
    cfg = deep_merge(base, overrides)
    exp_name = cfg.get("exp_name", "ablation_unknown_p2")

    ckpt = ckpt_path_for_exp(exp_name)
    if skip_existing and ckpt.exists():
        print(f"  [skip] Phase 2 ckpt exists: {ckpt}")
        return ckpt

    tmp_yaml = write_temp_yaml(cfg, f"{variant['name']}_p2")
    cmd = build_torchrun_cmd("train.2_any_variate", tmp_yaml, nproc)

    print(f"  [Phase 2] variant={variant['name']} exp={exp_name}")
    print(f"    resume: {phase1_ckpt}")
    print(f"    config: {tmp_yaml}")
    print(f"    cmd: {' '.join(cmd)}")
    if not dry_run:
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Phase 2 failed (variant={variant['name']}, rc={result.returncode})")
    return ckpt


def lookup_phase1_exp_name(variants: list[dict], name: str) -> str:
    """variant 의 phase1 exp_name 을 조회 (reuse 해결용)."""
    for v in variants:
        if v["name"] == name:
            p1 = v.get("phase1", {})
            if p1.get("reuse"):
                # 재귀 (chain reuse)
                return lookup_phase1_exp_name(variants, p1["reuse"])
            return p1.get("overrides", {}).get(
                "exp_name", f"ablation_{name}_p1"
            )
    raise KeyError(f"variant not found: {name}")


def run_variant(
    variant: dict,
    all_variants: list[dict],
    nproc: int,
    dry_run: bool,
    skip_existing: bool,
    phase1_only: bool,
    phase2_only: bool,
) -> None:
    """단일 variant 의 Phase 1 + Phase 2 실행."""
    p1 = variant.get("phase1", {})
    reuse_name = p1.get("reuse")

    if reuse_name:
        # Phase 1 재사용 — 학습 스킵, ckpt 경로만 조회
        reused_exp = lookup_phase1_exp_name(all_variants, reuse_name)
        phase1_ckpt = ckpt_path_for_exp(reused_exp)
        print(f"  [Phase 1] reuse: {reuse_name} → {phase1_ckpt}")
        if not phase1_ckpt.exists() and not dry_run:
            raise FileNotFoundError(
                f"Reused Phase 1 ckpt missing: {phase1_ckpt}. "
                f"먼저 variant '{reuse_name}' 의 Phase 1 을 학습하세요."
            )
    elif phase2_only:
        # phase2-only: phase1 학습 건너뛰고 ckpt 추정
        exp_name = p1.get("overrides", {}).get("exp_name", f"ablation_{variant['name']}_p1")
        phase1_ckpt = ckpt_path_for_exp(exp_name)
        print(f"  [Phase 1] skipped (--phase2-only). 기존 ckpt: {phase1_ckpt}")
    else:
        phase1_ckpt = run_phase1(variant, nproc, dry_run, skip_existing)

    if phase1_only:
        return
    run_phase2(variant, phase1_ckpt, nproc, dry_run, skip_existing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants-file", type=Path, default=VARIANTS_FILE)
    ap.add_argument("--variant", action="append", default=[],
                    help="실행할 variant name (반복 가능). 미지정 시 전체")
    ap.add_argument("--group", action="append", default=[],
                    help="실행할 ablation_group (반복 가능)")
    ap.add_argument("--nproc", type=int, default=2,
                    help="torchrun --nproc_per_node (default 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="merge 된 config 출력만, train 실행 안 함")
    ap.add_argument("--skip-existing", action="store_true",
                    help="best.pt 가 이미 있으면 해당 phase skip")
    ap.add_argument("--phase1-only", action="store_true")
    ap.add_argument("--phase2-only", action="store_true")
    args = ap.parse_args()

    if args.phase1_only and args.phase2_only:
        ap.error("--phase1-only 와 --phase2-only 동시 지정 불가")

    variants_doc = load_yaml(args.variants_file)
    all_variants: list[dict] = variants_doc["variants"]
    selected = resolve_variants(all_variants, args.variant, args.group)

    print(f"[run_ablation] {len(selected)}/{len(all_variants)} variant(s) selected")
    for v in selected:
        print(f"  - {v['name']} ({v.get('ablation_group', '?')}): {v.get('description', '')}")

    for v in selected:
        print()
        print(f"━━━ variant: {v['name']} ━━━")
        run_variant(
            v, all_variants,
            nproc=args.nproc,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            phase1_only=args.phase1_only,
            phase2_only=args.phase2_only,
        )

    print()
    print(f"[run_ablation] DONE — {len(selected)} variants processed.")


if __name__ == "__main__":
    main()
