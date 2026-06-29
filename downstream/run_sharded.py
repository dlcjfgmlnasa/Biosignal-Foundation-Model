# -*- coding:utf-8 -*-
"""GPU job-sharding 풀 런처 (A안) — fold/config 작업을 여러 GPU에 분산 실행.

downstream sweep(bash)이 명령을 **직렬**로 돌리던 것을, 명령 목록을 받아 GPU 풀에
slot 단위로 **동시 실행**한다. 각 작업은 ``CUDA_VISIBLE_DEVICES`` 로 단일 GPU에
핀(pin)된 **독립 프로세스**라 학습 코드·결과 정확성에 전혀 영향이 없다. 한 작업이
끝나면 빈 GPU에 다음 작업이 즉시 투입되는 work-stealing 풀이라, 작업 길이가 들쭉
날쭉해도 GPU가 놀지 않는다.

LoRA fine-tuning 의 5-fold × (signal × window × horizon) sweep 은 fold·config 가
서로 **완전 독립**이라, DDP 없이 이 런처만으로 GPU 수만큼(L40S×4 → ~4×) 처리량이
선형 증가한다. 단일 fold 하나를 더 빨리 끝내야 할 때만 torchrun+DDP(``_ddp_utils``)
를 추가로 쓴다.

명령 목록 입력 (둘 중 하나):
  - ``--jobs-file FILE`` : 한 줄에 하나의 셸 명령 (``#`` 주석·빈 줄 무시)
  - stdin 파이프            : 명령을 한 줄에 하나씩

사용 예
-------
    # bash sweep 이 명령만 생성(echo) → 런처가 4 GPU에 분산
    bash downstream/.../bash/run.sh --print-only > jobs.txt
    python -m downstream.run_sharded --gpus 0,1,2,3 --jobs-file jobs.txt

    # 5-fold 를 4 GPU 에 분산 (한 config)
    for f in 0 1 2 3 4; do
      echo "python -m downstream.acute_event.hypotension.run \\
              --mode lora --n-folds 5 --fold $f --data-path ... --device cuda"
    done | python -m downstream.run_sharded --gpus 0,1,2,3

각 작업의 stdout/stderr 은 ``--log-dir`` (기본 ``./shard_logs``) 아래
``job_{idx:03d}_gpu{g}.log`` 로 분리 저장된다. 하나라도 실패하면 런처는 비-0 으로
종료하고 실패 목록을 출력한다(이미 시작된 작업은 끝까지 진행).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _read_jobs(args: argparse.Namespace) -> list[str]:
    """jobs-file 또는 stdin 에서 명령 목록을 읽는다 (주석·빈 줄 제거)."""
    if args.jobs_file:
        raw = Path(args.jobs_file).read_text(encoding="utf-8").splitlines()
    else:
        if sys.stdin.isatty():
            print(
                "ERROR: --jobs-file 미지정 시 stdin 으로 명령을 파이프해야 합니다.",
                file=sys.stderr,
            )
            sys.exit(2)
        raw = sys.stdin.read().splitlines()
    jobs = [ln.strip() for ln in raw if ln.strip() and not ln.lstrip().startswith("#")]
    return jobs


def _parse_gpus(spec: str) -> list[str]:
    """"0,1,2,3" → ["0","1","2","3"]. 'auto' 면 nvidia-smi 로 자동 탐지."""
    if spec != "auto":
        return [g.strip() for g in spec.split(",") if g.strip() != ""]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        gpus = [ln.strip() for ln in out.splitlines() if ln.strip() != ""]
        if not gpus:
            raise RuntimeError("no gpus")
        return gpus
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: GPU 자동탐지 실패({e}) -> CPU 단일 슬롯으로 폴백", file=sys.stderr)
        return [""]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--gpus", type=str, default="auto",
        help="콤마구분 GPU 인덱스 (예: 0,1,2,3) 또는 'auto'(nvidia-smi 탐지)",
    )
    p.add_argument(
        "--jobs-per-gpu", type=int, default=1,
        help="GPU 한 장에 동시 투입할 작업 수 (VRAM 여유 시 2 이상; 기본 1)",
    )
    p.add_argument("--jobs-file", type=str, default=None,
                   help="명령 목록 파일 (한 줄에 하나). 미지정 시 stdin 사용")
    p.add_argument("--log-dir", type=str, default="./shard_logs",
                   help="작업별 로그 디렉토리 (기본 ./shard_logs)")
    p.add_argument("--poll-sec", type=float, default=1.0,
                   help="실행 중 작업 폴링 주기(초)")
    p.add_argument("--dry-run", action="store_true",
                   help="실행 없이 GPU 배정·명령만 출력")
    args = p.parse_args()

    gpus = _parse_gpus(args.gpus)
    jobs = _read_jobs(args)
    if not jobs:
        print("ERROR: 실행할 작업이 없습니다.", file=sys.stderr)
        sys.exit(2)

    # slot = (gpu 인덱스) 를 jobs_per_gpu 만큼 복제 → 동시 실행 가능 슬롯 풀.
    slots = [g for g in gpus for _ in range(max(1, args.jobs_per_gpu))]
    n_slots = len(slots)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  run_sharded: {len(jobs)} jobs x {len(gpus)} GPU "
          f"x {args.jobs_per_gpu}/GPU = {n_slots} concurrent slots")
    print(f"  GPUs: {gpus}")
    print(f"  Logs: {log_dir.resolve()}")
    print("=" * 60)

    if args.dry_run:
        for i, cmd in enumerate(jobs):
            print(f"[dry] job {i:03d} -> GPU {slots[i % n_slots]!r}: {cmd}")
        return

    # 슬롯별 점유 상태: slot_idx → (Popen, job_idx, logfile) 또는 None.
    running: list[tuple[subprocess.Popen, int, object] | None] = [None] * n_slots
    next_job = 0
    failures: list[tuple[int, int, str]] = []  # (job_idx, returncode, cmd)
    done = 0
    total = len(jobs)

    def _launch(slot_idx: int, job_idx: int) -> None:
        """슬롯에 작업 1개 투입. launch 실패(빈 명령/실행파일 없음 등)는 런처를
        죽이지 않고 failures 에 기록한 뒤 슬롯을 비운다(다음 작업이 채움)."""
        gpu = slots[slot_idx]
        cmd = jobs[job_idx]
        env = os.environ.copy()
        # 빈 문자열(CPU 폴백)이면 CUDA_VISIBLE_DEVICES 미설정.
        if gpu != "":
            env["CUDA_VISIBLE_DEVICES"] = gpu
        try:
            argv = shlex.split(cmd)
            if not argv:
                raise ValueError("빈 명령")
            logf = open(  # noqa: SIM115 — 프로세스 수명 동안 유지, 종료 시 close
                log_dir / f"job_{job_idx:03d}_gpu{gpu or 'cpu'}.log",
                "w", encoding="utf-8",
            )
            logf.write(f"# CUDA_VISIBLE_DEVICES={gpu}\n# CMD: {cmd}\n\n")
            logf.flush()
            # POSIX(서버 bash) 기준 shlex split. 셸 메타문자(&&,|)는 미지원 — 단일 명령만.
            popen = subprocess.Popen(
                argv, env=env, stdout=logf, stderr=subprocess.STDOUT,
            )
        except Exception as e:  # noqa: BLE001 — launch 실패를 치명적이지 않게
            print(f"  [ERROR] job {job_idx + 1}/{total} launch 실패: {e}",
                  file=sys.stderr)
            failures.append((job_idx, -1, cmd))
            running[slot_idx] = None
            return
        running[slot_idx] = (popen, job_idx, logf)
        print(f"  [start] job {job_idx + 1}/{total} -> GPU {gpu or 'cpu'} "
              f"(pid {popen.pid})")

    def _terminate_all() -> None:
        """실행 중 자식 프로세스를 모두 종료(Ctrl-C/예외 시 orphan/GPU 점유 방지)."""
        for entry in running:
            if entry is None:
                continue
            popen, _job_idx, logf = entry
            if popen.poll() is None:
                popen.terminate()
            try:
                logf.close()
            except Exception:  # noqa: BLE001
                pass

    try:
        # 초기 채움
        for s in range(min(n_slots, total)):
            _launch(s, next_job)
            next_job += 1

        # 폴링 루프: 끝난 슬롯에 다음 작업 투입.
        while any(r is not None for r in running):
            time.sleep(args.poll_sec)
            for s in range(n_slots):
                entry = running[s]
                if entry is None:
                    continue
                popen, job_idx, logf = entry
                rc = popen.poll()
                if rc is None:
                    continue  # still running
                logf.close()
                done += 1
                status = "ok" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  [done ] job {job_idx + 1}/{total} "
                      f"({done} finished) -> {status}")
                if rc != 0:
                    failures.append((job_idx, rc, jobs[job_idx]))
                if next_job < total:
                    _launch(s, next_job)
                    next_job += 1
                else:
                    running[s] = None
    except KeyboardInterrupt:
        # Ctrl-C: 실행 중 자식을 모두 종료(orphan/GPU 점유 방지) 후 재발생.
        print("\n  [interrupt] 실행 중 작업 종료 중...", file=sys.stderr)
        _terminate_all()
        raise

    print("=" * 60)
    if failures:
        print(f"  완료: {total - len(failures)}/{total} 성공, {len(failures)} 실패")
        for job_idx, rc, cmd in failures:
            print(f"    job {job_idx:03d} rc={rc}: {cmd}")
        print(f"  실패 로그: {log_dir.resolve()}")
        sys.exit(1)
    print(f"  완료: 전체 {total} 작업 성공")
    print("=" * 60)


if __name__ == "__main__":
    main()
