# -*- coding:utf-8 -*-
"""주경야독(web-orchestration) GPU 플랫폼 CLI.

`AGENTS.md` 의 공개 API 계약(https://web-orchestration.koreahealth.ai/AGENTS.md)을
감싼 얇은 클라이언트. 표준 라이브러리만 사용한다.

자격증명은 **레포 바깥** `~/.orch/orch.env` 에서 읽는다 (환경변수가 있으면 우선)::

    ORCH_BASE_URL=https://web-orchestration.koreahealth.ai
    ORCH_PAT=pat_...
    ORCH_NODE=<default node id>

사용 예::

    python scripts/orchctl.py me                     # 내 권한 확인
    python scripts/orchctl.py nodes                  # 쓸 수 있는 노드/GPU
    python scripts/orchctl.py up --gpus 1            # 세션 열기 (GPU 점유 시작)
    python scripts/orchctl.py run "nvidia-smi"       # 명령 실행 + 결과 대기
    python scripts/orchctl.py runfile train.sh       # 로컬 스크립트를 stdin 으로 주입
    python scripts/orchctl.py push out.csv           # ORCH_HOME 으로 업로드
    python scripts/orchctl.py pull runs/exp1 -o e.tgz
    python scripts/orchctl.py sync                   # 노드의 레포를 최신 브랜치로 갱신
    python scripts/orchctl.py down                   # 세션 반납 (GPU 해제)

주의: GPU 는 세션 단위 **배타 점유**다. 작업이 끝나면 반드시 ``down`` 으로 반납한다.

Git Bash 에서 ``api GET /usage`` 처럼 슬래시로 시작하는 인자를 넘길 때는
MSYS 경로 변환 때문에 ``MSYS_NO_PATHCONV=1`` 을 앞에 붙인다 (PowerShell 은 무관).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ENV_FILE = Path.home() / ".orch" / "orch.env"
STATE_FILE = Path.home() / ".orch" / "state.json"
TERMINAL_STATES = {"succeeded", "failed", "killed"}


# --------------------------------------------------------------------------- #
# config / state
# --------------------------------------------------------------------------- #
def load_env() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            cfg[key.strip().removeprefix("export ").strip()] = value.strip().strip('"').strip("'")
    for key in ("ORCH_BASE_URL", "ORCH_PAT", "ORCH_NODE"):
        if os.environ.get(key):
            cfg[key] = os.environ[key]
    if not cfg.get("ORCH_BASE_URL") or not cfg.get("ORCH_PAT"):
        sys.exit(f"ERROR: ORCH_BASE_URL / ORCH_PAT 이 없습니다. {ENV_FILE} 를 확인하세요.")
    cfg["ORCH_BASE_URL"] = cfg["ORCH_BASE_URL"].rstrip("/")
    return cfg


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def request(
    cfg: dict[str, str],
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    raw_out: Path | None = None,
    data: bytes | None = None,
    content_type: str | None = None,
) -> Any:
    url = cfg["ORCH_BASE_URL"] + path
    payload = data
    headers = {"Authorization": f"Bearer {cfg['ORCH_PAT']}"}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif content_type:
        headers["Content-Type"] = content_type

    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            if raw_out is not None:  # 다운로드 엔드포인트는 JSON 이 아니라 바이트를 준다
                raw_out.parent.mkdir(parents=True, exist_ok=True)
                with raw_out.open("wb") as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                return {"saved": str(raw_out), "bytes": raw_out.stat().st_size}
            text = resp.read().decode("utf-8")
            return json.loads(text) if text else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:  # 에러는 flat {"code","message","detail"}
            err = json.loads(detail)
            msg = err.get("message") or err.get("detail") or detail
            code = err.get("code", exc.code)
        except json.JSONDecodeError:
            msg, code = detail, exc.code
        hint = ""
        if exc.code == 403:
            hint = "  (권한 밖입니다 — 우회하지 말고 관리자에게 노드/시간 승인을 요청하세요)"
        elif exc.code == 401:
            hint = "  (PAT 이 없거나 폐기·만료되었습니다)"
        sys.exit(f"HTTP {exc.code} [{code}] {msg}{hint}")


def upload(cfg: dict[str, str], path_param: str, local: Path, node: str | None) -> Any:
    """multipart/form-data 업로드 (표준 라이브러리로 직접 조립)."""
    boundary = f"----orchctl{uuid.uuid4().hex}"
    ctype = mimetypes.guess_type(local.name)[0] or "application/octet-stream"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{local.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    payload = head + local.read_bytes() + tail
    query = {"path": path_param}
    if node:
        query["node"] = node
    return request(
        cfg,
        "POST",
        f"/me/data/upload?{urllib.parse.urlencode(query)}",
        data=payload,
        content_type=f"multipart/form-data; boundary={boundary}",
    )


def show(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# job helpers
# --------------------------------------------------------------------------- #
def current_session(cfg: dict[str, str], explicit: str | None) -> str:
    if explicit:
        return explicit
    sid = load_state().get("session_id")
    if not sid:
        sys.exit("ERROR: 활성 세션이 없습니다. 먼저 `orchctl.py up` 을 실행하세요.")
    return sid


def poll_job(cfg: dict[str, str], job_id: str, quiet: bool = False) -> dict[str, Any]:
    """터미널 상태가 될 때까지 폴링. 새로 도착한 stdout 만 증분 출력한다."""
    seen = 0
    delay = 1.0
    while True:
        res = request(cfg, "GET", f"/jobs/{job_id}")
        out = res.get("stdout_tail") or ""
        if not quiet and len(out) > seen:
            sys.stdout.write(out[seen:])
            sys.stdout.flush()
            seen = len(out)
        state = res["job"]["state"]
        if state in TERMINAL_STATES:
            err = res.get("stderr_tail") or ""
            if err.strip():
                print("\n--- stderr ---", file=sys.stderr)
                print(err, file=sys.stderr)
            print(f"\n[job {job_id}] state={state} exit_code={res['job'].get('exit_code')}")
            return res
        time.sleep(delay)
        delay = min(delay * 1.4, 10.0)


def exec_script(
    cfg: dict[str, str],
    sid: str,
    script: str,
    timeout: int,
    wait: bool = True,
) -> dict[str, Any] | str:
    """스크립트를 stdin 으로 주입해 실행한다.

    명령 문자열을 그대로 보내면 개행/따옴표가 깨지므로 항상 ``bash -s`` + stdin 을 쓴다.
    """
    res = request(
        cfg,
        "POST",
        f"/sessions/{sid}/exec",
        {"command": "bash -s", "stdin": script, "timeout_sec": timeout},
    )
    job_id = res["job_id"]
    if not wait:
        print(f"job_id={job_id}  (`orchctl.py job {job_id}` 로 확인)")
        return job_id
    return poll_job(cfg, job_id)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_me(cfg, args):
    show(request(cfg, "GET", "/me"))


def cmd_nodes(cfg, args):
    nodes = request(cfg, "GET", "/nodes")
    for node in nodes:
        caps = node.get("capabilities") or {}
        gpus = caps.get("gpus") or []
        model = gpus[0]["model"] if gpus else "?"
        mem = gpus[0]["memory_total_mib"] // 1024 if gpus else 0
        print(
            f"{node['name']:<12} {node['state']:<8} {node['gpu_count']}x {model} {mem}GB  "
            f"cpu={caps.get('cpu_cores')} ram={caps.get('ram_total_mib', 0) // 1024}GB  id={node['id']}"
        )


def cmd_meta(cfg, args):
    show(request(cfg, "GET", "/meta"))


def cmd_grants(cfg, args):
    show(request(cfg, "GET", "/grants"))


def cmd_storage(cfg, args):
    node = args.node or cfg.get("ORCH_NODE")
    show(request(cfg, "GET", f"/me/storage?node={node}" if node else "/me/storage"))


def cmd_up(cfg, args):
    node = args.node or cfg.get("ORCH_NODE")
    if not node:
        sys.exit("ERROR: --node 또는 ORCH_NODE 가 필요합니다.")
    body: dict[str, Any] = {"gpu_count": args.gpus}
    if args.gpu_model:
        body["gpu_model"] = args.gpu_model
    session = request(cfg, "POST", f"/nodes/{node}/sessions", body)
    save_state({"session_id": session["id"], "node_id": session["node_id"]})
    print(f"session={session['id']} state={session['state']} gpus={session['gpu_indices']}")
    print("작업이 끝나면 `orchctl.py down` 으로 반드시 반납하세요.")


def cmd_ps(cfg, args):
    for session in request(cfg, "GET", "/sessions"):
        if args.all or session["state"] != "released":
            print(
                f"{session['id']}  {session['state']:<9} gpus={session['gpu_indices']}  "
                f"created={session['created_at']}"
            )


def cmd_down(cfg, args):
    sid = current_session(cfg, args.session)
    request(cfg, "DELETE", f"/sessions/{sid}")
    state = load_state()
    if state.get("session_id") == sid:
        save_state({})
    print(f"released {sid}")


def cmd_reset(cfg, args):
    sid = current_session(cfg, args.session)
    show(request(cfg, "POST", f"/sessions/{sid}/reset"))


def _wrap(script: str, raw: bool) -> str:
    """기본으로 노드의 영속 환경(env.sh)을 source 한다."""
    if raw:
        return script
    return 'source "$ORCH_HOME/env.sh"\n' + script


def cmd_run(cfg, args):
    sid = current_session(cfg, args.session)
    exec_script(cfg, sid, _wrap(args.command, args.raw), args.timeout, wait=not args.detach)


def cmd_runfile(cfg, args):
    sid = current_session(cfg, args.session)
    script = Path(args.file).read_text(encoding="utf-8")
    exec_script(cfg, sid, _wrap(script, args.raw), args.timeout, wait=not args.detach)


def cmd_jobs(cfg, args):
    for job in request(cfg, "GET", f"/jobs?limit={args.limit}"):
        cmd = (job.get("command") or "").replace("\n", " ")[:70]
        print(f"{job['id']}  {job['state']:<10} exit={job.get('exit_code')}  {job['submitted_at']}  {cmd}")


def cmd_job(cfg, args):
    if args.follow:
        poll_job(cfg, args.job_id)
    else:
        res = request(cfg, "GET", f"/jobs/{args.job_id}")
        print(res.get("stdout_tail") or "")
        if (res.get("stderr_tail") or "").strip():
            print("--- stderr ---", file=sys.stderr)
            print(res["stderr_tail"], file=sys.stderr)
        print(f"[state={res['job']['state']} exit_code={res['job'].get('exit_code')}]")


def cmd_ls(cfg, args):
    node = args.node or cfg.get("ORCH_NODE")
    area = {"home": "/me/data", "shared": "/shared", "datasets": "/datasets"}[args.area]
    query = urllib.parse.urlencode({"path": args.path, "node": node} if node else {"path": args.path})
    res = request(cfg, "GET", f"{area}?{query}")
    for entry in res.get("entries", []):
        kind = "d" if entry["is_dir"] else "-"
        print(f"{kind} {entry['size']:>12}  {entry['name']}")


def cmd_push(cfg, args):
    node = args.node or cfg.get("ORCH_NODE")
    show(upload(cfg, args.dest, Path(args.file), node))


def cmd_pull(cfg, args):
    node = args.node or cfg.get("ORCH_NODE")
    area = {"home": "/me/data", "shared": "/shared", "datasets": "/datasets"}[args.area]
    endpoint = "archive" if args.dir else "file"
    query = urllib.parse.urlencode({"path": args.path, "node": node} if node else {"path": args.path})
    show(request(cfg, "GET", f"{area}/{endpoint}?{query}", raw_out=Path(args.out)))


def cmd_sync(cfg, args):
    sid = current_session(cfg, args.session)
    exec_script(cfg, sid, f'bash "$ORCH_HOME/bin/update_code.sh" {args.branch}\n', 600)


def cmd_api(cfg, args):
    """탈출구 — 임의의 엔드포인트 직접 호출."""
    body = json.loads(args.body) if args.body else None
    show(request(cfg, args.method.upper(), args.path, body))


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orchctl", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        return p

    add("me", cmd_me, "내 계정·역할·스코프")
    add("nodes", cmd_nodes, "쓸 수 있는 노드/GPU 목록")
    add("meta", cmd_meta, "플랫폼 설정(예약 on/off, idle timeout 등)")
    add("grants", cmd_grants, "내 노드 접근 승인 현황")

    p = add("storage", cmd_storage, "내 저장소 사용량")
    p.add_argument("--node")

    p = add("up", cmd_up, "세션 생성 (GPU 점유 시작)")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--node")
    p.add_argument("--gpu-model", dest="gpu_model")

    p = add("ps", cmd_ps, "세션 목록")
    p.add_argument("--all", action="store_true", help="반납된 세션까지")

    p = add("down", cmd_down, "세션 반납 (GPU 해제)")
    p.add_argument("--session")

    p = add("reset", cmd_reset, "세션 리셋 (잔여 프로세스·GPU 메모리 정리)")
    p.add_argument("--session")

    p = add("run", cmd_run, "명령 실행 + 결과 대기")
    p.add_argument("command")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--session")
    p.add_argument("--detach", action="store_true", help="job id 만 받고 즉시 반환")
    p.add_argument("--raw", action="store_true", help="env.sh 자동 source 안 함")

    p = add("runfile", cmd_runfile, "로컬 스크립트를 stdin 으로 주입해 실행")
    p.add_argument("file")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--session")
    p.add_argument("--detach", action="store_true")
    p.add_argument("--raw", action="store_true")

    p = add("jobs", cmd_jobs, "최근 잡 목록")
    p.add_argument("--limit", type=int, default=20)

    p = add("job", cmd_job, "잡 상세/로그")
    p.add_argument("job_id")
    p.add_argument("--follow", action="store_true", help="끝날 때까지 따라가기")

    p = add("ls", cmd_ls, "데이터 영역 목록")
    p.add_argument("path", nargs="?", default="")
    p.add_argument("--area", choices=["home", "shared", "datasets"], default="home")
    p.add_argument("--node")

    p = add("push", cmd_push, "파일 업로드 (ORCH_HOME)")
    p.add_argument("file")
    p.add_argument("--dest", default="uploads")
    p.add_argument("--node")

    p = add("pull", cmd_pull, "파일/폴더 다운로드")
    p.add_argument("path")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--dir", action="store_true", help="폴더를 tar.gz 로")
    p.add_argument("--area", choices=["home", "shared", "datasets"], default="home")
    p.add_argument("--node")

    p = add("sync", cmd_sync, "노드의 레포를 GitHub 최신 브랜치로 갱신")
    p.add_argument("branch", nargs="?", default="feat/ppg-abp-calibrated-absolute")
    p.add_argument("--session")

    p = add("api", cmd_api, "임의 엔드포인트 직접 호출")
    p.add_argument("method")
    p.add_argument("path")
    p.add_argument("--body")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(load_env(), args)


if __name__ == "__main__":
    main()
