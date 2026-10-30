"""服务端自测：不依赖 ``client/`` 与 ``examples/`` 的原始 socket 客户端。

跑通完整闭环（``docs/PROTOCOL.md`` v1）：

1. 错误 token → ``error BAD_AUTH``；
2. 正确 token 握手 → ``hello_ok``；
3. ``job_submit`` → ``job_accepted``（列 missing/present）；
4. 上传第一个文件后**直接断开**，重连带 ``resume_job_id`` 续传
   （present 跳过已完成文件）→ 补齐剩余文件；
5. ``job_start`` → ``job_started`` / ``job_progress`` / ``job_event`` / ``job_finished``；
6. ``output_list`` / ``output_chunk`` 下载产物并逐片 + 整文件校验 sha256；
7. 同一 spec 二次提交 → ``job_started.env.state == "cached"``；
8. ``job_status`` 查询终态。

用法::

    python server/selftest.py --server 127.0.0.1:18765 --token selftest

退出码：0 全部通过；1 失败。所有临时文件落在 ``<项目根>/_rgpu_data/selftest-*``
（``.gitignore`` 已排除），默认结束时清理；``--keep`` 保留现场。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import parse_host_port  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.models import (  # noqa: E402
    EnvSpec,
    FileEntry,
    JobSpec,
    Resources,
    validate_job_spec,
)
from common.protocol import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    FrameReader,
    FrameWriter,
)

PREFIX = "[selftest]"
TIMEOUT_S = 90.0


def log(message: str) -> None:
    print(f"{PREFIX} {message}", flush=True)


class Failure(RuntimeError):
    """断言失败。"""


def check(condition: Any, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# 极简原始 socket 客户端
# ---------------------------------------------------------------------------
class RawClient:
    def __init__(self, host: str, port: int, token: str, timeout: float = TIMEOUT_S) -> None:
        self.host = host
        self.port = int(port)
        self.token = token
        self.timeout = float(timeout)
        self.sock: Optional[socket.socket] = None
        self.reader: Optional[FrameReader] = None
        self.writer: Optional[FrameWriter] = None

    def connect(self) -> "RawClient":
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        self.reader = FrameReader(self.sock)
        self.writer = FrameWriter(self.sock)
        return self

    def close(self) -> None:
        sock, self.sock = self.sock, None
        self.reader = None
        self.writer = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def send_json(self, obj: Dict[str, Any]) -> None:
        assert self.writer is not None
        self.writer.send_json(obj)

    def send_blob(self, data: bytes) -> None:
        assert self.writer is not None
        self.writer.send_blob(data)

    def recv_json(self) -> Dict[str, Any]:
        assert self.reader is not None
        return self.reader.read_json()

    def recv_blob(self) -> bytes:
        assert self.reader is not None
        return self.reader.read_blob()

    def hello(self, token: Optional[str] = None) -> Dict[str, Any]:
        self.send_json(
            {
                "kind": "hello",
                "protocol": 1,
                "role": "client",
                "client_version": "0.1.0-selftest",
                "token": self.token if token is None else token,
            }
        )
        return self.recv_json()

    def __enter__(self) -> "RawClient":
        return self.connect()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# 玩具项目（纯标准库，服务端 CPU 模式可跑）
# ---------------------------------------------------------------------------
TOY_TRAIN_PY = '''"""selftest 玩具训练脚本：读取基座模型 + 增量数据，写出产物。"""
import json
import sys
from pathlib import Path


def main() -> int:
    model_path = Path("model.json")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    rows = [line for line in Path("data.csv").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    print(f"加载基座模型 counter={model.get('counter', 0)}", flush=True)
    print(f"读取增量数据 {len(rows)} 行", flush=True)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = int(model.get("counter", 0)) + len(rows)
    updated = {"counter": counter, "loss": round(1.0 / (counter + 1), 6)}
    (out_dir / "model.json").write_text(json.dumps(updated), encoding="utf-8")
    (out_dir / "train.log").write_text(f"counter={counter}\\n", encoding="utf-8")
    print(f"epoch 1/1 done counter={counter}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def make_toy_project(root: Path) -> Path:
    """创建玩具项目目录：train.py + model.json + data.csv。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text(TOY_TRAIN_PY, encoding="utf-8")
    (root / "model.json").write_text(
        json.dumps({"counter": 1, "loss": 0.5}, indent=2), encoding="utf-8"
    )
    (root / "data.csv").write_text(
        "x,y\n" + "".join(f"{i},{i * 2}\n" for i in range(1, 6)), encoding="utf-8"
    )
    return root


def build_spec(project: Path, *, job_name: str = "selftest-toy") -> JobSpec:
    env = EnvSpec(
        mode="venv",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        requirements=[],  # 零依赖 → 服务端用 venv --without-pip，不需要联网
        reuse=True,
    )
    env.fingerprint = env.compute_fingerprint()
    inputs: List[FileEntry] = []
    for name in ("train.py", "model.json", "data.csv"):
        path = project / name
        inputs.append(
            FileEntry(path=name, size=path.stat().st_size, sha256=sha256_file(path))
        )
    spec = JobSpec(
        job_name=job_name,
        entry_argv=["python", "train.py"],
        inputs=inputs,
        env=env,
        resources=Resources(gpu=False, gpu_count=1, timeout_s=120),
        outputs=["outputs/**"],
        workdir=".",
        meta={"mode": "selftest"},
        client_version="0.1.0-selftest",
    )
    validate_job_spec(spec)
    return spec


# ---------------------------------------------------------------------------
# 协议动作
# ---------------------------------------------------------------------------
def submit(client: RawClient, spec: JobSpec, resume_job_id: Optional[str] = None) -> Dict[str, Any]:
    client.send_json(
        {
            "kind": "job_submit",
            "spec": spec.to_dict(),
            "resume_job_id": resume_job_id,
        }
    )
    message = client.recv_json()
    check(
        message.get("kind") == "job_accepted",
        f"job_submit 期望 job_accepted，收到 {message!r}",
    )
    return message


def upload_file(
    client: RawClient, entry: FileEntry, project: Path, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Dict[str, Any]:
    data = (project / entry.path).read_bytes()
    check(len(data) == entry.size, f"{entry.path} 本地大小与清单不一致")
    offset = 0
    while offset < len(data):
        chunk = data[offset:offset + chunk_size]
        client.send_json(
            {
                "kind": "file_chunk",
                "path": entry.path,
                "offset": offset,
                "length": len(chunk),
                "sha256": sha256_bytes(chunk),
            }
        )
        client.send_blob(chunk)
        response = client.recv_json()
        check(
            response.get("kind") == "file_ok",
            f"{entry.path} 分片未被确认: {response!r}",
        )
        check(
            int(response.get("offset", -1)) == offset + len(chunk),
            f"{entry.path} file_ok.offset 不正确: {response!r}",
        )
        offset += len(chunk)
    client.send_json({"kind": "file_done", "path": entry.path})
    done = client.recv_json()
    check(done.get("kind") == "file_ok", f"{entry.path} file_done 未确认: {done!r}")
    check(bool(done.get("done")), f"{entry.path} 未标记 done: {done!r}")
    return done


def upload_all(
    client: RawClient, spec: JobSpec, project: Path, paths: List[str]
) -> None:
    index = spec.input_index()
    for path in paths:
        upload_file(client, index[path], project)
        log(f"已上传 {path}（{index[path].size} B）")


def wait_finished(
    client: RawClient, job_id: str, *, label: str = ""
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """发 job_start，读到 job_finished；返回 (finished, started, events_text)。"""
    client.send_json({"kind": "job_start", "job_id": job_id})
    started: Dict[str, Any] = {}
    events: List[str] = []
    progress: List[str] = []
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        message = client.recv_json()
        kind = message.get("kind")
        if kind == "job_started":
            started = message
            env = message.get("env") or {}
            log(
                f"job_started{(' ' + label) if label else ''}: "
                f"env.state={env.get('state')} fingerprint={str(env.get('fingerprint'))[:12]}…"
            )
        elif kind == "job_progress":
            text = f"{message.get('state')} {message.get('progress')} {message.get('message') or ''}"
            if not progress or progress[-1] != text:
                progress.append(text)
                log(f"job_progress: {text.strip()}")
        elif kind == "job_event":
            events.append(str(message.get("data") or ""))
        elif kind == "error":
            raise Failure(f"作业执行期间收到 error: {message!r}")
        elif kind == "job_finished":
            log(
                f"job_finished{(' ' + label) if label else ''}: "
                f"state={message.get('state')} exit_code={message.get('exit_code')} "
                f"duration={message.get('duration_s')}s outputs={len(message.get('outputs') or [])}"
            )
            return message, started, "".join(events)
    raise Failure(f"等待 job_finished 超时（{TIMEOUT_S}s）")


def download_outputs(
    client: RawClient, job_id: str, dest: Path
) -> Dict[str, Dict[str, Any]]:
    client.send_json({"kind": "output_list", "job_id": job_id})
    listing = client.recv_json()
    check(listing.get("kind") == "output_list_ok", f"output_list 失败: {listing!r}")
    outputs = [item for item in listing.get("outputs") or [] if isinstance(item, dict)]
    check(bool(outputs), "output_list 为空（期望至少有 outputs/model.json）")
    downloaded: Dict[str, Dict[str, Any]] = {}
    for item in outputs:
        path = str(item.get("path"))
        size = int(item.get("size"))
        expected_sha = str(item.get("sha256"))
        chunks: List[bytes] = []
        offset = 0
        while offset < size:
            client.send_json(
                {"kind": "output_chunk", "job_id": job_id, "path": path, "offset": offset}
            )
            header = client.recv_json()
            check(header.get("kind") == "output_chunk", f"output_chunk 头异常: {header!r}")
            blob = client.recv_blob()
            check(len(blob) == int(header.get("length", -1)),
                  f"{path} 分片长度与头不符")
            check(sha256_bytes(blob) == str(header.get("sha256")), f"{path} 分片 sha256 不符")
            chunks.append(blob)
            offset += len(blob)
            if header.get("eof"):
                break
        data = b"".join(chunks)
        check(len(data) == size, f"{path} 下载大小不符: {len(data)} != {size}")
        actual = sha256_bytes(data)
        check(actual == expected_sha, f"{path} 整文件 sha256 不符")
        local = dest.joinpath(*path.split("/"))
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        downloaded[path] = {"size": size, "sha256": actual, "local": str(local)}
        log(f"产物已下载并校验 {path}（{size} B，sha256={actual[:12]}…）")
    return downloaded


def query_status(client: RawClient, job_id: str) -> Dict[str, Any]:
    client.send_json({"kind": "job_status", "job_id": job_id})
    status = client.recv_json()
    check(status.get("kind") == "job_status_ok", f"job_status 异常: {status!r}")
    return status


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(server_spec: str, token: str, keep: bool = False) -> int:
    host, port = parse_host_port(server_spec, default_port=18765)
    workroot = PROJECT_ROOT / "_rgpu_data" / f"selftest-{uuid.uuid4().hex[:8]}"
    project = make_toy_project(workroot / "project")
    download_dir = workroot / "download"
    spec = build_spec(project)
    log(f"服务端 {host}:{port}，玩具项目 {project}")
    log(f"env.fingerprint={spec.env.fingerprint[:16]}…（requirements 为空 → venv --without-pip）")

    try:
        # 1) 错误 token → BAD_AUTH
        with RawClient(host, port, token) as bad:
            response = bad.hello(token="definitely-wrong-token")
            check(
                response.get("kind") == "error" and response.get("code") == "BAD_AUTH",
                f"错误 token 期望 error/BAD_AUTH，收到 {response!r}",
            )
            log("错误 token → error BAD_AUTH ✓")

        # 2) 正确握手
        client = RawClient(host, port, token).connect()
        hello = client.hello()
        check(hello.get("kind") == "hello_ok", f"握手失败: {hello!r}")
        log(
            "hello_ok: platform={} python={} gpus={} max_parallel={}".format(
                hello.get("platform"),
                hello.get("python"),
                len(hello.get("gpus") or []),
                hello.get("max_parallel_jobs"),
            )
        )

        # 3) 提交 + 上传第一个文件后「断线」
        accepted = submit(client, spec)
        job_id = str(accepted.get("job_id") or "")
        check(job_id.startswith("job-"), f"job_id 格式异常: {job_id!r}")
        missing = [str(item.get("path")) for item in accepted.get("missing") or []]
        check(len(missing) >= 2, f"玩具项目至少应有 2 个待传文件，实际 {missing}")
        log(f"job_accepted: job_id={job_id} missing={missing} chunk_size={accepted.get('chunk_size')}")
        first_path = missing[0]
        upload_file(client, spec.input_index()[first_path], project)
        log(f"已上传 {first_path}，随后模拟断线")
        client.close()

        # 4) 重连 + resume_job_id 续传
        client = RawClient(host, port, token).connect()
        hello = client.hello()
        check(hello.get("kind") == "hello_ok", f"重连握手失败: {hello!r}")
        resumed = submit(client, spec, resume_job_id=job_id)
        check(resumed.get("job_id") == job_id, f"续传应复用 job_id: {resumed!r}")
        present = [str(item) for item in resumed.get("present") or []]
        missing2 = [str(item.get("path")) for item in resumed.get("missing") or []]
        check(first_path in present, f"已完成文件应出现在 present: {present}")
        check(first_path not in missing2, "已完成文件不应再次出现在 missing")
        log(f"resume 成功: present={present} 仍需上传={missing2}")
        upload_all(client, spec, project, missing2)

        # 5) 启动 → 日志 → 完成
        finished, started, events = wait_finished(client, job_id, label="（首次构建）")
        check(finished.get("state") == "succeeded", f"作业未成功: {finished!r}")
        check(int(finished.get("exit_code", -999)) == 0, f"退出码非 0: {finished!r}")
        check("epoch 1/1 done" in events, f"未收到流式日志，events={events!r}")
        env_state = (started.get("env") or {}).get("state")
        check(env_state in ("built", "cached"), f"job_started.env.state 非法: {started!r}")
        outputs = finished.get("outputs") or []
        check(bool(outputs), "job_finished 未带产物清单")
        log(f"流式日志共 {len(events)} 字符；产物 {[o.get('path') for o in outputs]}")

        # 6) 下载产物并校验
        downloaded = download_outputs(client, job_id, download_dir)

        # 7) 状态查询
        status = query_status(client, job_id)
        check(status.get("state") == "succeeded", f"job_status 终态不正确: {status!r}")
        log(f"job_status: state={status.get('state')} outputs={len(status.get('outputs') or [])}")
        client.close()

        # 8) 同一 spec 二次提交 → 环境复用 cached
        client = RawClient(host, port, token).connect()
        hello = client.hello()
        check(hello.get("kind") == "hello_ok", f"第二次握手失败: {hello!r}")
        accepted2 = submit(client, spec)
        job2 = str(accepted2.get("job_id") or "")
        check(job2 and job2 != job_id, f"二次提交应创建新作业: {accepted2!r}")
        missing_all = [str(item.get("path")) for item in accepted2.get("missing") or []]
        upload_all(client, spec, project, missing_all)
        finished2, started2, _events2 = wait_finished(client, job2, label="（复用环境）")
        env_state2 = (started2.get("env") or {}).get("state")
        check(
            env_state2 == "cached",
            f"二次提交同 spec 应 env.state=cached，实际 {env_state2!r}",
        )
        check(finished2.get("state") == "succeeded", f"第二次作业未成功: {finished2!r}")
        downloaded2 = download_outputs(client, job2, workroot / "download2")
        check("outputs/model.json" in downloaded2, "第二次作业缺少 outputs/model.json")
        log("环境复用校验通过：job_started.env.state=cached ✓")
        client.close()

        log(
            f"PASS：提交 → 上传/断线续传 → 启动 → 流式日志 → 下载产物 → 环境复用 "
            f"（{len(downloaded)} 个产物）"
        )
        return 0
    finally:
        if keep:
            log(f"--keep 保留临时目录: {workroot}")
        else:
            shutil.rmtree(workroot, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    parser = argparse.ArgumentParser(description="RGPU 服务端原始 socket 自测")
    parser.add_argument("--server", default="127.0.0.1:18765", help="host:port")
    parser.add_argument("--token", default="selftest", help="鉴权 token")
    parser.add_argument("--keep", action="store_true", help="保留临时目录")
    args = parser.parse_args(argv)
    try:
        return run(args.server, args.token, keep=args.keep)
    except Failure as exc:
        print(f"{PREFIX} FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # pragma: no cover - 未预期异常
        print(f"{PREFIX} ERROR: {exc!r}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
