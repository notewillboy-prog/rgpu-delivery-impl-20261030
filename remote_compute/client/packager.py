"""项目打包：目录 → ``common.models.FileEntry`` 清单，以及 JobSpec 构建。

职责边界（与 PROTOCOL.md 3.1 对齐）：

- 递归遍历项目目录，路径一律转成 **POSIX 相对路径**（Windows/Linux 通用）；
- 每个文件计算 sha256、大小、可执行位；
- 默认忽略 ``.git/``、``.venv/``、``__pycache__/``、``node_modules/``、``outputs/``、
  ``*.log`` 等本地/产物文件；
- ``include`` / ``exclude`` glob：不含 ``/`` 的模式匹配「任意深度的文件名」，
  含 ``/`` 的模式匹配整条相对路径（目录前缀也视为命中，便于排除整棵子树）；
- 自动识别 ``<project>/requirements.txt`` 作为依赖清单（归一化由
  ``common.hashing.normalize_requirements`` 完成，保证不同注释/顺序指纹一致）；
- 增量训练语义：项目里同时存在基座模型与数据集时，自动填
  ``meta={"mode":"incremental_training","base_model":…,"dataset":…}``。

本模块不联网、不依赖 server/，可单独做函数级测试。
"""

from __future__ import annotations

import fnmatch
import os
import shlex
import stat
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from common.hashing import normalize_requirements, sha256_file
from common.models import (
    EnvSpec,
    FileEntry,
    JobSpec,
    Resources,
    validate_job_spec,
)

CLIENT_VERSION = "0.1.0"

#: 任意深度都会忽略的目录名（本地环境 / 缓存 / VCS 元数据）。
DEFAULT_IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        ".rgpu",
        # 远端产物/日志目录：不该作为下一次任务的输入再传上去
        "outputs",
        "rgpu-outputs",
    }
)

#: 任意深度都会忽略的文件名模式。
DEFAULT_IGNORED_FILE_PATTERNS: Tuple[str, ...] = (
    "*.log",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
)

REQUIREMENTS_FILENAME = "requirements.txt"

#: 增量训练：常见的基座模型 / 权重文件名（按优先级）。
DEFAULT_MODEL_CANDIDATES: Tuple[str, ...] = (
    "model.json",
    "base_model.json",
    "model.pt",
    "model.pth",
    "weights.pt",
    "weights.pth",
    "checkpoint.pt",
    "checkpoint.pth",
)

#: 增量训练：常见的数据集文件名（按优先级）。
DEFAULT_DATASET_CANDIDATES: Tuple[str, ...] = (
    "data.csv",
    "train.csv",
    "dataset.csv",
    "data.jsonl",
    "train.jsonl",
)

DEFAULT_OUTPUT_PATTERNS: Tuple[str, ...] = ("outputs/**",)


class PackagerError(ValueError):
    """打包/参数错误；CLI 映射为退出码 2。"""


WarningCallback = Callable[[str], None]


# ---------------------------------------------------------------------------
# 路径与 glob 匹配
# ---------------------------------------------------------------------------
def _to_posix(relpath: str) -> str:
    return str(relpath).strip().replace("\\", "/").strip("/")


def normalize_patterns(patterns: Optional[Sequence[str]]) -> List[str]:
    """清洗 glob 列表：去首尾空白、统一分隔符、去掉首尾 ``/``、去重保序。"""
    out: List[str] = []
    seen = set()
    for raw in patterns or []:
        pat = _to_posix(str(raw or ""))
        if not pat or pat in seen:
            continue
        seen.add(pat)
        out.append(pat)
    return out


def path_matches(relpath: str, pattern: str, is_dir: bool = False) -> bool:
    """glob 匹配。

    - 模式不含 ``/``：只匹配文件名（任意深度），例如 ``*.py``、``*.log``；
    - 模式含 ``/``：匹配整条相对路径，例如 ``src/*.py``、``docs/**``；
    - 目录前缀语义：``exclude="tmp"`` 或 ``"tmp/data"`` 会排除整棵子树。
    """
    rel = _to_posix(relpath)
    pat = _to_posix(pattern)
    if not rel or not pat:
        return False
    if "/" in pat:
        if fnmatch.fnmatchcase(rel, pat):
            return True
        if is_dir and fnmatch.fnmatchcase(rel + "/", pat + "/"):
            return True
        # 目录前缀：rel 位于该目录下（或被同名目录整体命中）
        if rel == pat or rel.startswith(pat + "/"):
            return True
        return False
    name = rel.rsplit("/", 1)[-1]
    return fnmatch.fnmatchcase(name, pat)


def is_default_ignored(relpath: str, is_dir: bool) -> bool:
    """默认忽略规则（POSIX 相对路径）。"""
    rel = _to_posix(relpath)
    if not rel:
        return False
    parts = rel.split("/")
    if is_dir:
        return parts[-1] in DEFAULT_IGNORED_DIR_NAMES
    name = parts[-1]
    return any(fnmatch.fnmatchcase(name, pat) for pat in DEFAULT_IGNORED_FILE_PATTERNS)


# ---------------------------------------------------------------------------
# 文件清单
# ---------------------------------------------------------------------------
def collect_files(
    project_dir: str | os.PathLike,
    *,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    ignore_defaults: bool = True,
    on_warning: Optional[WarningCallback] = None,
) -> List[FileEntry]:
    """递归收集项目文件，返回按路径排序的 ``FileEntry`` 列表。

    - ``include`` 为空表示「默认忽略规则之外的全部文件」；
    - ``exclude`` 优先于 ``include``；
    - 符号链接一律跳过（避免把项目目录之外的内容打包）；
    - 读取失败的文件给出 warning 后跳过（例如打包过程中被删除）。

    纯函数、可单独测试，不触碰网络。
    """
    warn = on_warning or (lambda _msg: None)
    root = Path(project_dir).expanduser()
    if not root.is_dir():
        raise PackagerError(f"项目目录不存在或不是目录: {root}")
    root = root.resolve()

    include_p = normalize_patterns(include)
    exclude_p = normalize_patterns(exclude)

    entries: List[FileEntry] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir

        kept_dirs: List[str] = []
        for name in sorted(dirnames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                warn(f"跳过符号链接目录: {rel}")
                continue
            if ignore_defaults and is_default_ignored(rel, True):
                continue
            if any(path_matches(rel, pat, True) for pat in exclude_p):
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs

        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                warn(f"跳过符号链接文件: {rel}")
                continue
            if ignore_defaults and is_default_ignored(rel, False):
                continue
            if any(path_matches(rel, pat, False) for pat in exclude_p):
                continue
            if include_p and not any(path_matches(rel, pat, False) for pat in include_p):
                continue
            try:
                st = os.stat(full)
            except OSError as exc:
                warn(f"跳过无法读取的文件 {rel}: {exc}")
                continue
            if not stat.S_ISREG(st.st_mode):
                warn(f"跳过非普通文件: {rel}")
                continue
            try:
                digest = sha256_file(full)
            except OSError as exc:
                warn(f"跳过无法读取的文件 {rel}: {exc}")
                continue
            entries.append(
                FileEntry(
                    path=rel,
                    size=int(st.st_size),
                    sha256=digest,
                    executable=bool(st.st_mode & stat.S_IXUSR),
                )
            )

    entries.sort(key=lambda item: item.path)
    return entries


# ---------------------------------------------------------------------------
# requirements 与 JobSpec
# ---------------------------------------------------------------------------
def read_requirements(
    project_dir: str | os.PathLike,
    requirements_path: Optional[str] = None,
) -> Tuple[List[str], Optional[str]]:
    """读取并归一化依赖清单。

    返回 ``(requirements, path)``；``path`` 为实际使用的文件路径（未找到则为 None）。
    显式传入的 ``requirements_path`` 不存在时抛 ``PackagerError``（配置错误）。
    相对路径按「项目目录优先、当前目录兜底」解析。
    """
    root = Path(project_dir).expanduser()
    path: Optional[Path]
    if requirements_path:
        raw = Path(requirements_path).expanduser()
        if raw.is_absolute():
            path = raw
        else:
            candidate = root / raw
            path = candidate if candidate.is_file() else raw
        if not path.is_file():
            raise PackagerError(f"requirements 文件不存在: {requirements_path}")
    else:
        candidate = root / REQUIREMENTS_FILENAME
        path = candidate if candidate.is_file() else None
    if path is None:
        return [], None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PackagerError(f"无法读取 requirements 文件 {path}: {exc}") from exc
    return normalize_requirements(text), str(path)


def find_first_existing(project_dir: str | os.PathLike, candidates: Iterable[str]) -> Optional[str]:
    root = Path(project_dir).expanduser()
    for name in candidates:
        if (root / name).is_file():
            return name
    return None


def infer_incremental_meta(project_dir: str | os.PathLike) -> Optional[Dict[str, str]]:
    """自动识别增量训练语义；模型与数据都存在时才返回 meta。"""
    model = find_first_existing(project_dir, DEFAULT_MODEL_CANDIDATES)
    dataset = find_first_existing(project_dir, DEFAULT_DATASET_CANDIDATES)
    if model and dataset:
        return {"mode": "incremental_training", "base_model": model, "dataset": dataset}
    return None


def incremental_meta(project_dir: str | os.PathLike) -> Dict[str, str]:
    """GUI 一键增量训练固定使用的 meta（缺省文件名用协议约定的默认值）。"""
    inferred = infer_incremental_meta(project_dir) or {}
    return {
        "mode": "incremental_training",
        "base_model": inferred.get("base_model", "model.json"),
        "dataset": inferred.get("dataset", "data.csv"),
    }


def parse_entry_command(entry: str | Sequence[str]) -> List[str]:
    """把 ``python train.py --epochs 3`` 解析成 argv 列表；不加 shell。"""
    if isinstance(entry, (list, tuple)):
        argv = [str(x) for x in entry if str(x) != ""]
    else:
        text = str(entry or "").strip()
        if not text:
            raise PackagerError("入口命令不能为空，例如: python train.py --epochs 3 --data data.csv")
        try:
            argv = shlex.split(text, posix=True)
        except ValueError as exc:
            raise PackagerError(f"入口命令解析失败: {exc}") from exc
    if not argv:
        raise PackagerError("入口命令不能为空")
    return argv


def build_env_spec(
    project_dir: str | os.PathLike,
    *,
    requirements_path: Optional[str] = None,
    env_mode: str = "venv",
    python_version: Optional[str] = None,
    docker_image: str = "",
) -> EnvSpec:
    requirements, _path = read_requirements(project_dir, requirements_path)
    pyver = (python_version or f"{sys.version_info.major}.{sys.version_info.minor}").strip()
    env = EnvSpec(
        mode=env_mode,
        python_version=pyver,
        requirements=requirements,
        docker_image=docker_image,
        reuse=True,
    )
    env.fingerprint = env.compute_fingerprint()
    return env


def make_job_spec(
    project_dir: str | os.PathLike,
    entry_argv: Sequence[str],
    *,
    job_name: str = "",
    requirements_path: Optional[str] = None,
    env_mode: str = "venv",
    python_version: Optional[str] = None,
    docker_image: str = "",
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    outputs: Optional[Sequence[str]] = None,
    gpu: bool = False,
    gpu_count: int = 1,
    timeout_s: int = 600,
    cpus: int = 0,
    memory_gb: float = 0.0,
    meta: Optional[Dict[str, object]] = None,
    client_version: str = CLIENT_VERSION,
    on_warning: Optional[WarningCallback] = None,
) -> JobSpec:
    """构建并本地校验 ``JobSpec``（不连接服务器）。"""
    root = Path(project_dir).expanduser()
    if not root.is_dir():
        raise PackagerError(f"项目目录不存在或不是目录: {root}")

    argv = [str(x) for x in entry_argv]
    if not argv:
        raise PackagerError("entry_argv 不能为空")

    files = collect_files(
        root, include=include, exclude=exclude, on_warning=on_warning
    )
    if not files:
        raise PackagerError("项目目录里没有可上传的文件（是否被 include/exclude 全部过滤了？）")

    env = build_env_spec(
        root,
        requirements_path=requirements_path,
        env_mode=env_mode,
        python_version=python_version,
        docker_image=docker_image,
    )

    name = (job_name or "").strip()
    if not name:
        name = root.name or "rgpu-job"

    output_patterns = [str(x) for x in (outputs or DEFAULT_OUTPUT_PATTERNS)]
    final_meta: Dict[str, object]
    if meta is None:
        final_meta = dict(infer_incremental_meta(root) or {})
    else:
        final_meta = dict(meta)

    spec = JobSpec(
        job_name=name,
        entry_argv=argv,
        inputs=files,
        env=env,
        resources=Resources(
            gpu=bool(gpu),
            gpu_count=int(gpu_count),
            timeout_s=int(timeout_s),
            cpus=int(cpus),
            memory_gb=float(memory_gb),
        ),
        outputs=output_patterns,
        workdir=".",
        meta=final_meta,
        client_version=client_version,
    )
    validate_job_spec(spec)
    return spec


def summarize(entries: Sequence[FileEntry]) -> Dict[str, object]:
    """清单摘要，供 CLI/GUI 打印。"""
    total = sum(entry.size for entry in entries)
    return {"count": len(entries), "bytes": total, "files": [e.path for e in entries]}
