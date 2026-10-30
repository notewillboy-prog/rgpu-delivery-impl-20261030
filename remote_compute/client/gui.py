"""tkinter 一键增量训练窗口。

设计要点：

- 单窗口，所有字段与按钮在一屏内完成：服务器地址/端口、token、项目目录、
  入口命令、GPU 勾选、产物下载目录、超时；
- 「开始增量训练」一键完成：打包项目（模型 + 当前权重 + 当前数据）→ 连接 →
  上传 → 远端执行 → 流式日志 → 下载产物；
- 网络与磁盘工作全部在**子线程**，通过 ``queue.Queue`` 回主线程刷新 UI
  （tkinter 不是线程安全的）；
- 运行期间禁用开始/保存按钮，避免重复点击；
- 配置可保存（服务器、token、下载目录等），下次启动自动回填。

``import client.gui`` 不会创建任何 Tk 对象；只有调用 ``main()`` 才打开窗口。
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from common.config import (
    ClientConfig,
    load_client_config,
    parse_host_port,
    save_client_config,
)

from . import __version__
from .runner import (
    DEFAULT_DOWNLOAD_DIR,
    Callbacks,
    RunResult,
    SubmitOptions,
    submit_project,
)

POLL_INTERVAL_MS = 80
WINDOW_TITLE = f"RGPU 远程算力调度 · 一键增量训练 v{__version__}"


class RgpuApp:
    """GUI 主窗口控制器。"""

    def __init__(self, root: tk.Tk, config_path: Optional[str] = None):
        self.root = root
        self.config_path = config_path
        self.queue: "queue.Queue[tuple]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self._config_warning = ""

        try:
            self.config = load_client_config(config_path)
        except ValueError as exc:
            self.config = ClientConfig()
            self._config_warning = str(exc)

        self._build_ui()
        self._apply_config(self.config)
        if self._config_warning:
            self._append_log(f"配置读取失败，使用默认值: {self._config_warning}", "error")

        self.root.title(WINDOW_TITLE)
        self.root.geometry("820x620")
        self.root.minsize(700, 520)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(POLL_INTERVAL_MS, self._poll_queue)

    # -- UI 构建 -----------------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        form = ttk.Frame(outer)
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)

        # 服务器 / token
        ttk.Label(form, text="服务器地址").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.server_var = tk.StringVar()
        self.server_entry = ttk.Entry(form, textvariable=self.server_var)
        self.server_entry.grid(row=0, column=1, sticky=tk.EW, padx=(4, 12), pady=3)

        ttk.Label(form, text="Token").grid(row=0, column=2, sticky=tk.W, pady=3)
        self.token_var = tk.StringVar()
        self.token_entry = ttk.Entry(form, textvariable=self.token_var, show="*")
        self.token_entry.grid(row=0, column=3, sticky=tk.EW, padx=(4, 0), pady=3)

        # 项目目录
        ttk.Label(form, text="项目目录").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.project_var = tk.StringVar()
        self.project_entry = ttk.Entry(form, textvariable=self.project_var)
        self.project_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=(4, 4), pady=3)
        self.project_btn = ttk.Button(form, text="浏览…", command=self._choose_project, width=8)
        self.project_btn.grid(row=1, column=3, sticky=tk.E, pady=3)

        # 入口命令
        ttk.Label(form, text="入口命令").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.entry_var = tk.StringVar(value="python train.py --epochs 3 --data data.csv")
        self.entry_entry = ttk.Entry(form, textvariable=self.entry_var)
        self.entry_entry.grid(row=2, column=1, columnspan=3, sticky=tk.EW, padx=(4, 0), pady=3)

        # 下载目录
        ttk.Label(form, text="产物下载目录").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.download_var = tk.StringVar()
        self.download_entry = ttk.Entry(form, textvariable=self.download_var)
        self.download_entry.grid(row=3, column=1, columnspan=2, sticky=tk.EW, padx=(4, 4), pady=3)
        self.download_btn = ttk.Button(form, text="浏览…", command=self._choose_download, width=8)
        self.download_btn.grid(row=3, column=3, sticky=tk.E, pady=3)

        # 选项行
        options = ttk.Frame(outer)
        options.pack(fill=tk.X, pady=(6, 4))
        self.gpu_var = tk.BooleanVar(value=False)
        self.gpu_check = ttk.Checkbutton(
            options, text="请求 GPU（服务端分配 CUDA_VISIBLE_DEVICES）", variable=self.gpu_var
        )
        self.gpu_check.pack(side=tk.LEFT)
        ttk.Label(options, text="  超时(秒)").pack(side=tk.LEFT, padx=(16, 2))
        self.timeout_var = tk.StringVar(value="600")
        self.timeout_spin = ttk.Spinbox(options, from_=10, to=86400, increment=30, width=8,
                                        textvariable=self.timeout_var)
        self.timeout_spin.pack(side=tk.LEFT)
        ttk.Label(options, text="  作业名").pack(side=tk.LEFT, padx=(16, 2))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(options, textvariable=self.name_var, width=16)
        self.name_entry.pack(side=tk.LEFT)

        # 按钮行
        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(4, 6))
        self.start_btn = ttk.Button(buttons, text="开始增量训练", command=self._on_start)
        self.start_btn.pack(side=tk.LEFT)
        self.save_btn = ttk.Button(buttons, text="保存配置", command=self._on_save_config)
        self.save_btn.pack(side=tk.LEFT, padx=8)
        self.clear_btn = ttk.Button(buttons, text="清空日志", command=self._clear_log)
        self.clear_btn.pack(side=tk.LEFT)

        # 状态 + 进度
        status_row = ttk.Frame(outer)
        status_row.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_row, textvariable=self.status_var).pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            outer, orient=tk.HORIZONTAL, mode="determinate", maximum=100.0,
            variable=self.progress_var,
        )
        self.progress.pack(fill=tk.X, pady=(2, 6))

        # 日志区
        log_frame = ttk.Frame(outer)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log = scrolledtext.ScrolledText(
            log_frame, height=18, wrap=tk.WORD, state=tk.DISABLED,
            font=("Consolas", 9),
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_configure("info", foreground="#666666")
        self.log.tag_configure("error", foreground="#c62828")
        self.log.tag_configure("stderr", foreground="#b26a00")
        self.log.tag_configure("success", foreground="#1b5e20")

    # -- 配置回填/收集 -----------------------------------------------------
    def _apply_config(self, config: ClientConfig) -> None:
        self.server_var.set(config.address)
        self.token_var.set(config.token)
        download = config.download_dir or ""
        if not download or download == "outputs":
            download = ""
        self.download_var.set(download)

    def _sync_config(self) -> ClientConfig:
        server = self.server_var.get().strip()
        host, port = parse_host_port(server)
        config = ClientConfig(
            host=host,
            port=port,
            token=self.token_var.get().strip(),
            download_dir=self.download_var.get().strip(),
        )
        return config

    def _save_config_best_effort(self, silent: bool = True) -> Optional[Path]:
        try:
            config = self._sync_config()
            path = save_client_config(config, self.config_path)
            if not silent:
                self._append_log(f"配置已保存: {path}", "info")
            return Path(path)
        except (ValueError, OSError) as exc:
            self._append_log(f"保存配置失败: {exc}", "error")
            return None

    def _on_save_config(self) -> None:
        path = self._save_config_best_effort(silent=False)
        if path is not None:
            messagebox.showinfo("保存配置", f"配置已保存到:\n{path}", parent=self.root)

    # -- 文件选择 ----------------------------------------------------------
    def _choose_project(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title="选择项目目录")
        if not path:
            return
        self.project_var.set(path)
        if not self.download_var.get().strip():
            self.download_var.set(str(Path(path) / DEFAULT_DOWNLOAD_DIR))
        # 入口命令自动适配：若项目里有 train.py 且当前入口没写 train.py，给出建议
        try:
            if (Path(path) / "train.py").is_file():
                current = self.entry_var.get().strip()
                if "train.py" not in current:
                    self.entry_var.set("python train.py --epochs 3 --data data.csv")
        except OSError:
            pass

    def _choose_download(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title="选择产物下载目录")
        if path:
            self.download_var.set(path)

    # -- 启动任务 ----------------------------------------------------------
    def _collect_options(self) -> SubmitOptions:
        project = self.project_var.get().strip()
        if not project:
            raise ValueError("请选择项目目录")
        project_path = Path(project).expanduser()
        if not project_path.is_dir():
            raise ValueError(f"项目目录不存在: {project}")
        if not self.token_var.get().strip():
            raise ValueError("请填写 token（服务端启动时的 --token）")
        entry = self.entry_var.get().strip()
        if not entry:
            raise ValueError("入口命令不能为空")
        try:
            timeout = int(self.timeout_var.get().strip() or "600")
        except ValueError as exc:
            raise ValueError(f"超时必须是整数秒: {self.timeout_var.get()!r}") from exc
        if timeout <= 0:
            raise ValueError("超时必须是正整数")
        download = self.download_var.get().strip()
        if not download:
            download = str(project_path / DEFAULT_DOWNLOAD_DIR)
        return SubmitOptions(
            project_dir=str(project_path),
            entry=entry,
            job_name=self.name_var.get().strip(),
            env_mode="venv",
            gpu=bool(self.gpu_var.get()),
            timeout_s=timeout,
            download_dir=download,
            watch=True,
            # 一键增量训练：项目目录（模型 + 权重 + 当前数据）全量上传
            incremental=True,
        )

    def _on_start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self._save_config_best_effort(silent=True)
        try:
            config = self._sync_config()
            options = self._collect_options()
        except ValueError as exc:
            self._append_log(f"参数错误: {exc}", "error")
            self.status_var.set(f"参数错误: {exc}")
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return

        self._set_running(True)
        self.progress_var.set(0.0)
        self.status_var.set("正在打包并连接服务器…")
        self._append_log("=" * 72, "info")
        self._append_log(f"开始增量训练：{options.project_dir}", "info")
        self._append_log(f"入口命令：{options.entry}", "info")
        if options.gpu:
            self._append_log("资源请求：GPU", "info")

        self.worker = threading.Thread(
            target=self._run_worker, args=(config, options), name="rgpu-gui-worker", daemon=True
        )
        self.worker.start()

    def _run_worker(self, config: ClientConfig, options: SubmitOptions) -> None:
        """子线程：执行完整链路，通过 queue 回主线程刷新 UI。"""

        callbacks = Callbacks(
            log=lambda msg: self.queue.put(("log", msg)),
            event=lambda stream, data: self.queue.put(("event", stream, data)),
            progress=lambda frac, text: self.queue.put(("progress", frac, text)),
            job_id=lambda job_id: self.queue.put(("job_id", job_id)),
        )
        try:
            result = submit_project(config, options, callbacks)
        except Exception as exc:  # 兜底：避免线程静默死亡
            self.queue.put(("fatal", f"{type(exc).__name__}: {exc}"))
            return
        self.queue.put(("done", result))

    # -- 主线程事件处理 ----------------------------------------------------
    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(str(item[1]))
                elif kind == "event":
                    stream = str(item[1])
                    data = str(item[2])
                    tag = "stderr" if stream in ("stderr", "system") else None
                    self._append_log(data, tag)
                elif kind == "progress":
                    frac = float(item[1])
                    text = str(item[2])
                    self.progress_var.set(max(0.0, min(1.0, frac)) * 100.0)
                    self.status_var.set(text)
                elif kind == "job_id":
                    self.status_var.set(f"作业 {item[1]} 已提交")
                elif kind == "done":
                    self._on_done(item[1])
                elif kind == "fatal":
                    self._append_log(f"未预期错误: {item[1]}", "error")
                    self.status_var.set("失败（未预期错误）")
                    self._set_running(False)
        except queue.Empty:
            pass
        self.root.after(POLL_INTERVAL_MS, self._poll_queue)

    def _on_done(self, result: RunResult) -> None:
        self._set_running(False)
        if result.ok:
            self.progress_var.set(100.0)
            self.status_var.set(
                f"成功：{result.state}（{result.duration_s:.1f}s，产物 {len(result.downloaded)} 个）"
            )
            self._append_log("", None)
            self._append_log(
                f"任务成功 job_id={result.job_id} state={result.state} "
                f"用时 {result.duration_s:.1f}s",
                "success",
            )
            for record in result.downloaded:
                self._append_log(f"  产物: {record.get('local_path')}", "success")
            if result.download_dir:
                self._append_log(f"下载目录: {result.download_dir}", "info")
        else:
            error = result.error or {}
            self.status_var.set(
                f"失败：{error.get('code', '')} {error.get('message', '')}".strip()
            )
            self._append_log(
                f"任务失败 job_id={result.job_id or '-'} state={result.state} "
                f"client_exit={result.client_exit} "
                f"error={error.get('code', '?')} {error.get('message', '')}",
                "error",
            )
            for line in result.log_tail[-20:]:
                self._append_log(f"  {line}", "error")

    def _set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        for widget in (
            self.start_btn,
            self.save_btn,
            self.project_btn,
            self.download_btn,
            self.project_entry,
            self.download_entry,
            self.entry_entry,
            self.server_entry,
            self.token_entry,
            self.gpu_check,
            self.timeout_spin,
            self.name_entry,
        ):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        if running:
            self.start_btn.configure(text="训练中…")
        else:
            self.start_btn.configure(text="开始增量训练")

    # -- 日志 --------------------------------------------------------------
    def _append_log(self, text: str, tag: Optional[str] = None) -> None:
        if not text:
            text = ""
        self.log.configure(state=tk.NORMAL)
        if text and not text.endswith("\n"):
            text += "\n"
        self.log.insert(tk.END, text, tag or ())
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    # -- 关闭 --------------------------------------------------------------
    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno(
                "任务进行中",
                "任务仍在运行，关闭窗口会中止客户端跟踪（远端任务可能继续）。确定关闭？",
                parent=self.root,
            ):
                return
        self._save_config_best_effort(silent=True)
        self.root.destroy()


def main(config_path: Optional[str] = None) -> int:
    """打开 GUI 主循环，返回进程退出码。"""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"无法初始化 tkinter（无图形环境？）: {exc}", file=__import__("sys").stderr)
        return 3
    try:
        RgpuApp(root, config_path)
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
