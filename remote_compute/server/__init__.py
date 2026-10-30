"""RGPU 服务端（Linux 目标平台，本机以 CPU 模式自测）。

模块划分（与 ``docs/ARCHITECTURE.md`` 一致）：

- ``main``       CLI 入口：``serve`` / ``gpus`` / ``token``
- ``app``        线程式 TCP 会话、握手鉴权、消息分发、错误统一为 ``error`` 消息
- ``jobstore``   作业目录与 ``state.json`` 原子读写、上传进度记录
- ``runtime``    作业运行态（状态机、订阅者广播、``job_finished`` 组装）
- ``envbuilder`` 环境指纹 → 目录复用 / venv 构建 / docker 镜像构建
- ``scheduler``  nvidia-smi 探测、FIFO 队列、GPU 分配（CUDA_VISIBLE_DEVICES）
- ``executor``   subprocess 执行、流式日志、超时 kill、产物收集
- ``selftest``   不依赖 ``client/`` 的原始 socket 端到端自测

只依赖 Python 标准库；``common/`` 为只读契约层。
"""

__version__ = "0.1.0"
