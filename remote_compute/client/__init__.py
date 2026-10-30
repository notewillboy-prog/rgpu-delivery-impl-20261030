"""RGPU 客户端包。

模块划分：
- ``packager``   项目目录 → 文件清单（sha256）+ requirements 识别 + JobSpec 构建
- ``connection`` TCP 帧连接：握手鉴权、心跳、超时、断线重连
- ``uploader``   分片上传（逐片 sha256、file_ok offset 推进、有限重传）
- ``runner``     提交 → 上传 → 启动 → 流式日志 → 产物下载 → 退出码映射
- ``gui``        tkinter 一键增量训练窗口（网络工作在子线程）
- ``main``       命令行入口（submit / status / gui / config）
- ``selftest``   本地假服务端自测（不依赖 server/）

与 ``common/`` 相同的协议版本：``common.protocol.PROTOCOL_VERSION``。
"""

from __future__ import annotations

__version__ = "0.1.0"

#: 与服务端 hello_ok 中 ``protocol`` 对比；不一致时客户端拒绝继续。
PROTOCOL_VERSION = 1

__all__ = ["__version__", "PROTOCOL_VERSION"]
