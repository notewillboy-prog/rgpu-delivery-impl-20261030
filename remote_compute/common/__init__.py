"""RGPU 远程算力调度 —— 客户端/服务端共用的公共层。

模块：
- ``protocol``  帧协议（TCP，4B 大端长度 + 1B 类型）、JSON 消息、错误码
- ``models``    JobSpec / FileEntry / EnvSpec 等数据模型与校验
- ``hashing``   文件 sha256、需求归一化、环境指纹
- ``config``    默认配置与本地配置文件读写

只依赖 Python 标准库，客户端（Windows/Linux）与服务端（Linux）共用同一份代码。
"""

from .protocol import (  # noqa: F401
    PROTOCOL_VERSION,
    MSG_JSON,
    MSG_BLOB,
    MAX_FRAME_PAYLOAD,
    MAX_CHUNK_SIZE,
    DEFAULT_CHUNK_SIZE,
    ProtocolError,
    FrameReader,
    FrameWriter,
    send_json,
    recv_json,
    send_blob,
    encode_frame,
)
from .models import (  # noqa: F401
    JobState,
    FileEntry,
    EnvSpec,
    Resources,
    JobSpec,
    OutputEntry,
    GpuInfo,
    validate_job_spec,
)
from .hashing import (  # noqa: F401
    sha256_bytes,
    sha256_file,
    sha256_stream,
    normalize_requirements,
    env_fingerprint,
    human_size,
)
from .config import (  # noqa: F401
    DEFAULT_PORT,
    CLIENT_CONFIG_NAME,
    ClientConfig,
    default_client_config_path,
    load_client_config,
    save_client_config,
    generate_token,
)

__version__ = "0.1.0"
