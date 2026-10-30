"""用协议参考实现 ``tests/fake_server.py`` 自测原始协议验收场景。

为什么需要：真 ``server/`` 由另一会话并行开发。在它落地前，先用这份按
``docs/PROTOCOL.md`` v1 写的最小实现，验证 ``tests/rawflow.py`` +
``tests/raw_scenarios.py`` 的断线续传 / 哈希篡改断言逻辑本身正确——
这样真服务端一落地，e2e 失败就能直接归因到服务端，而不是测试代码。

同一套场景会在 ``tests/test_e2e_local.py::E2ERawProtocolTest`` 对真服务端再跑一遍。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers  # noqa: E402
from fake_server import FakeServer  # noqa: E402
from raw_scenarios import RawFlowScenarios  # noqa: E402

FAKE_TOKEN = "e2e-token"


@unittest.skipUnless(helpers.EXAMPLE_AVAILABLE and helpers.CLIENT_AVAILABLE,
                     "需要 examples/ 与 client/ 才能构造真实 JobSpec")
class FakeServerRawFlowTest(RawFlowScenarios, unittest.TestCase):
    """对假服务端执行与真 e2e 完全相同的原始协议场景。"""

    token = FAKE_TOKEN

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = helpers.make_temp_dir("fake-raw-")
        cls.addClassCleanup(helpers.cleanup_dir, cls.tmp)
        cls._server = FakeServer(
            port=helpers.find_free_port(),
            token=FAKE_TOKEN,
            data_dir=cls.tmp / "server",
        ).start()
        cls.addClassCleanup(cls._server.stop)

    def server_address(self) -> str:
        return self._server.address

    def test_00_bad_auth(self) -> None:
        """错误 token 必须收到 error/BAD_AUTH（对应 e2e 的退出码 4 场景）。"""
        client = helpers.RawClient(self._server.address, token="wrong-token")
        client.connect()
        try:
            hello = client.hello()
            self.assertEqual(hello.get("kind"), "error", hello)
            self.assertEqual(hello.get("code"), "BAD_AUTH", hello)
        finally:
            client.close()

    def test_03_ping_pong(self) -> None:
        client, _hello = self._connect()
        try:
            client.send_json({"kind": "ping", "ts": 1})
            msg = client.recv_json()
            self.assertEqual(msg.get("kind"), "pong", msg)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
