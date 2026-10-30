"""协议层单元测试：帧编解码、坏帧、超大帧、非法 JSON、多帧连续、分片边界。

契约来源：``docs/PROTOCOL.md`` v1 + ``common/protocol.py``。
不依赖 server/ 与 client/，阶段 A 可独立跑。
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers  # noqa: E402

from common.protocol import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    ERROR_CODES,
    ERR_BAD_AUTH,
    ERR_BAD_PROTOCOL,
    ERR_BAD_REQUEST,
    ERR_BUSY,
    ERR_CONFLICT,
    ERR_ENV_BUILD,
    ERR_HASH_MISMATCH,
    ERR_INTERNAL,
    ERR_JOB_FAILED,
    ERR_NOT_FOUND,
    ERR_TIMEOUT,
    KIND_ERROR,
    MAX_CHUNK_SIZE,
    MAX_FRAME_PAYLOAD,
    MSG_BLOB,
    MSG_JSON,
    PROTOCOL_VERSION,
    ProtocolError,
    FrameReader,
    FrameWriter,
    check_version,
    choose_chunk_size,
    decode_json_payload,
    encode_frame,
    is_error,
    iter_chunks,
    make_error,
    require_fields,
)


class SocketPairTestCase(unittest.TestCase):
    """提供 socketpair 两端：A 端写、B 端读（或反过来）。"""

    def setUp(self) -> None:
        self.sock_a, self.sock_b = socket.socketpair()
        self.sock_a.settimeout(5.0)
        self.sock_b.settimeout(5.0)

    def tearDown(self) -> None:
        for sock in (self.sock_a, self.sock_b):
            try:
                sock.close()
            except OSError:
                pass

    def writer_on(self, sock: socket.socket) -> FrameWriter:
        return FrameWriter(sock)

    def reader_on(self, sock: socket.socket) -> FrameReader:
        return FrameReader(sock)

    def raw_send(self, sock: socket.socket, data: bytes) -> None:
        sock.sendall(data)


class FrameCodecTest(SocketPairTestCase):
    def test_json_roundtrip_including_unicode(self) -> None:
        payload = {"kind": "ping", "seq": 7, "note": "中文 UTF-8", "nested": {"a": [1, 2]}}
        self.writer_on(self.sock_a).send_json(payload)
        self.assertEqual(self.reader_on(self.sock_b).read_json(), payload)

    def test_blob_roundtrip_with_binary_bytes(self) -> None:
        data = bytes(range(256)) * 4 + b"\x00\xff\r\n"
        self.writer_on(self.sock_a).send_blob(data)
        self.assertEqual(self.reader_on(self.sock_b).read_blob(), data)

    def test_empty_blob_and_empty_json_object(self) -> None:
        writer = self.writer_on(self.sock_a)
        reader = self.reader_on(self.sock_b)
        writer.send_blob(b"")
        self.assertEqual(reader.read_blob(), b"")
        writer.send_json({})
        self.assertEqual(reader.read_json(), {})

    def test_wire_format_header_is_big_endian(self) -> None:
        # 协议：4 字节大端 payload 长度 + 1 字节 msg_type + payload
        payload = b'{"kind":"x"}'
        frame = encode_frame(MSG_JSON, payload)
        self.assertEqual(frame[:4], struct.pack("!I", len(payload)))
        self.assertEqual(frame[4:5], bytes([MSG_JSON]))
        self.assertEqual(frame[5:], payload)
        # 明确验证大端：长度 300 -> 00 00 01 2c
        frame300 = encode_frame(MSG_BLOB, b"a" * 300)
        self.assertEqual(frame300[:4], b"\x00\x00\x01\x2c")
        self.assertEqual(len(frame300), 5 + 300)

    def test_encoded_frame_can_be_decoded_by_reader(self) -> None:
        frame = encode_frame(MSG_JSON, b'{"kind":"hi"}')
        self.raw_send(self.sock_a, frame)
        self.assertEqual(self.reader_on(self.sock_b).read_json(), {"kind": "hi"})

    def test_encode_frame_rejects_unknown_type(self) -> None:
        for bad_type in (0, 3, 255):
            with self.assertRaises(ValueError):
                encode_frame(bad_type, b"x")

    def test_encode_frame_rejects_payload_over_limit(self) -> None:
        with self.assertRaises(ValueError):
            encode_frame(MSG_BLOB, b"\x00" * (MAX_FRAME_PAYLOAD + 1))

    def test_encode_frame_allows_payload_at_limit(self) -> None:
        # 上限本身是合法的（冻结契约：单帧 payload <= 16 MiB）
        data = b"\x00" * MAX_FRAME_PAYLOAD
        frame = encode_frame(MSG_BLOB, data)
        self.assertEqual(len(frame), 5 + MAX_FRAME_PAYLOAD)

    def test_writer_rejects_payload_over_limit(self) -> None:
        writer = self.writer_on(self.sock_a)
        with self.assertRaises(ProtocolError) as ctx:
            writer.send_blob(b"\x00" * (MAX_FRAME_PAYLOAD + 1))
        self.assertEqual(ctx.exception.code, ERR_BAD_REQUEST)

    def test_writer_send_after_socket_closed_raises_connection_error(self) -> None:
        self.sock_a.close()
        writer = FrameWriter(self.sock_a)
        with self.assertRaises(ConnectionError):
            writer.send_json({"kind": "ping"})


class FrameReaderFailuresTest(SocketPairTestCase):
    def test_unknown_frame_type_rejected(self) -> None:
        self.raw_send(self.sock_a, struct.pack("!IB", 2, 9) + b"{}")
        with self.assertRaises(ProtocolError) as ctx:
            self.reader_on(self.sock_b).read_frame()
        self.assertEqual(ctx.exception.code, ERR_BAD_PROTOCOL)
        self.assertIn("unknown frame type", str(ctx.exception))

    def test_oversized_frame_header_rejected_without_hanging(self) -> None:
        # 只发帧头，声明 length 超上限；reader 必须在读 payload 前就拒绝（不能阻塞等 16MiB+）
        self.raw_send(self.sock_a, struct.pack("!IB", MAX_FRAME_PAYLOAD + 1, MSG_JSON))
        reader = self.reader_on(self.sock_b)
        reader._sock.settimeout(1.0)
        with self.assertRaises(ProtocolError) as ctx:
            reader.read_frame()
        self.assertEqual(ctx.exception.code, ERR_BAD_PROTOCOL)
        self.assertIn("too large", str(ctx.exception))

    def test_invalid_json_rejected(self) -> None:
        self.raw_send(self.sock_a, encode_frame(MSG_JSON, b"this is not json"))
        with self.assertRaises(ProtocolError) as ctx:
            self.reader_on(self.sock_b).read_json()
        self.assertEqual(ctx.exception.code, ERR_BAD_PROTOCOL)
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_json_payload_must_be_object(self) -> None:
        for payload in (b"[]", b'"text"', b"42", b"null"):
            self.raw_send(self.sock_a, encode_frame(MSG_JSON, payload))
            with self.assertRaises(ProtocolError) as ctx:
                self.reader_on(self.sock_b).read_json()
            self.assertEqual(ctx.exception.code, ERR_BAD_PROTOCOL)
            self.assertIn("must be an object", str(ctx.exception))

    def test_invalid_utf8_json_rejected(self) -> None:
        self.raw_send(self.sock_a, encode_frame(MSG_JSON, b"\xff\xfe\x00"))
        with self.assertRaises(ProtocolError) as ctx:
            self.reader_on(self.sock_b).read_json()
        self.assertEqual(ctx.exception.code, ERR_BAD_PROTOCOL)

    def test_read_json_rejects_blob_frame(self) -> None:
        self.raw_send(self.sock_a, encode_frame(MSG_BLOB, b"binary"))
        with self.assertRaises(ProtocolError) as ctx:
            self.reader_on(self.sock_b).read_json()
        self.assertIn("expected JSON", str(ctx.exception))

    def test_read_blob_rejects_json_frame(self) -> None:
        self.raw_send(self.sock_a, encode_frame(MSG_JSON, b'{"kind":"x"}'))
        with self.assertRaises(ProtocolError) as ctx:
            self.reader_on(self.sock_b).read_blob()
        self.assertIn("expected blob", str(ctx.exception))

    def test_connection_closed_raises_connection_error(self) -> None:
        self.sock_a.close()
        with self.assertRaises(ConnectionError):
            self.reader_on(self.sock_b).read_frame()

    def test_socket_timeout_maps_to_protocol_error_timeout(self) -> None:
        self.sock_b.settimeout(0.1)
        reader = self.reader_on(self.sock_b)
        with self.assertRaises(ProtocolError) as ctx:
            reader.read_frame()
        self.assertEqual(ctx.exception.code, ERR_TIMEOUT)

    def test_decode_json_payload_reports_protocol_error(self) -> None:
        with self.assertRaises(ProtocolError) as ctx:
            decode_json_payload(b"{")
        self.assertEqual(ctx.exception.code, ERR_BAD_PROTOCOL)


class MultipleFramesTest(SocketPairTestCase):
    def test_json_blob_sequence_read_in_order(self) -> None:
        """模拟 protocol 3.3：file_chunk JSON 后紧跟 BLOB，再 file_ok JSON。"""
        writer = self.writer_on(self.sock_a)
        reader = self.reader_on(self.sock_b)
        chunk = b"0123456789" * 100
        writer.send_json({"kind": "file_chunk", "path": "data.csv", "offset": 0,
                          "length": len(chunk), "sha256": "deadbeef"})
        writer.send_blob(chunk)
        writer.send_json({"kind": "file_ok", "path": "data.csv", "offset": len(chunk)})

        head = reader.read_json()
        self.assertEqual(head["kind"], "file_chunk")
        self.assertEqual(reader.read_blob(), chunk)
        ack = reader.read_json()
        self.assertEqual(ack["kind"], "file_ok")
        self.assertEqual(ack["offset"], len(chunk))

    def test_many_small_frames_batched_together(self) -> None:
        # 一次性把所有帧写进 socket，验证 reader 的缓冲不会丢帧/错位
        payloads = [{"kind": "job_event", "seq": i, "data": f"line {i}\n"} for i in range(200)]
        raw = b"".join(encode_frame(MSG_JSON, json.dumps(p, ensure_ascii=False).encode("utf-8"))
                       for p in payloads)
        self.raw_send(self.sock_a, raw)
        reader = self.reader_on(self.sock_b)
        self.assertEqual([reader.read_json() for _ in payloads], payloads)

    def test_frame_split_across_tcp_segments(self) -> None:
        # 分成 3 段发送：帧头跨段、payload 跨段，reader 必须拼回完整帧
        frame = encode_frame(MSG_BLOB, b"x" * 5000)
        for piece in (frame[:2], frame[2:2000], frame[2000:]):
            self.raw_send(self.sock_a, piece)
        self.assertEqual(self.reader_on(self.sock_b).read_blob(), b"x" * 5000)

    def test_large_blob_then_json_frame(self) -> None:
        big = bytes(range(256)) * (DEFAULT_CHUNK_SIZE // 256)  # 1 MiB
        writer = self.writer_on(self.sock_a)
        writer.send_blob(big)
        writer.send_json({"kind": "done"})
        reader = self.reader_on(self.sock_b)
        self.assertEqual(reader.read_blob(), big)
        self.assertEqual(reader.read_json(), {"kind": "done"})


class ConstantsAndHelpersTest(unittest.TestCase):
    def test_protocol_constants_frozen(self) -> None:
        self.assertEqual(PROTOCOL_VERSION, 1)
        self.assertEqual(MSG_JSON, 1)
        self.assertEqual(MSG_BLOB, 2)
        self.assertEqual(MAX_FRAME_PAYLOAD, 16 * 1024 * 1024)
        self.assertEqual(DEFAULT_CHUNK_SIZE, 1024 * 1024)
        self.assertEqual(MAX_CHUNK_SIZE, 4 * 1024 * 1024)
        self.assertGreater(MAX_FRAME_PAYLOAD, MAX_CHUNK_SIZE)

    def test_error_codes_cover_protocol_table(self) -> None:
        # docs/PROTOCOL.md 第 4 节列出的错误码必须全部存在且有中文说明
        expected = {
            "BAD_PROTOCOL": ERR_BAD_PROTOCOL,
            "BAD_REQUEST": ERR_BAD_REQUEST,
            "BAD_AUTH": ERR_BAD_AUTH,
            "NOT_FOUND": ERR_NOT_FOUND,
            "CONFLICT": ERR_CONFLICT,
            "HASH_MISMATCH": ERR_HASH_MISMATCH,
            "ENV_BUILD_FAILED": ERR_ENV_BUILD,
            "JOB_FAILED": ERR_JOB_FAILED,
            "TIMEOUT": ERR_TIMEOUT,
            "BUSY": ERR_BUSY,
            "INTERNAL": ERR_INTERNAL,
        }
        self.assertEqual(set(ERROR_CODES), set(expected.values()))
        for code in expected.values():
            self.assertTrue(ERROR_CODES[code].strip(), f"{code} 缺少说明")

    def test_make_error_and_is_error(self) -> None:
        err = make_error(ERR_BAD_AUTH, "token 无效")
        self.assertEqual(err, {"kind": KIND_ERROR, "code": ERR_BAD_AUTH, "message": "token 无效"})
        self.assertTrue(is_error(err))
        self.assertFalse(is_error({"kind": "pong"}))
        err2 = make_error(ERR_HASH_MISMATCH, "bad", path="a.csv", offset=10)
        self.assertEqual(err2["detail"], {"path": "a.csv", "offset": 10})

    def test_require_fields(self) -> None:
        require_fields({"kind": "x", "a": 1, "b": None}, "kind", "a")  # 不抛
        with self.assertRaises(ProtocolError) as ctx:
            require_fields({"kind": "x"}, "job_id")
        self.assertEqual(ctx.exception.code, ERR_BAD_REQUEST)
        self.assertIn("job_id", str(ctx.exception))
        with self.assertRaises(ProtocolError) as ctx:
            require_fields({}, "a", "b")
        self.assertEqual(ctx.exception.code, ERR_BAD_REQUEST)
        self.assertIn("a", str(ctx.exception))
        self.assertIn("b", str(ctx.exception))

    def test_check_version(self) -> None:
        check_version({"protocol": PROTOCOL_VERSION})  # 不抛
        for bad in (0, 2, "1", None):
            with self.assertRaises(ProtocolError) as ctx:
                check_version({"protocol": bad})
            self.assertEqual(ctx.exception.code, ERR_BAD_PROTOCOL)
        with self.assertRaises(ProtocolError):
            check_version({})


class ChunkingTest(unittest.TestCase):
    def test_iter_chunks_boundaries(self) -> None:
        cases = [
            (0, 4, []),
            (1, 4, [(0, 1)]),
            (4, 4, [(0, 4)]),
            (5, 4, [(0, 4), (4, 1)]),
            (8, 4, [(0, 4), (4, 4)]),
            (10, 4, [(0, 4), (4, 4), (8, 2)]),
        ]
        for total, chunk, expected in cases:
            with self.subTest(total=total, chunk=chunk):
                self.assertEqual(list(iter_chunks(total, chunk)), expected)

    def test_iter_chunks_default_chunk_size(self) -> None:
        total = 2 * DEFAULT_CHUNK_SIZE + 1
        chunks = list(iter_chunks(total))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], (0, DEFAULT_CHUNK_SIZE))
        self.assertEqual(chunks[1], (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE))
        self.assertEqual(chunks[2], (2 * DEFAULT_CHUNK_SIZE, 1))
        self.assertEqual(sum(length for _, length in chunks), total)

    def test_iter_chunks_rejects_bad_chunk_size(self) -> None:
        for bad in (0, -1, MAX_CHUNK_SIZE + 1):
            with self.subTest(chunk=bad):
                with self.assertRaises(ValueError):
                    list(iter_chunks(10, bad))

    def test_choose_chunk_size(self) -> None:
        self.assertEqual(choose_chunk_size(0), 1)
        self.assertEqual(choose_chunk_size(1), 1)
        self.assertEqual(choose_chunk_size(100), 100)
        self.assertEqual(choose_chunk_size(DEFAULT_CHUNK_SIZE), DEFAULT_CHUNK_SIZE)
        self.assertEqual(choose_chunk_size(DEFAULT_CHUNK_SIZE + 1), DEFAULT_CHUNK_SIZE)
        self.assertEqual(choose_chunk_size(10, preferred=4), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
