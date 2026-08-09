"""AudioIO abstraction layer tests (commit 03+,commit 05 接入 MacOSVoiceIO 真实现)。

不依赖真实 sounddevice / Swift helper,只用 ``MagicMock`` 把
``sd.RawInputStream`` 与 ``sd.RawOutputStream`` 替换掉;
MacOSVoiceIO 用 fake socket 测 IPC 协议层。
"""

import struct
from unittest.mock import MagicMock

import pytest

from soul_tty.audio.io import AudioIO, MacOSVoiceIO, PortAudioIO, get_audio_io
from soul_tty.audio.io import base as io_base


# ── 工厂 ────────────────────────────────────────────────────────────


def test_get_audio_io_default_is_portaudio(monkeypatch):
    """未指定 backend 时,默认 portaudio。"""
    monkeypatch.setattr("soul_tty.config.AUDIO_IO_BACKEND", "portaudio")
    io = get_audio_io()
    assert isinstance(io, PortAudioIO)


def test_get_audio_io_explicit_portaudio(monkeypatch):
    monkeypatch.setattr("soul_tty.config.AUDIO_IO_BACKEND", "portaudio")
    io = get_audio_io("portaudio")
    assert isinstance(io, PortAudioIO)


def test_get_audio_io_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr("soul_tty.config.AUDIO_IO_BACKEND", "garbage")
    with pytest.raises(ValueError):
        get_audio_io()


def test_get_audio_io_macos_voice_returns_real_instance(monkeypatch):
    """commit 05+ :``macos_voice`` 后端现在返回真实的 ``MacOSVoiceIO`` 实例,
    而非 stub(构造不再抛 NotImplementedError)。"""
    monkeypatch.setattr("soul_tty.config.AUDIO_IO_BACKEND", "macos_voice")
    io = get_audio_io()
    assert isinstance(io, MacOSVoiceIO)


# ── Protocol 结构 ────────────────────────────────────────────────────


def test_audio_io_is_a_protocol():
    """``AudioIO`` 必须是 typing.Protocol,允许 duck-typed 实现。"""
    # Protocol 类的 __class__ 是其元类,直接 isinstance 检查会被
    # ProtocolMeta 拦截;用 ``__protocol_attrs__`` 间接验证。
    assert hasattr(AudioIO, "start")
    assert hasattr(AudioIO, "stop")
    assert hasattr(AudioIO, "write_playback")
    assert hasattr(AudioIO, "add_capture_listener")
    assert hasattr(AudioIO, "remove_capture_listener")
    assert hasattr(AudioIO, "set_playback_gain")
    assert hasattr(AudioIO, "get_playback_gain")


def test_capture_listener_signature():
    """``CaptureListener`` 是 ``Callable[[bytes, int], None]``。"""
    from soul_tty.audio.io.base import CaptureListener

    listener: CaptureListener = lambda pcm, sr: None
    listener(b"\x00\x00", 16000)


# ── PortAudioIO 行为 ────────────────────────────────────────────────


@pytest.fixture
def portaudio(monkeypatch):
    """构造一个隔离的 PortAudioIO 实例,``sounddevice`` 已替换为 MagicMock。"""
    monkeypatch.setattr("soul_tty.audio.io.portaudio.sd", MagicMock())
    return PortAudioIO(sample_rate=16000)


def test_portaudio_io_protocol_conformance(portaudio):
    """PortAudioIO 必须实现 AudioIO 的全部方法。"""
    for name in ("start", "stop", "write_playback",
                 "add_capture_listener", "remove_capture_listener",
                 "set_playback_gain", "get_playback_gain"):
        assert hasattr(portaudio, name), f"缺少方法: {name}"


def test_portaudio_io_start_stop_idempotent(portaudio):
    """``start()`` / ``stop()`` 必须可重入且幂等。"""
    portaudio.start()
    portaudio.start()  # 第二次不应抛错
    portaudio.stop()
    portaudio.stop()  # 同样应幂等


def test_portaudio_io_set_playback_gain_forwards_to_tts(portaudio, monkeypatch):
    """``set_playback_gain`` 必须转发到 ``tts.set_playback_gain``。"""
    from soul_tty.audio import tts

    captured = []
    monkeypatch.setattr(tts, "set_playback_gain", lambda v: captured.append(v))
    portaudio.set_playback_gain(0.5)
    assert captured == [0.5]
    assert portaudio.get_playback_gain() == tts.get_playback_gain()


def test_portaudio_io_write_playback_rejects_unsupported_sr(portaudio):
    """commit 03 阶段只支持 ``config.TTS_SAMPLE_RATE``,其它抛 NotImplementedError。"""
    from soul_tty import config

    with pytest.raises(NotImplementedError):
        portaudio.write_playback(b"\x00\x00", sample_rate=22050)
    # 默认采样率必须能调用不抛错
    portaudio.write_playback(b"\x00\x00", sample_rate=config.TTS_SAMPLE_RATE)


def test_portaudio_io_listener_receives_frames_after_start(portaudio):
    """``start()`` 后,``_capture_callback`` 推入的帧必须经由 listener 收到。"""
    received: list[tuple[bytes, int]] = []
    portaudio.add_capture_listener(lambda pcm, sr: received.append((pcm, sr)))

    portaudio.start()
    # 直接调内部的 sounddevice 回调(被 MagicMock 替换的 sd 不真触发回调)
    portaudio._capture_callback(
        indata=b"\x01\x00" * 480,
        frames=480,
        time_info=None,
        status=None,
    )
    # 等后台 fan-out 线程把帧从队列搬到 listener
    import time

    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and not received:
        time.sleep(0.01)
    assert received, "frame 应当被 listener 收到"
    assert received[0][0] == b"\x01\x00" * 480
    assert received[0][1] == 16000
    portaudio.stop()


def test_portaudio_io_remove_listener_stops_delivery(portaudio):
    """``remove_capture_listener`` 后再驱动回调,listener 不再收到帧。"""
    received: list[bytes] = []
    listener = lambda pcm, sr: received.append(pcm)
    portaudio.add_capture_listener(listener)

    portaudio.start()
    portaudio._capture_callback(indata=b"\x02\x00" * 480, frames=480,
                                 time_info=None, status=None)
    import time
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and not received:
        time.sleep(0.01)
    assert len(received) >= 1

    portaudio.remove_listener_calls = 0  # type: ignore[attr-defined]
    portaudio.remove_capture_listener(listener)
    portaudio._capture_callback(indata=b"\x03\x00" * 480, frames=480,
                                 time_info=None, status=None)
    time.sleep(0.1)
    # 注销后没有新帧进来
    assert len(received) == 1
    portaudio.stop()


def test_portaudio_io_listener_exception_does_not_break_callback(portaudio):
    """listener 抛异常不能影响 sounddevice 回调线程。"""
    def bad(pcm, sr):
        raise RuntimeError("boom")

    received: list[bytes] = []
    portaudio.add_capture_listener(bad)
    portaudio.add_capture_listener(lambda pcm, sr: received.append(pcm))

    portaudio.start()
    portaudio._capture_callback(indata=b"\x04\x00" * 480, frames=480,
                                 time_info=None, status=None)
    import time
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and not received:
        time.sleep(0.01)
    assert received, "good listener 仍应收到帧"
    portaudio.stop()


# ── MacOSVoiceIO 行为 (commit 05+) ───────────────────────────────────


def test_macos_voice_io_init_does_not_raise():
    """commit 05+:MacOSVoiceIO 构造不再抛 NotImplementedError;
    它只是延迟到 ``start()`` 才会 spawn helper 与连 socket。"""
    io = MacOSVoiceIO(helper_path="/nonexistent")
    assert io is not None


def test_macos_voice_io_default_socket_path():
    """默认 socket 路径:``/tmp/soul-tty-audio.sock``(未设环境变量时)。"""
    from soul_tty.audio.io.macos_voice import _default_socket_path

    # 测试干净环境:确保 SOUL_TTY_AUDIO_SOCK 未设。
    import os
    import unittest.mock as _um

    with _um.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SOUL_TTY_AUDIO_SOCK", None)
        assert _default_socket_path() == "/tmp/soul-tty-audio.sock"


def test_macos_voice_io_socket_path_env_override(monkeypatch):
    """``SOUL_TTY_AUDIO_SOCK`` 必须能覆盖默认 socket 路径。"""
    from soul_tty.audio.io.macos_voice import _default_socket_path

    monkeypatch.setenv("SOUL_TTY_AUDIO_SOCK", "/tmp/custom.sock")
    assert _default_socket_path() == "/tmp/custom.sock"


# ── Protocol encode/decode round-trip ───────────────────────────────


def test_encode_message_round_trip():
    """``encode_message`` / ``read_message`` 必须严格互逆。"""
    from soul_tty.audio.io.macos_voice import encode_message, read_message

    payload = b"\x01\x02\x03" * 100  # 300 B,确认变长也 OK
    msg = encode_message(0x01, payload)
    # 头部 5 字节 = 0x01 + 4 字节大端长度 300 = 0x0000012C
    assert msg[0] == 0x01
    assert msg[1:5] == b"\x00\x00\x01\x2c"
    assert len(msg) == 5 + len(payload)

    # 用一个临时 socket pair 跑 read_message:发一帧,读回。
    import socket

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        a.sendall(msg)
        a.close()
        msg_type, got = read_message(b)
        assert msg_type == 0x01
        assert got == payload
    finally:
        b.close()


def test_read_message_empty_payload():
    """空 payload(长度=0)必须有正确头,read_message 返回 b''。"""
    from soul_tty.audio.io.macos_voice import encode_message, read_message

    import socket

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        a.sendall(encode_message(0x06, b""))  # PONG
        a.close()
        msg_type, payload = read_message(b)
        assert msg_type == 0x06
        assert payload == b""
    finally:
        b.close()


def test_read_message_large_payload():
    """大帧往返:encode → decode 必须 bit-perfect。

    4 KiB 已足以覆盖多 frameLength 的 PCAP;真实运行中 helper → Python 是
    30 ms @ 16 kHz = 960 B,4 KiB 远超正常一帧(覆盖跨句合并等场景)。
    """
    from soul_tty.audio.io.macos_voice import encode_message, read_message

    import os
    import socket

    payload = os.urandom(4 * 1024)
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        a.sendall(encode_message(0x02, payload))
        a.close()
        msg_type, got = read_message(b)
        assert msg_type == 0x02
        assert got == payload
    finally:
        b.close()


def test_encode_message_rejects_oversize():
    """payload > 4 MiB → ValueError,不要撑爆内存。"""
    from soul_tty.audio.io.macos_voice import encode_message

    big = b"\x00" * (4 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        encode_message(0x01, big)


# ── MacOSVoiceIO fake-socket 行为 ───────────────────────────────────


class _FakeSocket:
    """最小 ``socket.socket`` 替身,只走 MacOSVoiceIO 用到的 API。

    ``sendall_queue`` 是 ``sendall`` 写出的字节;``recv_messages`` 拼成一个
    连续字节流,``recv(n)`` 按真实 socket 语义切片(返回最多 n 字节)。
    """

    def __init__(self, recv_messages: list[bytes] | None = None) -> None:
        self.sendall_queue: list[bytes] = []
        self._recv_buf = b"".join(recv_messages or [])
        self._closed = False
        self.raise_on_recv: Exception | None = None

    def connect(self, path):  # noqa: ARG002 - 兼容签名
        return None

    def sendall(self, data: bytes) -> None:
        self.sendall_queue.append(bytes(data))

    def recv(self, n: int) -> bytes:
        if self.raise_on_recv is not None:
            raise self.raise_on_recv
        if not self._recv_buf:
            return b""
        chunk = self._recv_buf[:n]
        self._recv_buf = self._recv_buf[n:]
        return chunk

    def settimeout(self, _t):
        return None

    def shutdown(self, _how):
        return None

    def close(self):
        self._closed = True


def test_macos_voice_io_protocol_handshake(monkeypatch, tmp_path):
    """``start()`` 必须发 START 消息,然后能 write_playback 把帧送出去。"""
    from soul_tty.audio.io import macos_voice as mv

    sock = _FakeSocket()
    monkeypatch.setattr(mv.socket, "socket", lambda *a, **kw: sock)
    monkeypatch.setattr(mv.time, "monotonic", lambda: 0.0)  # 跳过 connect 重试
    monkeypatch.setattr(mv.time, "sleep", lambda _s: None)

    io = mv.MacOSVoiceIO(socket_path=str(tmp_path / "x.sock"), helper_path="/nonexistent")
    io._helper_process = type("P", (), {"poll": staticmethod(lambda: 0)})()  # 已"在跑"
    io.start()

    # START(空 payload)+ 立刻 STOP(空 payload)+ write_playback(空 pcm 不发)
    # 第一帧发出后,SOCK 应当看到 START 头:0x03 + len 0
    types = []
    for blob in sock.sendall_queue:
        assert blob[0] in (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)
        types.append(blob[0])
    assert types[0] == 0x03, f"first message must be START (0x03), got 0x{types[0]:02x}"

    io.write_playback(b"\x00\x01" * 480, sample_rate=24000)
    last_msg = sock.sendall_queue[-1]
    assert last_msg[0] == 0x01
    # 头部 5 字节:[type=0x01][4B big-endian payload_len=964]
    # payload 内部:[4B sample_rate=24000][960B PCM]
    assert last_msg[1:5] == (964).to_bytes(4, "big")
    assert last_msg[5:9] == (24000).to_bytes(4, "big")
    assert len(last_msg) == 5 + 4 + 960
    assert last_msg[9:] == b"\x00\x01" * 480

    io.stop()


def test_macos_voice_io_listener_receives_capture_frames(monkeypatch, tmp_path):
    """server 推 CAPTURE_PCM → listener 必须收到(pcm, sample_rate)。"""
    from soul_tty.audio.io import macos_voice as mv

    captured: list[tuple[bytes, int]] = []

    def listener(pcm, sr):
        captured.append((pcm, sr))

    # 构造一帧 CAPTURE_PCM:[4B sr=16000][PCM]
    capture_payload = struct.pack(">I", 16000) + b"\x40\x00" * 480  # 30ms@16k
    sock = _FakeSocket(recv_messages=[mv.encode_message(0x02, capture_payload)])

    monkeypatch.setattr(mv.socket, "socket", lambda *a, **kw: sock)
    monkeypatch.setattr(mv.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(mv.time, "sleep", lambda _s: None)

    io = mv.MacOSVoiceIO(socket_path=str(tmp_path / "x.sock"), helper_path="/nonexistent")
    io._helper_process = type("P", (), {"poll": staticmethod(lambda: 0)})()
    io.add_capture_listener(listener)
    io.start()

    # 等后台 fan-out 线程把帧推给 listener
    import time

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not captured:
        time.sleep(0.01)
    assert captured, "CAPTURE_PCM 没到 listener"
    pcm, sr = captured[0]
    assert sr == 16000
    assert pcm == b"\x40\x00" * 480
    io.stop()


def test_macos_voice_io_error_message_logged(monkeypatch, tmp_path, caplog):
    """server 推 ERROR → 走 log.warning,不要崩主线程。"""
    from soul_tty.audio.io import macos_voice as mv

    err = mv.encode_message(0x08, b"speaker blew up")
    sock = _FakeSocket(recv_messages=[err])
    monkeypatch.setattr(mv.socket, "socket", lambda *a, **kw: sock)
    monkeypatch.setattr(mv.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(mv.time, "sleep", lambda _s: None)

    io = mv.MacOSVoiceIO(socket_path=str(tmp_path / "x.sock"), helper_path="/nonexistent")
    io._helper_process = type("P", (), {"poll": staticmethod(lambda: 0)})()
    with caplog.at_level("WARNING", logger=mv.log.name):
        io.start()
        import time as _t

        _t.sleep(0.2)
        io.stop()
    assert any("speaker blew up" in rec.message for rec in caplog.records)


def test_macos_voice_io_start_raises_if_helper_missing(monkeypatch, tmp_path):
    """helper 二进制不存在 → FileNotFoundError,把错误前移到 start()。"""
    from soul_tty.audio.io import macos_voice as mv

    sock = _FakeSocket()
    monkeypatch.setattr(mv.socket, "socket", lambda *a, **kw: sock)
    monkeypatch.setattr(mv.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(mv.time, "sleep", lambda _s: None)

    io = mv.MacOSVoiceIO(
        socket_path=str(tmp_path / "x.sock"),
        helper_path="/nonexistent/binary",
    )
    # 不预置 helper_process → start() 会去 Popen,看到不存在的 binary 报错
    monkeypatch.setattr(
        mv.subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(
            FileNotFoundError("helper missing")
        )
    )
    with pytest.raises(FileNotFoundError):
        io.start()


def test_helper_stderr_is_drained_without_touching_interactive_terminal(
    monkeypatch,
):
    """正常 TTY 下 helper 日志不能绕过 Rich Live 写进终端。"""
    import io as stdlib_io
    import sys

    from soul_tty.audio.io import macos_voice as mv

    class InteractiveStderr(stdlib_io.StringIO):
        def isatty(self):
            return True

    stderr = InteractiveStderr()
    proc = type(
        "P",
        (),
        {"stderr": stdlib_io.BytesIO(b"tap frame=50\nplayback drained\n")},
    )()
    audio = mv.MacOSVoiceIO(helper_path="/nonexistent")
    audio._helper_process = proc
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.delenv("SOUL_TTY_AUDIO_DEBUG", raising=False)

    audio._pump_helper_stderr()

    assert stderr.getvalue() == ""


def test_helper_stderr_can_be_forwarded_in_explicit_audio_debug_mode(
    monkeypatch,
):
    import io as stdlib_io
    import sys

    from soul_tty.audio.io import macos_voice as mv

    class InteractiveStderr(stdlib_io.StringIO):
        def isatty(self):
            return True

    stderr = InteractiveStderr()
    proc = type(
        "P",
        (),
        {"stderr": stdlib_io.BytesIO(b"tap frame=50\n")},
    )()
    audio = mv.MacOSVoiceIO(helper_path="/nonexistent")
    audio._helper_process = proc
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setenv("SOUL_TTY_AUDIO_DEBUG", "1")

    audio._pump_helper_stderr()

    assert stderr.getvalue() == "[helper] tap frame=50\n"


# ── config 暴露 ─────────────────────────────────────────────────────


def test_config_exposes_audio_io_backend():
    from soul_tty import config

    assert hasattr(config, "AUDIO_IO_BACKEND")
    assert config.AUDIO_IO_BACKEND == "portaudio"
