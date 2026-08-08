"""macOS voice-processing AEC backend。

设计要点(commit 05 真实落地):
- Python 通过 Unix domain socket 与 Swift helper ``macos-voice-io`` 通信。
- Swift 端负责 ``AVAudioEngine`` + ``setVoiceProcessingEnabled``,Apple 官方
  voice-processing 模式会接管 echo cancellation。
- 协议消息头:``[1 byte type][4 bytes uint32 BE payload_len][payload]``。
- Python 端看到的就是 16 kHz mono int16 AEC-clean PCM,与现有 Mic/PorteAudio
  路径字节级兼容,所以 Mic / DuplexListener 不用感知 backend 切换。

进程模型:
- ``MacOSVoiceIO`` 实例化时**不**自动 spawn Swift helper;调用方(``cli.py``)
  负责在 ``main()`` 启动早期用 ``subprocess.Popen`` 拉起它。原因:helper
  启动慢、可能在 sandbox / 无麦克风权限下启动失败;调用方需要能 catch
  ``CalledProcessError`` / ``FileNotFoundError``。
- start() 通过 socket 与 helper 握手(发 START),helper 内部才创建
  AudioEngine。
- stop() 发 STOP,helper 优雅退出。
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from .base import AudioIO, CaptureListener

log = logging.getLogger(__name__)

# 消息类型(与 native/macos_voice_io/Sources/macos-voice-io/Protocol.swift 一致)
_PLAYBACK_PCM = 0x01
_CAPTURE_PCM = 0x02
_START = 0x03
_STOP = 0x04
_PING = 0x05
_PONG = 0x06
_STATS = 0x07
_ERROR = 0x08

_MAX_PAYLOAD = 4 * 1024 * 1024


def _default_helper_path() -> Path:
    """Swift helper 可执行路径。"""
    env = os.environ.get("SOUL_TTY_MACOS_HELPER")
    if env:
        return Path(env)
    # 默认:仓库根/native/macos_voice_io/.build/debug/macos-voice-io
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "native" / "macos_voice_io" / ".build" / "debug" / "macos-voice-io"


def _default_socket_path() -> str:
    env = os.environ.get("SOUL_TTY_AUDIO_SOCK")
    if env:
        return env
    return "/tmp/soul-tty-audio.sock"


def encode_message(msg_type: int, payload: bytes = b"") -> bytes:
    """[1 byte type][4 bytes uint32 BE payload_len][payload]。"""
    if len(payload) > _MAX_PAYLOAD:
        raise ValueError(f"payload too large: {len(payload)} > {_MAX_PAYLOAD}")
    return struct.pack(">BI", msg_type, len(payload)) + payload


def read_message(sock: socket.socket) -> tuple[int, bytes]:
    """阻塞读一条消息。返回 (type, payload)。EOF 抛 ConnectionError。"""
    header = _read_exact(sock, 5)
    msg_type = header[0]
    payload_len = struct.unpack(">I", header[1:5])[0]
    if payload_len > _MAX_PAYLOAD:
        raise ValueError(f"protocol violation: payload_len={payload_len}")
    payload = _read_exact(sock, payload_len) if payload_len else b""
    return msg_type, payload


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed (got {} of {} bytes)".format(len(buf), n))
        buf.extend(chunk)
    return bytes(buf)


class MacOSVoiceIO(AudioIO):
    """macOS AVAudioEngine voice-processing 后端。

    工作模型:
    - ``__init__`` 仅记录参数;不连 socket、不启 helper。
    - ``start()`` spawn helper(若未提供 process 对象)、connect socket、
      发 START、启动 capture 读取线程。
    - ``stop()`` 发 STOP、关 socket、join helper process。
    """

    def __init__(
        self,
        socket_path: str | None = None,
        sample_rate: int = 16000,
        helper_path: str | os.PathLike[str] | None = None,
        helper_process: subprocess.Popen[bytes] | None = None,
    ) -> None:
        self._socket_path = socket_path or _default_socket_path()
        self._sample_rate = sample_rate
        self._helper_path = Path(helper_path) if helper_path else _default_helper_path()
        self._helper_process = helper_process  # 若调用方已启,start() 复用
        self._socket: Optional[socket.socket] = None
        self._capture_listeners: list[CaptureListener] = []
        self._listener_lock = threading.Lock()
        self._capture_queue: "queue.Queue[tuple[bytes, int]]" = queue.Queue()
        self._capture_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._capture_stop = threading.Event()
        self._started = False
        self._send_lock = threading.Lock()  # socket.send 不是 atomic

    # ── AudioIO protocol ───────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        self._capture_stop.clear()

        # 1) 如果调用方没启 helper,自己启。
        if self._helper_process is None:
            if not self._helper_path.exists():
                raise FileNotFoundError(
                    f"macos-voice-io helper not found at {self._helper_path}。"
                    "先用 `cd native/macos_voice_io && swift build` 编译。"
                )
            env = os.environ.copy()
            env["SOUL_TTY_AUDIO_SOCK"] = self._socket_path
            self._helper_process = subprocess.Popen(
                [str(self._helper_path)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # 持续把 helper stderr 转发到我们的 stderr;否则 PIPE buffer
            # 满后 helper 会阻塞。需要时把行加上 [helper] 前缀方便辨认。
            self._stderr_thread = threading.Thread(
                target=self._pump_helper_stderr,
                name="soul-tty-macos-voice-helper-stderr",
                daemon=True,
            )
            self._stderr_thread.start()

        # 2) 等 socket 文件出现 + connect。helper 启动要几十 ms。
        self._connect_with_retry(timeout_s=5.0)

        # 3) 发 START,helper 才创建 AVAudioEngine + tap。
        self._send(_START, b"")

        # 4) capture reader 从 socket 拉消息,fake 帧扇出。
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="soul-tty-macos-voice-reader",
            daemon=True,
        )
        self._reader_thread.start()
        self._capture_thread = threading.Thread(
            target=self._fanout_loop,
            name="soul-tty-macos-voice-fanout",
            daemon=True,
        )
        self._capture_thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._capture_stop.set()
        # 优雅关:helper 收 STOP 后 exit()。先关 socket 防止 read 卡死。
        try:
            if self._socket is not None:
                try:
                    self._send(_STOP, b"")
                except Exception:
                    pass
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    self._socket.close()
                except Exception:
                    pass
        finally:
            self._socket = None

        for t in (self._reader_thread, self._capture_thread):
            if t is not None:
                t.join(timeout=1.0)
        self._reader_thread = None
        self._capture_thread = None

        # 收 helper process;它应已优雅退出,这里再杀一下保险。
        proc = self._helper_process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._helper_process = None

    def _pump_helper_stderr(self) -> None:
        """持续把 helper stderr 转发到我们自己的 stderr。"""
        proc = self._helper_process
        if proc is None or proc.stderr is None:
            return
        import sys
        for line in iter(proc.stderr.readline, b""):
            try:
                sys.stderr.write(f"[helper] {line.decode('utf-8', 'replace')}")
            except Exception:
                pass

    def write_playback(self, pcm: bytes, sample_rate: int) -> None:
        """把 int16 mono PCM 推到 helper(helper 内部 resample 到硬件格式)。
        """
        if not pcm:
            return
        payload = struct.pack(">I", sample_rate) + pcm
        self._send(_PLAYBACK_PCM, payload)

    def add_capture_listener(self, listener: CaptureListener) -> None:
        with self._listener_lock:
            if listener not in self._capture_listeners:
                self._capture_listeners.append(listener)

    def remove_capture_listener(self, listener: CaptureListener) -> None:
        with self._listener_lock:
            if listener in self._capture_listeners:
                self._capture_listeners.remove(listener)

    def set_playback_gain(self, value: float) -> None:
        # helper 通过 AVAudioEngine.mainMixerNode.outputVolume 控制音量。
        # commit 05 暂不实现 IPC 消息(SET_GAIN 留到未来),但要确保 API
        # 仍可调用 —— 直接调 AVAudioEngine 会更优雅;此处 no-op。
        # 调用方(Mic / StreamingSpeaker)目前也只在 commit 02 调过,
        # 双工路径默认 gain=1.0。
        if value < 0:
            raise ValueError("playback gain must be >= 0")
        # no-op for now;helper volume 暂由 mainMixer 默认控制。

    def get_playback_gain(self) -> float:
        return 1.0

    # ── 内部 ────────────────────────────────────────────────────────

    def _connect_with_retry(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(self._socket_path)
                sock.settimeout(None)
                self._socket = sock
                return
            except (FileNotFoundError, ConnectionRefusedError) as e:
                last_err = e
                try:
                    sock.close()  # type: ignore[possibly-undefined]
                except Exception:
                    pass
                time.sleep(0.05)
        raise ConnectionError(
            f"failed to connect to {self._socket_path} within {timeout_s}s: {last_err}"
        )

    def _send(self, msg_type: int, payload: bytes) -> None:
        msg = encode_message(msg_type, payload)
        with self._send_lock:
            sock = self._socket
            if sock is None:
                raise ConnectionError("socket not connected")
            try:
                sock.sendall(msg)
            except Exception as e:
                raise ConnectionError(f"send failed: {e}") from e

    def _reader_loop(self) -> None:
        """从 socket 读消息。CAPTURE_PCM 入队,其它忽略(后续可加 STATS)。"""
        sock = self._socket
        if sock is None:
            return
        sock.settimeout(None)
        while not self._capture_stop.is_set():
            try:
                msg_type, payload = read_message(sock)
            except (ConnectionError, OSError):
                return
            if msg_type == _CAPTURE_PCM:
                if len(payload) < 4:
                    continue
                sample_rate = struct.unpack(">I", payload[:4])[0]
                pcm = payload[4:]
                if not pcm:
                    continue
                try:
                    self._capture_queue.put_nowait((pcm, sample_rate))
                except queue.Full:
                    pass  # 满 → 丢最旧
            elif msg_type == _PONG:
                pass  # 未来用于心跳
            elif msg_type == _ERROR:
                log.warning("macos-voice-io error: %r", payload)
            elif msg_type == _STATS:
                pass

    def _fanout_loop(self) -> None:
        while not self._capture_stop.is_set():
            try:
                pcm, sr = self._capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._listener_lock:
                listeners = list(self._capture_listeners)
            for fn in listeners:
                try:
                    fn(pcm, sr)
                except Exception:
                    pass