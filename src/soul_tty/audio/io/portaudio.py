"""PortAudio backend:用 sounddevice 包装当前 Mic + 播放能力。

commit 03 阶段:此 backend **不替换** ``Mic`` 与 ``StreamingSpeaker``;
只是把它能提供的能力(``AudioIO`` Protocol)摆出来,让 commit 04
可以"切路由"。

行为约定:
- 采集:16 kHz mono int16,30ms 帧,内部 ``queue.Queue`` + worker
  线程 fan-out 给所有 capture listener。
- 播放:长连接 ``sd.RawOutputStream`` + ``queue.Queue``。调用方
  (``StreamingSpeaker._play_loop``) 把每个 TTS chunk 调
  ``write_playback(pcm, sr)``,由内部 worker 持续写入;
  ``stop()`` 之前队尾会先 drain 完。
- 增益:转发到 ``tts.set_playback_gain``(单一权威)。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Optional

import sounddevice as sd

from ... import config
from .. import tts
from .base import AudioIO, CaptureListener

# commit 07+:playback queue 的"停止"信号。
_PB_SENTINEL = object()


class PortAudioIO(AudioIO):
    """基于 sounddevice 的 ``AudioIO`` 实现。"""

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._frame_bytes = sample_rate * 30 // 1000 * 2  # 30ms int16 mono
        self._capture_listeners: list[CaptureListener] = []
        self._listener_lock = threading.Lock()
        self._capture_queue: "queue.Queue[bytes]" = queue.Queue()
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_stop = threading.Event()
        self._stream: Optional[sd.RawInputStream] = None
        # commit 07+:playback 长连接 + 队列。
        self._playback_stream: Optional[sd.RawOutputStream] = None
        self._playback_queue: "queue.Queue[bytes | object]" = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._playback_stop = threading.Event()
        self._started = False

    # ── AudioIO protocol ───────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        self._capture_stop.clear()
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._frame_bytes // 2,
            dtype="int16",
            channels=1,
            callback=self._capture_callback,
        )
        self._stream.start()
        # commit 07+:开 playback 长连接,跟 input stream 并存。
        self._playback_stop.clear()
        self._playback_stream = sd.RawOutputStream(
            samplerate=config.TTS_SAMPLE_RATE,
            dtype="int16",
            channels=1,
        )
        self._playback_stream.start()
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            name="soul-tty-portaudio-playback",
            daemon=True,
        )
        self._playback_thread.start()
        self._capture_thread = threading.Thread(
            target=self._fanout_loop,
            name="soul-tty-portaudio-fanout",
            daemon=True,
        )
        self._capture_thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._capture_stop.set()
        self._playback_stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._playback_stream is not None:
            try:
                self._playback_stream.stop()
                self._playback_stream.close()
            except Exception:
                pass
            self._playback_stream = None
        # 唤醒 worker 退出(若它在 get() 上阻塞)。
        try:
            self._playback_queue.put_nowait(_PB_SENTINEL)
        except Exception:
            pass
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None
        self._started = False

    def write_playback(self, pcm: bytes, sample_rate: int) -> None:
        """把 PCM 推到内部队列,由 playback worker 持续写入长连接 stream。

        支持非 ``config.TTS_SAMPLE_RATE`` 的输入不在本 backend 范围(抛
        ``NotImplementedError``);macOS 后端通过 ``AVAudioConverter`` 处理
        重采样。
        """
        if sample_rate != config.TTS_SAMPLE_RATE:
            raise NotImplementedError(
                f"PortAudioIO.write_playback 仅支持 {config.TTS_SAMPLE_RATE} Hz,"
                f" got {sample_rate} Hz"
            )
        if not pcm:
            return
        if not self._started:
            return
        try:
            self._playback_queue.put_nowait(pcm)
        except queue.Full:
            # 队列满:丢最旧,避免阻塞调用方(StreamingSpeaker 的 _play_loop)
            try:
                self._playback_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._playback_queue.put_nowait(pcm)
            except queue.Full:
                pass

    def add_capture_listener(self, listener: CaptureListener) -> None:
        with self._listener_lock:
            if listener not in self._capture_listeners:
                self._capture_listeners.append(listener)

    def remove_capture_listener(self, listener: CaptureListener) -> None:
        with self._listener_lock:
            if listener in self._capture_listeners:
                self._capture_listeners.remove(listener)

    def set_playback_gain(self, value: float) -> None:
        # 单一权威:转发到 tts 模块。
        tts.set_playback_gain(value)

    def get_playback_gain(self) -> float:
        return tts.get_playback_gain()

    def flush_playback(self) -> None:
        """PortAudioIO 同步写,没有已排队未播放的 buffer。"""
        pass

    def wait_playback_drained(self, timeout: float | None = None) -> bool:
        """PortAudioIO 同步写,write_playback 返回时已播完。"""
        return True

    @property
    def playback_active(self) -> bool:
        return False

    # ── 内部 ────────────────────────────────────────────────────────

    def _capture_callback(self, indata, frames, time_info, status):
        # sounddevice 回调线程,必须非阻塞:只 put_nowait + 快照 list。
        chunk = bytes(indata)
        try:
            self._capture_queue.put_nowait(chunk)
        except queue.Full:
            # 队列满:丢最旧,避免阻塞回调线程。
            try:
                self._capture_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._capture_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def _fanout_loop(self) -> None:
        while not self._capture_stop.is_set():
            try:
                chunk = self._capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._listener_lock:
                listeners = list(self._capture_listeners)
            for fn in listeners:
                try:
                    fn(chunk, self._sample_rate)
                except Exception:
                    # 单 listener 抛异常不影响后续。
                    pass

    def _playback_loop(self) -> None:
        """commit 07+:持续从 _playback_queue 拉 PCM 写到长连接 stream。

        cancel 通过 stop() 唤醒(队列里 put SENTINEL + playback_stop)。
        """
        stream = self._playback_stream
        gain = tts.get_playback_gain()
        try:
            while not self._playback_stop.is_set():
                item = self._playback_queue.get()
                if item is _PB_SENTINEL:
                    break
                if stream is None:
                    break
                if not item:
                    continue
                # gain 实时读(支持 cancel 后调 set_playback_gain)
                cur_gain = tts.get_playback_gain()
                if cur_gain != gain:
                    gain = cur_gain
                if gain == 1.0:
                    stream.write(item)
                else:
                    import numpy as np
                    arr = np.frombuffer(item, dtype="<i2").astype(np.float32)
                    arr *= gain
                    np.clip(arr, -32768, 32767, out=arr)
                    stream.write(arr.astype("<i2").tobytes())
        except Exception:
            # playback 异常不传播(只影响本 stream)
            pass