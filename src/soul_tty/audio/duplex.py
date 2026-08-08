"""双工事件流:把麦克风帧持续送进一路 VAD-gated streaming ASR,
按事件(START/PARTIAL/FINAL/END)暴露给上层。

设计要点(commit 01):
- 不接管 Mic 的生命周期:由 caller 负责 start/stop 与 listener 注册。
- 复用 ``asr.VadGatedSherpaStream``:WebRTC VAD 触发 + pre-roll + endpoint
  静音都与半双工路径共享同一份代码。
- 事件队列有界,``Full`` 时丢最旧,保证 streaming partial 永远代表最新进度。
- ``events()`` 用带超时的 ``get``,让 ``Ctrl+C`` 能被响应。
"""

from __future__ import annotations

import enum
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass

from . import asr


class DuplexEventKind(str, enum.Enum):
    """双工事件类型。

    一次说话的事件顺序大致为::

        SPEECH_START → PARTIAL... → FINAL → SPEECH_END

    短到没有触发 VAD 时不会有任何事件;超长段会被 sherpa
    endpoint 强制截断,产生 FINAL + SPEECH_END,然后下一段又从
    SPEECH_START 开始。
    """

    SPEECH_START = "speech_start"
    PARTIAL = "partial"
    FINAL = "final"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class DuplexEvent:
    """一条双工事件。

    ``text`` 仅在 PARTIAL/FINAL 时有意义;FINAL 同时携带完整 utterance PCM
    (含 pre-roll),供 SenseVoice / Reflection 等旁路消费。
    """

    kind: DuplexEventKind
    text: str = ""
    pcm: bytes | None = None


class DuplexListener:
    """一路 VAD-gated streaming ASR + 事件队列。

    用法::

        listener = DuplexListener()
        mic.add_frame_listener(listener.on_frame)
        try:
            for event in listener.events():
                ...
        finally:
            mic.remove_frame_listener(listener.on_frame)
            listener.stop()

    ``on_frame`` 必须非阻塞(只 ``queue.put_nowait``)。Sounddevice 回调线程
    上任何长时间计算都会触发 PortAudio xrun。
    """

    def __init__(
        self,
        session: asr.VadGatedSherpaStream | None = None,
        *,
        queue_maxsize: int = 64,
    ) -> None:
        self._session = session or asr.VadGatedSherpaStream()
        self._events: queue.Queue[DuplexEvent] = queue.Queue(
            maxsize=max(1, queue_maxsize)
        )
        self._stop = threading.Event()
        self._was_active = False  # 用于在 endpoint 处发 SPEECH_END

    def on_frame(self, pcm: bytes, sample_rate: int) -> None:
        """Sounddevice 回调调用入口:把帧灌进 streaming ASR 并入队事件。

        listener 协议要求非阻塞。本方法本身只做 ``session.accept``(<1ms)
        + ``put_nowait``;``session.accept`` 内部已做过 ``numpy.frombuffer``
        + ``accept_waveform`` 的小开销,实测单帧 < 0.5ms。
        """
        try:
            updates = self._session.accept(pcm)
        except Exception:
            # 单帧识别失败不应打断后续帧;记一次 partial 丢弃即可。
            return

        for update in updates:
            events = self._convert(update)
            for event in events:
                self._enqueue(event)

    def events(self) -> Iterator[DuplexEvent]:
        """阻塞产出事件;``stop()`` 后退出循环。"""
        while not self._stop.is_set():
            try:
                yield self._events.get(timeout=0.5)
            except queue.Empty:
                continue

    def stop(self) -> None:
        """标记停止;``events()`` 在下一个超时点退出。"""
        self._stop.set()

    # ── 内部 ────────────────────────────────────────────────────────

    def _convert(self, update: asr.TranscriptUpdate) -> list[DuplexEvent]:
        """把 ``TranscriptUpdate`` 翻译成 ``DuplexEvent`` 序列。

        翻译规则:
        - VAD 从非 active → active 的那一刻,先发 ``SPEECH_START``。
        - ``final=False`` ⇒ ``PARTIAL``(只发 text 变化的部分)。
        - ``final=True`` ⇒ ``FINAL``(带 PCM) + ``SPEECH_END``。
        - 帧之间的 endpoint(流被 sherpa 截断)⇒ ``SPEECH_END``(不重复发 FINAL)。
        """
        events: list[DuplexEvent] = []

        active_now = bool(update.text) or self._session.active
        if active_now and not self._was_active:
            events.append(DuplexEvent(DuplexEventKind.SPEECH_START))
            self._was_active = True

        if not update.final:
            if update.text:
                events.append(
                    DuplexEvent(DuplexEventKind.PARTIAL, text=update.text)
                )
        else:
            events.append(
                DuplexEvent(
                    DuplexEventKind.FINAL,
                    text=update.text,
                    pcm=update.pcm,
                )
            )
            # 触发 SPEECH_END 后立刻清空 active 标记,允许下一段说话
            # 重新走 START 流程。
            if self._was_active:
                events.append(DuplexEvent(DuplexEventKind.SPEECH_END))
                self._was_active = False

        return events

    def _enqueue(self, event: DuplexEvent) -> None:
        """``put_nowait`` 入队;满时丢最旧,保留最新事件流。"""
        try:
            self._events.put_nowait(event)
        except queue.Full:
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._events.put_nowait(event)
            except queue.Full:
                # 极端并发:连续 full 时直接丢弃,不阻塞回调线程。
                pass