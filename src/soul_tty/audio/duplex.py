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
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .. import config
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
    # 播放期间是否检测到持续近端人声。明确制止词不依赖它，普通自然插话
    # 必须有该声学证据，避免 AEC 残差仅凭错误文本抢走话权。
    near_end: bool = True


class PlaybackCaptureGate:
    """播放期间抑制 AEC 后仍残留的低能量远端声音。

    VPIO 的 AEC 能显著降低扬声器回灌，却不能保证残差为绝对静音。残差一旦
    进入 Paraformer，识别结果常与实际播放文本完全不同，因此不能只依赖
    文本相似度。门控关闭时返回等长静音帧而非丢帧，使已启动的 VAD/endpoint
    仍可正常收尾；检测到近端较强声音后短暂保持打开，避免切碎真人语音。
    """

    def __init__(
        self,
        playback_active,
        *,
        peak_threshold: float,
        hold_ms: int,
        confirm_frames: int = 1,
        tail_ms: int = 0,
    ) -> None:
        self._playback_active = playback_active
        self._peak_threshold = max(0.0, min(float(peak_threshold), 1.0))
        self._hold_s = max(0, int(hold_ms)) / 1000
        self._confirm_frames = max(1, int(confirm_frames))
        self._tail_s = max(0, int(tail_ms)) / 1000
        self._open_until = 0.0
        self._playback_until = 0.0
        self._loud_frames = 0

    def observe(self, pcm: bytes) -> bool:
        """更新门控状态，返回当前帧是否已有持续近端人声证据。"""
        try:
            playing = bool(self._playback_active())
        except Exception:
            # 状态探针失效时宁可保留采集，不让用户永久失去打断能力。
            playing = False

        now = time.monotonic()
        if playing:
            # 流式 TTS 的 chunk 之间可能短暂显示 drained；锁存一小段时间，
            # 同时覆盖扬声器停止后的房间混响与 ASR endpoint 延迟。
            self._playback_until = now + self._tail_s
        elif now >= self._playback_until:
            self._open_until = 0.0
            self._loud_frames = 0
            return True

        if now < self._open_until:
            return True

        samples = np.frombuffer(pcm, dtype="<i2")
        peak = (
            float(np.max(np.abs(samples.astype(np.int32)))) / 32768.0
            if samples.size
            else 0.0
        )
        if peak >= self._peak_threshold:
            self._loud_frames += 1
        else:
            self._loud_frames = 0

        if self._loud_frames >= self._confirm_frames:
            self._loud_frames = 0
            self._open_until = now + self._hold_s
            return True

        return False

    def process(self, pcm: bytes) -> bytes:
        """兼容原门控 API：无近端证据时返回等长静音帧。"""
        return pcm if self.observe(pcm) else bytes(len(pcm))


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
        playback_active=None,
    ) -> None:
        self._session = session or asr.VadGatedSherpaStream()
        self._events: queue.Queue[DuplexEvent] = queue.Queue(
            maxsize=max(1, queue_maxsize)
        )
        self._stop = threading.Event()
        self._was_active = False  # 用于在 endpoint 处发 SPEECH_END
        self._utterance_near_end = False
        self._playback_gate = (
            PlaybackCaptureGate(
                playback_active,
                peak_threshold=config.DUPLEX_PLAYBACK_GATE_PEAK,
                hold_ms=config.DUPLEX_PLAYBACK_GATE_HOLD_MS,
                confirm_frames=config.DUPLEX_PLAYBACK_GATE_CONFIRM_FRAMES,
                tail_ms=config.DUPLEX_PLAYBACK_GATE_TAIL_MS,
            )
            if playback_active is not None
            and config.DUPLEX_PLAYBACK_GATE_ENABLED
            else None
        )

    def on_frame(self, pcm: bytes, sample_rate: int) -> None:
        """Sounddevice 回调调用入口:把帧灌进 streaming ASR 并入队事件。

        listener 协议要求非阻塞。本方法本身只做 ``session.accept``(<1ms)
        + ``put_nowait``;``session.accept`` 内部已做过 ``numpy.frombuffer``
        + ``accept_waveform`` 的小开销,实测单帧 < 0.5ms。
        """
        near_end = (
            self._playback_gate.observe(pcm)
            if self._playback_gate is not None
            else True
        )
        try:
            # ASR 始终看见 AEC-clean PCM，使低音量的“停下/别说了”仍能走
            # 明确制止词快路径。普通文本能否打断由 near_end 声学证据决定。
            updates = self._session.accept(pcm)
        except Exception:
            # 单帧识别失败不应打断后续帧;记一次 partial 丢弃即可。
            return

        for update in updates:
            if near_end and (bool(update.text) or self._session.active):
                self._utterance_near_end = True
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
            events.append(
                DuplexEvent(
                    DuplexEventKind.SPEECH_START,
                    near_end=self._utterance_near_end,
                )
            )
            self._was_active = True

        if not update.final:
            if update.text:
                events.append(
                    DuplexEvent(
                        DuplexEventKind.PARTIAL,
                        text=update.text,
                        near_end=self._utterance_near_end,
                    )
                )
        else:
            events.append(
                DuplexEvent(
                    DuplexEventKind.FINAL,
                    text=update.text,
                    pcm=update.pcm,
                    near_end=self._utterance_near_end,
                )
            )
            # 触发 SPEECH_END 后立刻清空 active 标记,允许下一段说话
            # 重新走 START 流程。
            if self._was_active:
                events.append(DuplexEvent(DuplexEventKind.SPEECH_END))
                self._was_active = False
            self._utterance_near_end = False

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
