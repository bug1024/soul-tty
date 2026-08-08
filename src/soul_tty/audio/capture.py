"""音频采集 + VAD:从麦克风持续采集,按"一句话"切分,产出 PCM 段。"""

import queue
import threading
from collections.abc import Callable, Iterator

import sounddevice as sd
import webrtcvad

from .. import config


# 单帧回调签名:(int16 mono PCM, sample_rate)
FrameListener = Callable[[bytes, int], None]


class Mic:
    """麦克风采集:VAD 按句切分。

    flush() 丢弃队列里积压的音频——TTS 播放期间采集到的声音
    (包括扬声器回声)不应进入下一轮识别。

    add_frame_listener() 注册额外的帧级回调,用于在 half-duplex 之外
    给 duplex 流式 ASR / SenseVoice 等旁路提供实时 PCM。
    listener 必须在 sounddevice 回调线程上非阻塞(≤1ms),否则会触发
    PortAudio underrun / xrun。
    """

    FRAME_BYTES = config.SAMPLE_RATE * config.FRAME_MS // 1000 * 2  # 30ms int16 mono

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        self._running = False
        self._vad_generation = 0
        self._frame_listeners: list[FrameListener] = []
        self._listener_lock = threading.Lock()
        self._stream = sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=self.FRAME_BYTES // 2,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        chunk = bytes(indata)
        # 队列保留旧行为不变;listener 在队列推送之后分发,
        # 保证 _q 与 frames() 的语义不被任何下游 bug 污染。
        self._q.put(chunk)
        # listener 列表快照后逐个调用,期间允许 listener 自己 unregister。
        with self._listener_lock:
            listeners = list(self._frame_listeners)
        for fn in listeners:
            try:
                fn(chunk, config.SAMPLE_RATE)
            except Exception:
                # 单个 listener 抛异常不能影响 sounddevice 回调,
                # 否则后续所有 PCM 都会被丢。
                pass

    def start(self):
        self._closed.clear()
        self._stream.start()
        self._running = True

    def pause(self):
        """回答期间停止硬件采集，从源头避免将 TTS 回声送入 VAD。"""
        if self._running:
            self._stream.stop()
            self._running = False

    def resume(self):
        if not self._running and not self._closed.is_set():
            self._stream.start()
            self._running = True

    def stop(self):
        self._closed.set()
        if self._running:
            self._stream.stop()
            self._running = False
        self._stream.close()

    def flush(self):
        with self._q.mutex:
            self._q.queue.clear()

    def reset_vad(self):
        """通知切句生成器丢弃尚未成句的局部缓冲。"""
        self._vad_generation += 1

    def add_frame_listener(self, listener: FrameListener) -> None:
        """注册帧级 listener。listener 必须非阻塞,见类 docstring。"""
        with self._listener_lock:
            if listener not in self._frame_listeners:
                self._frame_listeners.append(listener)

    def remove_frame_listener(self, listener: FrameListener) -> None:
        """注销帧级 listener。"""
        with self._listener_lock:
            if listener in self._frame_listeners:
                self._frame_listeners.remove(listener)

    def frames(self) -> Iterator[bytes]:
        """从采集队列持续取 30ms 帧。

        必须用带超时的 get:无限阻塞的 queue.get() 在 macOS 上无法被
        Ctrl+C(KeyboardInterrupt)打断,会导致退不出程序。
        """
        while not self._closed.is_set():
            try:
                yield self._q.get(timeout=0.5)
            except queue.Empty:
                continue

    def utterances(self) -> Iterator[bytes]:
        """生成器:每检测到一句完整说话,产出其 int16 PCM 字节。

        切句逻辑:webrtcvad 判定有声后起句,连续静音 SILENCE_MS 收句;
        超过 MAX_UTTERANCE_S 强制收句;短于 MIN_UTTERANCE_MS 的段丢弃。
        """
        vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        silence_frames_needed = config.SILENCE_MS // config.FRAME_MS
        max_frames = int(config.MAX_UTTERANCE_S * 1000 / config.FRAME_MS)
        min_frames = config.MIN_UTTERANCE_MS // config.FRAME_MS

        buf: list[bytes] = []
        triggered = False
        silence = 0
        generation = self._vad_generation
        for frame in self.frames():
            if generation != self._vad_generation:
                buf = []
                triggered = False
                silence = 0
                generation = self._vad_generation
            is_speech = vad.is_speech(frame, config.SAMPLE_RATE)
            if not triggered:
                if is_speech:
                    triggered = True
                    buf = [frame]
                    silence = 0
            else:
                buf.append(frame)
                silence = 0 if is_speech else silence + 1
                if silence >= silence_frames_needed or len(buf) >= max_frames:
                    if len(buf) >= min_frames:
                        yield b"".join(buf)
                    buf = []
                    triggered = False
                    silence = 0


class BackgroundListener:
    """在后台持续完成 VAD 切句，让主线程在 LLM/TTS 回答期间仍能收到插话。"""

    def __init__(self, mic: Mic):
        self._mic = mic
        self._utterances: queue.Queue[bytes] = queue.Queue()
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _listen(self) -> None:
        for pcm in self._mic.utterances():
            self._utterances.put(pcm)

    def get(self, timeout: float | None = None) -> bytes:
        return self._utterances.get(timeout=timeout)

    def flush(self) -> None:
        with self._utterances.mutex:
            self._utterances.queue.clear()

    def join(self) -> None:
        self._thread.join(timeout=1)
