"""MLX Qwen3-TTS 与 macOS 系统音色播放客户端。

StreamingSpeaker:句子队列 -> 合成线程 -> 播放线程的流水线,
LLM 边生成边切句送入,边合成边播放,降低首音延迟。
"""

import queue
import re
import subprocess
import threading
from collections.abc import Callable, Iterator

import httpx
import numpy as np
import sounddevice as sd

from .. import config

_SENTINEL = object()
_MLX_SENTENCE = re.compile(r".+?(?:[。！？!?；;\n]+|$)", re.DOTALL)
_SPEAKABLE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")
_MARKDOWN_LINK = re.compile(r"!?\[([^]]*)\]\([^)]+\)")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MARKDOWN_PREFIX = re.compile(r"(?m)^\s{0,3}(?:#{1,6}|[-+>]\s+)\s*")
_ELONGATED_INTERJECTION = re.compile(r"([嗯啊呀哦噢唔哎诶哈])\s*[—–-]{2,}")
_LONG_PAUSE_PUNCTUATION = re.compile(r"(?:\.[ \t]*){2,}|…{2,}")
_TRAILING_LONG_PAUSE = re.compile(
    r"(?:(?:\.[ \t]*){2,}|…{2,})(?=[ \t]*(?:\n|$))"
)
_SHORT_TAIL_MAX_CHARS = 8
_SHORT_TAIL_MIN_PREFIX_CHARS = 8
_PLAYBACK_LEVEL_FRAME_MS = 50

# 播放线性增益（commit 02 引入）。module 级单值 + lock 让
# set_playback_gain 跨线程安全；播放线程在 _write_metered_pcm 内
# 读一次就锁定整个 frame 的增益，避免逐 sample 抖动。
_PLAYBACK_GAIN: float = config.TTS_PLAYBACK_GAIN
_PLAYBACK_GAIN_LOCK = threading.Lock()


def set_playback_gain(value: float) -> None:
    """设置 TTS 线性播放增益。

    - ``value == 0`` 合法：等效静音，便于 sanity check。
    - ``value < 0`` 抛 ``ValueError``。
    - ``value > 1`` 会保留 clip，避免 int16 上溢爆音。
    """
    if value < 0:
        raise ValueError(f"playback gain must be >= 0, got {value}")
    with _PLAYBACK_GAIN_LOCK:
        global _PLAYBACK_GAIN
        _PLAYBACK_GAIN = float(value)


def get_playback_gain() -> float:
    """读取当前 TTS 播放增益（默认 1.0）。"""
    with _PLAYBACK_GAIN_LOCK:
        return _PLAYBACK_GAIN


def apply_playback_gain(pcm: bytes) -> bytes:
    """对 int16 PCM 应用当前播放增益。

    与 ``_write_metered_pcm`` 内的增益逻辑一致,但返回处理后的 bytes,
    不写 stream。供 AudioIO 播放路径使用。
    """
    gain = get_playback_gain()
    if not pcm or gain == 1.0:
        return pcm
    usable = len(pcm) - len(pcm) % 2
    if usable == 0:
        return pcm
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32)
    samples = np.clip(samples * gain, -32768, 32767).astype("<i2")
    return samples.tobytes() + pcm[usable:]


# commit 07+ 修:把 TTS 播放也接进 AudioIO,让 macos_voice 后端的
# AVAudioEngine 拿到 playback reference(完整 AEC)。非 duplex 路径
# 仍然走默认的 sd.RawOutputStream(由 ``_play_loop`` 内部 fallback)。
_AUDIO_IO = None
_AUDIO_IO_LOCK = threading.Lock()


def set_audio_io(audio_io) -> None:
    """``cli.py`` 早期把 ``AudioIO`` 实例绑到这里。

    StreamingSpeaker 创建时如果发现这里有实例,``_play_loop`` 会改用
    ``audio_io.write_playback(pcm, sr)``,而不是直接 ``sd.RawOutputStream``。
    """
    if audio_io is not None and not hasattr(audio_io, "write_playback"):
        raise TypeError(
            f"audio_io must implement AudioIO (have write_playback), "
            f"got {type(audio_io)}"
        )
    with _AUDIO_IO_LOCK:
        global _AUDIO_IO
        _AUDIO_IO = audio_io


def get_audio_io():
    """读取当前绑定的 ``AudioIO``(供 StreamingSpeaker 内部使用)。"""
    with _AUDIO_IO_LOCK:
        return _AUDIO_IO


class PlaybackLevelMeter:
    """把 int16 PCM 转为平滑的 0-1 口型驱动值。"""

    def __init__(self, callback: Callable[[float], None] | None = None):
        self.callback = callback
        self.value = 0.0

    def update(self, pcm: bytes) -> None:
        if self.callback is None or not pcm:
            return
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        if not samples.size:
            return
        rms = float(np.sqrt(np.mean(samples * samples))) / 32768.0
        target = min(1.0, max(0.0, (rms - 0.003) / 0.10))
        # 张口快速跟随人声，闭口也要能捕捉到字词间的短暂停顿。
        alpha = 0.72 if target > self.value else 0.85
        self.value += (target - self.value) * alpha
        try:
            self.callback(self.value)
        except Exception:
            # 视觉反馈绝不能中断音频播放。
            self.callback = None

    def close(self) -> None:
        if self.callback is not None:
            try:
                self.callback(0.0)
            except Exception:
                pass


def _write_metered_pcm(stream, pcm: bytes, meter: PlaybackLevelMeter) -> None:
    """按接近口型刷新周期的小窗播放，避免一个 HTTP 大块只更新一次。

    增益在这里一次性应用：先 clip 再 cast 回 int16，避免溢出爆音；
    meter 必须读到 gain 之后的样本，否则口型会与听感错位。
    gain == 1.0 时整段乘法 + clip 是 no-op，与 commit 02 之前逐字节等价。
    """
    samples_per_frame = max(
        1,
        int(config.TTS_SAMPLE_RATE * _PLAYBACK_LEVEL_FRAME_MS / 1000),
    )
    frame_bytes = samples_per_frame * 2
    gain = get_playback_gain()
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset : offset + frame_bytes]
        if not frame:
            continue
        if gain != 1.0 and len(frame) >= 2:
            usable = len(frame) - len(frame) % 2
            if usable:
                samples = np.frombuffer(frame[:usable], dtype="<i2").astype(np.float32)
                samples = np.clip(samples * gain, -32768, 32767).astype("<i2")
                frame = samples.tobytes() + frame[usable:]
        meter.update(frame)
        stream.write(frame)


# commit 06+ 注:TTS 完整 AEC 需要把播放也走 AudioIO.write_playback,
# 这样 AVAudioEngine 才能拿到 playback reference。当前默认 portaudio 路径
# 仍走 sd.RawOutputStream;commit 07+ 才会把 speak() / StreamingSpeaker 切到
# AudioIO(并加 playback-drained 同步)。


def _aligned_pcm(chunks: Iterator[bytes], cancel: threading.Event | None):
    """合并可能在 int16 样本中间断开的 HTTP 块。"""
    remainder = b""
    for chunk in chunks:
        if cancel is not None and cancel.is_set():
            return
        chunk = remainder + chunk
        aligned = len(chunk) - len(chunk) % 2
        if aligned:
            yield chunk[:aligned]
        remainder = chunk[aligned:]


def _split_mlx_text(text: str) -> list[str]:
    """清除不可朗读的 Markdown，并按完整句拆分请求。"""
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MARKDOWN_PREFIX.sub("", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # 引号本身没有朗读价值；拉长符号会诱导模型持续发声或生成足以被
    # 尾部静音保护误判的内部长停顿。
    text = _ELONGATED_INTERJECTION.sub(r"\1", text)
    text = _TRAILING_LONG_PAUSE.sub("。", text)
    text = _LONG_PAUSE_PUNCTUATION.sub("，", text)
    text = re.sub(r"[“”‘’\"']", "", text)
    sentences = [
        match.group().strip()
        for match in _MLX_SENTENCE.finditer(text)
        if _SPEAKABLE.search(match.group())
    ]
    segments: list[str] = []
    for sentence in sentences:
        # Qwen3-TTS 偶尔会把长句末尾的短逗号分句说得极快甚至近似吞掉。
        # 只拆这种“长前缀 + 很短句尾”，避免恢复成所有逗号都切请求的顿挫感。
        comma = max(sentence.rfind("，"), sentence.rfind(","))
        if comma >= 0:
            prefix = sentence[: comma + 1].strip()
            tail = sentence[comma + 1 :].strip()
            prefix_chars = len(re.sub(r"\s", "", prefix))
            tail_chars = len(re.sub(r"\s", "", tail))
            if (
                prefix_chars >= _SHORT_TAIL_MIN_PREFIX_CHARS
                and 1 <= tail_chars <= _SHORT_TAIL_MAX_CHARS
                and _SPEAKABLE.search(tail)
            ):
                segments.extend((prefix, tail))
                continue
        segments.append(sentence)
    return segments


def _trim_trailing_silence(
    chunks: Iterator[bytes], max_audio_s: float | None = None
) -> Iterator[bytes]:
    """丢弃长尾静音，并限制非静音退化产生的异常超长音频。"""
    pending: list[bytes] = []
    pending_bytes = 0
    received_bytes = 0
    heard_voice = False
    bytes_per_second = config.TTS_SAMPLE_RATE * 2

    for chunk in chunks:
        if max_audio_s is not None:
            remaining = int(max_audio_s * bytes_per_second) - received_bytes
            if remaining <= 0:
                return
            chunk = chunk[: remaining - remaining % 2]
            if not chunk:
                return
        received_bytes += len(chunk)
        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
        rms = (
            float(np.sqrt(np.mean(samples * samples))) / 32768.0
            if samples.size
            else 0.0
        )
        if heard_voice and rms < config.MLX_TTS_SILENCE_RMS:
            pending.append(chunk)
            pending_bytes += len(chunk)
            if pending_bytes / bytes_per_second >= config.MLX_TTS_TRAILING_SILENCE_S:
                return
            continue

        if pending:
            yield from pending
            pending.clear()
            pending_bytes = 0
        if rms >= config.MLX_TTS_SILENCE_RMS:
            heard_voice = True
        yield chunk
        if max_audio_s is not None and received_bytes >= max_audio_s * bytes_per_second:
            return


def _resolve_instruct(instruct: str | None) -> str:
    """非空覆盖 config 默认值；空串或 None 时回退到 persona 默认。"""
    if instruct:
        return instruct
    return config.MLX_TTS_INSTRUCT


def _synthesize_mlx_segment(
    text: str,
    cancel: threading.Event | None,
    client: httpx.Client,
    *,
    instruct: str = "",
) -> Iterator[bytes]:
    """合成一个完整句，并为随机采样退化设置硬上限。

    `instruct` 为本次回答的 TTS 指令（来自 EmotionService.current_tts_instruct）；
    空串时回退到 config.MLX_TTS_INSTRUCT（persona 默认）。
    """
    payload = {
        "model": config.MLX_TTS_MODEL,
        "input": text,
        "lang_code": "Chinese",
        "response_format": "pcm",
        "stream": True,
        "streaming_interval": config.MLX_TTS_STREAMING_INTERVAL,
        "max_tokens": config.MLX_TTS_MAX_TOKENS,
        "temperature": config.MLX_TTS_TEMPERATURE,
        "top_p": config.MLX_TTS_TOP_P,
        "top_k": config.MLX_TTS_TOP_K,
        "repetition_penalty": config.MLX_TTS_REPETITION_PENALTY,
    }
    if config.MLX_TTS_VOICE:
        payload["voice"] = config.MLX_TTS_VOICE
        effective = _resolve_instruct(instruct)
        if effective:
            payload["instruct"] = effective
    else:
        raise RuntimeError("MLX_TTS_VOICE 不能为空；音色克隆后端已移除")
    with client.stream(
        "POST", f"{config.MLX_TTS_URL}/v1/audio/speech", json=payload
    ) as resp:
        resp.raise_for_status()
        aligned = _aligned_pcm(resp.iter_bytes(chunk_size=8192), cancel)
        char_count = len(re.sub(r"\s", "", text))
        max_audio_s = min(
            config.MLX_TTS_MAX_AUDIO_S,
            max(
                config.MLX_TTS_MIN_AUDIO_S,
                char_count * config.MLX_TTS_AUDIO_S_PER_CHAR
                + config.MLX_TTS_AUDIO_PADDING_S,
            ),
        )
        yield from _trim_trailing_silence(aligned, max_audio_s)


def synthesize_mlx_stream(
    text: str,
    cancel: threading.Event | None = None,
    *,
    instruct: str = "",
) -> Iterator[bytes]:
    """调用常驻 MLX Qwen3-TTS，返回 24kHz int16 裸 PCM。

    `instruct` 传入 EmotionService.current_tts_instruct() 结果；
    一段回答内不应变化，避免同一句话中途换语气。
    """
    segments = _split_mlx_text(text)
    if not segments:
        return
    # 一轮播报内复用本地 HTTP 连接，同时保留逐句模型请求和异常隔离。
    with httpx.Client(timeout=config.REQUEST_TIMEOUT) as client:
        for segment in segments:
            if cancel is not None and cancel.is_set():
                return
            yield from _synthesize_mlx_segment(
                segment, cancel, client, instruct=instruct
            )


def synthesize_stream(
    text: str,
    cancel: threading.Event | None = None,
    *,
    instruct: str = "",
) -> Iterator[bytes]:
    yield from synthesize_mlx_stream(text, cancel, instruct=instruct)


def speak(
    text: str,
    cancel: threading.Event | None = None,
    on_audio_level: Callable[[float], None] | None = None,
    *,
    instruct: str = "",
) -> None:
    """整段合成并播放:服务端按段流式返回 PCM,边收边播。

    与按句流水线相比,整段送合成能保留跨句韵律,听感更连贯流畅。
    `instruct` 同 synthesize_stream：空串时回退 persona 默认。

    commit 07+ fix:优先走 AudioIO(让 macos_voice 后端拿到 playback
    reference,完整 AEC)。无 AudioIO 时 fallback 到 sd.RawOutputStream。
    """
    if config.TTS_BACKEND == "macos":
        if on_audio_level is not None:
            on_audio_level(0.55)
        try:
            _speak_macos(text, cancel)
        finally:
            if on_audio_level is not None:
                on_audio_level(0.0)
        return

    audio_io = get_audio_io()
    meter = PlaybackLevelMeter(on_audio_level)
    try:
        if audio_io is not None:
            # commit 07+ fix:走 AudioIO,让 macos_voice/AVAudioEngine
            # 拿到 playback reference,完整 AEC
            for pcm in synthesize_stream(text, cancel, instruct=instruct):
                if cancel is not None and cancel.is_set():
                    audio_io.flush_playback()
                    return
                if pcm:
                    pcm = apply_playback_gain(pcm)
                    meter.update(pcm)
                    audio_io.write_playback(pcm, config.TTS_SAMPLE_RATE)
            # 等扬声器真正播完(最长 15s)
            audio_io.wait_playback_drained(timeout=15.0)
            return
        # fallback:无 AudioIO 时走 sd.RawOutputStream
        with sd.RawOutputStream(
            samplerate=config.TTS_SAMPLE_RATE, dtype="int16", channels=1
        ) as stream:
            for pcm in synthesize_stream(text, cancel, instruct=instruct):
                if pcm:
                    _write_metered_pcm(stream, pcm, meter)
    finally:
        meter.close()


def _speak_macos(text: str, cancel: threading.Event | None = None) -> None:
    """使用系统音色播报，并允许插话事件终止当前进程。"""
    process = subprocess.Popen(
        [
            "say",
            "-v",
            config.MACOS_VOICE,
            "-r",
            str(config.MACOS_SPEECH_RATE),
            text,
        ]
    )
    while process.poll() is None:
        if cancel is not None and cancel.wait(0.05):
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
            return


class StreamingSpeaker:
    """流式播报器:say() 送入句子,后台线程合成并按序播放。

    用作上下文管理器,退出时等待队列里剩余的句子播报完毕。
    `instruct` 在构造时锁住，整段回答共享同一份 TTS 指令，
    避免同一句话念到一半突然换语气。

    commit 07+:如果显式传 ``audio_io``(或通过 ``tts.set_audio_io``
    模块级注册),``_play_loop`` 会改走 ``audio_io.write_playback``,
    让 macos_voice 后端拿到 playback reference —— 完整 AEC。
    """

    def __init__(
        self,
        cancel: threading.Event | None = None,
        on_audio_level: Callable[[float], None] | None = None,
        *,
        instruct: str = "",
        audio_io=None,
    ):
        self._cancel = cancel or threading.Event()
        self._on_audio_level = on_audio_level
        self._instruct = instruct
        self._audio_io = audio_io if audio_io is not None else get_audio_io()
        self._sent_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue(maxsize=8)
        target = self._macos_loop if config.TTS_BACKEND == "macos" else self._synth_loop
        self._synth_t = threading.Thread(target=target, daemon=True)
        self._play_t = (
            None
            if config.TTS_BACKEND == "macos"
            else threading.Thread(target=self._play_loop, daemon=True)
        )

    def __enter__(self):
        self._synth_t.start()
        if self._play_t is not None:
            self._play_t.start()
        return self

    def __exit__(self, *exc):
        self._sent_q.put(_SENTINEL)
        if self._cancel.is_set():
            self._drain(self._sent_q)
            self._sent_q.put(_SENTINEL)
        join_timeout = 2 if self._cancel.is_set() else None
        self._synth_t.join(timeout=join_timeout)
        if self._play_t is not None:
            self._play_t.join(timeout=join_timeout)
        return False

    def say(self, sentence: str):
        if not self._cancel.is_set():
            self._sent_q.put(sentence)

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        with q.mutex:
            q.queue.clear()

    def _put_audio(self, item: bytes | object) -> bool:
        while not self._cancel.is_set():
            try:
                self._audio_q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _synth_loop(self):
        while True:
            s = self._sent_q.get()
            if s is _SENTINEL:
                break
            try:
                for pcm in synthesize_stream(
                    s, self._cancel, instruct=self._instruct
                ):
                    if not self._put_audio(pcm):
                        break
            except Exception as e:
                if not self._cancel.is_set():
                    print(f"(TTS 合成失败: {e})")
            if self._cancel.is_set():
                break
        if self._cancel.is_set():
            self._drain(self._audio_q)
            try:
                self._audio_q.put_nowait(_SENTINEL)
            except queue.Full:
                pass
        else:
            self._audio_q.put(_SENTINEL)

    def _play_loop(self):
        meter = PlaybackLevelMeter(self._on_audio_level)
        try:
            if self._audio_io is not None:
                # commit 07+:把 PCM 推到 AudioIO(由它决定走 macos_voice
                # 的 playerNode / portaudio 的 sd.RawOutputStream 长连接)。
                # 这样 macos_voice 后端 AVAudioEngine 能拿到 playback
                # reference,voice-processing 才能完整 echo cancellation。
                while not self._cancel.is_set():
                    pcm = self._audio_q.get()
                    if pcm is _SENTINEL:
                        break
                    if pcm:
                        meter.update(pcm)
                        try:
                            self._audio_io.write_playback(
                                pcm, config.TTS_SAMPLE_RATE
                            )
                        except Exception as e:
                            if not self._cancel.is_set():
                                print(f"(TTS 播放失败: {e})")
                            break
            else:
                # 兼容路径:非 duplex(默认半双工/插话模式)仍自己开
                # sd.RawOutputStream,跟 commit 02+ 行为一致。
                with sd.RawOutputStream(
                    samplerate=config.TTS_SAMPLE_RATE, dtype="int16", channels=1
                ) as stream:
                    while not self._cancel.is_set():
                        pcm = self._audio_q.get()
                        if pcm is _SENTINEL:
                            break
                        if pcm:
                            _write_metered_pcm(stream, pcm, meter)
        except Exception as e:
            if not self._cancel.is_set():
                print(f"(TTS 播放失败: {e})")
        finally:
            meter.close()

    def _macos_loop(self) -> None:
        """使用 macOS 系统音色直接播放。"""
        while not self._cancel.is_set():
            sentence = self._sent_q.get()
            if sentence is _SENTINEL:
                break
            if self._on_audio_level is not None:
                self._on_audio_level(0.55)
            try:
                _speak_macos(sentence, self._cancel)
            finally:
                if self._on_audio_level is not None:
                    self._on_audio_level(0.0)
