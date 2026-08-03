"""进程内 sherpa-onnx 流式语音识别。"""

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import webrtcvad

from .. import config


@dataclass(frozen=True)
class TranscriptUpdate:
    """在线识别更新。只有 final 文本可以送进 LLM。"""

    text: str
    final: bool


def _pcm_samples(pcm: bytes) -> np.ndarray:
    """int16 little-endian PCM 转 sherpa 需要的 [-1, 1] float32。"""
    usable = len(pcm) - len(pcm) % 2
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0


def _load_sherpa_recognizer():
    try:
        import sherpa_onnx
    except ImportError as e:
        raise RuntimeError(
            "sherpa-onnx 运行时加载失败，请在 soul-tty 目录执行 uv sync"
        ) from e

    model_dir = Path(config.SHERPA_MODEL_DIR).expanduser()
    tokens = model_dir / "tokens.txt"
    encoder = model_dir / "encoder.int8.onnx"
    decoder = model_dir / "decoder.int8.onnx"
    missing = [str(path) for path in (tokens, encoder, decoder) if not path.is_file()]
    if missing:
        raise RuntimeError(f"sherpa 模型文件不存在: {', '.join(missing)}")

    return sherpa_onnx.OnlineRecognizer.from_paraformer(
        tokens=str(tokens),
        encoder=str(encoder),
        decoder=str(decoder),
        num_threads=config.SHERPA_NUM_THREADS,
        provider="cpu",
        enable_endpoint_detection=True,
        # 空白噪声段用较宽松的 rule1；已有文字后用 rule2 快速提交。
        rule1_min_trailing_silence=1.2,
        rule2_min_trailing_silence=config.SHERPA_ENDPOINT_SILENCE_S,
        rule3_min_utterance_length=config.MAX_UTTERANCE_S,
    )


_sherpa_recognizer = None
_sherpa_lock = threading.Lock()


def get_sherpa_recognizer():
    """惰性加载并常驻；启动后的每句话不再承担模型加载成本。"""
    global _sherpa_recognizer
    if _sherpa_recognizer is None:
        with _sherpa_lock:
            if _sherpa_recognizer is None:
                _sherpa_recognizer = _load_sherpa_recognizer()
    return _sherpa_recognizer


class SherpaStream:
    """一个持续接收麦克风帧的在线识别会话。"""

    def __init__(self, recognizer=None):
        self.recognizer = recognizer or get_sherpa_recognizer()
        self.stream = self.recognizer.create_stream()
        self._last_partial = ""
        self.last_endpoint = False

    def reset(self) -> None:
        # 新建流比保留前一句 encoder 状态更稳妥，也能彻底丢掉播放尾音。
        self.stream = self.recognizer.create_stream()
        self._last_partial = ""
        self.last_endpoint = False

    def accept(self, pcm: bytes) -> list[TranscriptUpdate]:
        self.last_endpoint = False
        samples = _pcm_samples(pcm)
        if samples.size == 0:
            return []
        self.stream.accept_waveform(config.SAMPLE_RATE, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        text = self.recognizer.get_result(self.stream).strip()
        endpoint = self.recognizer.is_endpoint(self.stream)
        updates: list[TranscriptUpdate] = []
        if text and text != self._last_partial:
            self._last_partial = text
            if not endpoint:
                updates.append(TranscriptUpdate(text, final=False))
        if endpoint:
            if text:
                updates.append(TranscriptUpdate(text, final=True))
            self.reset()
        self.last_endpoint = endpoint
        return updates

    def finish(self) -> str:
        """结束文件/固定 PCM 输入，并冲刷模型剩余上下文。"""
        tail = np.zeros(int(config.SAMPLE_RATE * 0.8), dtype=np.float32)
        self.stream.accept_waveform(config.SAMPLE_RATE, tail)
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream).strip()
        self.reset()
        return text


class VadGatedSherpaStream:
    """只在检测到人声后唤醒 Sherpa，同时保留句首与 endpoint 静音。"""

    def __init__(
        self,
        session: SherpaStream | None = None,
        vad=None,
        *,
        pre_roll_ms: int | None = None,
        trigger_ms: int | None = None,
    ) -> None:
        self.session = session or SherpaStream()
        self.vad = vad or webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        pre_roll_ms = (
            config.SHERPA_VAD_PRE_ROLL_MS
            if pre_roll_ms is None
            else pre_roll_ms
        )
        trigger_ms = (
            config.SHERPA_VAD_TRIGGER_MS if trigger_ms is None else trigger_ms
        )
        self._pre_roll: deque[bytes] = deque(
            maxlen=max(1, pre_roll_ms // config.FRAME_MS)
        )
        self._trigger_frames = max(1, trigger_ms // config.FRAME_MS)
        self._speech_frames = 0
        self.active = False

    def reset(self) -> None:
        self.session.reset()
        self._pre_roll.clear()
        self._speech_frames = 0
        self.active = False

    def accept(self, pcm: bytes) -> list[TranscriptUpdate]:
        if self.active:
            updates = self.session.accept(pcm)
            if self.session.last_endpoint:
                self._pre_roll.clear()
                self._speech_frames = 0
                self.active = False
            return updates

        self._pre_roll.append(pcm)
        try:
            speech = self.vad.is_speech(pcm, config.SAMPLE_RATE)
        except ValueError:
            # 非标准尾帧只会出现在固定文件输入；实时流应始终是 30ms。
            speech = False
        self._speech_frames = self._speech_frames + 1 if speech else 0
        if self._speech_frames < self._trigger_frames:
            return []

        self.active = True
        buffered = b"".join(self._pre_roll)
        self._pre_roll.clear()
        updates = self.session.accept(buffered)
        if self.session.last_endpoint:
            self.active = False
            self._speech_frames = 0
        return updates


def transcribe(pcm: bytes) -> str:
    """识别固定 PCM；麦克风实时模式应直接使用 SherpaStream。"""
    session = SherpaStream()
    # 分块模拟真实在线输入，避免文件路径和麦克风走出两套行为。
    frame_bytes = config.SAMPLE_RATE * config.FRAME_MS // 1000 * 2
    final_parts: list[str] = []
    for offset in range(0, len(pcm), frame_bytes):
        for update in session.accept(pcm[offset : offset + frame_bytes]):
            if update.final:
                final_parts.append(update.text)
    tail = session.finish()
    if tail:
        final_parts.append(tail)
    return "".join(final_parts)
