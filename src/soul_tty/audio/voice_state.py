"""异步声音感知旁路：SenseVoiceSmall 离线分析用户语气/情绪/声学事件。

设计原则：
- 不替换 Streaming Paraformer，两路独立
- 不阻塞主对话，submit(pcm) < 1~5ms
- 结果作为弱证据供 Reflection 消费，不直接修改 Emotion/Bond/Memory
- VoiceObservation 不是用户真实心理状态，只是"这句话听起来像什么"
- 模型懒加载，默认关闭 VOICE_STATE_ENABLED=0
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

# 内部 emotion 标签 → 规范化名
_EMOTION_MAP: dict[str, str] = {
    "<|HAPPY|>": "happy",
    "<|SAD|>": "sad",
    "<|ANGRY|>": "angry",
    "<|NEUTRAL|>": "neutral",
    "<|SURPRISE|>": "surprise",
    "<|FEAR|>": "fear",
    "<|DISGUST|>": "disgust",
}

# 内部 event 标签 → 规范化名
_EVENT_MAP: dict[str, str] = {
    "<|Speech|>": "speech",
    "<|Laughter|>": "laughter",
    "<|Cry|>": "crying",
    "<|Cough|>": "cough",
    "<|Sneeze|>": "sneeze",
    "<|Applause|>": "applause",
}

# 实际能测到的 event 标签集可能更少，但保留完整映射不做硬截断。


@dataclass(frozen=True)
class VoiceObservation:
    """一段声音的感知结果。

    此数据是"听起来像什么"，不是"用户实际是什么"。
    """

    emotion: str  # happy / sad / angry / neutral / unknown
    event: str  # speech / laughter / crying / cough / ...
    language: str  # zh / en / ja / ko / yue / ...
    duration_ms: int


def _normalize_emotion(raw: str) -> str:
    return _EMOTION_MAP.get(raw.strip(), "unknown")


def _normalize_event(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned in _EVENT_MAP:
        return _EVENT_MAP[cleaned]
    # 尝试去掉 < > 再查一次
    no_angle = cleaned.strip("<>")
    if no_angle in _EVENT_MAP:
        return _EVENT_MAP[no_angle]
    return no_angle.strip("|") or "unknown"


def _normalize_lang(raw: str) -> str:
    cleaned = raw.strip().strip("<>")
    return cleaned.strip("|") or "unknown"


# ── SenseVoice 加载与解码 ──────────────────────────────────────────────


def _load_recognizer():
    try:
        import sherpa_onnx
    except ImportError as e:
        raise RuntimeError(
            "sherpa-onnx 运行时加载失败，请在 soul-tty 目录执行 uv sync"
        ) from e

    model_dir = Path(config.SENSEVOICE_MODEL_DIR).expanduser()
    # 兼容 ModelScope 版本（model_quant.onnx + tokens.json）
    # 和官方 sherpa-onnx 版本（model.int8.onnx + tokens.txt）
    model_path = model_dir / "model_quant.onnx"
    if not model_path.is_file():
        model_path = model_dir / "model.int8.onnx"
    tokens = model_dir / "tokens.json"
    if not tokens.is_file():
        tokens = model_dir / "tokens.txt"
    missing = [
        str(path) for path in (model_path, tokens) if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"SenseVoice 模型文件不存在: {', '.join(missing)}"
        )

    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_path),
        tokens=str(tokens),
        num_threads=config.SENSEVOICE_NUM_THREADS,
        provider=config.SENSEVOICE_PROVIDER,
        language="auto",
        use_itn=False,
    )


def _decode(pcm: bytes, recognizer) -> VoiceObservation | None:
    """对一段完整 PCM 做一次 SenseVoice 推理，返回 VoiceObservation。

    返回 None 表示解码失败或输入太短。
    """
    import sherpa_onnx

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        return None
    duration_ms = int(len(samples) / 16)  # 16kHz → ms

    try:
        stream = recognizer.create_stream()
        stream.accept_waveform(config.SAMPLE_RATE, samples)
        recognizer.decode_stream(stream)
        result: sherpa_onnx.OfflineRecognizerResult = stream.result
    except Exception:
        logger.warning("SenseVoice decode 异常", exc_info=True)
        return None

    emotion = _normalize_emotion(result.emotion) if result.emotion else "unknown"
    event = _normalize_event(result.event) if result.event else "unknown"
    language = _normalize_lang(result.lang) if result.lang else "unknown"
    return VoiceObservation(emotion=emotion, event=event, language=language, duration_ms=duration_ms)


# ── VoiceStateService ──────────────────────────────────────────────────


VoiceRef = int


class VoiceStateService:
    """异步声音感知服务。

    用法：
        service = VoiceStateService()
        ref = service.submit(pcm)          # < 1~5ms，不阻塞
        obs = service.get(ref)              # 结果就绪则返回，否则 None
        service.close()
    """

    def __init__(self) -> None:
        self._queue: Queue[tuple[VoiceRef, bytes]] = Queue(
            maxsize=config.VOICE_STATE_QUEUE_SIZE
        )
        self._cache: OrderedDict[VoiceRef, VoiceObservation] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._ref_counter = 0
        self._ref_lock = threading.Lock()
        self._recognizer = None
        self._recognizer_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            name="soul-tty-voice-state",
            daemon=True,
        )
        self._worker.start()

    # ── 公开 API ────────────────────────────────────────────────────

    def submit(self, pcm: bytes) -> VoiceRef | None:
        """提交一段完整的 utterance PCM 做异步分析。

        返回 VoiceRef，可用于后续查询结果。队列满时丢弃最旧任务。
        语音太短（< VOICE_STATE_MIN_UTTERANCE_MS）时返回 None。
        """
        if not pcm or not config.VOICE_STATE_ENABLED:
            return None
        duration_ms = int(len(pcm) / 16)
        if duration_ms < config.VOICE_STATE_MIN_UTTERANCE_MS:
            return None
        with self._ref_lock:
            self._ref_counter += 1
            ref = self._ref_counter
        try:
            self._queue.put_nowait((ref, pcm))
        except Full:
            # 队列满：丢最旧任务
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((ref, pcm))
            except (Full, Empty):
                pass
        return ref

    def get(self, ref: VoiceRef) -> VoiceObservation | None:
        """查询指定 ref 的观察结果。结果就绪则返回，否则 None。"""
        with self._cache_lock:
            return self._cache.get(ref)

    def get_many(self, refs: tuple[VoiceRef | None, ...]) -> list[VoiceObservation]:
        """批量查询多个 ref，跳过 None 和未就绪的。"""
        with self._cache_lock:
            return [
                self._cache[ref]
                for ref in refs
                if ref is not None and ref in self._cache
            ]

    def latest(self) -> VoiceObservation | None:
        """返回最近一次完成的观察结果。"""
        with self._cache_lock:
            if not self._cache:
                return None
            return next(reversed(self._cache.values()))

    def close(self) -> None:
        self._stop.set()
        # 塞一个空任务让 worker 退出等待
        try:
            self._queue.put_nowait((0, b""))
        except Full:
            pass

    # ── 内部 ────────────────────────────────────────────────────────

    def _get_recognizer(self):
        """懒加载。第一次 submit 后才 load，不阻塞启动。"""
        if self._recognizer is None:
            with self._recognizer_lock:
                if self._recognizer is None:
                    self._recognizer = _load_recognizer()
        return self._recognizer

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ref, pcm = self._queue.get(timeout=1.0)
            except Exception:
                continue
            if self._stop.is_set():
                return
            try:
                recognizer = self._get_recognizer()
            except Exception as exc:
                logger.warning("SenseVoice 加载失败: %s", exc)
                continue
            obs = _decode(pcm, recognizer)
            if obs is None:
                continue
            with self._cache_lock:
                self._cache[ref] = obs
                # TTL 清理
                self._evict_old()

    def _evict_old(self) -> None:
        """移除超出 TTL 的缓存条目，最多保留 2x 队列大小。"""
        max_cache = config.VOICE_STATE_QUEUE_SIZE * 2
        while len(self._cache) > max_cache:
            self._cache.popitem(last=False)