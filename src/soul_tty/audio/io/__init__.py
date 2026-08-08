"""音频 I/O 后端抽象层。

commit 03 引入 ``AudioIO`` Protocol,把"采集 + 播放"从 ``Mic`` /
``StreamingSpeaker`` 里抽出来,留出未来 macOS voice-processing AEC 后端
的接入点。当前默认 backend 仍是 PortAudio,行为逐字节等价。
"""

from __future__ import annotations

from ... import config
from .base import AudioIO
from .macos_voice import MacOSVoiceIO
from .portaudio import PortAudioIO


def get_audio_io(backend: str | None = None) -> AudioIO:
    """根据 ``AUDIO_IO_BACKEND`` 配置构造 AudioIO 实例。

    未知 backend 抛 ``ValueError``;``macos_voice`` 当前是 stub,
    构造时立即抛 ``NotImplementedError``,把错误前移到 import / 启动阶段。
    """
    backend = backend or config.AUDIO_IO_BACKEND
    if backend == "portaudio":
        return PortAudioIO()
    if backend == "macos_voice":
        return MacOSVoiceIO()
    raise ValueError(f"unknown AUDIO_IO_BACKEND: {backend!r}")


__all__ = ["AudioIO", "PortAudioIO", "MacOSVoiceIO", "get_audio_io"]