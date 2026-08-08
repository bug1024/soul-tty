"""PlaybackTranscript:跟踪 TTS 实际送出的文本片段。

commit 09 引入。FloorManager 用它做回声判定(比单纯累加 ``agent_text``
更接近"扬声器实际播放的内容"):只有真正送进 TTS 的文本才视为 echo source。

设计要点:
- 单线程写(_answer → _print_answer → speaker.say);多线程读(FloorManager)。
- ``played_text()`` 是累加的副本,不在调用方缓存。
- 打断(cancel)后,``clear()`` 不自动调用:让上层决定何时重置。
"""

from __future__ import annotations

import threading


class PlaybackTranscript:
    """TTS 播放 transcript —— 累加所有送进 TTS 的文本片段。"""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        """记录一段文本已送 TTS。空字符串也保留(标记分句位置)。"""
        with self._lock:
            if text:
                self._chunks.append(text)

    def played_text(self) -> str:
        """当前已记录的完整文本(回声判定 / 用户回看 用)。"""
        with self._lock:
            return "".join(self._chunks)

    def clear(self) -> None:
        """新一轮开始时清空。"""
        with self._lock:
            self._chunks.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)