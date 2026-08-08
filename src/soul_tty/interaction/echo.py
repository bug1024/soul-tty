"""纯文本回声判定 —— 不依赖音频/ASR,可在任何模块使用。

被 ``conversation.py`` 与 ``interaction/floor.py`` 复用,避免 floor 模块
引入 ``soul_tty.conversation`` 时把 webrtcvad / sherpa / sounddevice
一起拉进来,导致测试在没装音频依赖的环境里 import 就炸。
"""

from __future__ import annotations

import difflib
import re

# 句子结束符 / 切句送 TTS 也在 conversation.py 里复用同一份 regex,
# 但这里只暴露与回声判定相关的常量。
_NON_SPEECH_TEXT = re.compile(r"[^0-9a-z一-鿿]+")


def is_probable_echo(
    heard: str, spoken: str, similarity: float = 0.72
) -> bool:
    """判断 ``heard`` 是否更像扬声器回声而非用户插话。

    Args:
        heard: ASR 这次听到的文本。
        spoken: agent 已播放的累计文本。
        similarity: 序列相似度阈值（0~1）。

    Returns:
        True → 当作回声过滤掉；False → 可能是真插话。
    """
    heard_n = _NON_SPEECH_TEXT.sub("", heard.lower())
    spoken_n = _NON_SPEECH_TEXT.sub("", spoken.lower())
    if len(heard_n) < 3 or not spoken_n:
        return False
    if heard_n in spoken_n:
        return True
    window = spoken_n[-max(len(heard_n) * 2, 24):]
    return difflib.SequenceMatcher(None, heard_n, window).ratio() >= similarity


__all__ = ["is_probable_echo", "_NON_SPEECH_TEXT"]