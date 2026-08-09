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


def normalize_speech_text(text: str) -> str:
    """移除标点与空白，得到适合回声比较的紧凑文本。"""
    return _NON_SPEECH_TEXT.sub("", text.lower())


def _best_window_similarity(heard: str, spoken: str) -> float:
    """在整段播放文本中寻找与 ASR 结果最相似的局部窗口。

    回声不一定来自播放尾部，ASR 也经常产生一两个同音错字；只比较播放
    transcript 的尾部会漏掉长回答中间位置的残余回声。
    """
    if not heard or not spoken:
        return 0.0
    tolerance = max(2, len(heard) // 4)
    min_size = max(3, len(heard) - tolerance)
    max_size = min(len(spoken), len(heard) + tolerance)
    if min_size > max_size:
        return difflib.SequenceMatcher(None, heard, spoken).ratio()
    best = 0.0
    for size in range(min_size, max_size + 1):
        for start in range(0, len(spoken) - size + 1):
            ratio = difflib.SequenceMatcher(
                None, heard, spoken[start : start + size]
            ).ratio()
            if ratio > best:
                best = ratio
    return best


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
    heard_n = normalize_speech_text(heard)
    spoken_n = normalize_speech_text(spoken)
    if not heard_n or not spoken_n:
        return False
    # 两个汉字的精确片段也可以安全识别为回声；单字信息量太低。
    if len(heard_n) >= 2 and heard_n in spoken_n:
        return True
    if len(heard_n) < 3:
        return False
    return _best_window_similarity(heard_n, spoken_n) >= similarity


__all__ = ["is_probable_echo", "normalize_speech_text", "_NON_SPEECH_TEXT"]
