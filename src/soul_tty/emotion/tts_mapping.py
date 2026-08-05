"""Mood + Expression → MLX_TTS_INSTRUCT 文案。"""

from __future__ import annotations

_MOOD_INSTRUCT: dict[str, str] = {
    "happy": "用开心上扬的语气说",
    "excited": "用兴奋激动的语气说",
    "sad": "用低沉平缓的语气说",
    "tired": "用轻柔缓慢的语气说",
    "curious": "用好奇询问的语气说",
    "numb": "用平淡低能量的语气说",
}

_EXPRESSION_OVERRIDE: dict[str, str] = {
    "caring": "用温柔关切的语气说",
}


def build_tts_instruct(
    mood: str,
    intensity: float,
    *,
    expression: str = "neutral",
) -> str:
    """expression 优先于 mood；calm + neutral 返回空。"""
    if expression == "caring":
        return _EXPRESSION_OVERRIDE["caring"]
    return _MOOD_INSTRUCT.get(mood, "")