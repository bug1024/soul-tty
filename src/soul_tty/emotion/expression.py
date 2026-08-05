"""Expression 解析：Soul 对用户的表达方式（独立于 Mood）。"""

from __future__ import annotations

from .state import EmotionVector

EXPRESSIONS = ("neutral", "caring")

_ALLOWED_EXPRESSIONS = set(EXPRESSIONS)


def resolve_expression(emotion: EmotionVector, hint: str = "") -> str:
    """根据 LLM 输出的 hint 字段解析 expression；hint 非法时回退 neutral。

    V1 规则：完全信任 hint 字段。hint 由 InteractionAnalyzer 在同一份输出里给出。
    """
    hint = (hint or "").strip().lower()
    if hint in _ALLOWED_EXPRESSIONS:
        return hint
    return "neutral"