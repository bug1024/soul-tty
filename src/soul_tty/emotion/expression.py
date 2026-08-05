"""Expression 服务：Soul 对用户的表达方式，独立于 Mood。

V1 只输出 `expression` 字符串（neutral / caring），其它表达字段
（voice_style / avatar_expression）暂不扩展。
"""

from __future__ import annotations

from .state import EmotionVector

EXPRESSIONS = ("neutral", "caring")
_ALLOWED_EXPRESSIONS = set(EXPRESSIONS)


def resolve_expression(emotion: EmotionVector, hint: str = "") -> str:
    """根据 LLM 输出的 hint 字段解析 expression；hint 非法时回退 neutral。

    兼容旧调用方式；ExpressionService.resolve() 的轻量包装。
    """
    return ExpressionService().resolve(emotion, hint)


class ExpressionService:
    """无状态服务：只负责把 hint 收敛到合法值并保持线程安全。

    之所以不是 dataclass：当前没有需要持久化的字段；后续若要加
    `voice_style` / `avatar_expression`，再升级为带状态的 dataclass。
    """

    def resolve(self, emotion: EmotionVector, hint: str = "") -> str:
        """收敛 expression；hint 非法或缺失一律回退 neutral。"""
        normalized = (hint or "").strip().lower()
        if normalized in _ALLOWED_EXPRESSIONS:
            return normalized
        return "neutral"
