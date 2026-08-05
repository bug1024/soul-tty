"""实时情绪系统：五维情绪值、Mood/Expression 解析、Prompt 注入。"""

from .service import EmotionService, EmotionSnapshot

__all__ = ["EmotionService", "EmotionSnapshot"]