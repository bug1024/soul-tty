"""实时情绪系统：五维情绪值、Mood/Expression 解析、Prompt 注入。"""

from .state import EmotionVector, load_emotion_state, save_emotion_state, load_runtime, save_runtime

__all__ = [
    "EmotionVector",
    "load_emotion_state",
    "save_emotion_state",
    "load_runtime",
    "save_runtime",
]