"""五维情绪值 → (mood, intensity)。"""

from __future__ import annotations

from .state import EmotionVector

MOODS = ("numb", "tired", "sad", "excited", "curious", "happy", "calm")


def _avg(*values: float) -> float:
    return sum(values) / len(values)


def resolve_mood(emotion: EmotionVector) -> tuple[str, float]:
    """按优先级返回 (mood, intensity)；未命中则返回 ("calm", calmness)。"""
    h, c, q, s, e = (
        emotion.happiness,
        emotion.calmness,
        emotion.curiosity,
        emotion.stress,
        emotion.energy,
    )

    if s >= 0.75 and e <= 0.25:
        return "numb", _avg(s, 1 - e)
    if e <= 0.35:
        return "tired", 1 - e
    if h <= 0.35 and s >= 0.45:
        return "sad", _avg(1 - h, s)
    if h >= 0.75 and e >= 0.75:
        return "excited", _avg(h, e)
    if q >= 0.7:
        return "curious", q
    if h >= 0.65 and e >= 0.4:
        return "happy", h
    return "calm", c