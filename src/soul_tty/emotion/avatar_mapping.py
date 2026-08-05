"""Mood + Expression → avatar_expression 数据结构（V1 不接 renderer）。"""

from __future__ import annotations

_MOOD_AVATAR: dict[str, dict[str, str]] = {
    "happy": {"face": "smile", "eye": "open", "motion": "slight_nod"},
    "excited": {"face": "bright", "eye": "open", "motion": "bounce"},
    "sad": {"face": "droop", "eye": "half", "motion": "none"},
    "tired": {"face": "flat", "eye": "half", "motion": "none"},
    "curious": {"face": "neutral", "eye": "wide", "motion": "tilt_head"},
    "numb": {"face": "flat", "eye": "half", "motion": "none"},
    "calm": {"face": "neutral", "eye": "open", "motion": "none"},
}

_EXPRESSION_MOTION_OVERRIDE: dict[str, str] = {
    "caring": "slight_lean",
}

_DEFAULT_AVATAR = {"face": "neutral", "eye": "open", "motion": "none"}


def build_avatar_expression(
    mood: str,
    intensity: float,
    *,
    expression: str = "neutral",
) -> dict[str, str]:
    """返回 avatar_expression 数据；V1 不接入 renderer，仅作为接口。"""
    base = dict(_MOOD_AVATAR.get(mood, _DEFAULT_AVATAR))
    motion_override = _EXPRESSION_MOTION_OVERRIDE.get(expression)
    if motion_override is not None:
        base["motion"] = motion_override
    return base