"""Mood + Expression → Emotion Context 文本段。"""

from __future__ import annotations


def _intensity_word(intensity: float) -> str:
    if intensity >= 0.8:
        return "明显"
    if intensity >= 0.5:
        return "中等"
    return ""


_MOOD_TEMPLATES: dict[str, tuple[str, str]] = {
    # state sentence, behavior lines
    "numb": (
        "你处于麻木低能状态。",
        "用词平淡简洁，避免夸张情绪；不要刻意表演感受；保持低调陪伴。",
    ),
    "tired": (
        "你有些疲惫。",
        "语速放缓，少用感叹号；用更短的句子；允许表达轻微倦意。",
    ),
    "sad": (
        "你正在体会低落的情绪。",
        "语气沉稳共情，避免强行转欢乐；多倾听少建议；用温和短句。",
    ),
    "excited": (
        "你正处于兴奋状态。",
        "语气上扬有活力，可以有惊喜感；保持真诚，不要过度夸张。",
    ),
    "curious": (
        "你对当前话题充满好奇。",
        "多提问、主动延伸；语气探索感强；可以鼓励用户继续讲。",
    ),
    "happy": (
        "你处于愉悦状态。",
        "语气积极温暖，可以表达开心；保持自然，不过度兴奋。",
    ),
    "calm": (
        "你处于放松平稳的状态。",
        "语气柔和、节奏平缓；保持陪伴感；不强行制造话题。",
    ),
}


_EXPRESSION_LINES: dict[str, str] = {
    "caring": "表达方式偏向温柔关切，主动关心用户的当下感受。",
}


def build_emotion_context(
    mood: str,
    intensity: float,
    *,
    expression: str = "neutral",
) -> str:
    """生成 Emotion Context 文本段，追加到 system_prompt 末尾。

    不暴露原始数值；intensity 仅控制修饰词强度。
    """
    state, behavior = _MOOD_TEMPLATES.get(
        mood,
        ("你保持稳定的语气。", "按当前对话节奏自然回应。"),
    )
    intensity_word = _intensity_word(intensity)
    if intensity_word and state.endswith("状态。"):
        state = state[:-3] + f"状态，{intensity_word}。"
    elif intensity_word and state.endswith("情绪。"):
        state = state[:-3] + f"情绪，{intensity_word}。"
    expression_line = _EXPRESSION_LINES.get(expression, "")

    lines = [
        "当前情绪状态：",
        state,
        "行为倾向：",
        f"- {behavior}",
    ]
    if expression_line:
        lines.append(f"- {expression_line}")
    return "\n".join(lines)