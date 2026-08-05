"""从 LLM 输出中提取并清洗 emotion_delta。"""

from __future__ import annotations

from .state import DIMENSIONS


def parse_emotion_delta(
    raw: object,
    delta_cap: float,
) -> dict[str, float]:
    """清洗 LLM 输出的 emotion_delta：剔除未知维度、夹到 [-delta_cap, +delta_cap]。"""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for dim in DIMENSIONS:
        value = raw.get(dim)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        out[dim] = max(-delta_cap, min(delta_cap, number))
    return out