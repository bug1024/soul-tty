"""情绪值更新算法：EMA 平滑 + idle decay + 启动扰动。"""

from __future__ import annotations

import random

from .state import DIMENSIONS, DEFAULT_BASELINE, EmotionVector


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def apply_delta(
    old: EmotionVector,
    delta: dict[str, float],
    rate: float,
    delta_cap: float = 0.3,
) -> EmotionVector:
    """对每一维执行 EMA 平滑趋向 clamp(old+delta, 0, 1)。

    target = clamp(old + delta_capped, 0, 1)
    new = old + (target - old) * rate
    """
    new_values: dict[str, float] = {}
    for dim in DIMENSIONS:
        old_v = getattr(old, dim)
        raw = float(delta.get(dim, 0.0) or 0.0)
        capped = max(-delta_cap, min(delta_cap, raw))
        target = _clamp_unit(old_v + capped)
        new_values[dim] = old_v + (target - old_v) * rate
    return EmotionVector(**new_values)


def apply_decay(
    current: EmotionVector,
    baseline: EmotionVector,
    rate: float,
) -> EmotionVector:
    """idle 衰减：每维向 baseline 回归 rate 步长。"""
    new_values: dict[str, float] = {}
    for dim in DIMENSIONS:
        cur_v = getattr(current, dim)
        base_v = getattr(baseline, dim)
        new_values[dim] = cur_v + (base_v - cur_v) * rate
    return EmotionVector(**new_values)


def perturb_baseline(
    baseline: EmotionVector,
    jitter: float = 0.1,
    seed: int | None = None,
) -> EmotionVector:
    """给每个维度叠加 ±jitter 的随机扰动并 clamp。"""
    rng = random.Random(seed)
    new_values: dict[str, float] = {}
    for dim in DIMENSIONS:
        base_v = getattr(baseline, dim)
        if jitter > 0:
            new_values[dim] = _clamp_unit(base_v + rng.uniform(-jitter, jitter))
        else:
            new_values[dim] = base_v
    return EmotionVector(**new_values)


def default_baseline() -> EmotionVector:
    return DEFAULT_BASELINE