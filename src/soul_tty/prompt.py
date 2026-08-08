"""System prompt 组装：各状态模块只提供文本，顺序与拼装在这里统一。

为什么需要它：此前 `apply_persona()` 直接改写全局 `config.SYSTEM_PROMPT`，
Emotion 是唯一的额外贡献者，尚能工作。再加入 Memory 之后会出现两个问题：

1. 签名退化成 `apply_persona(persona, emotion, memory, bond, ...)`
2. 两个后台线程各自重建 prompt 时会互相抹掉对方的段落

Builder 保持纯组装——不感知任何状态如何计算，也不反向依赖任何状态模块。
状态服务只暴露 `render_*_context()` 返回文本，由调用方接线。
"""

from __future__ import annotations

import os
import threading
from datetime import datetime

# 段落顺序对应模型理解的语义层级：
# 我是谁 → 现在什么模式 → 用户是谁 → 我们关系如何 → 我此刻的状态
_SECTION_ORDER: tuple[str, ...] = (
    "persona",
    "mode",
    "profile",
    "bond",
    "emotion",
    "datetime",
)

_SECTION_TITLES: dict[str, str] = {
    "profile": "[User Context]",
    "bond": "[Bond Context]",
    "emotion": "[Emotion Context]",
    "datetime": "[Date & Time]",
}


class SystemPromptBuilder:
    """按固定顺序拼接 system prompt 的各个文本段。线程安全。"""

    def __init__(self) -> None:
        self._sections: dict[str, str] = {}
        self._lock = threading.RLock()

    def set_section(self, name: str, text: str | None) -> None:
        """写入或清除一个段落；空文本等同于清除。"""
        if name not in _SECTION_ORDER:
            raise ValueError(
                f"未知 prompt 段落: {name}（可用: {', '.join(_SECTION_ORDER)}）"
            )
        cleaned = (text or "").strip()
        with self._lock:
            if cleaned:
                self._sections[name] = cleaned
            else:
                self._sections.pop(name, None)

    def render(self) -> str:
        with self._lock:
            parts = []
            for name in _SECTION_ORDER:
                body = self._sections.get(name)
                if not body:
                    continue
                title = _SECTION_TITLES.get(name)
                parts.append(f"{title}\n{body}" if title else body)
        return "\n\n".join(parts)


_builder = SystemPromptBuilder()


def builder() -> SystemPromptBuilder:
    """进程级单例；与 config 的全局配置模型保持一致。"""
    return _builder


def render_datetime_context() -> str:
    """渲染当前日期时间上下文字符串，供 LLM 感知"现在是什么时候"。"""
    now = datetime.now()
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    period = _day_period(now.hour)
    return (
        f"当前时间：{now.year}年{now.month}月{now.day}日 {weekday_names[now.weekday()]}，{period}。"
        f"现在是 {now.hour}点{now.minute}分。"
    )


def _day_period(hour: int) -> str:
    if 5 <= hour < 11:
        return "早上"
    if 11 <= hour < 14:
        return "中午"
    if 14 <= hour < 18:
        return "下午"
    if 18 <= hour < 24:
        return "晚上"
    return "深夜"


def refresh() -> str:
    """把当前段落渲染进 `config.SYSTEM_PROMPT` 并返回。

    `SYSTEM_PROMPT` 环境变量拥有最高优先级：设置了就完全不动它，
    与改造前 `apply_persona()` 的行为一致。
    """
    from . import config

    if "SYSTEM_PROMPT" not in os.environ:
        config.SYSTEM_PROMPT = _builder.render()
    return config.SYSTEM_PROMPT
