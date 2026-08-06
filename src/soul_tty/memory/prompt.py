"""Memory → Prompt 文本段。

两条路径，对应两种内存性质：

- `render_resident` 返回 [User Context] 段落正文（标题由
  SystemPromptBuilder 统一添加）；只有画像/偏好走这里。
- `render_recall` 返回完整的 [Relevant Memories] 块（含标题与指令），
  因为它走临时 system message，不经过 Builder。
"""

from __future__ import annotations

from datetime import datetime

from .models import (
    TYPE_EXPERIENCE,
    TYPE_PREFERENCE,
    TYPE_PROFILE,
    Memory,
)

_TYPE_SECTIONS: dict[str, str] = {
    TYPE_PROFILE: "关于用户：",
    TYPE_PREFERENCE: "交流偏好：",
}


def render_resident(memories: list[Memory]) -> str:
    """画像与偏好 → [User Context] 段落正文。

    按 `type` 分两组渲染：profile → 「关于用户：」，
    preference → 「交流偏好：」。某组为空则不渲染对应小标题；
    全空则不渲染整个段落（让 Builder 也跳过 [User Context] 标题）。
    """
    profiles = [m for m in memories if m.type == TYPE_PROFILE]
    preferences = [m for m in memories if m.type == TYPE_PREFERENCE]
    if not profiles and not preferences:
        return ""

    blocks: list[str] = []
    for memory_type, label in _TYPE_SECTIONS.items():
        rows = profiles if memory_type == TYPE_PROFILE else preferences
        if not rows:
            continue
        lines = [f"- {row.content}" for row in rows]
        blocks.append(f"{label}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _format_date(created_at: str) -> str:
    try:
        return datetime.fromisoformat(created_at).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return ""


def render_recall(memories: list[Memory]) -> str:
    """经历 → [Relevant Memories] 整段文本（含标题）。"""
    if not memories:
        return ""
    lines = []
    for memory in memories:
        if memory.type != TYPE_EXPERIENCE:
            # 防御性：只把 experience 注入临时 message
            continue
        date = _format_date(memory.created_at)
        suffix = f"（{date}）" if date else ""
        lines.append(f"- {memory.content}{suffix}")
    if not lines:
        return ""
    return (
        "[Relevant Memories]\n"
        "你和用户过去相关的经历：\n"
        + "\n".join(lines)
        + "\n\n自然地使用这些信息，不要说"
        "「你曾经告诉过我」「我记得你说过」这类话。"
    )
