"""Memory 数据模型：三类记忆、两种作用域。

作用域是独立字段而不是从 type 推导出来的临时值。V1 的映射固定
（画像/偏好 → global，经历 → persona），但显式落库让「用户喜欢
Serena 说短句」这种人格作用域的 preference 在 V2 落地时不必迁移表。
"""

from __future__ import annotations

from dataclasses import dataclass

SCOPE_GLOBAL = "global"
SCOPE_PERSONA = "persona"
SCOPES: tuple[str, ...] = (SCOPE_GLOBAL, SCOPE_PERSONA)

TYPE_PROFILE = "profile"
TYPE_PREFERENCE = "preference"
TYPE_EXPERIENCE = "experience"
MEMORY_TYPES: tuple[str, ...] = (TYPE_PROFILE, TYPE_PREFERENCE, TYPE_EXPERIENCE)

# V1 只有 reflection 会写入；另两个是手工录入与导入的预留取值。
SOURCE_REFLECTION = "reflection"

_TYPE_SCOPES: dict[str, str] = {
    TYPE_PROFILE: SCOPE_GLOBAL,
    TYPE_PREFERENCE: SCOPE_GLOBAL,
    TYPE_EXPERIENCE: SCOPE_PERSONA,
}

TYPE_LABELS: dict[str, str] = {
    TYPE_PROFILE: "用户画像",
    TYPE_PREFERENCE: "交流偏好",
    TYPE_EXPERIENCE: "共同经历",
}


def scope_for_type(memory_type: str) -> str:
    """V1 的 type → scope 映射。

    profile / preference 是关于用户的，换人格依然成立；
    experience 是「用户与某个 Agent 的共同经历」，换人格不该继承。
    """
    try:
        return _TYPE_SCOPES[memory_type]
    except KeyError:
        raise ValueError(
            f"未知记忆类型: {memory_type}（可用: {', '.join(MEMORY_TYPES)}）"
        ) from None


@dataclass(frozen=True)
class Memory:
    id: int
    scope: str
    persona_id: str
    type: str
    content: str
    importance: float
    source: str
    created_at: str
    updated_at: str

    @property
    def type_label(self) -> str:
        return TYPE_LABELS.get(self.type, self.type)
