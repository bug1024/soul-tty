"""长期记忆：用户画像、交流偏好与共同经历。

Memory 是上下文，不是控制权——它只向 Prompt 提供信息，
不修改 Bond、Emotion 或任何其他状态。
"""

from .models import (
    MEMORY_TYPES,
    SCOPE_GLOBAL,
    SCOPE_PERSONA,
    SOURCE_REFLECTION,
    TYPE_EXPERIENCE,
    TYPE_PREFERENCE,
    TYPE_PROFILE,
    Memory,
    scope_for_type,
)
from .store import MemoryStore

__all__ = [
    "MEMORY_TYPES",
    "SCOPE_GLOBAL",
    "SCOPE_PERSONA",
    "SOURCE_REFLECTION",
    "TYPE_EXPERIENCE",
    "TYPE_PREFERENCE",
    "TYPE_PROFILE",
    "Memory",
    "MemoryStore",
    "scope_for_type",
]
