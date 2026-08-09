"""MemoryService：把 store / retriever / prompt 装配成业务接口。

业务规则集中在这一层：
- importance 门槛（`MEMORY_MIN_IMPORTANCE`，默认 0.7）
- 落库前的兜底去重（与同类已有记忆 bigram 重叠 > 0.8 即跳过）
- 按 type 分组的 resident 上限
- persona 隔离的 recall 范围

降级：store 不可用时所有方法静默成功 0 / 返回空串，调用方
无需做空值判断。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from .. import config
from .models import (
    MEMORY_TYPES,
    SCOPE_GLOBAL,
    SCOPE_PERSONA,
    TYPE_EXPERIENCE,
    TYPE_PREFERENCE,
    TYPE_PROFILE,
    Memory,
)
from .prompt import render_recall, render_resident
from .retriever import bigrams, is_recall_query, search
from .store import MemoryStore


def _bigram_overlap(a: str, b: str) -> float:
    """与 retriever 不同的归一化：`|A∩B| / min(|A|,|B|)`，使短的新记忆
    被长记忆完全包含时也算重复。检索用「短 query 完全被覆盖」的目标
    归一化；落库去重要的是「这个事实是不是说过」——长记忆已经包含它。"""
    a_grams, b_grams = bigrams(a), bigrams(b)
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))


@dataclass(frozen=True)
class MemoryPresence:
    """供 Presence Panel 使用的克制摘要，不暴露存储实现。"""

    count: int = 0
    experience_count: int = 0
    recent_recall: str = ""
    latest_id: int | None = None


class MemoryService:
    def __init__(self, path) -> None:
        self._store = MemoryStore(path)
        self.available = self._store.available
        self._presence_lock = threading.RLock()
        self._recent_recall = ""

    # --- 写入 -------------------------------------------------------

    def remember(
        self,
        memory: dict,
        *,
        persona_id: str = "",
        source: str | None = None,
    ) -> int:
        """写入单条记忆；返回 0 / 1。"""
        return self.remember_many([memory], persona_id=persona_id, source=source)

    def remember_many(
        self,
        memories: list[dict],
        *,
        persona_id: str = "",
        source: str | None = None,
    ) -> int:
        """批量写入；返回成功落库条数。全部失败返回 0。"""
        if not self.available:
            return 0
        accepted = 0
        for item in memories:
            accepted += self._remember_one(
                item, persona_id=persona_id, source=source
            )
        return accepted

    def _remember_one(
        self,
        item: dict,
        *,
        persona_id: str,
        source: str | None,
    ) -> int:
        if not isinstance(item, dict):
            return 0
        memory_type = item.get("type")
        if memory_type not in MEMORY_TYPES:
            return 0
        content = (item.get("content") or "").strip()
        if not content:
            return 0
        try:
            importance = float(item.get("importance", 0))
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(importance):
            return 0
        importance = max(0.0, min(1.0, importance))
        if importance < config.MEMORY_MIN_IMPORTANCE:
            return 0
        owner = persona_id
        # 与同类已有记忆做兜底去重；LLM 也会有重复输出
        if self._is_duplicate(memory_type, content, owner):
            return 0
        written = self._store.add(
            type=memory_type,
            content=content,
            importance=importance,
            persona_id=owner,
            source=source or "reflection",
        )
        return 1 if written is not None else 0

    def _is_duplicate(
        self, memory_type: str, content: str, persona_id: str
    ) -> bool:
        if memory_type == TYPE_EXPERIENCE:
            scope = SCOPE_PERSONA
            # experience 在 persona 内部去重，跨人格不查
            existing = self._store.list(
                scope=scope, persona_id=persona_id, types=(TYPE_EXPERIENCE,)
            )
        else:
            scope = SCOPE_GLOBAL
            existing = self._store.list(scope=scope, types=(memory_type,))
        threshold = config.MEMORY_DEDUPE_THRESHOLD
        for row in existing:
            if _bigram_overlap(row.content, content) > threshold:
                return True
        return False

    # --- 读取 -------------------------------------------------------

    def known_facts(
        self, *, persona_id: str, recent_experience: int = 10
    ) -> list[dict]:
        """供抽取器拼 prompt：全部 global + 目标人格最近 N 条 experience。

        以 dict 返回（不是 Memory）让抽取器层不必感知 dataclass 字段。
        """
        if not self.available:
            return []
        rows = self._store.list(scope=SCOPE_GLOBAL)
        rows.extend(
            self._store.list(
                scope=SCOPE_PERSONA,
                persona_id=persona_id,
                types=(TYPE_EXPERIENCE,),
                limit=recent_experience,
                order_by="id DESC",
            )
        )
        return [
            {
                "scope": row.scope,
                "persona_id": row.persona_id,
                "type": row.type,
                "content": row.content,
            }
            for row in rows
        ]

    def render_resident_context(self) -> str:
        """[User Context] 段落正文：画像 + 偏好，按 importance 取前 N 条。"""
        if not self.available:
            return ""
        rows = self._store.list(
            scope=SCOPE_GLOBAL,
            types=(TYPE_PROFILE, TYPE_PREFERENCE),
            limit=config.MEMORY_MAX_RESIDENT,
        )
        return render_resident(rows)

    def recall(self, query: str, *, persona_id: str = "") -> str:
        """命中召回词时按 persona 检索 experience，返回完整 [Relevant Memories] 段。

        未命中或检索为空都返回空串，调用方应直接跳过注入。
        """
        if not self.available or not is_recall_query(query):
            return ""
        rows = self._store.list(
            scope=SCOPE_PERSONA,
            persona_id=persona_id,
            types=(TYPE_EXPERIENCE,),
        )
        if not rows:
            return ""
        found = search(query, rows)
        if not found:
            return ""
        with self._presence_lock:
            self._recent_recall = found[0].content
        return render_recall(found)

    def presence(self, *, persona_id: str = "") -> MemoryPresence:
        """返回当前人格可见的长期记忆摘要。

        全局画像/偏好与当前人格的共同经历合并计数；“最近想起”只来自本次
        会话真实发生过的 recall，不拿“最近写入”冒充“最近想起”。
        """
        if not self.available:
            return MemoryPresence()
        global_rows = self._store.list(scope=SCOPE_GLOBAL)
        experiences = self._store.list(
            scope=SCOPE_PERSONA,
            persona_id=persona_id,
            types=(TYPE_EXPERIENCE,),
        )
        rows = [*global_rows, *experiences]
        latest_id = max((row.id for row in rows), default=None)
        with self._presence_lock:
            recent_recall = self._recent_recall
        return MemoryPresence(
            count=len(rows),
            experience_count=len(experiences),
            recent_recall=recent_recall,
            latest_id=latest_id,
        )

    # --- 管理 -------------------------------------------------------

    def list(self, **kwargs) -> list[Memory]:
        return self._store.list(**kwargs)

    def get(self, memory_id: int) -> Memory | None:
        return self._store.get(memory_id)

    def delete(self, memory_id: int) -> bool:
        return self._store.delete(memory_id)

    def clear(self) -> int:
        return self._store.clear()
