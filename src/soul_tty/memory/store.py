"""Memory 的 SQLite 存储。本模块是唯一执行 SQL 的地方。

连接策略：不持有长连接，每次操作开一个新连接后关闭。
ReflectionWorker 线程写入、主线程检索、CLI 独立进程读写，三方并发；
表规模在千行量级时单次连接开销约 0.1ms，换取彻底不必管理线程安全。
WAL 适配「读多写少」的实际负载。

降级策略：任何 sqlite 错误都不向上抛。打不开或文件损坏时 `available`
为 False，所有操作退化成 no-op，主对话表现与 MEMORY_ENABLED=0 一致。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from ..reflection.relationship import safe_persona_id
from .models import (
    MEMORY_TYPES,
    SOURCE_REFLECTION,
    Memory,
    scope_for_type,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope       TEXT NOT NULL,
    persona_id  TEXT NOT NULL DEFAULT '',
    type        TEXT NOT NULL,
    content     TEXT NOT NULL,
    importance  REAL NOT NULL,
    source      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_scope
    ON memories(scope, persona_id, type);
"""

_COLUMNS = (
    "id, scope, persona_id, type, content, importance, source, "
    "created_at, updated_at"
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _row_to_memory(row: tuple) -> Memory:
    return Memory(
        id=int(row[0]),
        scope=str(row[1]),
        persona_id=str(row[2]),
        type=str(row[3]),
        content=str(row[4]),
        importance=float(row[5]),
        source=str(row[6]),
        created_at=str(row[7]),
        updated_at=str(row[8]),
    )


class MemoryStore:
    def __init__(self, path: Path, *, timeout: float = 5.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.available = self._ensure_schema()

    # --- 连接与建表 -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=self.timeout)

    def _ensure_schema(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect()
        except (OSError, sqlite3.Error):
            return False
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.commit()
        except sqlite3.Error:
            return False
        finally:
            connection.close()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        return True

    # --- 写入 -------------------------------------------------------

    def add(
        self,
        *,
        type: str,
        content: str,
        importance: float,
        persona_id: str = "",
        source: str = SOURCE_REFLECTION,
    ) -> Memory | None:
        """写入一条记忆；类型非法、内容为空或存储不可用时返回 None。"""
        if not self.available or type not in MEMORY_TYPES:
            return None
        content = (content or "").strip()
        if not content:
            return None
        scope = scope_for_type(type)
        # global 作用域不绑人格，即使调用方传了 persona_id 也丢弃。
        owner = safe_persona_id(persona_id) if scope != "global" else ""
        if scope != "global" and not persona_id.strip():
            return None
        stamp = _now()
        try:
            connection = self._connect()
        except sqlite3.Error:
            return None
        try:
            cursor = connection.execute(
                "INSERT INTO memories "
                "(scope, persona_id, type, content, importance, source, "
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scope,
                    owner,
                    type,
                    content,
                    float(importance),
                    source,
                    stamp,
                    stamp,
                ),
            )
            connection.commit()
            new_id = int(cursor.lastrowid)
        except sqlite3.Error:
            return None
        finally:
            connection.close()
        return Memory(
            id=new_id,
            scope=scope,
            persona_id=owner,
            type=type,
            content=content,
            importance=float(importance),
            source=source,
            created_at=stamp,
            updated_at=stamp,
        )

    # --- 读取 -------------------------------------------------------

    def list(
        self,
        *,
        scope: str | None = None,
        persona_id: str | None = None,
        types: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[Memory]:
        """按作用域与类型过滤；`scope=None` 表示不限作用域。

        排序固定为 importance DESC, id DESC，调用方需要别的顺序自行重排。
        """
        if not self.available:
            return []
        clauses: list[str] = []
        params: list[object] = []
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if persona_id is not None:
            clauses.append("persona_id = ?")
            params.append(safe_persona_id(persona_id) if persona_id else "")
        if types:
            clauses.append(f"type IN ({', '.join('?' * len(types))})")
            params.extend(types)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        tail = " ORDER BY importance DESC, id DESC"
        if limit is not None:
            tail += " LIMIT ?"
            params.append(int(limit))
        try:
            connection = self._connect()
        except sqlite3.Error:
            return []
        try:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM memories{where}{tail}", params
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            connection.close()
        return [_row_to_memory(row) for row in rows]

    def get(self, memory_id: int) -> Memory | None:
        if not self.available:
            return None
        try:
            connection = self._connect()
        except sqlite3.Error:
            return None
        try:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (int(memory_id),)
            ).fetchone()
        except sqlite3.Error:
            return None
        finally:
            connection.close()
        return _row_to_memory(row) if row is not None else None

    # --- 删除 -------------------------------------------------------

    def delete(self, memory_id: int) -> bool:
        if not self.available:
            return False
        try:
            connection = self._connect()
        except sqlite3.Error:
            return False
        try:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?", (int(memory_id),)
            )
            connection.commit()
            return cursor.rowcount > 0
        except sqlite3.Error:
            return False
        finally:
            connection.close()

    def clear(self) -> int:
        """清空全部记忆，返回删除条数。"""
        if not self.available:
            return 0
        try:
            connection = self._connect()
        except sqlite3.Error:
            return 0
        try:
            cursor = connection.execute("DELETE FROM memories")
            connection.commit()
            return int(cursor.rowcount)
        except sqlite3.Error:
            return 0
        finally:
            connection.close()
