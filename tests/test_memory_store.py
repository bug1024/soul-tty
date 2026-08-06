"""Memory 存储层：SQLite 读写、作用域隔离、损坏降级。"""

import os
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from soul_tty.memory.models import (
    SCOPE_GLOBAL,
    SCOPE_PERSONA,
    SOURCE_REFLECTION,
    TYPE_EXPERIENCE,
    TYPE_PREFERENCE,
    TYPE_PROFILE,
    scope_for_type,
)
from soul_tty.memory.store import MemoryStore


class ScopeMappingTests(unittest.TestCase):
    def test_profile_and_preference_are_global(self):
        self.assertEqual(scope_for_type(TYPE_PROFILE), SCOPE_GLOBAL)
        self.assertEqual(scope_for_type(TYPE_PREFERENCE), SCOPE_GLOBAL)

    def test_experience_is_persona_scoped(self):
        self.assertEqual(scope_for_type(TYPE_EXPERIENCE), SCOPE_PERSONA)

    def test_unknown_type_rejected(self):
        with self.assertRaises(ValueError):
            scope_for_type("mood")


class StoreBaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "memory.db"
        self.store = MemoryStore(self.path)

    def tearDown(self):
        self._tmp.cleanup()


class AddTests(StoreBaseTest):
    def test_add_returns_row_with_id(self):
        memory = self.store.add(
            type=TYPE_PROFILE, content="用户是医药研发工程师", importance=0.9
        )
        self.assertIsNotNone(memory)
        self.assertGreater(memory.id, 0)
        self.assertEqual(memory.content, "用户是医药研发工程师")
        self.assertEqual(memory.importance, 0.9)
        self.assertEqual(memory.source, SOURCE_REFLECTION)
        self.assertTrue(memory.created_at)
        self.assertEqual(memory.created_at, memory.updated_at)

    def test_profile_is_stored_global_without_persona(self):
        memory = self.store.add(
            type=TYPE_PROFILE,
            content="用户有一个5岁的女儿",
            importance=0.8,
            persona_id="serena",
        )
        # 用户画像与人格无关，即使传了 persona_id 也落 global
        self.assertEqual(memory.scope, SCOPE_GLOBAL)
        self.assertEqual(memory.persona_id, "")

    def test_experience_is_bound_to_persona(self):
        memory = self.store.add(
            type=TYPE_EXPERIENCE,
            content="用户完成了 Emotion 系统设计",
            importance=0.85,
            persona_id="serena",
        )
        self.assertEqual(memory.scope, SCOPE_PERSONA)
        self.assertEqual(memory.persona_id, "serena")

    def test_persona_id_is_sanitized(self):
        memory = self.store.add(
            type=TYPE_EXPERIENCE,
            content="一起做了件事",
            importance=0.8,
            persona_id="a/b c",
        )
        # 与 relationships/{persona}.json 的命名保持一致
        self.assertEqual(memory.persona_id, "a-b-c")

    def test_unknown_type_returns_none(self):
        self.assertIsNone(
            self.store.add(type="mood", content="今天很累", importance=0.9)
        )

    def test_blank_content_returns_none(self):
        self.assertIsNone(
            self.store.add(type=TYPE_PROFILE, content="   ", importance=0.9)
        )

    def test_custom_source_is_kept(self):
        memory = self.store.add(
            type=TYPE_PROFILE,
            content="手工录入的事实",
            importance=0.9,
            source="manual",
        )
        self.assertEqual(memory.source, "manual")


class ListTests(StoreBaseTest):
    def setUp(self):
        super().setUp()
        self.store.add(type=TYPE_PROFILE, content="用户是工程师", importance=0.9)
        self.store.add(type=TYPE_PREFERENCE, content="喜欢结构化表达", importance=0.7)
        self.store.add(
            type=TYPE_EXPERIENCE,
            content="和 Serena 完成了 Emotion 系统",
            importance=0.85,
            persona_id="serena",
        )
        self.store.add(
            type=TYPE_EXPERIENCE,
            content="和 Coder 重构了解析器",
            importance=0.8,
            persona_id="coder",
        )

    def test_list_all(self):
        self.assertEqual(len(self.store.list()), 4)

    def test_list_global_scope(self):
        rows = self.store.list(scope=SCOPE_GLOBAL)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.scope == SCOPE_GLOBAL for row in rows))

    def test_list_persona_scope_isolates_personas(self):
        rows = self.store.list(scope=SCOPE_PERSONA, persona_id="serena")
        self.assertEqual([row.content for row in rows], ["和 Serena 完成了 Emotion 系统"])

    def test_switching_persona_keeps_global_memories(self):
        """换人格不该让用户画像丢失，但共同经历不继承。"""
        coder_global = self.store.list(scope=SCOPE_GLOBAL)
        coder_experience = self.store.list(scope=SCOPE_PERSONA, persona_id="coder")
        self.assertEqual(len(coder_global), 2)
        self.assertEqual([row.content for row in coder_experience], ["和 Coder 重构了解析器"])

    def test_list_filters_by_type(self):
        rows = self.store.list(types=(TYPE_PROFILE,))
        self.assertEqual([row.content for row in rows], ["用户是工程师"])

    def test_list_orders_by_importance_desc(self):
        rows = self.store.list(scope=SCOPE_GLOBAL)
        self.assertEqual([row.importance for row in rows], [0.9, 0.7])

    def test_list_respects_limit(self):
        self.assertEqual(len(self.store.list(limit=2)), 2)


class MutationTests(StoreBaseTest):
    def test_get_returns_row(self):
        added = self.store.add(type=TYPE_PROFILE, content="用户是工程师", importance=0.9)
        fetched = self.store.get(added.id)
        self.assertEqual(fetched, added)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get(9999))

    def test_delete_removes_row(self):
        added = self.store.add(type=TYPE_PROFILE, content="用户是工程师", importance=0.9)
        self.assertTrue(self.store.delete(added.id))
        self.assertIsNone(self.store.get(added.id))

    def test_delete_missing_returns_false(self):
        self.assertFalse(self.store.delete(9999))

    def test_clear_removes_everything_and_returns_count(self):
        self.store.add(type=TYPE_PROFILE, content="用户是工程师", importance=0.9)
        self.store.add(type=TYPE_PREFERENCE, content="喜欢简洁", importance=0.8)
        self.assertEqual(self.store.clear(), 2)
        self.assertEqual(self.store.list(), [])


class DurabilityTests(unittest.TestCase):
    def test_data_survives_reopen(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            MemoryStore(path).add(
                type=TYPE_PROFILE, content="用户是工程师", importance=0.9
            )
            rows = MemoryStore(path).list()
            self.assertEqual([row.content for row in rows], ["用户是工程师"])

    def test_db_file_is_owner_only(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            MemoryStore(path).add(
                type=TYPE_PROFILE, content="用户是工程师", importance=0.9
            )
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_schema_version_is_recorded(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            MemoryStore(path)
            connection = sqlite3.connect(path)
            try:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(version, 1)

    def test_parent_directory_is_created(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "deeper" / "memory.db"
            store = MemoryStore(path)
            self.assertTrue(store.available)
            self.assertIsNotNone(
                store.add(type=TYPE_PROFILE, content="用户是工程师", importance=0.9)
            )


class DegradationTests(unittest.TestCase):
    """存储任何环节失败，调用方都必须能继续跑，不抛异常。"""

    def test_corrupted_file_degrades_to_noop(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            path.write_bytes(b"this is definitely not a sqlite database" * 10)

            store = MemoryStore(path)
            self.assertFalse(store.available)
            self.assertIsNone(
                store.add(type=TYPE_PROFILE, content="用户是工程师", importance=0.9)
            )
            self.assertEqual(store.list(), [])
            self.assertIsNone(store.get(1))
            self.assertFalse(store.delete(1))
            self.assertEqual(store.clear(), 0)

    def test_unwritable_directory_degrades_to_noop(self):
        with TemporaryDirectory() as tmp:
            locked = Path(tmp) / "locked"
            locked.mkdir()
            os.chmod(locked, 0o500)
            try:
                store = MemoryStore(locked / "memory.db")
                self.assertFalse(store.available)
                self.assertEqual(store.list(), [])
            finally:
                os.chmod(locked, 0o700)


if __name__ == "__main__":
    unittest.main()
