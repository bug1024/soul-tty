"""MemoryService：把 store、retriever、prompt 装配成业务接口。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from soul_tty import config
from soul_tty.memory.models import (
    SOURCE_REFLECTION,
    TYPE_EXPERIENCE,
    TYPE_PREFERENCE,
    TYPE_PROFILE,
)
from soul_tty.memory.retriever import bigrams
from soul_tty.memory.service import MemoryService


class ServiceBaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "memory.db"
        self.service = MemoryService(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def _bigram_overlap(self, a: str, b: str) -> float:
        a_grams, b_grams = bigrams(a), bigrams(b)
        if not a_grams or not b_grams:
            return 0.0
        return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))


class ImportanceGateTests(ServiceBaseTest):
    def test_below_threshold_is_rejected(self):
        self.assertEqual(
            self.service.remember({"type": TYPE_PROFILE, "content": "x", "importance": 0.5}),
            0,
        )
        self.assertEqual(self.service._store.list(), [])

    def test_at_threshold_is_accepted(self):
        self.service.remember({
            "type": TYPE_PROFILE,
            "content": "用户是工程师",
            "importance": config.MEMORY_MIN_IMPORTANCE,
        })
        self.assertEqual(len(self.service._store.list()), 1)


class DeduplicationTests(ServiceBaseTest):
    def test_high_overlap_drops_new_memory(self):
        # 第一次落库
        self.service.remember({
            "type": TYPE_PROFILE,
            "content": "用户是医药研发数字化方向工程师",
            "importance": 0.9,
        })
        # 第二次内容：几乎完全是第一次的字符 bigram 复刻
        second = "用户是医药研发数字化工程师"
        overlap = self._bigram_overlap(
            "用户是医药研发数字化方向工程师", second
        )
        # 用来设计这个 case 的重叠值应明显高于 0.8
        self.assertGreater(overlap, config.MEMORY_DEDUPE_THRESHOLD)
        self.assertEqual(
            self.service.remember({
                "type": TYPE_PROFILE,
                "content": second,
                "importance": 0.85,
            }),
            0,
        )
        self.assertEqual(len(self.service._store.list()), 1)

    def test_dedupe_is_scoped_per_type(self):
        """profile / preference / experience 三类之间不互相去重。"""
        self.service.remember({
            "type": TYPE_PROFILE,
            "content": "用户喜欢简洁",
            "importance": 0.9,
        })
        # preference 里的「喜欢简洁」是交流偏好，不该被 profile 那条挡掉
        self.service.remember({
            "type": TYPE_PREFERENCE,
            "content": "喜欢简洁",
            "importance": 0.9,
        })
        self.assertEqual(len(self.service._store.list()), 2)


class BulkRememberTests(ServiceBaseTest):
    def test_skips_unknown_types(self):
        accepted = self.service.remember_many([
            {"type": TYPE_PROFILE, "content": "是工程师", "importance": 0.9},
            {"type": "mood", "content": "今天很累", "importance": 0.9},
            {"type": TYPE_PREFERENCE, "content": "喜欢列表", "importance": 0.9},
        ])
        self.assertEqual(accepted, 2)
        self.assertEqual(len(self.service._store.list()), 2)

    def test_blank_content_skipped(self):
        accepted = self.service.remember_many([
            {"type": TYPE_PROFILE, "content": "   ", "importance": 0.9},
            {"type": TYPE_PROFILE, "content": "是工程师", "importance": 0.9},
        ])
        self.assertEqual(accepted, 1)

    def test_known_facts_returns_global_and_recent_experience(self):
        self.service.remember({
            "type": TYPE_PROFILE,
            "content": "用户是工程师",
            "importance": 0.9,
        })
        self.service.remember({
            "type": TYPE_EXPERIENCE,
            "content": "完成 Emotion 系统",
            "importance": 0.85,
            "persona_id": "serena",
        }, persona_id="serena")
        self.service.remember({
            "type": TYPE_EXPERIENCE,
            "content": "和 Coder 调试 TTS",
            "importance": 0.8,
            "persona_id": "coder",
        }, persona_id="coder")
        # known_facts 给抽取器用：全部 global + 目标人格最近 experience
        facts = self.service.known_facts(persona_id="serena", recent_experience=5)
        self.assertEqual(len(facts), 2)  # 1 profile + 1 experience
        self.assertTrue(all(f["scope"] == "global" or f["persona_id"] == "serena" for f in facts))


class ResidentContextTests(ServiceBaseTest):
    def test_returns_profile_and_preference_only(self):
        self.service.remember({
            "type": TYPE_PROFILE,
            "content": "是工程师",
            "importance": 0.9,
        })
        self.service.remember({
            "type": TYPE_EXPERIENCE,
            "content": "完成 Emotion 系统",
            "importance": 0.85,
            "persona_id": "serena",
        }, persona_id="serena")
        text = self.service.render_resident_context()
        self.assertIn("是工程师", text)
        self.assertNotIn("完成 Emotion 系统", text)

    def test_caps_at_max_resident(self):
        # 灌 5 条 profile，进 resident 段的总条数不超过 MEMORY_MAX_RESIDENT
        original = config.MEMORY_MAX_RESIDENT
        config.MEMORY_MAX_RESIDENT = 3
        try:
            for index in range(5):
                self.service.remember({
                    "type": TYPE_PROFILE,
                    "content": f"事实 {index}",
                    "importance": 0.9 - index * 0.05,
                })
            text = self.service.render_resident_context()
            self.assertEqual(text.count("- 事实"), 3)
        finally:
            config.MEMORY_MAX_RESIDENT = original


class RecallTests(ServiceBaseTest):
    def test_recall_skips_non_recall_queries(self):
        self.service.remember({
            "type": TYPE_EXPERIENCE,
            "content": "完成 Emotion 系统",
            "importance": 0.85,
            "persona_id": "serena",
        }, persona_id="serena")
        self.assertEqual(
            self.service.recall("今天天气怎么样", persona_id="serena"), ""
        )

    def test_recall_renders_related_experiences(self):
        self.service.remember({
            "type": TYPE_EXPERIENCE,
            "content": "用户完成了 Soul-TTY Emotion 系统设计",
            "importance": 0.85,
            "persona_id": "serena",
        }, persona_id="serena")
        text = self.service.recall(
            "你还记得我的 Soul-TTY 项目吗", persona_id="serena"
        )
        self.assertIn("[Relevant Memories]", text)
        self.assertIn("Soul-TTY Emotion 系统设计", text)

    def test_recall_filters_by_persona(self):
        """换人格的 experience 不会被对方检索到。"""
        self.service.remember({
            "type": TYPE_EXPERIENCE,
            "content": "Soul-TTY 项目",
            "importance": 0.85,
            "persona_id": "serena",
        }, persona_id="serena")
        self.service.remember({
            "type": TYPE_EXPERIENCE,
            "content": "Soul-TTY 项目",
            "importance": 0.85,
            "persona_id": "coder",
        }, persona_id="coder")
        text = self.service.recall(
            "你还记得我的 Soul-TTY 项目吗", persona_id="serena"
        )
        self.assertIn("[Relevant Memories]", text)
        # 只命中 serena 自己的那一条
        self.assertEqual(text.count("Soul-TTY 项目"), 1)

    def test_presence_counts_only_visible_memories_and_real_recall(self):
        self.service.remember({
            "type": TYPE_PROFILE,
            "content": "用户是工程师",
            "importance": 0.9,
        })
        self.service.remember({
            "type": TYPE_EXPERIENCE,
            "content": "用户完成了 Soul-TTY 状态页设计",
            "importance": 0.9,
        }, persona_id="serena")
        self.service.remember({
            "type": TYPE_EXPERIENCE,
            "content": "和其他人格完成了另一个项目",
            "importance": 0.9,
        }, persona_id="coder")

        before = self.service.presence(persona_id="serena")
        self.assertEqual(before.count, 2)
        self.assertEqual(before.experience_count, 1)
        self.assertEqual(before.recent_recall, "")

        self.service.recall("你还记得 Soul-TTY 吗", persona_id="serena")
        after = self.service.presence(persona_id="serena")
        self.assertIn("状态页设计", after.recent_recall)
        self.assertIsNotNone(after.latest_id)


class CorruptedStoreTests(unittest.TestCase):
    """Store 不可用时 Service 全程静默降级。"""

    def test_all_methods_safe_on_corrupted_db(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            path.write_bytes(b"not a sqlite database")
            service = MemoryService(path)
            self.assertFalse(service.available)
            self.assertEqual(
                service.remember({"type": TYPE_PROFILE, "content": "x", "importance": 0.9}),
                0,
            )
            self.assertEqual(service.remember_many([]), 0)
            self.assertEqual(service.render_resident_context(), "")
            self.assertEqual(service.recall("你还记得吗"), "")
            self.assertEqual(service.presence().count, 0)


if __name__ == "__main__":
    unittest.main()
