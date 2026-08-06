"""Memory → Prompt 文本段。"""

import unittest

from soul_tty.memory.models import (
    SCOPE_GLOBAL,
    SCOPE_PERSONA,
    SOURCE_REFLECTION,
    TYPE_EXPERIENCE,
    TYPE_PREFERENCE,
    TYPE_PROFILE,
    Memory,
)
from soul_tty.memory.prompt import render_recall, render_resident


def make(memory_type, content, *, memory_id=1, created_at="2026-08-02T10:00:00+08:00"):
    global_scope = memory_type != TYPE_EXPERIENCE
    return Memory(
        id=memory_id,
        scope=SCOPE_GLOBAL if global_scope else SCOPE_PERSONA,
        persona_id="" if global_scope else "serena",
        type=memory_type,
        content=content,
        importance=0.85,
        source=SOURCE_REFLECTION,
        created_at=created_at,
        updated_at=created_at,
    )


class ResidentTests(unittest.TestCase):
    def test_groups_profile_and_preference(self):
        text = render_resident([
            make(TYPE_PROFILE, "用户是医药研发数字化方向工程师", memory_id=1),
            make(TYPE_PROFILE, "用户有一个5岁的女儿", memory_id=2),
            make(TYPE_PREFERENCE, "用户喜欢结构化、列表化的信息表达", memory_id=3),
        ])
        self.assertIn("关于用户：", text)
        self.assertIn("- 用户是医药研发数字化方向工程师", text)
        self.assertIn("- 用户有一个5岁的女儿", text)
        self.assertIn("交流偏好：", text)
        self.assertIn("- 用户喜欢结构化、列表化的信息表达", text)

    def test_profile_group_comes_first(self):
        text = render_resident([
            make(TYPE_PREFERENCE, "喜欢简洁", memory_id=1),
            make(TYPE_PROFILE, "是工程师", memory_id=2),
        ])
        self.assertLess(text.index("关于用户："), text.index("交流偏好："))

    def test_empty_group_header_is_omitted(self):
        text = render_resident([make(TYPE_PROFILE, "是工程师")])
        self.assertIn("关于用户：", text)
        self.assertNotIn("交流偏好：", text)

    def test_no_memories_renders_empty_string(self):
        self.assertEqual(render_resident([]), "")

    def test_experience_is_never_resident(self):
        """经历不常驻：数量会持续增长，全塞进去会污染上下文。"""
        text = render_resident([make(TYPE_EXPERIENCE, "一起完成了 Emotion 系统")])
        self.assertEqual(text, "")

    def test_no_section_title_in_body(self):
        """标题由 SystemPromptBuilder 统一加，这里只出正文。"""
        text = render_resident([make(TYPE_PROFILE, "是工程师")])
        self.assertNotIn("[User Context]", text)


class RecallTests(unittest.TestCase):
    def test_renders_experiences_with_date(self):
        text = render_recall([
            make(
                TYPE_EXPERIENCE,
                "用户完成了 Soul-TTY Emotion 系统设计",
                created_at="2026-08-02T10:00:00+08:00",
            )
        ])
        self.assertIn("[Relevant Memories]", text)
        self.assertIn("- 用户完成了 Soul-TTY Emotion 系统设计（2026-08-02）", text)

    def test_includes_instruction_against_quoting(self):
        text = render_recall([make(TYPE_EXPERIENCE, "一起做了件事")])
        self.assertIn("不要说", text)

    def test_no_memories_renders_empty_string(self):
        self.assertEqual(render_recall([]), "")

    def test_unparseable_date_is_omitted_not_crashing(self):
        text = render_recall([
            make(TYPE_EXPERIENCE, "一起做了件事", created_at="garbage")
        ])
        self.assertIn("- 一起做了件事", text)
        self.assertNotIn("（garbage）", text)

    def test_carries_own_title_because_it_bypasses_the_builder(self):
        """recall 走临时 message，不经过 SystemPromptBuilder，标题得自带。"""
        text = render_recall([make(TYPE_EXPERIENCE, "一起做了件事")])
        self.assertTrue(text.startswith("[Relevant Memories]"))


if __name__ == "__main__":
    unittest.main()
