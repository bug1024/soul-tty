"""System prompt 组装层。

Builder 只负责按固定顺序拼接文本段，不感知任何状态如何计算。
"""

import threading
import unittest

from soul_tty.prompt import SystemPromptBuilder


class SectionOrderTests(unittest.TestCase):
    def test_renders_in_fixed_order_regardless_of_set_order(self):
        builder = SystemPromptBuilder()
        # 乱序写入
        builder.set_section("emotion", "情绪文本")
        builder.set_section("persona", "人格文本")
        builder.set_section("bond", "关系文本")
        builder.set_section("mode", "模式文本")
        builder.set_section("profile", "画像文本")

        rendered = builder.render()
        positions = [
            rendered.index("人格文本"),
            rendered.index("模式文本"),
            rendered.index("画像文本"),
            rendered.index("关系文本"),
            rendered.index("情绪文本"),
        ]
        # persona → mode → profile → bond → emotion
        self.assertEqual(positions, sorted(positions))

    def test_titled_sections_get_headers(self):
        builder = SystemPromptBuilder()
        builder.set_section("profile", "用户是工程师")
        builder.set_section("bond", "已建立一定联系")
        builder.set_section("emotion", "当前平静")
        rendered = builder.render()
        self.assertIn("[User Context]\n用户是工程师", rendered)
        self.assertIn("[Bond Context]\n已建立一定联系", rendered)
        self.assertIn("[Emotion Context]\n当前平静", rendered)

    def test_persona_and_mode_have_no_headers(self):
        builder = SystemPromptBuilder()
        builder.set_section("persona", "你是 Serena")
        builder.set_section("mode", "陪伴模式")
        rendered = builder.render()
        self.assertEqual(rendered, "你是 Serena\n\n陪伴模式")

    def test_missing_sections_are_skipped(self):
        builder = SystemPromptBuilder()
        builder.set_section("persona", "你是 Serena")
        builder.set_section("emotion", "当前平静")
        rendered = builder.render()
        self.assertNotIn("[User Context]", rendered)
        self.assertNotIn("[Bond Context]", rendered)
        self.assertEqual(rendered, "你是 Serena\n\n[Emotion Context]\n当前平静")

    def test_empty_builder_renders_empty_string(self):
        self.assertEqual(SystemPromptBuilder().render(), "")


class SectionMutationTests(unittest.TestCase):
    def test_empty_text_clears_section(self):
        builder = SystemPromptBuilder()
        builder.set_section("persona", "你是 Serena")
        builder.set_section("emotion", "当前平静")
        builder.set_section("emotion", "")
        self.assertEqual(builder.render(), "你是 Serena")

    def test_whitespace_only_text_clears_section(self):
        builder = SystemPromptBuilder()
        builder.set_section("persona", "你是 Serena")
        builder.set_section("emotion", "   \n  ")
        self.assertEqual(builder.render(), "你是 Serena")

    def test_set_section_overwrites(self):
        builder = SystemPromptBuilder()
        builder.set_section("emotion", "旧的")
        builder.set_section("emotion", "新的")
        rendered = builder.render()
        self.assertIn("新的", rendered)
        self.assertNotIn("旧的", rendered)

    def test_unknown_section_rejected(self):
        builder = SystemPromptBuilder()
        with self.assertRaises(ValueError):
            builder.set_section("memory", "x")

    def test_none_text_clears_section(self):
        builder = SystemPromptBuilder()
        builder.set_section("emotion", "有内容")
        builder.set_section("emotion", None)
        self.assertEqual(builder.render(), "")


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_writes_from_two_threads_never_lose_a_section(self):
        """EmotionService decay 线程与 ReflectionWorker 会并发写不同段落。"""
        builder = SystemPromptBuilder()
        builder.set_section("persona", "你是 Serena")
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def writer(name: str, marker: str) -> None:
            try:
                barrier.wait()
                for _ in range(300):
                    builder.set_section(name, marker)
                    builder.render()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=("emotion", "情绪段")),
            threading.Thread(target=writer, args=("profile", "画像段")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(errors, [])
        rendered = builder.render()
        self.assertIn("你是 Serena", rendered)
        self.assertIn("情绪段", rendered)
        self.assertIn("画像段", rendered)


if __name__ == "__main__":
    unittest.main()
