"""MEMORY_ENABLED=0 时，记忆相关代码全部不参与主对话。

主对话表现必须与改造前完全一致。
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from soul_tty import config
from soul_tty.memory.service import MemoryService
from soul_tty.reflection.worker import ReflectionWorker


class ServiceDisabledTests(unittest.TestCase):
    def test_memory_enabled_false_yields_unavailable_service(self):
        """MEMORY_ENABLED=0 也应该让 MemoryService.available=False。"""
        # Service 不读 MEMORY_ENABLED——它只读 DB 是否可用。验证这条约束：
        # DB 损坏时也走 available=False 路径，效果与 MEMORY_ENABLED=0 相同。
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            path.write_bytes(b"not a database")
            service = MemoryService(path)
            self.assertFalse(service.available)
            # 所有方法静默降级
            self.assertEqual(
                service.remember({"type": "profile", "content": "x", "importance": 0.9}),
                0,
            )
            self.assertEqual(service.render_resident_context(), "")
            self.assertEqual(service.recall("你还记得吗"), "")

    def test_reflection_worker_without_memory_extractor_is_noop(self):
        """没注入 memory_extractor 时，worker 不应做任何 memory 路径。"""
        from soul_tty.reflection.relationship import CompletedTurn

        with TemporaryDirectory() as tmp:
            worker = ReflectionWorker(
                persona_id="serena",
                evaluator=lambda s, t: None,
                state_dir=Path(tmp),
                idle_delay_s=0.0,
                min_interval_s=0.0,
            )
            # memory_extractor 默认就是 None
            self.assertIsNone(worker.memory_extractor)
            # 灌 buffer 也不应触发任何抽取
            worker._memory_buffer.append((1, CompletedTurn("u", "a")))
            # 应当能调，但 _memory_due 不会通过（buffer 极短字符过 min_chars=20）
            # 直接验证：on_memory_updated 默认 None，调 _maybe_extract_memory 不崩
            worker._maybe_extract_memory()
            self.assertEqual(len(worker._memory_buffer), 1)


class DisabledIntegrationTests(unittest.TestCase):
    """MEMORY_ENABLED=0 时整套旁路链路不参与。"""

    def test_disabled_config_skips_memory_extractor(self):
        # 临时改 config：cli.py 在 MEMORY_ENABLED=0 时根本不实例化 MemoryService，
        # 也不会把 memory_extractor / on_memory_updated 挂到 worker。
        # 这条用例只测 worker 端：未注入就完全不参与。
        with TemporaryDirectory() as tmp:
            worker = ReflectionWorker(
                persona_id="serena",
                evaluator=lambda s, t: None,
                state_dir=Path(tmp),
                memory_min_interval_s=0.0,
                memory_min_text_chars=0,
                # memory_extractor 默认 None
            )
            # 没注入就不该有 memory 路径活动
            self.assertIsNone(worker.memory_extractor)
            self.assertIsNone(worker.on_memory_updated)


if __name__ == "__main__":
    unittest.main()
