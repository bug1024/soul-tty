"""ReflectionWorker 调度 + 记忆 buffer 的不变量。

queue 溢出丢轮可以接受，记忆不能丢——这是 Memory 与关系评估的根本分离点。
这些测试只覆盖记忆 buffer 路径，直接调 _maybe_extract_memory，
不起后台线程、不走关系评估。
"""

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from soul_tty import config
from soul_tty.memory.models import ExtractionStatus
from soul_tty.reflection.relationship import CompletedTurn
from soul_tty.reflection.worker import ReflectionWorker


def _make_worker(
    tmp_path,
    *,
    memory_extractor=None,
    on_memory_updated=None,
    **overrides,
) -> ReflectionWorker:
    defaults = dict(
        persona_id="serena",
        evaluator=lambda state, turn: None,
        state_dir=Path(tmp_path),
        queue_size=4,
        idle_delay_s=0.0,
        min_interval_s=0.0,
        memory_min_interval_s=0.0,
        memory_min_text_chars=0,
        memory_buffer_turns=20,
    )
    defaults.update(overrides)
    return ReflectionWorker(
        memory_extractor=memory_extractor,
        on_memory_updated=on_memory_updated,
        **defaults,
    )


class BufferRetentionTests(unittest.TestCase):
    """queue 溢出不影响 memory buffer——buffer 是无条件的 append。"""

    def test_buffer_keeps_all_submitted_turns_regardless_of_queue_size(self):
        with TemporaryDirectory() as tmp:
            captured: list[list[CompletedTurn]] = []

            def extractor(turns):
                captured.append(list(turns))
                return False  # 不 ack，buffer 保留

            worker = _make_worker(tmp, memory_extractor=extractor)
            # 直接灌 buffer，绕过 queue（测的是 buffer 的物理不变量）
            for index in range(25):
                worker._memory_buffer.append(
                    (index + 1, CompletedTurn(f"user {index}", f"agent {index}"))
                )
            worker._memory_seq = 25

            worker._maybe_extract_memory()

            # 25 条全部一次性抽出去，buffer.maxlen=20 会让前 5 条被 deque 静默挤出
            self.assertEqual(len(captured), 1)
            batch = captured[0]
            # 取出来的是 buffer 的副本；maxlen 20 已经把最早 5 条丢了
            self.assertEqual(len(batch), 20)
            # 剩下的 20 条按 seq 升序、用户文本不重复不丢
            self.assertEqual(
                [t.user_text for t in batch],
                [f"user {i}" for i in range(5, 25)],
            )


class ExtractorFailureTests(unittest.TestCase):
    """抽取失败时 buffer 不清空，下次连带重试。"""

    def test_failed_extraction_preserves_buffer(self):
        with TemporaryDirectory() as tmp:
            worker = _make_worker(
                tmp,
                memory_extractor=lambda turns: False,
            )
            worker._memory_buffer.append((1, CompletedTurn("alpha", "A")))
            original_size = len(worker._memory_buffer)

            worker._maybe_extract_memory()

            # 返回 False 不触发 ack
            self.assertEqual(len(worker._memory_buffer), original_size)
            # 再次调用，buffer 还在
            worker._maybe_extract_memory()
            self.assertEqual(len(worker._memory_buffer), original_size)

    def test_exception_in_extractor_preserves_buffer(self):
        with TemporaryDirectory() as tmp:
            def boom(turns):
                raise RuntimeError("LLM 抖了")

            worker = _make_worker(tmp, memory_extractor=boom)
            worker._memory_buffer.append((1, CompletedTurn("alpha", "A")))

            worker._maybe_extract_memory()  # 不应抛

            self.assertEqual(len(worker._memory_buffer), 1)


class SuccessfulAckTests(unittest.TestCase):
    """成功抽取后只 ack 已消费的那部分，新进的 turn 保留。"""

    def test_successful_extraction_acks_only_consumed_turns(self):
        """LLM 推理期间到达的 turn 不会被本次 ack 误清。"""
        with TemporaryDirectory() as tmp:
            # 在第一次抽取时「LLM 推理期间」追加 seq 100
            def extractor(turns):
                worker._memory_buffer.append(
                    (100, CompletedTurn("late arrival", "L"))
                )
                worker._memory_seq = 100
                return ExtractionStatus.UPDATED

            worker = _make_worker(tmp, memory_extractor=extractor)
            worker._memory_buffer.append((1, CompletedTurn("first", "A")))
            worker._memory_buffer.append((2, CompletedTurn("second", "B")))
            worker._memory_seq = 2

            worker._maybe_extract_memory()

            # seq 1, 2 是 drain 时的 max_seq=2；seq 100 是 LLM 期间新增
            # ack 条件是 seq > max_seq=2，seq 100 满足，保留
            remaining = list(worker._memory_buffer)
            self.assertEqual([seq for seq, _ in remaining], [100])
            self.assertEqual(remaining[0][1].user_text, "late arrival")


class ThrottleTests(unittest.TestCase):
    """_memory_due 的两条独立门控。"""

    def test_within_min_interval_does_not_extract(self):
        with TemporaryDirectory() as tmp:
            calls = []
            worker = _make_worker(
                tmp,
                memory_extractor=lambda turns: (calls.append(turns) or ExtractionStatus.FAILED),
                memory_min_interval_s=60.0,
            )
            # 刚抽过——再调不应触发
            worker._last_memory_at = time.monotonic()
            worker._memory_buffer.append((1, CompletedTurn("a", "A")))
            worker._maybe_extract_memory()
            self.assertEqual(calls, [])

    def test_past_min_interval_with_chars_extracts(self):
        with TemporaryDirectory() as tmp:
            calls = []
            worker = _make_worker(
                tmp,
                memory_extractor=lambda turns: (calls.append(turns) or ExtractionStatus.FAILED),
            )
            worker._last_memory_at = 0.0
            worker._memory_buffer.append((1, CompletedTurn("alpha beta", "A")))
            worker._maybe_extract_memory()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0].user_text, "alpha beta")

    def test_past_min_interval_but_no_chars_does_not_extract(self):
        with TemporaryDirectory() as tmp:
            calls = []
            worker = _make_worker(
                tmp,
                memory_extractor=lambda turns: (calls.append(turns) or ExtractionStatus.FAILED),
                memory_min_text_chars=50,
            )
            worker._last_memory_at = 0.0
            worker._memory_buffer.append((1, CompletedTurn("嗯", "嗯")))
            worker._maybe_extract_memory()
            self.assertEqual(calls, [])

    def test_empty_buffer_does_not_extract(self):
        with TemporaryDirectory() as tmp:
            calls = []
            worker = _make_worker(
                tmp,
                memory_extractor=lambda turns: (calls.append(turns) or ExtractionStatus.FAILED),
            )
            worker._last_memory_at = 0.0
            worker._maybe_extract_memory()
            self.assertEqual(calls, [])


class OnMemoryUpdatedTests(unittest.TestCase):
    """回调只在真的落库后才触发。"""

    def test_callback_fires_only_on_landing(self):
        with TemporaryDirectory() as tmp:
            updates: list[int] = []

            def extractor(turns):
                return ExtractionStatus.FAILED

            worker = _make_worker(
                tmp,
                memory_extractor=extractor,
                on_memory_updated=lambda: updates.append(1),
            )
            worker._last_memory_at = 0.0
            worker._memory_buffer.append((1, CompletedTurn("a", "A")))
            worker._maybe_extract_memory()
            self.assertEqual(updates, [])

            def extractor_landed(turns):
                return ExtractionStatus.UPDATED

            worker.memory_extractor = extractor_landed
            worker._last_memory_at = 0.0
            worker._memory_buffer.append((2, CompletedTurn("b", "B")))
            worker._maybe_extract_memory()
            self.assertEqual(updates, [1])


class DisabledMemoryTests(unittest.TestCase):
    """memory_extractor=None 时整套记忆路径不参与。"""

    def test_no_extractor_no_op(self):
        with TemporaryDirectory() as tmp:
            worker = _make_worker(tmp, memory_extractor=None)
            worker._last_memory_at = 0.0
            worker._memory_buffer.append((1, CompletedTurn("a", "A")))
            worker._maybe_extract_memory()  # 不应崩
            # 没抽也无所谓——buffer 原封不动
            self.assertEqual(len(worker._memory_buffer), 1)


if __name__ == "__main__":
    unittest.main()
