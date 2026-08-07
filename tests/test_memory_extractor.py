"""MemoryExtractor：把多轮对话拼成一次 LLM 抽取，交给 MemoryService 落库。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from soul_tty.clients.llm import extract_memories
from soul_tty.memory.extractor import extract_from_turns
from soul_tty.memory.models import ExtractionStatus, TYPE_EXPERIENCE, TYPE_PROFILE
from soul_tty.memory.service import MemoryService
from soul_tty.reflection.relationship import CompletedTurn


class LLMJsonShapeTests(unittest.TestCase):
    """确认抽出的 memories 数组结构稳定（空数组也是有意义的结构）。"""

    def test_no_memories_returns_empty_list(self):
        from unittest.mock import patch as _patch
        with _patch("httpx.Client") as client_cls:
            client = client_cls.return_value.__enter__.return_value
            client.post.return_value.json.return_value = {
                "choices": [{"message": {"content": '{"memories":[]}'}}]
            }
            client.post.return_value.raise_for_status = lambda: None
            result = extract_memories("model", "Serena", [], "user", "agent")
        self.assertEqual(result, {"memories": []})

    def test_parses_valid_memories(self):
        from unittest.mock import patch as _patch
        with _patch("httpx.Client") as client_cls:
            client = client_cls.return_value.__enter__.return_value
            client.post.return_value.json.return_value = {
                "choices": [{
                    "message": {
                        "content": (
                            '{"memories":['
                            '{"type":"profile","content":"用户是工程师",'
                            '"importance":0.9}]}'
                        )
                    }
                }]
            }
            client.post.return_value.raise_for_status = lambda: None
            result = extract_memories("model", "Serena", [], "user", "agent")
        self.assertEqual(len(result["memories"]), 1)
        self.assertEqual(result["memories"][0]["type"], TYPE_PROFILE)


class ExtractorFlowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.service = MemoryService(Path(self._tmp.name) / "memory.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_extracts_and_persists(self):
        turns = [CompletedTurn("我女儿今年5岁", "真的吗？")]
        with patch(
            "soul_tty.memory.extractor.extract_memories",
            return_value={
                "memories": [
                    {
                        "type": "profile",
                        "content": "用户有一个5岁的女儿",
                        "importance": 0.8,
                    }
                ]
            },
        ):
            extract_from_turns(
                self.service,
                persona_id="serena",
                display_name="Serena",
                model="m",
                turns=turns,
            )
        rows = self.service._store.list()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].content, "用户有一个5岁的女儿")

    def test_coalesces_multiple_turns_into_one_call(self):
        turns = [
            CompletedTurn("我女儿5岁", "嗯嗯"),
            CompletedTurn("她喜欢踢足球", "真好"),
            CompletedTurn("我们最近搬了家", "搬到哪里了"),
        ]
        with patch(
            "soul_tty.memory.extractor.extract_memories",
            return_value={"memories": []},
        ) as extract:
            extract_from_turns(
                self.service,
                persona_id="serena",
                display_name="Serena",
                model="m",
                turns=turns,
            )
        self.assertEqual(extract.call_count, 1)
        # 三个 turn 都被拼进 prompt
        called = extract.call_args.kwargs
        self.assertIn("我女儿5岁", called["user_text"])
        self.assertIn("她喜欢踢足球", called["user_text"])
        self.assertIn("我们最近搬了家", called["user_text"])

    def test_passes_known_facts_to_extractor(self):
        with patch(
            "soul_tty.memory.extractor.extract_memories",
            return_value={"memories": []},
        ) as extract:
            extract_from_turns(
                self.service,
                persona_id="serena",
                display_name="Serena",
                model="m",
                turns=[CompletedTurn("普通闲聊", "嗯")],
                known_facts=[
                    {"scope": "global", "type": "profile", "content": "是工程师"}
                ],
            )
        self.assertEqual(
            extract.call_args.kwargs["known_facts"],
            [
                {
                    "scope": "global",
                    "type": "profile",
                    "content": "是工程师",
                }
            ],
        )

    def test_uses_memory_llm_url_when_configured(self):
        with patch(
            "soul_tty.memory.extractor.extract_memories",
            return_value={"memories": []},
        ) as extract:
            with patch("soul_tty.config.MEMORY_LLM_URL", "http://mem:9999"):
                extract_from_turns(
                    self.service,
                    persona_id="serena",
                    display_name="Serena",
                    model="m",
                    turns=[CompletedTurn("x", "y")],
                )
        self.assertEqual(extract.call_args.kwargs["model"], "m")

    def test_remember_result_tells_caller_whether_something_landed(self):
        with patch(
            "soul_tty.memory.extractor.extract_memories",
            return_value={"memories": []},
        ):
            landed = extract_from_turns(
                self.service,
                persona_id="serena",
                display_name="Serena",
                model="m",
                turns=[CompletedTurn("x", "y")],
            )
        self.assertIs(landed, ExtractionStatus.NO_CHANGE)

    def test_returns_true_when_at_least_one_memory_lands(self):
        with patch(
            "soul_tty.memory.extractor.extract_memories",
            return_value={
                "memories": [
                    {
                        "type": TYPE_EXPERIENCE,
                        "content": "一起完成 Soul-TTY",
                        "importance": 0.9,
                    }
                ]
            },
        ):
            landed = extract_from_turns(
                self.service,
                persona_id="serena",
                display_name="Serena",
                model="m",
                turns=[CompletedTurn("我们完成 Soul-TTY 了", "恭喜")],
            )
        self.assertIs(landed, ExtractionStatus.UPDATED)

    def test_service_unavailable_returns_false_silently(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            path.write_bytes(b"not a database")
            service = MemoryService(path)
            with patch(
                "soul_tty.memory.extractor.extract_memories",
                return_value={"memories": []},
            ) as extract:
                landed = extract_from_turns(
                    service,
                    persona_id="serena",
                    display_name="Serena",
                    model="m",
                    turns=[CompletedTurn("x", "y")],
                )
        self.assertIs(landed, ExtractionStatus.FAILED)
        # 不可用时连 LLM 都不发，省一次尾延迟
        self.assertEqual(extract.call_count, 0)


if __name__ == "__main__":
    unittest.main()
