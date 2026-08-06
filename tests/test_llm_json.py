"""旁路 LLM 的 JSON 响应清洗。

本地小模型的 JSON 输出常见三种污染：思考标签、Markdown 围栏、前后闲聊。
关系评估与记忆抽取共用同一套清洗逻辑。
"""

import unittest

from soul_tty.clients.llm import _parse_json_object


class ParseJsonObjectTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(_parse_json_object('{"a": 1}'), {"a": 1})

    def test_strips_think_tags(self):
        text = '<think>让我想想\n用户说了什么</think>{"confidence": 0.8}'
        self.assertEqual(_parse_json_object(text), {"confidence": 0.8})

    def test_strips_code_fence(self):
        self.assertEqual(
            _parse_json_object('```json\n{"memories": []}\n```'),
            {"memories": []},
        )

    def test_strips_bare_fence(self):
        self.assertEqual(_parse_json_object('```\n{"a": 1}\n```'), {"a": 1})

    def test_extracts_from_surrounding_chatter(self):
        text = '好的，我的判断如下：{"event": "分享"} 希望有帮助。'
        self.assertEqual(_parse_json_object(text), {"event": "分享"})

    def test_keeps_nested_objects(self):
        text = '{"relationship_delta": {"bond": 0.02}, "confidence": 0.9}'
        self.assertEqual(
            _parse_json_object(text),
            {"relationship_delta": {"bond": 0.02}, "confidence": 0.9},
        )

    def test_returns_none_without_object(self):
        self.assertIsNone(_parse_json_object("我不知道该说什么"))

    def test_returns_none_on_malformed_json(self):
        self.assertIsNone(_parse_json_object('{"a": }'))

    def test_returns_none_on_json_array(self):
        # 顶层必须是对象；数组不是合法 schema。
        self.assertIsNone(_parse_json_object("[1, 2, 3]"))

    def test_returns_none_on_empty(self):
        self.assertIsNone(_parse_json_object(""))


if __name__ == "__main__":
    unittest.main()
