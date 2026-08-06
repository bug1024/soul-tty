"""recall 注入：临时 system message，不进 system prompt / 不进 history。"""

import unittest
from unittest.mock import MagicMock, patch

from soul_tty import config
from soul_tty.clients.llm import Chat


def _fake_client():
    """构造一个支持 `with httpx.Client(...) as client` 语法的 fake。"""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)

    # 至少产一个有效 token（带句号），让 Chat.ask_stream 不走 messages.pop()
    # 流式增量是逐句 yield 的，所以单条有效响应即可
    line = (
        'data: {"choices":[{"delta":{"content":"ok。"}}]}'
    )
    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.raise_for_status = MagicMock()
    stream.iter_lines = MagicMock(return_value=iter([line, "data: [DONE]"]))
    client.stream = MagicMock(return_value=stream)
    return client


class RecallInjectionTests(unittest.TestCase):
    def setUp(self):
        self._client = _fake_client()
        import soul_tty.clients.llm as llm_mod
        self._cls_patcher = patch.object(
            llm_mod.httpx, "Client", return_value=self._client
        )
        self._cls_patcher.start()
        self.addCleanup(self._cls_patcher.stop)

    def _last_payload_messages(self):
        # client.stream 被以关键字 json= 调用，payload 在那里
        return self._client.stream.call_args.kwargs["json"]["messages"]

    def test_recall_appears_in_payload_between_history_and_user(self):
        chat = Chat("m")
        list(chat.ask_stream(
            "你还记得我上次说的项目吗",
            recall="[Relevant Memories]\n- x",
        ))
        messages = self._last_payload_messages()
        # 倒数第二条 = recall 临时段，倒数第一条 = 本轮 user
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "你还记得我上次说的项目吗")
        self.assertEqual(messages[-2]["role"], "system")
        self.assertIn("[Relevant Memories]", messages[-2]["content"])

    def test_recall_does_not_pollute_messages_history(self):
        chat = Chat("m")
        list(chat.ask_stream("你还记得吗", recall="[Relevant Memories]\n- x"))
        # self.messages 不能含 [Relevant Memories]——recall 是临时段
        roles = [m["role"] for m in chat.messages]
        self.assertEqual(roles, ["system", "user", "assistant"])
        for message in chat.messages:
            self.assertNotIn("[Relevant Memories]", message["content"])
        # 整个 history 都不带 recall
        for message in chat.messages:
            self.assertNotIn("[Relevant Memories]", message["content"])

    def test_empty_recall_keeps_history_intact(self):
        chat = Chat("m")
        list(chat.ask_stream("今天天气", recall=""))
        messages = self._last_payload_messages()
        self.assertEqual(messages, chat.messages)

    def test_recall_is_invisible_to_next_turn(self):
        """recall 临时段只影响本轮请求，不能跨轮复用。"""
        chat = Chat("m")
        list(chat.ask_stream("你还记得吗", recall="[Relevant Memories]\n- x"))
        list(chat.ask_stream("继续", recall=""))
        messages = self._last_payload_messages()
        for message in messages:
            if message["role"] == "system":
                self.assertNotIn("[Relevant Memories]", message["content"])

    def test_recall_sits_after_history_before_user(self):
        chat = Chat("m")
        list(chat.ask_stream("first", recall=""))
        list(chat.ask_stream("second", recall="[Relevant Memories]\n- y"))
        messages = self._last_payload_messages()
        # system, user(first), assistant(first), system(recall), user(second)
        roles = [m["role"] for m in messages]
        self.assertEqual(
            roles,
            ["system", "user", "assistant", "system", "user"],
        )
        recall_index = next(
            i
            for i, m in enumerate(messages)
            if "[Relevant Memories]" in m.get("content", "")
        )
        last_user_index = len(messages) - 1
        self.assertLess(recall_index, last_user_index)


if __name__ == "__main__":
    unittest.main()

