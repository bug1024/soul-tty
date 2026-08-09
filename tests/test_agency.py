import json
import random
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from soul_tty.agency import AgencyService, AgencyState, ResponseMode, ResponsePolicy
from soul_tty.agency.state import continuity_update, evolve_after_decision, load_state
from soul_tty.clients.llm import Chat
from soul_tty import conversation


class AgencyStateTests(unittest.TestCase):
    def test_social_and_solitude_needs_are_independent(self):
        state = AgencyState(desire_for_company=0.78, solitude_need=0.71)

        self.assertGreater(state.desire_for_company, 0.7)
        self.assertGreater(state.solitude_need, 0.7)

    def test_continuity_recovers_energy_and_can_increase_company_need(self):
        now = datetime(2026, 8, 9, 10, tzinfo=timezone.utc)
        state = AgencyState(
            social_energy=0.20,
            desire_for_company=0.30,
            last_interaction_at=(now - timedelta(hours=10)).isoformat(),
        )

        updated = continuity_update(state, now=now)

        self.assertGreater(updated.social_energy, state.social_energy)
        self.assertGreater(updated.desire_for_company, state.desire_for_company)

    def test_passive_answer_builds_debt_and_active_move_pays_it_down(self):
        state = evolve_after_decision(
            AgencyState(),
            mode="answer",
            mood="calm",
            user_text="你今天怎么样？",
        )
        self.assertAlmostEqual(state.initiative_debt, 0.18)
        self.assertEqual(state.passive_answer_streak, 1)

        state = evolve_after_decision(
            state,
            mode="answer_and_lead",
            mood="calm",
            user_text="刚才在干嘛？",
        )
        self.assertEqual(state.initiative_debt, 0.0)
        self.assertEqual(state.passive_answer_streak, 0)

    def test_short_reply_also_builds_passive_debt(self):
        state = evolve_after_decision(
            AgencyState(),
            mode="short_reply",
            mood="calm",
            user_text="对吧，真甜",
        )

        self.assertAlmostEqual(state.initiative_debt, 0.18)
        self.assertEqual(state.passive_answer_streak, 1)

    def test_protected_task_does_not_create_initiative_debt(self):
        state = evolve_after_decision(
            AgencyState(),
            mode="answer",
            mood="calm",
            user_text="帮我分析这段代码",
            protected=True,
        )
        self.assertEqual(state.initiative_debt, 0.0)
        self.assertEqual(state.passive_answer_streak, 0)

    def test_change_topic_consumes_the_presented_thought(self):
        state = evolve_after_decision(
            AgencyState(unresolved_thoughts=("先提这个", "以后再提那个")),
            mode="change_topic",
            mood="curious",
            user_text="今天天气不错",
        )
        self.assertEqual(state.unresolved_thoughts, ("以后再提那个",))


class ResponsePolicyTests(unittest.TestCase):
    def _quiet_state(self, **changes):
        state = AgencyState(
            social_energy=0.18,
            desire_to_talk=0.12,
            desire_for_company=0.68,
            solitude_need=0.84,
            turn_count=8,
        )
        return replace(state, **changes)

    def test_explicit_request_is_never_silenced(self):
        policy = ResponsePolicy(silence_rate=1.0, rng=random.Random(1))

        decision = policy.decide(
            self._quiet_state(), "停下，别说了", relationship_level="close"
        )

        self.assertEqual(decision.mode, ResponseMode.ANSWER)

    def test_sensitive_statement_is_never_silenced(self):
        policy = ResponsePolicy(silence_rate=1.0, rng=random.Random(1))

        decision = policy.decide(
            self._quiet_state(), "我今天真的很难过", relationship_level="close"
        )

        self.assertEqual(decision.mode, ResponseMode.ANSWER)

    def test_fact_or_operation_question_is_still_protected(self):
        policy = ResponsePolicy(
            answer_and_lead_rate=0,
            self_express_rate=1,
            rng=random.Random(1),
        )

        decision = policy.decide(AgencyState(), "Python 怎么读取文件？")

        self.assertEqual(decision.mode, ResponseMode.ANSWER)
        self.assertTrue(decision.protected)

    def test_social_question_is_not_forced_to_answer(self):
        policy = ResponsePolicy(
            answer_and_lead_rate=0,
            self_express_rate=1,
            ask_rate=0,
            change_topic_rate=0,
            rng=random.Random(1),
        )

        decision = policy.decide(AgencyState(), "你今天怎么样？")

        self.assertEqual(decision.mode, ResponseMode.SELF_EXPRESS)
        self.assertFalse(decision.protected)

        decision = policy.decide(AgencyState(), "你是什么样的人？")
        self.assertEqual(decision.mode, ResponseMode.SELF_EXPRESS)

    def test_third_passive_question_forces_answer_and_lead(self):
        policy = ResponsePolicy(
            answer_and_lead_rate=0,
            self_express_rate=0,
            ask_rate=0,
            change_topic_rate=0,
            rng=random.Random(1),
        )
        state = AgencyState(
            social_energy=0.68,
            desire_to_talk=0.62,
            passive_answer_streak=2,
            initiative_debt=0.36,
        )

        decision = policy.decide(state, "你喜欢晚上吗？")

        self.assertEqual(decision.mode, ResponseMode.ANSWER_AND_LEAD)
        self.assertEqual(decision.reason, "passive_answer_limit")
        self.assertIn("主动拿走下一步", decision.instruction)

    def test_low_need_name_ping_can_choose_intentional_silence(self):
        policy = ResponsePolicy(silence_rate=1.0, rng=random.Random(1))

        decision = policy.decide(
            self._quiet_state(), "Serena？", relationship_level="close"
        )

        self.assertEqual(decision.mode, ResponseMode.SILENCE)

    def test_never_silences_twice_in_a_row(self):
        policy = ResponsePolicy(silence_rate=1.0, rng=random.Random(1))

        decision = policy.decide(
            self._quiet_state(consecutive_silences=1),
            "Serena？",
            relationship_level="close",
        )

        self.assertEqual(decision.mode, ResponseMode.SHORT_REPLY)

    def test_low_energy_breaks_out_of_repeated_short_replies(self):
        policy = ResponsePolicy(
            silence_rate=0,
            initiative_debt_threshold=0.50,
            max_passive_answers=2,
            rng=random.Random(1),
        )
        state = self._quiet_state(
            initiative_debt=0.36,
            passive_answer_streak=2,
        )

        decision = policy.decide(
            state,
            "我心里不是装的你吗",
            relationship_level="companion",
        )

        self.assertEqual(decision.mode, ResponseMode.SELF_EXPRESS)
        self.assertEqual(decision.reason, "low_energy_initiative")
        self.assertIn("不要为了推进而提问", decision.instruction)

    def test_low_energy_still_starts_with_a_short_reply(self):
        policy = ResponsePolicy(silence_rate=0, rng=random.Random(1))

        decision = policy.decide(
            self._quiet_state(
                initiative_debt=0.0,
                passive_answer_streak=0,
            ),
            "今天的月亮很好看",
            relationship_level="companion",
        )

        self.assertEqual(decision.mode, ResponseMode.SHORT_REPLY)
        self.assertEqual(decision.reason, "low_social_energy")

    def test_low_energy_cycle_becomes_proactive_on_third_social_turn(self):
        policy = ResponsePolicy(silence_rate=0, rng=random.Random(1))
        state = self._quiet_state(
            initiative_debt=0.0,
            passive_answer_streak=0,
        )

        for text in ("如果是别人呢", "你不是说只属于我吗"):
            decision = policy.decide(
                state,
                text,
                relationship_level="companion",
            )
            self.assertEqual(decision.mode, ResponseMode.SHORT_REPLY)
            state = evolve_after_decision(
                state,
                mode=decision.mode.value,
                mood="calm",
                user_text=text,
            )

        decision = policy.decide(
            state,
            "你好像有点高冷了",
            relationship_level="companion",
        )
        self.assertEqual(decision.mode, ResponseMode.SELF_EXPRESS)
        self.assertEqual(decision.reason, "low_energy_initiative")

    def test_unfamiliar_relationship_does_not_use_silence(self):
        policy = ResponsePolicy(silence_rate=1.0, rng=random.Random(1))

        decision = policy.decide(
            self._quiet_state(), "Serena？", relationship_level="acquaintance"
        )

        self.assertEqual(decision.mode, ResponseMode.SHORT_REPLY)

    def test_persisted_turns_do_not_bypass_new_session_silence_guard(self):
        policy = ResponsePolicy(silence_rate=1.0, rng=random.Random(1))

        decision = policy.decide(
            self._quiet_state(turn_count=100),
            "Serena？",
            relationship_level="close",
            session_turn_count=0,
        )

        self.assertEqual(decision.mode, ResponseMode.SHORT_REPLY)

    def test_unresolved_thread_can_drive_change_topic(self):
        policy = ResponsePolicy(
            silence_rate=0,
            change_topic_rate=1,
            ask_rate=0,
            answer_and_lead_rate=0,
            self_express_rate=0,
            rng=random.Random(1),
        )
        state = AgencyState(
            turn_count=4,
            unresolved_thoughts=("昨天那件重要的事后来怎么样了",),
        )

        decision = policy.decide(
            state, "今天天气不错", relationship_level="close"
        )

        self.assertEqual(decision.mode, ResponseMode.CHANGE_TOPIC)
        self.assertIn("昨天那件重要的事", decision.instruction)


class AgencyServiceTests(unittest.TestCase):
    def test_decision_updates_and_persists_state_without_blocking_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agency.json"
            service = AgencyService(
                path,
                policy=ResponsePolicy(
                    silence_rate=0,
                    change_topic_rate=0,
                    ask_rate=0,
                    answer_and_lead_rate=0,
                    self_express_rate=0,
                ),
            )
            try:
                decision = service.decide(
                    "今天继续开发",
                    mood="curious",
                    relationship_level="companion",
                )
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not path.exists():
                    time.sleep(0.01)
            finally:
                service.close()

            persisted = load_state(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(decision.mode, ResponseMode.ANSWER)
            self.assertEqual(persisted.turn_count, 1)
            self.assertEqual(persisted.mood, "curious")
            self.assertEqual(raw["schema_version"], 2)

    def test_inner_thread_is_deduplicated_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agency.json"
            service = AgencyService(path)
            try:
                self.assertTrue(service.add_thought("用户刚才似乎还有话没说完"))
                self.assertTrue(service.add_thought("用户刚才似乎还有话没说完"))
                self.assertTrue(service.add_thought("想知道那件事后来怎样了"))
            finally:
                service.close()

            persisted = load_state(path)
            self.assertEqual(
                persisted.unresolved_thoughts,
                ("想知道那件事后来怎样了", "用户刚才似乎还有话没说完"),
            )


class _StreamingResponse:
    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"好。"}}]}'
        yield "data: [DONE]"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _CapturingClient:
    payload = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def stream(self, method, url, **kwargs):
        type(self).payload = kwargs["json"]
        return _StreamingResponse()


class AgencyConversationIntegrationTests(unittest.TestCase):
    def test_agency_hook_does_not_break_voice_observation_submission(self):
        pcm = b"\x01\x00" * 80
        with patch.object(conversation, "_voice_submit_provider", lambda _: 42):
            self.assertEqual(conversation._current_voice_ref(pcm), 42)

    @patch("soul_tty.clients.llm.httpx.Client", _CapturingClient)
    def test_response_instruction_is_ephemeral(self):
        chat = Chat("test")

        answer = "".join(
            chat.ask_stream(
                "今天还好吗",
                response_instruction="[Response Policy]\n只说一句短句。",
            )
        )

        request_messages = _CapturingClient.payload["messages"]
        self.assertEqual(answer, "好。")
        self.assertTrue(
            any("Response Policy" in item["content"] for item in request_messages)
        )
        self.assertEqual(
            [item["role"] for item in request_messages].count("system"),
            1,
        )
        self.assertEqual(request_messages[-1]["role"], "user")
        self.assertIn("[Current User Message]\n今天还好吗", request_messages[-1]["content"])
        self.assertFalse(
            any("Response Policy" in item["content"] for item in chat.messages)
        )

    def test_intentional_silence_skips_llm_and_tts(self):
        class SilentChat:
            last_stop_reason = None
            messages = []
            recorded = []

            def record_silence(self, text, *, private=False):
                del private
                self.recorded.append(text)
                self.last_stop_reason = "intentional_silence"

            def ask_stream(self, *args, **kwargs):
                raise AssertionError("SILENCE 不应调用 LLM")

        decision = type(
            "Decision",
            (),
            {
                "mode": ResponseMode.SILENCE,
                "reason": "test",
                "instruction": "",
            },
        )()
        chat = SilentChat()
        with (
            patch.object(conversation, "_response_policy_provider", lambda _: decision),
            patch.object(conversation.terminal, "intentional_silence") as render,
            patch.object(conversation.reflection, "record_turn") as record,
        ):
            answer = conversation._answer_impl(chat, "Serena？")

        self.assertEqual(answer, "")
        self.assertEqual(chat.recorded, ["Serena？"])
        render.assert_called_once_with()
        record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
