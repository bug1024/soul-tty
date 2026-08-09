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
from soul_tty.agency.state import continuity_update, load_state
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
            self.assertEqual(raw["schema_version"], 1)


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
