import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from soul_tty import config
from soul_tty.relationship import (
    CompletedTurn,
    RelationshipService,
    RelationshipState,
    apply_evaluation,
    load_state,
    save_state,
    tier_for,
)


class RelationshipStateTests(unittest.TestCase):
    def test_tier_boundaries(self):
        self.assertEqual(tier_for(0), "初识")
        self.assertEqual(tier_for(15), "熟悉")
        self.assertEqual(tier_for(35), "亲近")
        self.assertEqual(tier_for(60), "默契")
        self.assertEqual(tier_for(85), "灵魂共鸣")

    def test_applies_only_confident_bounded_llm_evaluation(self):
        state = RelationshipState(score=20, session_count=2)
        result = {
            "event": "真诚关心",
            "delta": 99,
            "mood": "warm",
            "inner_voice": "被你惦记着真好。",
            "confidence": 0.9,
        }
        with patch.object(config, "RELATIONSHIP_MAX_DELTA", 2):
            updated = apply_evaluation(state, result)

        self.assertIsNotNone(updated)
        self.assertEqual(updated.score, 22)
        self.assertEqual(updated.mood, "warm")
        self.assertEqual(updated.inner_voice, "被你惦记着真好。")
        self.assertEqual(updated.session_count, 3)

    def test_rejects_low_confidence_evaluation(self):
        with patch.object(config, "RELATIONSHIP_MIN_CONFIDENCE", 0.65):
            updated = apply_evaluation(
                RelationshipState(score=20),
                {"delta": 2, "confidence": 0.2},
            )
        self.assertIsNone(updated)

    def test_rejects_non_finite_confidence_and_long_voice_over(self):
        state = RelationshipState(score=20)
        self.assertIsNone(
            apply_evaluation(state, {"delta": 2, "confidence": "nan"})
        )
        updated = apply_evaluation(
            state,
            {
                "delta": 0,
                "mood": "calm",
                "inner_voice": "这是一句明显超过界面允许长度的关系画外音文本",
                "confidence": 0.9,
            },
        )
        self.assertEqual(updated.inner_voice, "")

    def test_state_persistence_round_trip(self):
        state = RelationshipState(
            score=47,
            mood="shy",
            event="共同玩笑",
            inner_voice="好像更懂你一点了。",
            session_count=8,
            updated_at="2026-08-03T12:00:00+08:00",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "relationships" / "serena.json"
            save_state(path, state)
            restored = load_state(path)

        self.assertEqual(restored, state)


class RelationshipServiceTests(unittest.TestCase):
    def test_llm_evaluation_runs_off_the_main_path(self):
        entered = threading.Event()
        release = threading.Event()
        updated = threading.Event()

        def evaluate(state, turn):
            entered.set()
            release.wait(timeout=1)
            return {
                "event": "关心",
                "delta": 1,
                "mood": "warm",
                "inner_voice": "被你惦记着真好。",
                "confidence": 0.9,
            }

        with TemporaryDirectory() as directory:
            service = RelationshipService(
                "serena",
                evaluate,
                lambda state: updated.set(),
                state_dir=Path(directory),
                queue_size=2,
                idle_delay_s=0,
            )
            service.start()
            started_at = time.perf_counter()
            accepted = service.submit("今天好吗", "见到你就很好。")
            submit_elapsed = time.perf_counter() - started_at

            self.assertTrue(accepted)
            self.assertLess(submit_elapsed, 0.05)
            self.assertTrue(entered.wait(timeout=1))

            second_started_at = time.perf_counter()
            self.assertTrue(service.submit("还在吗", "我在。"))
            self.assertLess(time.perf_counter() - second_started_at, 0.05)

            release.set()
            self.assertTrue(updated.wait(timeout=1))
            service.stop()

            persisted = load_state(
                Path(directory) / "relationships" / "serena.json"
            )

        self.assertGreaterEqual(
            persisted.score,
            config.RELATIONSHIP_INITIAL_SCORE + 1,
        )
        self.assertLessEqual(
            persisted.score,
            config.RELATIONSHIP_INITIAL_SCORE + 2,
        )
        self.assertEqual(persisted.mood, "warm")

    def test_evaluator_failure_is_silently_ignored(self):
        called = threading.Event()

        def evaluate(state: RelationshipState, turn: CompletedTurn):
            called.set()
            raise RuntimeError("temporary model failure")

        with TemporaryDirectory() as directory:
            service = RelationshipService(
                "serena",
                evaluate,
                state_dir=Path(directory),
                idle_delay_s=0,
            )
            service.start()
            self.assertTrue(service.submit("你好", "你好呀"))
            self.assertTrue(called.wait(timeout=1))
            service.stop()

        self.assertEqual(service.state.score, config.RELATIONSHIP_INITIAL_SCORE)


if __name__ == "__main__":
    unittest.main()
