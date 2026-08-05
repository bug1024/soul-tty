import json
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
    level_for,
    load_state,
    save_state,
)


class RelationshipStateTests(unittest.TestCase):
    def test_level_boundaries(self):
        self.assertEqual(level_for(0.0), "stranger")
        self.assertEqual(level_for(0.05), "stranger")
        self.assertEqual(level_for(0.1), "acquaintance")
        self.assertEqual(level_for(0.3), "familiar")
        self.assertEqual(level_for(0.5), "companion")
        self.assertEqual(level_for(0.7), "close")
        self.assertEqual(level_for(0.9), "bonded")
        self.assertEqual(level_for(1.0), "bonded")
        # 边界保护：超过 [0,1] 范围的输入也要落在合法 label
        self.assertEqual(level_for(-0.5), "stranger")
        self.assertEqual(level_for(1.5), "bonded")

    def test_applies_only_confident_bounded_llm_evaluation(self):
        state = RelationshipState(bond=0.20, session_count=2)
        result = {
            "event": "真诚关心",
            "relationship_delta": {"bond": 99},
            "inner_voice": "被你惦记着真好。",
            "confidence": 0.9,
        }
        with patch.object(config, "RELATIONSHIP_MAX_DELTA", 0.03):
            payload = apply_evaluation(state, result)

        self.assertIsNotNone(payload)
        updated = payload["relationship"]
        # 边际递减：0.20 + 0.03 * (1 - 0.20) = 0.224
        self.assertAlmostEqual(updated.bond, 0.224, places=4)
        self.assertEqual(updated.inner_voice, "被你惦记着真好。")
        # session_count 由 RelationshipService 在每轮评估时统一递增；
        # apply_evaluation 不再触碰它。
        self.assertEqual(updated.session_count, 2)

    def test_rejects_low_confidence_evaluation(self):
        with patch.object(config, "RELATIONSHIP_MIN_CONFIDENCE", 0.65):
            updated = apply_evaluation(
                RelationshipState(bond=0.20),
                {"delta": 2, "confidence": 0.2},
            )
        self.assertIsNone(updated)

    def test_rejects_non_finite_confidence_and_long_voice_over(self):
        state = RelationshipState(bond=0.20)
        self.assertIsNone(
            apply_evaluation(
                state, {"relationship_delta": {"bond": 0.02}, "confidence": "nan"}
            )
        )
        payload = apply_evaluation(
            state,
            {
                "relationship_delta": {"bond": 0},
                "inner_voice": "这是一句明显超过界面允许长度的关系画外音文本",
                "confidence": 0.9,
            },
        )
        self.assertEqual(payload["relationship"].inner_voice, "")

    def test_rejects_mechanism_language_and_third_person_narration(self):
        state = RelationshipState(bond=0.20)
        for voice in ("关系更甜蜜了。", "亲密度提升了。", "她似乎很开心。"):
            payload = apply_evaluation(
                state,
                {
                    "relationship_delta": {"bond": 0},
                    "inner_voice": voice,
                    "confidence": 0.9,
                },
            )
            self.assertEqual(payload["relationship"].inner_voice, "")

    def test_persistent_relationship_resets_session_only_voice(self):
        state = RelationshipState(
            bond=0.47,
            event="共同玩笑",
            inner_voice="好像更懂你一点了。",
            session_count=8,
            updated_at="2026-08-03T12:00:00+08:00",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "relationships" / "serena.json"
            save_state(path, state)
            raw = json.loads(path.read_text(encoding="utf-8"))
            restored = load_state(path)

        self.assertEqual(restored.bond, state.bond)
        self.assertEqual(restored.session_count, state.session_count)
        self.assertEqual(restored.event, state.event)
        self.assertEqual(restored.updated_at, state.updated_at)
        self.assertEqual(restored.inner_voice, "")
        self.assertNotIn("inner_voice", raw)

    def test_legacy_score_field_migrates_to_bond(self):
        """旧 JSON 用 score (0~100) → 启动加载时映射为 bond (0~1)，不写回。"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_score.json"
            path.write_text(
                json.dumps(
                    {"score": 75, "event": "老数据", "session_count": 5},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            restored = load_state(path)
            # 75 / 100 = 0.75
            self.assertAlmostEqual(restored.bond, 0.75, places=4)
            self.assertEqual(restored.event, "老数据")
            self.assertEqual(restored.session_count, 5)
            # 内存迁移不写回原文件
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("score", raw)
            self.assertNotIn("bond", raw)

    def test_bond_delta_follows_diminishing_returns(self):
        """同样 delta=0.03，bond 越高增长越少（边际递减）。"""
        low_state = RelationshipState(bond=0.05)
        high_state = RelationshipState(bond=0.90)
        result = {
            "relationship_delta": {"bond": 0.03},
            "confidence": 0.9,
        }
        with patch.object(config, "RELATIONSHIP_MAX_DELTA", 0.03):
            low = apply_evaluation(low_state, result)["relationship"]
            high = apply_evaluation(high_state, result)["relationship"]
        # 0.05 + 0.03*(1-0.05) = 0.0785
        self.assertAlmostEqual(low.bond, 0.0785, places=4)
        # 0.90 + 0.03*(1-0.90) = 0.903
        self.assertAlmostEqual(high.bond, 0.903, places=4)
        self.assertGreater(low.bond - 0.05, high.bond - 0.90)


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
                min_interval_s=0,
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

        # 至少发生一次增长；evaluate 可能被调用 1~2 次（取决于 release 时机）。
        # 每次最大增长受 MAX_DELTA 限制且边际递减，用 3 次上限覆盖所有路径。
        self.assertGreater(persisted.bond, config.RELATIONSHIP_INITIAL_BOND)
        self.assertLessEqual(
            persisted.bond,
            # 3 次最大增长，叠加浮点容差
            0.05 + 0.03 * 3 + 1e-4,
        )

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
                min_interval_s=0,
            )
            service.start()
            self.assertTrue(service.submit("你好", "你好呀"))
            self.assertTrue(called.wait(timeout=1))
            service.stop()

        self.assertEqual(service.state.bond, config.RELATIONSHIP_INITIAL_BOND)

    def test_pending_turns_are_coalesced_into_one_llm_evaluation(self):
        evaluated = []
        completed = threading.Event()

        def evaluate(state, turn):
            evaluated.append(turn)
            completed.set()
            return {
                "event": "连续交流",
                "delta": 1,
                "inner_voice": "和你聊得很开心。",
                "confidence": 0.9,
            }

        with TemporaryDirectory() as directory:
            service = RelationshipService(
                "serena",
                evaluate,
                state_dir=Path(directory),
                idle_delay_s=0,
                min_interval_s=0,
            )
            service.submit("第一问", "第一答")
            service.submit("第二问", "第二答")
            service.start()
            self.assertTrue(completed.wait(timeout=1))
            service.stop()

        self.assertEqual(len(evaluated), 1)
        self.assertIn("第1轮：第一问", evaluated[0].user_text)
        self.assertIn("第2轮：第二答", evaluated[0].agent_text)

    def test_session_count_increments_per_evaluation_regardless_of_confidence(self):
        """每轮 LLM 评估都让 session_count +1（不管 confidence 够不够）。"""
        updated = threading.Event()

        def evaluate(state, turn):
            return {
                "event": "弱信号",
                "relationship_delta": {"bond": 0.0},
                "inner_voice": "我还在。",
                "confidence": 0.1,  # 低于默认 0.65
                "emotion_delta": {},
                "expression": "neutral",
            }

        with TemporaryDirectory() as directory:
            service = RelationshipService(
                "serena",
                evaluate,
                lambda state: updated.set(),
                state_dir=Path(directory),
                queue_size=2,
                idle_delay_s=0,
                min_interval_s=0,
            )
            self.assertEqual(service.state.session_count, 0)
            service.start()
            self.assertTrue(service.submit("hi", "hello back"))
            self.assertTrue(updated.wait(timeout=1))
            service.stop()

        # 即便 confidence 不足，session_count 也应该 +1
        self.assertEqual(service.state.session_count, 1)
        # bond 不变：低 confidence 不扣分也不加分
        self.assertEqual(service.state.bond, config.RELATIONSHIP_INITIAL_BOND)


if __name__ == "__main__":
    unittest.main()


# --- Task 12: LLM prompt schema ---

def test_evaluate_relationship_system_prompt_declares_three_state_lanes():
    from src.soul_tty.clients import llm as llm_mod
    import inspect

    src = inspect.getsource(llm_mod.evaluate_relationship)
    # 三路状态拆分：relationship_delta / emotion_delta / expression
    assert "relationship_delta" in src
    assert "emotion_delta" in src
    assert "expression" in src


# --- Task 13/14: apply_evaluation returns payload + emotion hook ---

def test_apply_evaluation_returns_emotion_payload():
    from src.soul_tty.relationship import (
        RelationshipState,
        apply_evaluation,
    )

    state = RelationshipState(bond=0.10)
    result = {
        "event": "user shared",
        "relationship_delta": {"bond": 0.01},
        "inner_voice": "替你高兴",
        "confidence": 0.85,
        "emotion_delta": {"happiness": 0.15, "stress": -0.05},
        "expression": "caring",
    }
    payload = apply_evaluation(state, result)
    assert payload is not None
    # 边际递减：0.10 + 0.01 * (1 - 0.10) = 0.109
    assert abs(payload["relationship"].bond - 0.109) < 1e-6
    assert payload["emotion_delta"] == {"happiness": 0.15, "stress": -0.05}
    assert payload["expression"] == "caring"


def test_relationship_service_calls_emotion_apply():
    import time
    import tempfile
    from src.soul_tty.relationship import RelationshipService
    from src.soul_tty.emotion.state import EmotionVector
    from src.soul_tty.emotion.service import EmotionSnapshot

    calls = []

    class FakeEmotion:
        baseline = EmotionVector(
            happiness=0.5, calmness=0.5, curiosity=0.5, stress=0.5, energy=0.5
        )

        def apply_delta(self, delta, *, expression_hint="neutral"):
            calls.append((dict(delta), expression_hint))
            return EmotionSnapshot(
                baseline=self.baseline,
                emotion=self.baseline,
                mood="calm",
                intensity=0.5,
                expression=expression_hint,
                should_update_prompt=False,
                context_text="",
            )

    fake = FakeEmotion()

    def fake_evaluator(state, turn):
        return {
            "relationship_delta": {"bond": 0.01},
            "inner_voice": "替你高兴",
            "confidence": 0.9,
            "emotion_delta": {"happiness": 0.1},
            "expression": "caring",
        }

    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        svc = RelationshipService(
            persona_id="serena",
            evaluator=fake_evaluator,
            on_update=None,
            state_dir=Path(tmp),
            queue_size=4,
            idle_delay_s=0.0,
            min_interval_s=0.0,
        )
        svc.emotion = fake
        svc.start()
        assert svc.submit("hi", "hello back") is True
        for _ in range(50):
            if calls:
                break
            time.sleep(0.05)
        svc.stop()
        assert len(calls) >= 1
        assert calls[0][0] == {"happiness": 0.1}
        assert calls[0][1] == "caring"
