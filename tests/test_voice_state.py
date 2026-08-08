"""VoiceStateService：异步声音感知旁路单元测试。"""

import time
import unittest
from unittest.mock import patch

from soul_tty import config
from soul_tty.audio.voice_state import (
    VoiceObservation,
    VoiceStateService,
    _normalize_emotion,
    _normalize_event,
    _normalize_lang,
    render_voice_context,
)


class NormalizeTests(unittest.TestCase):
    """emotion / event 标签规范化。"""

    def test_emotion_happy(self):
        self.assertEqual(_normalize_emotion("<|HAPPY|>"), "happy")

    def test_emotion_sad(self):
        self.assertEqual(_normalize_emotion("<|SAD|>"), "sad")

    def test_emotion_angry(self):
        self.assertEqual(_normalize_emotion("<|ANGRY|>"), "angry")

    def test_emotion_neutral(self):
        self.assertEqual(_normalize_emotion("<|NEUTRAL|>"), "neutral")

    def test_emotion_unknown(self):
        self.assertEqual(_normalize_emotion("<|BORED|>"), "unknown")

    def test_emotion_empty(self):
        self.assertEqual(_normalize_emotion(""), "unknown")

    def test_event_laughter(self):
        self.assertEqual(_normalize_event("<|Laughter|>"), "laughter")

    def test_event_crying(self):
        self.assertEqual(_normalize_event("<|Cry|>"), "crying")

    def test_event_speech(self):
        self.assertEqual(_normalize_event("<|Speech|>"), "speech")

    def test_event_unknown(self):
        self.assertEqual(_normalize_event("<|Slam|>"), "Slam")

    def test_lang_zh(self):
        self.assertEqual(_normalize_lang("<|zh|>"), "zh")

    def test_lang_en(self):
        self.assertEqual(_normalize_lang("<|en|>"), "en")

    def test_lang_empty(self):
        self.assertEqual(_normalize_lang(""), "unknown")


class VoiceObservationDataclassTests(unittest.TestCase):
    def test_frozen(self):
        obs = VoiceObservation(
            emotion="happy", event="speech", language="zh", duration_ms=1000
        )
        with self.assertRaises(AttributeError):
            obs.emotion = "sad"  # type: ignore[misc]


class VoiceStateServiceSubmitTests(unittest.TestCase):
    """submit 的基本行为：返回 ref、短语音跳过、disabled 时跳过。"""

    def setUp(self):
        self._patcher = patch.object(config, "VOICE_STATE_ENABLED", True)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_submit_returns_ref(self):
        service = VoiceStateService()
        try:
            ref = service.submit(b"\x00\x00" * 16000 * 2)  # 2 秒静音
            self.assertIsInstance(ref, int)
            self.assertGreater(ref, 0)
        finally:
            service.close()

    def test_short_utterance_skipped(self):
        service = VoiceStateService()
        try:
            ref = service.submit(b"\x00\x00" * 160)  # 10ms
            self.assertIsNone(ref)
        finally:
            service.close()

    def test_empty_pcm_skipped(self):
        service = VoiceStateService()
        try:
            ref = service.submit(b"")
            self.assertIsNone(ref)
        finally:
            service.close()

    def test_get_returns_none_for_unresolved(self):
        service = VoiceStateService()
        try:
            obs = service.get(9999)
            self.assertIsNone(obs)
        finally:
            service.close()

    def test_get_many_skips_none(self):
        service = VoiceStateService()
        try:
            result = service.get_many((None, None, 9999))
            self.assertEqual(result, [])
        finally:
            service.close()


class VoiceStateServiceDisabledTests(unittest.TestCase):
    """VOICE_STATE_ENABLED=0 时 submit 静默返回 None。"""

    def test_disabled_skips_submit(self):
        with patch(
            "soul_tty.audio.voice_state.config.VOICE_STATE_ENABLED", False
        ):
            service = VoiceStateService()
            try:
                ref = service.submit(b"\x00\x00" * 16000 * 2)
                self.assertIsNone(ref)
            finally:
                service.close()


class VoiceStateServiceCacheTTLTests(unittest.TestCase):
    """缓存 TTL 行为。"""

    def setUp(self):
        self._patcher = patch.object(config, "VOICE_STATE_ENABLED", True)
        self._patcher.start()
        # 保存原值，测试完恢复
        self._orig_ttl = config.VOICE_STATE_RESULT_TTL_S
        self._orig_qsize = config.VOICE_STATE_QUEUE_SIZE
        config.VOICE_STATE_RESULT_TTL_S = 1  # 1 秒 TTL 方便测试
        config.VOICE_STATE_QUEUE_SIZE = 4

    def tearDown(self):
        config.VOICE_STATE_RESULT_TTL_S = self._orig_ttl
        config.VOICE_STATE_QUEUE_SIZE = self._orig_qsize
        self._patcher.stop()

    def _put(self, service, ref, obs):
        """直接写 cache：(obs, timestamp)"""
        service._cache[ref] = (obs, time.monotonic())

    def test_get_returns_fresh_entry(self):
        service = VoiceStateService()
        try:
            obs = VoiceObservation(
                emotion="sad", event="speech", language="zh", duration_ms=1000
            )
            self._put(service, 1, obs)
            result = service.get(1)
            self.assertIsNotNone(result)
            self.assertEqual(result.emotion, "sad")
        finally:
            service.close()

    def test_get_returns_none_after_ttl(self):
        service = VoiceStateService()
        try:
            obs = VoiceObservation(
                emotion="sad", event="speech", language="zh", duration_ms=1000
            )
            service._cache[1] = (obs, time.monotonic() - 2)
            result = service.get(1)
            self.assertIsNone(result)
        finally:
            service.close()

    def test_get_many_skips_expired(self):
        service = VoiceStateService()
        try:
            obs1 = VoiceObservation(
                emotion="happy", event="laughter", language="zh", duration_ms=500
            )
            obs2 = VoiceObservation(
                emotion="sad", event="speech", language="zh", duration_ms=800
            )
            service._cache[1] = (obs1, time.monotonic())  # fresh
            service._cache[2] = (obs2, time.monotonic() - 2)  # expired
            results = service.get_many((1, 2))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].emotion, "happy")
        finally:
            service.close()

    def test_latest_returns_none_after_ttl(self):
        service = VoiceStateService()
        try:
            obs = VoiceObservation(
                emotion="sad", event="speech", language="zh", duration_ms=1000
            )
            # 用旧时间戳放入，使其过期
            service._cache[1] = (obs, time.monotonic() - 10)
            result = service.latest()
            self.assertIsNone(result)
        finally:
            service.close()

    def test_evict_removes_expired(self):
        service = VoiceStateService()
        try:
            obs = VoiceObservation(
                emotion="neutral", event="speech", language="zh", duration_ms=1000
            )
            # 放入 10 条，5 条过期，5 条 fresh
            now = time.monotonic()
            for i in range(10):
                age = 2 if i < 5 else 0  # 前 5 条过期
                service._cache[i] = (obs, now - age)
            service._evict_old()
            # 过期 5 条被清除，剩余 5 条 fresh
            self.assertEqual(len(service._cache), 5)
            for ref in range(5):
                self.assertNotIn(ref, service._cache)
            for ref in range(5, 10):
                self.assertIn(ref, service._cache)
        finally:
            service.close()

    def test_get_indexed_returns_fresh_only(self):
        service = VoiceStateService()
        try:
            obs1 = VoiceObservation(
                emotion="sad", event="speech", language="zh", duration_ms=1000
            )
            obs2 = VoiceObservation(
                emotion="happy", event="laughter", language="zh", duration_ms=500
            )
            service._cache[1] = (obs1, time.monotonic() - 2)  # expired
            service._cache[2] = (obs2, time.monotonic())  # fresh
            indexed = service.get_indexed(((1, 1), (2, 2)))
            self.assertEqual(len(indexed), 1)
            self.assertEqual(indexed[0][0], 2)  # turn_index=2
            self.assertEqual(indexed[0][1].emotion, "happy")
        finally:
            service.close()


class RenderVoiceContextTests(unittest.TestCase):
    def test_renders_indexed(self):
        obs = VoiceObservation(
            emotion="sad", event="speech", language="zh", duration_ms=1000
        )
        indexed = [(2, obs)]
        text = render_voice_context(observations=[], indexed=indexed)
        self.assertIn("turn=2", text)
        self.assertIn("emotion=sad", text)
        self.assertIn("event=speech", text)
        self.assertIn("language=zh", text)

    def test_renders_non_indexed(self):
        obs = VoiceObservation(
            emotion="happy", event="laughter", language="zh", duration_ms=500
        )
        text = render_voice_context(observations=[obs])
        self.assertIn("emotion=happy", text)
        self.assertIn("event=laughter", text)
        self.assertNotIn("turn=", text)

    def test_empty(self):
        self.assertEqual(render_voice_context(observations=[], indexed=[]), "")
        self.assertEqual(render_voice_context(observations=[]), "")


if __name__ == "__main__":
    unittest.main()
