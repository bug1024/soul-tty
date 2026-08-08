"""VoiceStateService：异步声音感知旁路单元测试。"""

import time
import unittest
from unittest.mock import patch, MagicMock

from soul_tty import config
from soul_tty.audio.voice_state import (
    VoiceObservation,
    VoiceStateService,
    _normalize_emotion,
    _normalize_event,
    _normalize_lang,
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
        obs = VoiceObservation(emotion="happy", event="speech", language="zh", duration_ms=1000)
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
        with patch("soul_tty.audio.voice_state.config.VOICE_STATE_ENABLED", False):
            service = VoiceStateService()
            try:
                ref = service.submit(b"\x00\x00" * 16000 * 2)
                self.assertIsNone(ref)
            finally:
                service.close()


class VoiceStateServiceCacheTTLTests(unittest.TestCase):
    """缓存不会无限增长。"""

    def setUp(self):
        self._patcher = patch.object(config, "VOICE_STATE_ENABLED", True)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_cache_evicts_old(self):
        service = VoiceStateService()
        try:
            # 模拟多次 submit，但 worker 不实际 decode
            # 我们直接往缓存里写入条目来测试 TTL 清理
            with service._cache_lock:
                for i in range(20):
                    service._cache[i] = VoiceObservation(
                        emotion="neutral", event="speech", language="zh",
                        duration_ms=1000,
                    )
                service._evict_old()
                # 最多保留 2x queue_size = 8 条
                self.assertLessEqual(len(service._cache), 8)
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()