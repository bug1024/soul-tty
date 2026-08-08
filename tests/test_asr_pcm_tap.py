"""ASR PCM tap：TranscriptUpdate.pcm 不变量测试。

必须锁死：
- partial update: pcm is None
- final update: pcm is not None
- reset() 后上一句 PCM 完全清空
"""

import unittest
from unittest.mock import patch, MagicMock

from soul_tty.audio.asr import TranscriptUpdate, SherpaStream, VadGatedSherpaStream


class TranscriptUpdatePCMTests(unittest.TestCase):
    """TranscriptUpdate 的 pcm 字段不变量。"""

    def test_partial_has_no_pcm(self):
        update = TranscriptUpdate(text="hello", final=False)
        self.assertIsNone(update.pcm)

    def test_final_has_no_pcm_by_default(self):
        update = TranscriptUpdate(text="hello", final=True)
        self.assertIsNone(update.pcm)

    def test_final_with_pcm(self):
        update = TranscriptUpdate(text="hello", final=True, pcm=b"\x00\x00" * 100)
        self.assertIsNotNone(update.pcm)
        self.assertEqual(len(update.pcm), 200)

    def test_frozen(self):
        with self.assertRaises(AttributeError):
            update = TranscriptUpdate(text="h", final=True)
            update.pcm = b""  # type: ignore[misc]


class VadGatedSherpaStreamPCMTests(unittest.TestCase):
    """VadGatedSherpaStream 积累 PCM 并附加到 final update。"""

    def setUp(self):
        # 使用纯 mock 绕过真实 sherpa 加载
        self._stream_patcher = patch(
            "soul_tty.audio.asr.SherpaStream", autospec=True
        )
        self._mock_stream_cls = self._stream_patcher.start()
        self._mock_stream = self._mock_stream_cls.return_value
        self._mock_stream._last_partial = ""
        self._mock_stream.last_endpoint = False

    def tearDown(self):
        self._stream_patcher.stop()

    def _make_vad_stream(self):
        vad = MagicMock()
        vad.is_speech.return_value = True
        return VadGatedSherpaStream(
            session=self._mock_stream,
            vad=vad,
            pre_roll_ms=0,
            trigger_ms=30,
        )

    def test_initial_state(self):
        stream = self._make_vad_stream()
        self.assertFalse(stream.active)
        self.assertEqual(stream._utterance_pcm, [])

    def test_after_speech_trigger_utterance_pcm_accumulates(self):
        stream = self._make_vad_stream()
        self._mock_stream.accept.return_value = []
        self._mock_stream.last_endpoint = False

        # 触发语音
        stream.accept(b"\x00\x00" * 480)  # 30ms @ 16kHz
        self.assertTrue(stream.active)
        self.assertGreater(len(stream._utterance_pcm), 0)

    def test_final_update_contains_pcm(self):
        stream = self._make_vad_stream()
        self._mock_stream.accept.return_value = [
            TranscriptUpdate("hello world", final=True)
        ]
        self._mock_stream.last_endpoint = True

        # 触发语音 + 模拟 endpoint
        updates = stream.accept(b"\x00\x00" * 480)
        self.assertEqual(len(updates), 1)
        self.assertTrue(updates[0].final)
        self.assertIsNotNone(updates[0].pcm)
        # PCM 至少包含触发帧
        self.assertGreater(len(updates[0].pcm), 0)

    def test_partial_update_has_no_pcm(self):
        stream = self._make_vad_stream()
        self._mock_stream.accept.return_value = [
            TranscriptUpdate("hello", final=False)
        ]
        self._mock_stream.last_endpoint = False

        updates = stream.accept(b"\x00\x00" * 480)
        for u in updates:
            if not u.final:
                self.assertIsNone(u.pcm)

    def test_reset_clears_utterance_pcm(self):
        stream = self._make_vad_stream()
        self._mock_stream.accept.return_value = []
        self._mock_stream.last_endpoint = False

        stream.accept(b"\x00\x00" * 480)
        self.assertTrue(stream.active)
        self.assertGreater(len(stream._utterance_pcm), 0)

        stream.reset()
        self.assertFalse(stream.active)
        self.assertEqual(stream._utterance_pcm, [])

    def test_endpoint_clears_utterance_pcm(self):
        stream = self._make_vad_stream()
        self._mock_stream.accept.return_value = [
            TranscriptUpdate("done", final=True, pcm=b"\x00\x00" * 480)
        ]
        self._mock_stream.last_endpoint = True

        stream.accept(b"\x00\x00" * 480)
        # 清空后可以开始新的一句话
        self.assertFalse(stream.active)
        self.assertEqual(stream._utterance_pcm, [])


if __name__ == "__main__":
    unittest.main()