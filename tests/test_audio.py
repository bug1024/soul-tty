"""TTS audio layer tests：聚焦 instruct 注入链路。"""

import threading
from unittest.mock import MagicMock, patch

import pytest


def _make_streaming_response(chunks):
    """构造一个能被 client.stream 用作 response 的 mock 上下文。"""

    class _Resp:
        def __init__(self, chunks):
            self._chunks = chunks

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size=8192):
            for c in self._chunks:
                yield c

    return _Resp(chunks)


def _make_fake_client(captured: list[dict]):
    """tts 内部用 `with httpx.Client(...) as client:`，所以 FakeClient 必须支持 ctx mgr。"""

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url, json):
            captured.append(json)
            return _make_streaming_response([b"\x00\x00"] * 4)

    return FakeClient


def test_synthesize_mlx_segment_uses_passed_instruct(monkeypatch):
    """instruct 参数必须出现在 payload 中，覆盖 config 默认值。"""
    from src.soul_tty import config
    from src.soul_tty.audio import tts

    captured: list[dict] = []
    monkeypatch.setattr(config, "MLX_TTS_VOICE", "Serena")
    monkeypatch.setattr(config, "MLX_TTS_INSTRUCT", "persona 默认语气")
    monkeypatch.setattr(tts.httpx, "Client", _make_fake_client(captured))

    chunks = list(
        tts.synthesize_mlx_stream("今天天气好。", instruct="用温柔关切的语气说")
    )
    assert chunks, "应当至少 yield 一个 chunk"
    assert captured, "fake client 应至少收到一次 POST"
    payload = captured[0]
    assert payload["instruct"] == "用温柔关切的语气说"


def test_synthesize_mlx_segment_falls_back_to_config(monkeypatch):
    """instruct 为空时回退到 config.MLX_TTS_INSTRUCT。"""
    from src.soul_tty import config
    from src.soul_tty.audio import tts

    captured: list[dict] = []
    monkeypatch.setattr(config, "MLX_TTS_VOICE", "Serena")
    monkeypatch.setattr(config, "MLX_TTS_INSTRUCT", "persona 默认语气")
    monkeypatch.setattr(tts.httpx, "Client", _make_fake_client(captured))

    list(tts.synthesize_mlx_stream("你好。", instruct=""))
    assert captured[0]["instruct"] == "persona 默认语气"


def test_synthesize_mlx_segment_omits_instruct_when_both_empty(monkeypatch):
    """persona 默认也空 → payload 不带 instruct 字段（让音色自然发声）。"""
    from src.soul_tty import config
    from src.soul_tty.audio import tts

    captured: list[dict] = []
    monkeypatch.setattr(config, "MLX_TTS_VOICE", "Serena")
    monkeypatch.setattr(config, "MLX_TTS_INSTRUCT", "")
    monkeypatch.setattr(tts.httpx, "Client", _make_fake_client(captured))

    list(tts.synthesize_mlx_stream("你好。"))
    assert "instruct" not in captured[0]


def test_streaming_speaker_locks_instruct_at_construction(monkeypatch):
    """StreamingSpeaker 构造时锁定 instruct，避免同一回答中途换语气。"""
    from src.soul_tty import config
    from src.soul_tty.audio import tts

    captured_payloads: list[dict] = []
    monkeypatch.setattr(config, "TTS_BACKEND", "mlx")
    monkeypatch.setattr(config, "TTS_SAMPLE_RATE", 24000)
    monkeypatch.setattr(config, "MLX_TTS_VOICE", "Serena")
    monkeypatch.setattr(config, "MLX_TTS_INSTRUCT", "")
    monkeypatch.setattr(tts.httpx, "Client", _make_fake_client(captured_payloads))
    monkeypatch.setattr(tts.sd, "RawOutputStream", MagicMock())

    with tts.StreamingSpeaker(
        threading.Event(),
        on_audio_level=None,
        instruct="用开心上扬的语气说",
    ) as speaker:
        speaker.say("第一句。")
        speaker.say("第二句。")
        # 即使外部模块改了 config 默认值也不该影响
        monkeypatch.setattr(config, "MLX_TTS_INSTRUCT", "后注入的不该生效")
        speaker.say("第三句。")

    assert len(captured_payloads) == 3
    for payload in captured_payloads:
        assert payload["instruct"] == "用开心上扬的语气说"


def test_conversation_provider_round_trip(monkeypatch):
    """conversation.set_emotion_instruct_provider 写入后能被 _current_tts_instruct 读到。"""
    from soul_tty import conversation

    calls = {"n": 0}
    sequence = ["", "用温柔关切的语气说", "用开心上扬的语气说", ""]

    def provider():
        calls["n"] += 1
        return sequence[min(calls["n"] - 1, len(sequence) - 1)]

    monkeypatch.setattr(conversation, "_emotion_instruct_provider", None)
    conversation.set_emotion_instruct_provider(provider)
    assert conversation._current_tts_instruct() == ""
    assert conversation._current_tts_instruct() == "用温柔关切的语气说"
    assert conversation._current_tts_instruct() == "用开心上扬的语气说"
    # 关闭后回退到空
    conversation.set_emotion_instruct_provider(None)
    assert conversation._current_tts_instruct() == ""


def test_conversation_provider_exception_returns_empty():
    """Provider 抛异常时不打断对话，返回空让 persona 默认生效。"""
    from soul_tty import conversation

    def bad_provider():
        raise RuntimeError("emotion 临时挂了")

    conversation.set_emotion_instruct_provider(bad_provider)
    try:
        assert conversation._current_tts_instruct() == ""
    finally:
        conversation.set_emotion_instruct_provider(None)


# --- TTS playback gain (commit 02) --------------------------------------


def _frame_bytes(samples):
    """把 int16 样本列表打包成 30ms 帧（480 样本 = 960 字节）。"""
    import struct

    return b"".join(struct.pack("<h", s) for s in samples)


@pytest.fixture(autouse=True)
def _reset_playback_gain():
    """每个 case 跑前/后重置 gain，避免互相污染。"""
    from src.soul_tty.audio import tts

    tts.set_playback_gain(1.0)
    yield
    tts.set_playback_gain(1.0)


def test_default_playback_gain_is_one():
    """未调用 set_playback_gain 时默认 1.0，与历史行为一致。"""
    from src.soul_tty.audio import tts

    assert tts.get_playback_gain() == 1.0


def test_set_playback_gain_rejects_negative():
    """负数必须拒绝：gain < 0 在声学上没有意义。"""
    from src.soul_tty.audio import tts

    with pytest.raises(ValueError):
        tts.set_playback_gain(-0.1)


def test_set_playback_gain_accepts_zero_for_mute():
    """gain == 0 合法：sanity check / 调试静音场景。"""
    from src.soul_tty.audio import tts

    tts.set_playback_gain(0.0)
    assert tts.get_playback_gain() == 0.0


def test_write_metered_pcm_passes_through_at_gain_one(monkeypatch):
    """gain == 1.0 时输出逐字节等价（numpy × 1.0 + clip 是 no-op）。"""
    import numpy as np

    from src.soul_tty.audio import tts

    # 一段正弦波式 PCM，避开全零 / 全 1 让乘法有可视痕迹
    pcm = _frame_bytes([1000, 2000, -1500, 3000, -2000, 4000])

    written: list[bytes] = []

    class FakeStream:
        def write(self, frame):
            written.append(frame)

    meter = tts.PlaybackLevelMeter()
    tts._write_metered_pcm(FakeStream(), pcm, meter)
    assert written, "should have written at least one frame"

    # 把所有写入拼起来按 int16 数组比较
    out_bytes = b"".join(written)[: len(pcm)]
    out = np.frombuffer(out_bytes, dtype="<i2")
    expected = np.frombuffer(pcm, dtype="<i2")
    assert np.array_equal(out, expected)


def test_playback_gain_scales_below_one():
    """gain = 0.5 时输出振幅减半。"""
    import numpy as np

    from src.soul_tty.audio import tts

    pcm = _frame_bytes([1000, -2000, 4000, -8000] * 4)
    tts.set_playback_gain(0.5)

    written: list[bytes] = []

    class FakeStream:
        def write(self, frame):
            written.append(frame)

    tts._write_metered_pcm(FakeStream(), pcm, tts.PlaybackLevelMeter())
    out = np.frombuffer(b"".join(written)[: len(pcm)], dtype="<i2")
    expected = np.clip(np.frombuffer(pcm, dtype="<i2") * 0.5, -32768, 32767).astype("<i2")
    assert np.array_equal(out, expected)


def test_playback_gain_clips_to_prevent_overflow():
    """gain > 1.0 时 int16 振幅会被 clip 到 ±32767，避免爆音。"""
    import numpy as np

    from src.soul_tty.audio import tts

    pcm = _frame_bytes([32767, -32768, 16384, -16384] * 4)
    tts.set_playback_gain(2.0)

    written: list[bytes] = []

    class FakeStream:
        def write(self, frame):
            written.append(frame)

    tts._write_metered_pcm(FakeStream(), pcm, tts.PlaybackLevelMeter())
    out = np.frombuffer(b"".join(written)[: len(pcm)], dtype="<i2")
    assert int(out.max()) <= 32767
    assert int(out.min()) >= -32768


def test_meter_runs_after_gain():
    """meter 必须读到 gain 之后的样本，否则口型与听感错位。"""
    from src.soul_tty.audio import tts

    seen: list[float] = []

    class ProbeMeter(tts.PlaybackLevelMeter):
        def update(self, pcm):
            seen.append(self.value)
            super().update(pcm)

    pcm = _frame_bytes([16000] * 480 * 4)  # 高幅值，让 RMS 明显
    tts.set_playback_gain(0.25)

    class FakeStream:
        def write(self, frame):
            pass

    tts._write_metered_pcm(FakeStream(), pcm, ProbeMeter())
    # 至少有一次 meter 看到的是缩放后的样本（value 应被填入），
    # 而不是 raw 16000 / 32768 的全幅 RMS
    assert seen, "meter 应该被调过"
    # 0.25 gain 后 RMS ≈ 0.146，远低于 raw 0.488
    assert seen[-1] < 0.3, f"meter 看到的应是被缩放后的 RMS，got {seen[-1]}"


def test_set_playback_gain_is_thread_safe():
    """多线程并发写 gain 必须安全，且最终值是某次写入的结果。"""
    from src.soul_tty.audio import tts

    values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    errors: list[Exception] = []

    def writer(v):
        try:
            for _ in range(1000):
                tts.set_playback_gain(v)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(v,)) for v in values]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    final = tts.get_playback_gain()
    assert final in values, f"final gain 必须是写入过的某个值，got {final}"


def test_config_exposes_tts_playback_gain():
    """TTS_PLAYBACK_GAIN 必须在 config.py 中存在，默认 1.0。"""
    from src.soul_tty import config

    assert hasattr(config, "TTS_PLAYBACK_GAIN")
    assert config.TTS_PLAYBACK_GAIN == 1.0

