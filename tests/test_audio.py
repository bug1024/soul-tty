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

