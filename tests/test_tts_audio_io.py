"""commit 07+:TTS 播放接 AudioIO 的测试。

- tts.set_audio_io / get_audio_io 模块级绑定
- StreamingSpeaker 收到 audio_io 时走 audio_io.write_playback
- 短 PCM 也能端到端通过 audio_io
"""

import queue
import threading
from unittest.mock import MagicMock

import pytest

from soul_tty.audio import tts
from soul_tty.audio.io.base import AudioIO


@pytest.fixture(autouse=True)
def _reset_audio_io():
    """每个 case 后清掉模块级 audio_io,免污染下游测试。"""
    yield
    tts.set_audio_io(None)


def test_set_and_get_audio_io_round_trip():
    """set_audio_io / get_audio_io 模块级绑定可逆。"""
    fake = MagicMock(spec=AudioIO)
    tts.set_audio_io(fake)
    assert tts.get_audio_io() is fake
    tts.set_audio_io(None)
    assert tts.get_audio_io() is None


def test_set_audio_io_rejects_non_audio_io():
    """非 AudioIO 实例必须抛 TypeError(防止 StreamingSpeaker 内部炸)。"""
    import pytest

    with pytest.raises(TypeError):
        tts.set_audio_io("not an audio io")
    with pytest.raises(TypeError):
        tts.set_audio_io(object())


def test_streaming_speaker_uses_audio_io_when_set():
    """StreamingSpeaker 收到 audio_io 时,合成后的 PCM 应推到 audio_io
    而不是自己开 sd.RawOutputStream。"""
    from soul_tty.audio.tts import StreamingSpeaker

    received: list[bytes] = []
    io_lock = threading.Lock()

    class _FakeIO:
        def write_playback(self, pcm: bytes, sr: int):
            with io_lock:
                received.append(pcm)

    audio_io = _FakeIO()
    cancel = threading.Event()
    with StreamingSpeaker(cancel, audio_io=audio_io) as speaker:
        # 模拟一句"测试"被合成线程合成出来:直接送 PCM 进 _audio_q
        speaker._audio_q.put(b"\x00\x01" * 16)
        speaker._audio_q.put(object())  # 不是 _SENTINEL,但让队列一直有数据
        # 不实际等 _SENTINEL 退出(避免等合成线程),cancel 强制退出
        cancel.set()

    # 等 worker 处理完
    deadline = threading.Event()
    threading.Thread(target=lambda: (deadline.wait(0.5), None), daemon=True).start()
    import time
    time.sleep(0.2)
    assert len(received) >= 1, "audio_io.write_playback 必须被调用"


def test_streaming_speaker_falls_back_when_no_audio_io(monkeypatch):
    """不传 audio_io 时,_play_loop 仍走 sd.RawOutputStream 旧路径。"""
    from soul_tty.audio.tts import StreamingSpeaker

    opened_streams: list = []
    monkeypatch.setattr(
        "sounddevice.RawOutputStream",
        lambda **kw: (opened_streams.append(kw), _FakeStream())[1],
    )

    cancel = threading.Event()
    speaker = StreamingSpeaker(cancel)
    # 不启动合成线程,直接推 PCM 进 _audio_q 并启动 _play_t
    speaker._play_t = threading.Thread(target=speaker._play_loop, daemon=True)
    speaker._play_t.start()
    speaker._audio_q.put(b"\x00\x01" * 16)
    time.sleep(0.1)
    cancel.set()
    # 让 worker 退出
    from soul_tty.audio.tts import _SENTINEL
    speaker._audio_q.put(_SENTINEL)
    speaker._play_t.join(timeout=1.0)

    assert opened_streams, "无 audio_io 时必须自己开 sd.RawOutputStream"


class _FakeStream:
    """最小 sd.RawOutputStream 替身(只记调用)。"""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, pcm):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


import time  # noqa: E402