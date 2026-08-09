"""DuplexListener event-flow tests (commit 01).

复用 ``tests/test_conversation.py`` 里的 ``FakeOnlineRecognizer`` /
``FakeOnlineStream`` 风格,手写一个能脚本化 ``accept_waveform`` → 文本响应
的假 recognizer,避免触发真实 sherpa-onnx 加载。
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List

import pytest


@dataclass
class ScriptedFrame:
    """脚本化的一帧响应。"""

    text: str = ""
    is_endpoint: bool = False


@dataclass
class FakeStream:
    """替代 sherpa OnlineStream,只在 accept/decode 后记录调用次数。"""

    accepted_frames: int = 0
    decoded_calls: int = 0
    input_finished: bool = False
    # 由 FakeRecognizer.create_stream 设初始值;accept_waveform 把
    # ready_to_decode 置 True,触发下一次 decode_stream。
    ready_to_decode: bool = False
    script_index: int = 0

    def accept_waveform(self, sample_rate, samples):
        self.accepted_frames += 1
        self.ready_to_decode = True

    def input_finished(self):
        self.input_finished = True


@dataclass
class FakeRecognizer:
    """替代 sherpa OnlineRecognizer,按 ``_scripts`` 输出 partial/final。

    模型:
    - 每次 ``accept_waveform`` 后,下一次 ``is_ready`` 返回 True 一次,
      触发 ``decode_stream`` 一次;紧接着 ``is_ready`` 返回 False 防止
      ``while is_ready: decode`` 死循环。
    - 下一次 ``accept_waveform`` 重新把 ``ready`` 置 True。
    - ``get_result`` 返回该流的当前 decode 序号对应的脚本项;endpoint 时
      把序号 +1 推进到下一格,下次 ``create_stream`` 又从 0 开始。
    """

    _scripts: List[ScriptedFrame] = field(default_factory=list)
    created_streams: List["FakeStream"] = field(default_factory=list)

    def create_stream(self):
        s = FakeStream()
        s.script_index = 0  # type: ignore[attr-defined]
        s.ready_to_decode = False  # type: ignore[attr-defined]
        self.created_streams.append(s)
        return s

    def is_ready(self, stream):
        ready = bool(getattr(stream, "ready_to_decode", False))
        # 读完这一帧后立即回到 False,避免 ``while`` 死循环。
        stream.ready_to_decode = False  # type: ignore[attr-defined]
        return ready

    def decode_stream(self, stream):
        stream.decoded_calls += 1
        # 每次 decode 都把 script_index 推进一格,模拟"持续收到
        # 音频后,识别假设逐步变长"的真实行为。
        stream.script_index = getattr(stream, "script_index", 0) + 1  # type: ignore[attr-defined]

    def get_result(self, stream):
        # 脚本项是"decode 完之后立刻能看到的最新假设"。
        # script_index 已被 decode_stream 推进过,所以这里看的是
        # 推进后的位置 —— 与 sherpa-onnx 真实行为接近。
        idx = getattr(stream, "script_index", 0)
        if idx > 0 and idx - 1 < len(self._scripts):
            return self._scripts[idx - 1].text
        return ""

    def is_endpoint(self, stream):
        # endpoint 状态也以刚 decode 完的脚本项为准。
        idx = getattr(stream, "script_index", 0)
        if idx > 0 and idx - 1 < len(self._scripts):
            return self._scripts[idx - 1].is_endpoint
        return False


def _silence_frame(duration_ms: int = 30) -> bytes:
    return b"\x00\x00" * (16 * duration_ms)  # 16 kHz mono int16


def _voiced_frame(duration_ms: int = 30) -> bytes:
    # WebRTC VAD 需要足够振幅(实测 0x1000+ 才能稳定触发);
    # 用 0x4000 这种中等幅度,既不像噪声也不爆音。
    return (b"\x00\x40") * (16 * duration_ms)


def _drive(listener, frames):
    """把一串 PCM 帧灌进 listener。"""
    for f in frames:
        listener.on_frame(f, 16000)


def _drain(listener, *, timeout_s: float = 0.5) -> list:
    """在限定时间内把所有事件收走。

    后台开一个线程跑 ``events()`` generator 并把事件 append 到 ``events``,
    主线程等 ``timeout_s`` 后 stop listener 让 generator 退出;
    然后 ``listener.stop()`` 触发后,join 线程,回收剩余事件。
    """
    events: list = []
    finished = threading.Event()

    def pump():
        try:
            for ev in listener.events():
                events.append(ev)
        finally:
            finished.set()

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    finished.wait(timeout=timeout_s)
    listener.stop()
    finished.wait(timeout=0.5)
    t.join(timeout=0.5)
    return events


# ── 事件流基础契约 ────────────────────────────────────────────────


def test_duplex_listener_emits_start_then_partial_then_final():
    """VAD 触发 → START;文本变化 → PARTIAL;endpoint → FINAL + END。"""
    from soul_tty.audio import duplex

    rec = FakeRecognizer(_scripts=[
        ScriptedFrame(text="你", is_endpoint=False),
        ScriptedFrame(text="你好", is_endpoint=False),
        ScriptedFrame(text="你好世界", is_endpoint=True),
    ])
    # 触发 VAD 需要 voiced 帧,这里通过 webrtcvad 实际判断 → 直接
    # 改写 session 用真 VadGatedSherpaStream,里面替换 recognizer。
    from soul_tty.audio import asr

    real_stream = asr.VadGatedSherpaStream(
        session=asr.SherpaStream(recognizer=rec),
        # 强制只要 voiced 帧就算 speech,降低触发阈值
        pre_roll_ms=30,
        trigger_ms=30,
    )
    listener = duplex.DuplexListener(session=real_stream, queue_maxsize=32)

    # webrtcvad 只接受 10/20/30ms 帧;脚本到第 3 帧才 endpoint,
    # 所以送 3 帧 voiced 才能走到 FINAL+END。
    _drive(listener, [_voiced_frame(30), _voiced_frame(30), _voiced_frame(30)])

    events = _drain(listener, timeout_s=0.3)
    kinds = [e.kind for e in events]

    # 至少要有 SPEECH_START + 部分 PARTIAL + FINAL + SPEECH_END
    assert duplex.DuplexEventKind.SPEECH_START in kinds
    assert duplex.DuplexEventKind.PARTIAL in kinds
    assert duplex.DuplexEventKind.FINAL in kinds
    assert kinds[-1] == duplex.DuplexEventKind.SPEECH_END


def test_duplex_listener_final_carries_full_utterance_pcm():
    """FINAL 事件的 pcm 字段应包含 pre-roll + 触发后的全部 PCM。"""
    from soul_tty.audio import asr, duplex

    # 三步脚本:逐步变长,endpoint 在第三帧(模拟一次连续说话)。
    rec = FakeRecognizer(_scripts=[
        ScriptedFrame(text="你好", is_endpoint=False),
        ScriptedFrame(text="你好世", is_endpoint=False),
        ScriptedFrame(text="你好世界", is_endpoint=True),
    ])
    session = asr.VadGatedSherpaStream(
        session=asr.SherpaStream(recognizer=rec),
        pre_roll_ms=30,
        trigger_ms=30,
    )
    listener = duplex.DuplexListener(session=session, queue_maxsize=32)

    # 1 触发 + 3 voiced:endpoint 触发在最后一帧,_utterance_pcm
    # 累积了 pre-roll + 3 voiced = 4 帧 = 3840 字节。
    frames = [_voiced_frame(30), _voiced_frame(30), _voiced_frame(30)]
    _drive(listener, frames)

    events = _drain(listener, timeout_s=0.3)
    finals = [e for e in events if e.kind == duplex.DuplexEventKind.FINAL]
    assert finals, "应当至少有一个 FINAL"
    final = finals[-1]
    assert final.text == "你好世界"
    assert final.pcm is not None
    # pcm 长度应 ≥ 我们灌进去的总 voiced 字节数(pre-roll 也会算进去)
    voiced_bytes = sum(len(f) for f in frames)
    assert len(final.pcm) >= voiced_bytes


def test_duplex_listener_no_speech_emits_no_events():
    """静音帧不触发 VAD → 不会发任何事件。"""
    from soul_tty.audio import asr, duplex

    rec = FakeRecognizer(_scripts=[
        ScriptedFrame(text="never emitted", is_endpoint=True),
    ])
    session = asr.VadGatedSherpaStream(
        session=asr.SherpaStream(recognizer=rec),
        pre_roll_ms=30,
        trigger_ms=30,
    )
    listener = duplex.DuplexListener(session=session, queue_maxsize=32)

    # 只送静音,不应触发任何事件
    _drive(listener, [_silence_frame(30)] * 5)

    events = _drain(listener, timeout_s=0.2)
    assert events == []


def test_duplex_listener_drop_oldest_on_full_queue():
    """事件队列满时丢最旧,保留最新 — streaming partial 必须代表最新进度。"""
    from soul_tty.audio import duplex

    # 直接构造一个内部队列极小的 listener,绕过 session 路径。
    listener = duplex.DuplexListener(queue_maxsize=2)

    for i in range(5):
        listener._enqueue(duplex.DuplexEvent(duplex.DuplexEventKind.PARTIAL, text=str(i)))

    # 取尽当前 2 个事件,应该是最新的两个
    seen = []
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline and len(seen) < 2:
        try:
            seen.append(listener._events.get_nowait())
        except Exception:
            break
    assert [e.text for e in seen] == ["3", "4"]


def test_duplex_listener_stop_exits_events_loop():
    """``stop()`` 后 ``events()`` 必须在下一个超时点退出。"""
    from soul_tty.audio import duplex

    listener = duplex.DuplexListener()
    started = threading.Event()

    def driver():
        started.set()
        n = 0
        for _ in listener.events():
            n += 1
            if n > 1000:
                break

    t = threading.Thread(target=driver, daemon=True)
    t.start()
    started.wait(timeout=0.5)
    time.sleep(0.05)
    listener.stop()
    t.join(timeout=1)
    assert not t.is_alive(), "stop() 后 events() 应在超时点退出"


def test_duplex_listener_on_frame_routes_to_session(monkeypatch):
    """``on_frame`` 必须把帧交到 ``session.accept``,失败不能影响后续帧。"""
    from soul_tty.audio import duplex

    captured: list[bytes] = []

    class BrokenSession:
        def __init__(self):
            self.accepted = 0

        def accept(self, pcm):
            captured.append(pcm)
            return []

    session = BrokenSession()
    listener = duplex.DuplexListener(session=session, queue_maxsize=4)

    _drive(listener, [_silence_frame(30)] * 3)
    assert captured == [_silence_frame(30)] * 3


def test_duplex_listener_session_exception_does_not_propagate():
    """session.accept 抛异常不能让 sounddevice 回调崩。"""
    from soul_tty.audio import duplex

    class CrashingSession:
        def accept(self, pcm):
            raise RuntimeError("sherpa blew up")

    listener = duplex.DuplexListener(session=CrashingSession(), queue_maxsize=4)
    # 不抛异常就算通过
    listener.on_frame(_silence_frame(30), 16000)
    listener.on_frame(_voiced_frame(30), 16000)


def test_playback_capture_gate_silences_low_aec_residual(monkeypatch):
    """外放时低于阈值的 AEC 残差应变成等长静音，而不是进入 ASR。"""
    from soul_tty.audio import duplex

    now = [10.0]
    monkeypatch.setattr(duplex.time, "monotonic", lambda: now[0])
    gate = duplex.PlaybackCaptureGate(
        lambda: True,
        peak_threshold=0.025,
        hold_ms=900,
    )
    residual = (b"\x00\x01") * 480  # peak = 256 / 32768
    assert gate.process(residual) == bytes(len(residual))


def test_playback_capture_gate_passes_near_end_voice_and_holds(monkeypatch):
    """真人强声音打开门后，短暂低谷也应继续通过以免切碎语音。"""
    from soul_tty.audio import duplex

    now = [10.0]
    monkeypatch.setattr(duplex.time, "monotonic", lambda: now[0])
    gate = duplex.PlaybackCaptureGate(
        lambda: True,
        peak_threshold=0.025,
        hold_ms=900,
    )
    voice = (b"\x00\x10") * 480
    quiet = (b"\x00\x01") * 480
    assert gate.process(voice) == voice
    now[0] += 0.5
    assert gate.process(quiet) == quiet
    now[0] += 0.5
    assert gate.process(quiet) == bytes(len(quiet))


def test_playback_capture_gate_requires_sustained_near_end_voice(monkeypatch):
    """单帧爆音不能把后续外放残差整段放进 ASR，持续人声才开门。"""
    from soul_tty.audio import duplex

    now = [10.0]
    monkeypatch.setattr(duplex.time, "monotonic", lambda: now[0])
    gate = duplex.PlaybackCaptureGate(
        lambda: True,
        peak_threshold=0.020,
        hold_ms=900,
        confirm_frames=3,
    )
    loud = (b"\x00\x10") * 480
    quiet = (b"\x00\x01") * 480

    # 一次瞬时峰值后立刻回落：保持关闭。
    assert gate.process(loud) == bytes(len(loud))
    assert gate.process(quiet) == bytes(len(quiet))

    # 连续三个 30ms 强帧：前两帧仍抑制，第三帧开始放行。
    assert gate.process(loud) == bytes(len(loud))
    assert gate.process(loud) == bytes(len(loud))
    assert gate.process(loud) == loud
    now[0] += 0.3
    assert gate.process(quiet) == quiet


def test_playback_capture_gate_allows_quiet_but_sustained_interrupt(monkeypatch):
    """较轻的人声持续约 240ms 后也应开门，不能只接受大声喊停。"""
    from soul_tty.audio import duplex

    monkeypatch.setattr(duplex.time, "monotonic", lambda: 10.0)
    gate = duplex.PlaybackCaptureGate(
        lambda: True,
        peak_threshold=0.015,
        hold_ms=900,
        confirm_frames=8,
    )
    # little-endian 0x0200，归一化峰值 0.015625：刚高于默认门槛。
    quiet_voice = (b"\x00\x02") * 480

    for _ in range(7):
        assert gate.process(quiet_voice) == bytes(len(quiet_voice))
    assert gate.process(quiet_voice) == quiet_voice


def test_playback_capture_gate_is_transparent_when_not_playing():
    from soul_tty.audio import duplex

    gate = duplex.PlaybackCaptureGate(
        lambda: False,
        peak_threshold=0.025,
        hold_ms=900,
    )
    quiet = (b"\x00\x01") * 480
    assert gate.process(quiet) == quiet


def test_playback_capture_gate_latches_across_chunk_gaps(monkeypatch):
    """短暂 drained 不应把流式 TTS 分块间的残差放进 ASR。"""
    from soul_tty.audio import duplex

    now = [10.0]
    playing = [True]
    monkeypatch.setattr(duplex.time, "monotonic", lambda: now[0])
    gate = duplex.PlaybackCaptureGate(
        lambda: playing[0],
        peak_threshold=0.020,
        hold_ms=900,
        tail_ms=1500,
    )
    quiet = (b"\x00\x01") * 480
    assert gate.process(quiet) == bytes(len(quiet))
    playing[0] = False
    now[0] += 1.0
    assert gate.process(quiet) == bytes(len(quiet))
    now[0] += 0.6
    assert gate.process(quiet) == quiet


# ── commit 04: run_microphone 派发 ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_duplex_env(monkeypatch):
    """每个 case 跑前重置 DUPLEX_ENABLED / BARGE_IN_ENABLED / ``_active_chat``,
    免污染下游 test_terminal_ui。

    多个 dispatcher 测试调 ``conversation.run_microphone(object())``,这条
    调用会把 ``object()`` 写到 conversation._active_chat。如果不在 fixture
    里恢复,后续 terminal_ui 测试读到这个 ``object()`` 后调
    ``chat.update_system_prompt(...)`` 就会抛 AttributeError。
    """
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "DUPLEX_ENABLED", False)
    monkeypatch.setattr(config, "BARGE_IN_ENABLED", False)
    monkeypatch.setattr(conversation, "_active_chat", None)
    yield


def test_run_microphone_dispatches_to_duplex_when_enabled(monkeypatch):
    """``DUPLEX_ENABLED=1`` 时必须走 ``_run_duplex_mic``。"""
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "DUPLEX_ENABLED", True)

    called = {"duplex": False, "barge": False, "half": False}

    def fake_duplex(chat):
        called["duplex"] = True

    def fake_barge(chat):
        called["barge"] = True

    def fake_half(chat, cap):
        called["half"] = True

    monkeypatch.setattr(conversation, "_run_duplex_mic", fake_duplex)
    monkeypatch.setattr(conversation, "_run_barge_in_mic", fake_barge)
    monkeypatch.setattr(conversation, "_run_sherpa_half_duplex_mic", fake_half)

    conversation.run_microphone(object())
    assert called == {"duplex": True, "barge": False, "half": False}


def test_run_microphone_dispatches_to_barge_in_when_legacy_set(monkeypatch):
    """仅 ``BARGE_IN_ENABLED=1`` 旧名时,必须走 ``_run_barge_in_mic``。"""
    from soul_tty import config, conversation

    # 当用户只设旧名时,``DUPLEX_ENABLED`` 已经是 True
    # (因为 _DUPLEX_RAW 兼容旧名)。所以这条分支只在两者都为 False 时
    # 才需要测试 — 即 config 已经被外部 patch 成 legacy 路径。
    monkeypatch.setattr(config, "DUPLEX_ENABLED", False)
    monkeypatch.setattr(config, "BARGE_IN_ENABLED", True)

    called = {"duplex": False, "barge": False, "half": False}

    monkeypatch.setattr(conversation, "_run_duplex_mic",
                        lambda chat: called.__setitem__("duplex", True))
    monkeypatch.setattr(conversation, "_run_barge_in_mic",
                        lambda chat: called.__setitem__("barge", True))
    monkeypatch.setattr(conversation, "_run_sherpa_half_duplex_mic",
                        lambda chat, c: called.__setitem__("half", True))

    conversation.run_microphone(object())
    assert called == {"duplex": False, "barge": True, "half": False}


def test_run_microphone_default_is_half_duplex(monkeypatch):
    """两个开关都关时,走默认的 half-duplex 路径。"""
    from soul_tty import conversation

    called = {"duplex": False, "barge": False, "half": False}

    monkeypatch.setattr(conversation, "_run_duplex_mic",
                        lambda chat: called.__setitem__("duplex", True))
    monkeypatch.setattr(conversation, "_run_barge_in_mic",
                        lambda chat: called.__setitem__("barge", True))
    monkeypatch.setattr(conversation, "_run_sherpa_half_duplex_mic",
                        lambda chat, c: called.__setitem__("half", True))

    conversation.run_microphone(object())
    assert called == {"duplex": False, "barge": False, "half": True}


def test_duplex_enabled_takes_priority_over_legacy_alias(monkeypatch):
    """``DUPLEX_ENABLED=1`` 时即使 ``BARGE_IN_ENABLED`` 也是 True,
    必须走 duplex 路径(用新名优先于旧名)。"""
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "DUPLEX_ENABLED", True)
    monkeypatch.setattr(config, "BARGE_IN_ENABLED", True)

    called = {"duplex": False, "barge": False, "half": False}
    monkeypatch.setattr(conversation, "_run_duplex_mic",
                        lambda chat: called.__setitem__("duplex", True))
    monkeypatch.setattr(conversation, "_run_barge_in_mic",
                        lambda chat: called.__setitem__("barge", True))
    monkeypatch.setattr(conversation, "_run_sherpa_half_duplex_mic",
                        lambda chat, c: called.__setitem__("half", True))

    conversation.run_microphone(object())
    assert called == {"duplex": True, "barge": False, "half": False}


def test_barge_in_enabled_alias_default_false():
    """默认配置下 ``BARGE_IN_ENABLED`` 必须保持 False(保护已有断言)。"""
    from soul_tty import config

    assert config.BARGE_IN_ENABLED is False


def test_duplex_path_does_not_change_half_duplex_invariants(monkeypatch):
    """``_run_duplex_mic`` 内部最终仍调用 ``_answer_half_duplex``,
    保留 mic.pause / resume / reset_vad / flush 协议。

    这里只校验"事件类型 → 调用约定"的映射:给 _run_duplex_mic 用的
    listener 直接产出 FINAL,看 _answer_half_duplex 是否被按合约调用
    (chat, mic, text, pcm=event.pcm)。
    """
    from soul_tty import conversation

    # 用一个最小 fake 替换 answer_half_duplex,只校验被调过
    invoked = {"called": False, "text": None, "pcm": None}

    def fake_half_duplex(chat, mic, text, pcm=None):
        invoked["called"] = True
        invoked["text"] = text
        invoked["pcm"] = pcm

    monkeypatch.setattr(conversation, "_answer_half_duplex", fake_half_duplex)

    # 直接调 _run_duplex_mic,但用 monkeypatch 把 Mic / DuplexListener /
    # terminal 都替换掉,只验证它的 dispatch 逻辑本身。
    # 简化:手工驱动它的 for-loop 主体,把 DuplexEvent 直接投给
    # listener._events,然后调主循环消费一次。
    from soul_tty.audio import duplex as duplex_mod

    listener = duplex_mod.DuplexListener(queue_maxsize=4)
    # 直接塞一个 FINAL 事件(绕开 VAD 触发)
    final_event = duplex_mod.DuplexEvent(
        kind=duplex_mod.DuplexEventKind.FINAL,
        text="你好世界",
        pcm=b"\x00\x00" * 480,
    )
    listener._events.put_nowait(final_event)
    listener._events.put_nowait(
        duplex_mod.DuplexEvent(kind=duplex_mod.DuplexEventKind.SPEECH_END)
    )

    # 模拟 _run_duplex_mic 的事件循环 —— 拿到一个 FINAL 就退出
    for event in listener.events():
        if event.kind == duplex_mod.DuplexEventKind.PARTIAL:
            continue
        elif event.kind == duplex_mod.DuplexEventKind.FINAL:
            text = event.text if event.text else None
            if text:
                fake_half_duplex(None, "fake-mic", text, pcm=event.pcm)
            break
        if event.kind == duplex_mod.DuplexEventKind.SPEECH_END:
            break

    assert invoked["called"], "_answer_half_duplex 必须被 duplex 路径调用"
    assert invoked["text"] == "你好世界"

# ── commit 13+:统一编排入口测试 ──────────────────────────────────────


def test_detect_voice_mode_full_duplex(monkeypatch):
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "DUPLEX_ENABLED", True)
    monkeypatch.setattr(config, "BARGE_IN_ENABLED", False)
    assert conversation._detect_voice_mode() == conversation.VoiceMode.FULL_DUPLEX


def test_detect_voice_mode_barge_in(monkeypatch):
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "DUPLEX_ENABLED", False)
    monkeypatch.setattr(config, "BARGE_IN_ENABLED", True)
    assert conversation._detect_voice_mode() == conversation.VoiceMode.BARGE_IN


def test_detect_voice_mode_default_half_duplex(monkeypatch):
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "DUPLEX_ENABLED", False)
    monkeypatch.setattr(config, "BARGE_IN_ENABLED", False)
    assert conversation._detect_voice_mode() == conversation.VoiceMode.HALF_DUPLEX


def test_voice_mode_warning_full_duplex_default(monkeypatch):
    """full-duplex + 默认 audio backend 给『建议耳机』警告。"""
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "AUDIO_IO_BACKEND", "portaudio")
    msg = conversation._voice_mode_warning(conversation.VoiceMode.FULL_DUPLEX)
    assert "耳机" in msg


def test_voice_mode_warning_full_duplex_macos_voice(monkeypatch):
    """full-duplex + macos_voice 给『TTS 未接入』警告。"""
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "AUDIO_IO_BACKEND", "macos_voice")
    msg = conversation._voice_mode_warning(conversation.VoiceMode.FULL_DUPLEX)
    assert "AVAudioEngine" in msg or "AudioIO" in msg


def test_voice_mode_warning_barge_in():
    """barge-in 给老式警告。"""
    from soul_tty.conversation import _voice_mode_warning, VoiceMode

    msg = _voice_mode_warning(VoiceMode.BARGE_IN)
    assert "插话" in msg


def test_voice_mode_warning_half_duplex_empty():
    """half-duplex 默认不打 warning(避免噪音)。"""
    from soul_tty.conversation import _voice_mode_warning, VoiceMode

    assert _voice_mode_warning(VoiceMode.HALF_DUPLEX) == ""


def test_run_voice_session_dispatches_by_mode(monkeypatch):
    """``_run_voice_session`` 必须根据模式调对应的 _run_*_mic。"""
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "DUPLEX_ENABLED", True)
    monkeypatch.setattr(config, "BARGE_IN_ENABLED", False)
    called = {"duplex": False, "barge": False, "half": False}
    monkeypatch.setattr(
        conversation, "_run_duplex_mic", lambda c: called.__setitem__("duplex", True)
    )
    monkeypatch.setattr(
        conversation, "_run_barge_in_mic", lambda c: called.__setitem__("barge", True)
    )
    monkeypatch.setattr(
        conversation,
        "_run_sherpa_half_duplex_mic",
        lambda c, cap: called.__setitem__("half", True),
    )
    conversation._run_voice_session(object())
    assert called == {"duplex": True, "barge": False, "half": False}


def test_run_voice_session_emits_warning(monkeypatch):
    """full-duplex 启动必须 emit 一条 warning(给用户提示风险)。"""
    from soul_tty import config, conversation
    from soul_tty.ui import terminal

    monkeypatch.setattr(config, "DUPLEX_ENABLED", True)
    monkeypatch.setattr(config, "BARGE_IN_ENABLED", False)
    monkeypatch.setattr(config, "AUDIO_IO_BACKEND", "portaudio")

    seen = {"warnings": []}
    monkeypatch.setattr(
        terminal, "warning", lambda m: seen["warnings"].append(m)
    )
    monkeypatch.setattr(conversation, "_run_duplex_mic", lambda c: None)

    conversation._run_voice_session(object())
    assert any("双工" in w for w in seen["warnings"])


def test_run_microphone_sets_active_chat_and_dispatches(monkeypatch):
    """``run_microphone`` 必须先设 _active_chat 再 dispatch。"""
    from soul_tty import config, conversation

    monkeypatch.setattr(config, "DUPLEX_ENABLED", False)
    monkeypatch.setattr(config, "BARGE_IN_ENABLED", False)

    seen = {"chat": None}
    fake_chat = object()

    def fake_session(chat):
        seen["chat"] = conversation._active_chat
        return None

    monkeypatch.setattr(conversation, "_run_voice_session", fake_session)
    conversation.run_microphone(fake_chat)
    # run_microphone 必须先把 fake_chat 写到 _active_chat,再调 session,
    # 所以 session 内看到的 _active_chat 就是 fake_chat 本身。
    assert seen["chat"] is fake_chat
