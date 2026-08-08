"""对话编排：麦克风 -> ASR -> LLM -> TTS。"""

import difflib
import enum
import queue
import re
import sys
import threading
import wave
from collections.abc import Callable

from . import config, prompt, reflection
from .audio import asr, capture, duplex, tts
from .clients import llm
from .ui import terminal

# 全局 Chat 实例引用，供 terminal.py 换模式时热更新 system prompt
_active_chat: llm.Chat | None = None
# Emotion 注入：cli.py 把 EmotionService.current_tts_instruct 绑到这里；
# 每段 LLM 回答开始时取一次快照，整段播放期间保持同一份 TTS 指令。
_emotion_instruct_provider: Callable[[], str] | None = None
# Memory recall：cli.py 把 MemoryService.recall 绑到这里；
# 每次主对话取一次，命中召回词时返回临时 [Relevant Memories] 段。
_recall_provider: Callable[[str], str] | None = None
# Voice perception：cli.py 把 VoiceStateService.submit 绑到这里；
# 每次 ASR final 后提交 PCM，不阻塞主对话。
_voice_submit_provider: Callable[[bytes], int | None] | None = None


def set_emotion_instruct_provider(
    provider: Callable[[], str] | None,
) -> None:
    """cli.py 在 EmotionService 启动后调用；None 表示关闭 emotion TTS 注入。"""
    global _emotion_instruct_provider
    _emotion_instruct_provider = provider


def set_recall_provider(provider: Callable[[str], str] | None) -> None:
    """cli.py 在 MemoryService 启动后调用；None 表示关闭 recall 注入。

    provider 接受用户文本、返回 [Relevant Memories] 段或空串。
    """
    global _recall_provider
    _recall_provider = provider


def set_voice_submit_provider(
    provider: Callable[[bytes], int | None] | None,
) -> None:
    """cli.py 在 VoiceStateService 启动后调用；None 表示关闭声音感知。"""
    global _voice_submit_provider
    _voice_submit_provider = provider


def _current_tts_instruct() -> str:
    provider = _emotion_instruct_provider
    if provider is None:
        return ""
    try:
        return provider() or ""
    except Exception:
        return ""


def _current_recall(user_text: str) -> str:
    provider = _recall_provider
    if provider is None:
        return ""
    try:
        return provider(user_text) or ""
    except Exception:
        return ""


def _current_voice_ref(pcm: bytes | None) -> int | None:
    """提交 utterance PCM 给 VoiceStateService，返回 VoiceRef。"""
    if pcm is None or _voice_submit_provider is None:
        return None
    try:
        return _voice_submit_provider(pcm)
    except Exception:
        return None


def emit_emotion_update(emotion_service, snap) -> None:
    """Re-build Emotion Context and hot-update active Chat system_prompt."""
    if _active_chat is None:
        return
    prompt.builder().set_section("emotion", snap.context_text)
    _active_chat.update_system_prompt(prompt.refresh())

# ASR 在静音/噪声段上的典型幻觉文本，直接丢弃不送 LLM。
HALLUCINATIONS = (
    "谢谢收看", "謝謝收看", "谢谢观看", "謝謝觀看",
    "感谢收看", "感謝收看", "请订阅", "請訂閱", "字幕",
)

# 句子结束符:切句送 TTS 用。只在句末标点切——按逗号切会让每个分句
# 独立合成,分句间有合成间隔且韵律不连贯,听起来一顿一顿的。
_SENT_END = re.compile(r"[。!！?？;；…\n]")
_NON_SPEECH_TEXT = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")


def _print_answer(
    chat: llm.Chat,
    text: str,
    speaker: tts.StreamingSpeaker | None = None,
    cancel: threading.Event | None = None,
    on_token: Callable[[str], None] | None = None,
    *,
    recall: str = "",
) -> str:
    """流式打印 LLM 回答;给了 speaker 就按句切分实时送 TTS。"""
    terminal.answer_start()
    parts = []
    buf = ""
    playback_started = False
    for token in chat.ask_stream(text, cancel, recall=recall):
        terminal.answer_chunk(token)
        parts.append(token)
        if on_token is not None:
            on_token(token)
        buf += token
        while speaker is not None and (m := _SENT_END.search(buf)):
            sent = buf[:m.end()].strip()
            buf = buf[m.end():]
            if sent:
                if not playback_started:
                    terminal.speaking()
                    playback_started = True
                speaker.say(sent)
    terminal.answer_end()
    if chat.last_stop_reason == "repetition":
        terminal.notice("检测到模型重复生成，已自动截断")
    if speaker is not None and buf.strip():
        if not playback_started:
            terminal.speaking()
        speaker.say(buf.strip())
    return "".join(parts).strip()


def _answer(
    chat: llm.Chat,
    text: str,
    cancel: threading.Event | None = None,
    on_token: Callable[[str], None] | None = None,
    pcm: bytes | None = None,
) -> str:
    cancel = cancel or threading.Event()
    # 一段回答内锁定同一份 TTS 指令；中途换 mood 不影响本句。
    # Provider 未注册时（emotion 关闭）回退到 config.MLX_TTS_INSTRUCT。
    instruct = _current_tts_instruct()
    # 本轮检索到的相关记忆；临时插入，不进 system prompt / 不进 history。
    recall = _current_recall(text)
    # 提交 utterance PCM 给 VoiceStateService（不阻塞）
    voice_ref = _current_voice_ref(pcm)
    if config.TTS_ENABLED and not config.TTS_WHOLE_ANSWER:
        with tts.StreamingSpeaker(
            cancel, terminal.audio_level, instruct=instruct
        ) as speaker:
            answer = _print_answer(
                chat, text, speaker, cancel, on_token, recall=recall
            )
    else:
        answer = _print_answer(
            chat, text, cancel=cancel, on_token=on_token, recall=recall
        )
        if config.TTS_ENABLED and answer and not cancel.is_set():
            try:
                terminal.speaking()
                tts.speak(answer, cancel, terminal.audio_level, instruct=instruct)
            except Exception as e:
                if not cancel.is_set():
                    terminal.notice(f"TTS 失败: {e}")
    if answer and not cancel.is_set():
        reflection.record_turn(text, answer, voice_ref=voice_ref)
    return answer


def _usable_transcript(text: str) -> bool:
    """过滤空文本和 ASR 在静音/噪声上的常见幻觉。"""
    normalized = text.strip("()（） \t。.!！?？~～,")
    return bool(normalized) and not any(h in normalized for h in HALLUCINATIONS)


def _is_probable_echo(heard: str, spoken: str) -> bool:
    """判断回答期间收到的文本是否更像扬声器回声而非插话。

    commit 11+ 改用 ``interaction.echo.is_probable_echo``（纯文本,
    不依赖音频模块,便于单独被 floor 引用）。
    """
    from .interaction.echo import is_probable_echo

    return is_probable_echo(heard, spoken, config.BARGE_IN_ECHO_SIMILARITY)


def _transcribe(pcm: bytes) -> str | None:
    try:
        text = asr.transcribe(pcm)
    except Exception as e:
        terminal.notice(f"识别失败: {e}，继续聆听")
        return None
    return text if _usable_transcript(text) else None


def _answer_interruptibly(
    chat: llm.Chat, text: str, listener, pcm: bytes | None = None
) -> str | None:
    """后台回答，前台同时确认插话；返回已确认的下一句用户文本。"""
    cancel = threading.Event()
    answer_parts: list[str] = []
    outcome: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

    def run_answer() -> None:
        try:
            _answer(chat, text, cancel, answer_parts.append, pcm=pcm)
            outcome.put(None)
        except BaseException as e:
            outcome.put(e)

    worker = threading.Thread(target=run_answer, daemon=True)
    worker.start()
    interruption = None
    while worker.is_alive():
        try:
            pcm = listener.get(timeout=0.1)
        except queue.Empty:
            continue
        if not config.BARGE_IN_ENABLED:
            continue
        heard = _transcribe(pcm)
        if not heard:
            continue
        if _is_probable_echo(heard, "".join(answer_parts)):
            continue
        interruption = heard
        cancel.set()
        terminal.interrupted(heard)
        break

    worker.join()
    error = outcome.get()
    if error is not None:
        raise error
    return interruption


def _run_file(chat: llm.Chat, path: str) -> None:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            sys.exit("--file 需要 16bit 单声道 WAV")
        pcm = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
    if rate != config.SAMPLE_RATE:
        terminal.notice(f"文件采样率 {rate}，ASR 期望 16kHz，识别效果可能受影响")
    text = asr.transcribe(pcm)
    terminal.recognized(text)
    if text:
        _answer(chat, text)


def _show_partial(text: str) -> None:
    if text:
        reflection.user_activity()
    if config.SHERPA_PARTIAL_ENABLED:
        terminal.partial(text)


def _show_final(text: str) -> None:
    reflection.user_activity()
    terminal.user_text(text)


def _answer_half_duplex(
    chat: llm.Chat, mic, text: str, pcm: bytes | None = None
) -> None:
    mic.pause()
    try:
        _answer(chat, text, pcm=pcm)
    except Exception as e:
        terminal.notice(f"LLM 调用失败: {e}")
    finally:
        mic.reset_vad()
        mic.flush()
        mic.resume()
    terminal.listening()


def _run_sherpa_half_duplex_mic(chat: llm.Chat, audio) -> None:
    """PCM 持续送入在线模型；partial 只展示，endpoint final 才触发回答。"""
    terminal.model_loading(True)
    session = (
        asr.VadGatedSherpaStream()
        if config.SHERPA_VAD_GATE_ENABLED
        else asr.SherpaStream()
    )
    terminal.model_loading(False)
    mic = audio.Mic()
    mic.start()
    terminal.listening(initial=True)
    try:
        for frame in mic.frames():
            for update in session.accept(frame):
                if not update.final:
                    _show_partial(update.text)
                    continue
                text = update.text if _usable_transcript(update.text) else None
                if not text:
                    continue
                _show_final(text)
                _answer_half_duplex(chat, mic, text, pcm=update.pcm)
                session.reset()
    finally:
        mic.stop()


def _run_barge_in_mic(chat: llm.Chat) -> None:
    """显式开启的实验模式：仅建议在耳机或具备 AEC 的设备上使用。"""
    mic = capture.Mic()
    mic.start()
    listener = capture.BackgroundListener(mic)
    listener.start()
    terminal.listening(initial=True)
    pending_text = None
    try:
        while True:
            if pending_text is None:
                try:
                    pcm = listener.get(timeout=0.5)
                except queue.Empty:
                    continue
                text = _transcribe(pcm)
            else:
                text, pending_text = pending_text, None
            if not text:
                continue
            reflection.user_activity()
            terminal.user_text(text)
            try:
                pending_text = _answer_interruptibly(chat, text, listener, pcm=pcm)
            except Exception as e:
                terminal.notice(f"LLM 调用失败: {e}")
                continue
            # 不论正常结束还是被打断，都必须清掉播放尾音与 VAD 局部状态。
            mic.reset_vad()
            mic.flush()
            listener.flush()
            if pending_text is None:
                terminal.listening()
    finally:
        mic.stop()
        listener.join()


def _run_duplex_mic(chat: llm.Chat) -> None:
    """真双工数据通路(commit 06+:FloorManager 状态机 + 真中断)。

    行为:
    - 采集源按 ``AUDIO_IO_BACKEND`` 选:
      * ``portaudio``:``Mic`` + sounddevice(无 AEC,需耳机)
      * ``macos_voice``:``MacOSVoiceIO`` 通过 Swift helper 取 AEC-clean PCM
    - 用户说话时持续 partial 流给 UI;同时后台跑 LLM/TTS answer。
    - FloorManager 决定用户 partial 是否打断 agent(回声过滤)。
    - 真正打断:cancel 当前 LLM streaming + 中断 TTS 播放。
    - Mic 不再 pause/resume — 真全双工。
    - 用户说完(FINAL)后,无论是否打断,都走下一轮 answer。

    AEC 状态(commit 07+):
    - macos_voice 后端:TTS 播放也通过 AudioIO 推到 helper,Swift playerNode
      与 inputNode 处于同一 AVAudioEngine,voice-processing 能拿到 playback
      reference → 完整外放 AEC。
    - portaudio 后端:sounddevice 无硬件 AEC,需配合耳机。
    """
    from .audio.io import get_audio_io
    from .interaction.floor import FloorManager

    listener = duplex.DuplexListener(queue_maxsize=64)
    floor = FloorManager()

    # commit 07+:复用 cli.py 提前创建的 audio_io(用于 TTS 播放回灌)。
    # cli 没创建(duplex 但 AUDIO_IO_BACKEND=portaudio)时,这里自己起一个
    # PortAudioIO,保证 StreamingSpeaker._play_loop 一定有 audio_io 可用。
    audio_io = tts.get_audio_io()
    owns_audio_io = False
    if audio_io is None and config.AUDIO_IO_BACKEND == "portaudio":
        audio_io = get_audio_io("portaudio")
        audio_io.start()
        tts.set_audio_io(audio_io)
        owns_audio_io = True

    if config.AUDIO_IO_BACKEND == "macos_voice":
        assert audio_io is not None  # cli.py 必须提前注入
        audio_io.add_capture_listener(listener.on_frame)
        # macos_voice:AudioIO 接管采集,Mic 没有 pause/resume 的概念
        audio_io_holder: list = [audio_io]
        mic_for_answer: object = None  # type: ignore[assignment]
    else:
        mic = capture.Mic()
        mic.start()
        mic.add_frame_listener(listener.on_frame)
        audio_io_holder = []  # type: ignore[assignment]
        mic_for_answer = mic

    terminal.listening(initial=True)
    try:
        for event in listener.events():
            if event.kind == duplex.DuplexEventKind.SPEECH_START:
                # 用户开始说话 → 通知 FloorManager
                floor.user_start()
                # 如果当前有 answer 在跑,cancel 它
                answer_state = _current_answer_state()
                if answer_state is not None:
                    answer_state.cancel.set()
                continue

            if event.kind == duplex.DuplexEventKind.PARTIAL:
                _show_partial(event.text)
                # FloorManager 决定是否打断
                if floor.user_partial(event.text):
                    answer_state = _current_answer_state()
                    if answer_state is not None and not answer_state.cancel.is_set():
                        answer_state.cancel.set()
                        terminal.interrupted(event.text)
                else:
                    # 没打断:看一下是不是 backchannel("嗯"之类),给 UI 一个轻量提示
                    bc = floor.pending_backchannel
                    if bc is not None and bc == event.text.strip():
                        terminal.notice(f"…{bc}")
                continue

            if event.kind == duplex.DuplexEventKind.FINAL:
                text = event.text if _usable_transcript(event.text) else None
                was_interrupted = floor.user_final(text or "", pcm=event.pcm)
                if text:
                    _show_final(text)
                    # 等待上一轮 answer 真正结束(cancel 后还会跑完流清理)
                    _wait_answer_done(timeout_s=0.5)
                    # 起新 answer(真全双工:不再 mic.pause)
                    _spawn_answer(chat, text, pcm=event.pcm)
                elif was_interrupted:
                    # 没拿到 final 但 floor 说被打断了:丢掉,等下一轮
                    pass
                continue

            # SPEECH_END 当前不单独消费(都在 FINAL 里走完)
    finally:
        # 关停当前 answer
        answer_state = _current_answer_state()
        if answer_state is not None:
            answer_state.cancel.set()
        _wait_answer_done(timeout_s=1.0)
        # 关停采集
        if config.AUDIO_IO_BACKEND == "macos_voice":
            io = audio_io_holder[0] if audio_io_holder else None
            if io is not None:
                _safe_cleanup(
                    lambda: io.remove_capture_listener(listener.on_frame),
                )
        else:
            mic = mic_for_answer
            if mic is not None:
                _safe_cleanup(
                    lambda: mic.remove_frame_listener(listener.on_frame),
                    lambda: mic.stop(),
                )
        listener.stop()
        # commit 07+:只有当 _run_duplex_mic 自己 start 的 audio_io 才
        # 在这里 stop;cli 注入的由 cli 负责关停。
        if owns_audio_io and audio_io is not None:
            try:
                audio_io.stop()
                tts.set_audio_io(None)
            except Exception:
                pass


def _safe_cleanup(*callables) -> None:
    """顺序调多个清理动作;任一抛异常不影响后续。"""
    for fn in callables:
        try:
            fn()
        except Exception:
            pass


# ── commit 06+:answer 线程化 + cancel 同步 ─────────────────────────


class _AnswerState:
    """当前在跑的 answer 状态:cancel 事件 + 完成事件 + 错误。"""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.error: BaseException | None = None


_active_answer_state: _AnswerState | None = None
_active_answer_lock = threading.Lock()


def _current_answer_state() -> _AnswerState | None:
    with _active_answer_lock:
        return _active_answer_state


def _wait_answer_done(timeout_s: float) -> None:
    """等当前 answer 线程结束(cancel 后还要清理流)。"""
    state = _current_answer_state()
    if state is None:
        return
    state.done.wait(timeout=timeout_s)


def _spawn_answer(chat: llm.Chat, text: str, pcm: bytes | None = None) -> None:
    """后台跑 ``_answer``,cancel 可由 ``_current_answer_state().cancel`` 触发。"""
    state = _AnswerState()
    with _active_answer_lock:
        global _active_answer_state
        _active_answer_state = state

    def _run() -> None:
        try:
            _answer(chat, text, state.cancel, pcm=pcm)
        except BaseException as e:
            state.error = e
        finally:
            state.done.set()
            with _active_answer_lock:
                global _active_answer_state
                # 只清掉自己;防止被新 answer 覆盖后误清
                if _active_answer_state is state:
                    _active_answer_state = None

    threading.Thread(target=_run, name="soul-tty-duplex-answer", daemon=True).start()


# ── commit 13+:统一编排入口 ────────────────────────────────────────


class VoiceMode(str, enum.Enum):
    """当前会话的语音交互模式(commit 13+ 单一入口后用)。"""

    HALF_DUPLEX = "half_duplex"
    BARGE_IN = "barge_in"
    FULL_DUPLEX = "full_duplex"


def _detect_voice_mode() -> VoiceMode:
    """根据 config 选择模式,优先级 FULL_DUPLEX > BARGE_IN > HALF_DUPLEX。"""
    if config.DUPLEX_ENABLED:
        return VoiceMode.FULL_DUPLEX
    if config.BARGE_IN_ENABLED:
        return VoiceMode.BARGE_IN
    return VoiceMode.HALF_DUPLEX


def _voice_mode_warning(mode: VoiceMode) -> str:
    """模式对应的一次性启动警告(给 UI 提示用)。"""
    if mode == VoiceMode.FULL_DUPLEX:
        if config.AUDIO_IO_BACKEND == "macos_voice":
            from .audio import tts as _tts

            # commit 07+:cli 已经把 MacOSVoiceIO 注入 tts,TTS 播放也走
            # AVAudioEngine 的 playerNode,voice-processing 能拿到
            # playback reference → 完整外放 AEC。
            if _tts.get_audio_io() is not None:
                return ""
            return (
                "双工 + macos_voice 模式:TTS 播放未注入 AVAudioEngine,"
                "完整 AEC 失效(打断可能不触发)"
            )
        return "双工模式已开启 — 使用麦克风直采,建议使用耳机防止外放自激"
    if mode == VoiceMode.BARGE_IN:
        return "插话模式已开启,请使用耳机或带 AEC 的设备"
    return ""  # half-duplex:默认模式,不打 warning


def _run_voice_session(chat: llm.Chat) -> None:
    """统一编排入口(commit 13+):根据 config 选模式,转给具体实现。"""
    mode = _detect_voice_mode()
    warn = _voice_mode_warning(mode)
    if warn:
        terminal.warning(warn)
    if mode == VoiceMode.FULL_DUPLEX:
        _run_duplex_mic(chat)
    elif mode == VoiceMode.BARGE_IN:
        _run_barge_in_mic(chat)
    else:
        _run_sherpa_half_duplex_mic(chat, capture)


def run_microphone(chat: llm.Chat) -> None:
    """对外唯一入口。设置 _active_chat,然后交给 _run_voice_session。"""
    global _active_chat
    _active_chat = chat
    _run_voice_session(chat)


def answer_text(chat: llm.Chat, text: str) -> str:
    global _active_chat
    _active_chat = chat
    return _answer(chat, text)


def run_file(chat: llm.Chat, path: str) -> None:
    _run_file(chat, path)
