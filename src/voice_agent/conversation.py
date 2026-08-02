"""对话编排：麦克风 -> ASR -> LLM -> TTS。"""

import difflib
import queue
import re
import sys
import threading
import wave
from collections.abc import Callable

from . import config
from .audio import asr, capture, tts
from .clients import llm
from .ui import terminal

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
) -> str:
    """流式打印 LLM 回答;给了 speaker 就按句切分实时送 TTS。"""
    terminal.answer_start()
    parts = []
    buf = ""
    playback_started = False
    for token in chat.ask_stream(text, cancel):
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
) -> str:
    cancel = cancel or threading.Event()
    if config.TTS_ENABLED and not config.TTS_WHOLE_ANSWER:
        with tts.StreamingSpeaker(
            cancel, on_audio_level=terminal.mouth_level
        ) as speaker:
            return _print_answer(chat, text, speaker, cancel, on_token)
    answer = _print_answer(chat, text, cancel=cancel, on_token=on_token)
    if config.TTS_ENABLED and answer and not cancel.is_set():
        try:
            terminal.speaking()
            tts.speak(answer, cancel, on_audio_level=terminal.mouth_level)
        except Exception as e:
            if not cancel.is_set():
                terminal.notice(f"TTS 失败: {e}")
    return answer


def _usable_transcript(text: str) -> bool:
    """过滤空文本和 ASR 在静音/噪声上的常见幻觉。"""
    normalized = text.strip("()（） \t。.!！?？~～,")
    return bool(normalized) and not any(h in normalized for h in HALLUCINATIONS)


def _is_probable_echo(heard: str, spoken: str) -> bool:
    """判断回答期间收到的文本是否更像扬声器回声而非插话。"""
    heard_n = _NON_SPEECH_TEXT.sub("", heard.lower())
    spoken_n = _NON_SPEECH_TEXT.sub("", spoken.lower())
    if len(heard_n) < 3 or not spoken_n:
        return False
    if heard_n in spoken_n:
        return True
    window = spoken_n[-max(len(heard_n) * 2, 24):]
    return difflib.SequenceMatcher(None, heard_n, window).ratio() >= config.BARGE_IN_ECHO_SIMILARITY


def _transcribe(pcm: bytes) -> str | None:
    try:
        text = asr.transcribe(pcm)
    except Exception as e:
        terminal.notice(f"识别失败: {e}，继续聆听")
        return None
    return text if _usable_transcript(text) else None


def _answer_interruptibly(chat: llm.Chat, text: str, listener) -> str | None:
    """后台回答，前台同时确认插话；返回已确认的下一句用户文本。"""
    cancel = threading.Event()
    answer_parts: list[str] = []
    outcome: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

    def run_answer() -> None:
        try:
            _answer(chat, text, cancel, answer_parts.append)
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
    if config.SHERPA_PARTIAL_ENABLED:
        terminal.partial(text)


def _show_final(text: str) -> None:
    terminal.user_text(text)


def _answer_half_duplex(chat: llm.Chat, mic, text: str) -> None:
    mic.pause()
    try:
        _answer(chat, text)
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
    session = asr.SherpaStream()
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
                _answer_half_duplex(chat, mic, text)
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
            terminal.user_text(text)
            try:
                pending_text = _answer_interruptibly(chat, text, listener)
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


def run_microphone(chat: llm.Chat) -> None:
    if config.BARGE_IN_ENABLED:
        terminal.warning("插话模式已开启，请使用耳机或带 AEC 的设备")
        _run_barge_in_mic(chat)
    else:
        _run_sherpa_half_duplex_mic(chat, capture)


def answer_text(chat: llm.Chat, text: str) -> str:
    return _answer(chat, text)


def run_file(chat: llm.Chat, path: str) -> None:
    _run_file(chat, path)
