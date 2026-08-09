"""对话编排：麦克风 -> ASR -> LLM -> TTS。"""

import difflib
import enum
import queue
import re
import sys
import threading
import time
import wave
from collections.abc import Callable

import numpy as np

from . import config, observability, prompt, reflection
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
# Agency：每轮在 Memory / LLM 之前决定回应方式；本地常数时间，不走网络。
_response_policy_provider: Callable[[str], object] | None = None
# 当前模式是否允许把完成轮次送入长期 Memory。由 UI 换装时原子切换；
# 每轮回答开始时取快照，避免回答途中切换模式造成隐私竞态。
_memory_persistence_allowed = True

_SECRET_ATMOSPHERE_INSTRUCTION = (
    "[Secret Mode Response Policy]\n"
    "本轮不要采用普通助手的一问一答结构，也不要只说一句确认。主动承接并推进当前"
    "私密场景，用三至六句适合口语朗读的话持续营造氛围；把环境、距离、声音、呼吸、"
    "触感、身体反应与下一步意图自然串联起来。提问可省略，确有必要时最多一个。"
    "不要解释策略，不要输出 Markdown。若用户表达停止、拒绝或不适，则立即停止"
    "场景推进并简短尊重边界。"
)


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


def set_response_policy_provider(
    provider: Callable[[str], object] | None,
) -> None:
    """注册 Agency ResponsePolicy；None 表示保持传统的有问必答。"""
    global _response_policy_provider
    _response_policy_provider = provider


def set_memory_persistence_allowed(allowed: bool) -> None:
    global _memory_persistence_allowed
    _memory_persistence_allowed = bool(allowed)


def _current_tts_instruct(*, private: bool = False) -> str:
    if private:
        # 秘密套装的声学指令经过稳定性验证；不要让 excited 等会话情绪
        # 覆盖成高张力指令，否则 Qwen3-TTS 容易退化为拉长气声或持续音。
        return config.MLX_TTS_INSTRUCT
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
        observability.exception(
            "memory.recall.error",
            "记忆召回失败，已降级为空",
        )
        return ""


def _current_voice_ref(pcm: bytes | None) -> int | None:
    """提交 utterance PCM 给 VoiceStateService，返回 VoiceRef。"""
    if pcm is None or _voice_submit_provider is None:
        return None
    try:
        return _voice_submit_provider(pcm)
    except Exception:
        observability.exception(
            "voice_state.submit.error",
            "声音感知任务提交失败",
        )
        return None


def _current_response_decision(user_text: str):
    provider = _response_policy_provider
    if provider is None:
        return None
    try:
        return provider(user_text)
    except Exception:
        observability.exception(
            "agency.policy.error",
            "Response Policy 决策失败，已降级为正常回答",
        )
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

_NON_SPEECH_TEXT = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")
_STRONG_SPEECH_END = frozenset("。!！?？;；\n")
_SOFT_SPEECH_END = frozenset("，,、：:")
_TRAILING_CLOSERS = frozenset("”’\"'）)]】》〉」』")


def _speakable_chars(text: str) -> int:
    return len(_NON_SPEECH_TEXT.sub("", text.lower()))


def _pcm_energy(pcm: bytes | None) -> tuple[float, float]:
    """返回 int16 PCM 的 peak / RMS（0~1），用于定位假唤醒。"""
    if not pcm:
        return 0.0, 0.0
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if not samples.size:
        return 0.0, 0.0
    peak = float(np.max(np.abs(samples))) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) / 32768.0
    return round(peak, 6), round(rms, 6)


def _punctuation_cuts(text: str, punctuation: frozenset[str]) -> list[int]:
    """返回包含尾随引号/括号的自然切点，避免下一段只剩一个闭合符号。"""
    cuts: list[int] = []
    for index, char in enumerate(text):
        if char not in punctuation:
            continue
        cut = index + 1
        while cut < len(text) and text[cut] in _TRAILING_CLOSERS:
            cut += 1
        if not cuts or cuts[-1] != cut:
            cuts.append(cut)
    return cuts


def _hard_cut(text: str, max_chars: int) -> int | None:
    """按可朗读字符数兜底切分，并避免截断连续英文、数字和小数。"""
    seen = 0
    cut = None
    for index, char in enumerate(text):
        if _NON_SPEECH_TEXT.sub("", char.lower()):
            seen += 1
        if seen >= max_chars:
            cut = index + 1
            break
    if cut is None:
        return None
    # 不在 ASCII 单词、版本号或小数中间下刀；最长只向后保护 16 个字符，
    # 防止异常的超长无空格 token 让缓冲无限增长。
    protected_until = min(len(text), cut + 16)
    while (
        cut < protected_until
        and cut > 0
        and text[cut - 1].isascii()
        and text[cut - 1].isalnum()
        and text[cut].isascii()
        and (text[cut].isalnum() or text[cut] in "._-")
    ):
        cut += 1
    return cut


class _SemanticSpeechBuffer:
    """把 LLM token 聚合成适合自然发音的语义段。

    优先完整句；过长句子在逗号等自然短语边界提前提交。第一段更短，
    后续段稍长，让首音速度和跨句情绪之间保持平衡。
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._pending_since: float | None = None
        self._segments = 0

    def feed(self, token: str, *, now: float | None = None) -> list[str]:
        if not token:
            return []
        current = time.monotonic() if now is None else now
        if not self._buffer:
            self._pending_since = current
        self._buffer += token
        emitted: list[str] = []
        while (cut := self._choose_cut(current)) is not None:
            segment = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:].lstrip()
            if segment and _speakable_chars(segment):
                emitted.append(segment)
                self._segments += 1
            self._pending_since = current if self._buffer else None
        return emitted

    def flush(self) -> list[str]:
        segment = self._buffer.strip()
        self._buffer = ""
        self._pending_since = None
        if not segment or not _speakable_chars(segment):
            return []
        self._segments += 1
        return [segment]

    def _choose_cut(self, now: float) -> int | None:
        if not self._buffer:
            return None
        minimum = (
            config.TTS_SEMANTIC_FIRST_MIN_CHARS
            if self._segments == 0
            else config.TTS_SEMANTIC_MIN_CHARS
        )
        speakable = _speakable_chars(self._buffer)
        waited_ms = (
            (now - self._pending_since) * 1000
            if self._pending_since is not None
            else 0.0
        )
        soft = _punctuation_cuts(self._buffer, _SOFT_SPEECH_END)
        usable_soft = [
            cut for cut in soft
            if _speakable_chars(self._buffer[:cut]) >= minimum
        ]
        strong = _punctuation_cuts(self._buffer, _STRONG_SPEECH_END)
        strong_cut = next(
            (
                cut for cut in strong
                if _speakable_chars(self._buffer[:cut]) >= minimum
            ),
            None,
        )
        if strong_cut is not None:
            strong_chars = _speakable_chars(self._buffer[:strong_cut])
            if strong_chars <= config.TTS_SEMANTIC_TARGET_CHARS:
                return strong_cut
            # 一个完整句已经明显过长时，不因已经看见句号而放弃前面的
            # 自然逗号；优先在目标长度附近启动首段。
            before_strong = [
                cut for cut in usable_soft
                if cut < strong_cut
                and _speakable_chars(self._buffer[:cut])
                <= config.TTS_SEMANTIC_MAX_CHARS
            ]
            if before_strong:
                return min(
                    before_strong,
                    key=lambda cut: abs(
                        _speakable_chars(self._buffer[:cut])
                        - config.TTS_SEMANTIC_TARGET_CHARS
                    ),
                )
            return strong_cut

        if usable_soft and (
            speakable >= config.TTS_SEMANTIC_TARGET_CHARS
            or waited_ms >= config.TTS_SEMANTIC_MAX_WAIT_MS
        ):
            # 取最靠近目标长度、但不超过硬上限的自然边界。
            bounded = [
                cut for cut in usable_soft
                if _speakable_chars(self._buffer[:cut])
                <= config.TTS_SEMANTIC_MAX_CHARS
            ]
            if bounded:
                return min(
                    bounded,
                    key=lambda cut: abs(
                        _speakable_chars(self._buffer[:cut])
                        - config.TTS_SEMANTIC_TARGET_CHARS
                    ),
                )

        if speakable >= config.TTS_SEMANTIC_MAX_CHARS:
            if usable_soft:
                bounded = [
                    cut for cut in usable_soft
                    if _speakable_chars(self._buffer[:cut])
                    <= config.TTS_SEMANTIC_MAX_CHARS
                ]
                if bounded:
                    return bounded[-1]
            return _hard_cut(self._buffer, config.TTS_SEMANTIC_MAX_CHARS)
        return None


def _print_answer(
    chat: llm.Chat,
    text: str,
    speaker: tts.StreamingSpeaker | None = None,
    cancel: threading.Event | None = None,
    on_token: Callable[[str], None] | None = None,
    *,
    recall: str = "",
    response_instruction: str = "",
    private: bool = False,
    on_speech_queued: Callable[[str], None] | None = None,
) -> str:
    """流式打印 LLM 回答；给了 speaker 就按语义段送入 TTS 流水线。

    ``on_speech_queued``:每句实际送进 TTS 的文本(commit 07+ 用于回声参考,
    比 ``on_token`` 更准确,因为 LLM token 可能还没送到 TTS 就中断了)。
    """
    terminal.answer_start()
    parts = []
    speech_buffer = _SemanticSpeechBuffer() if speaker is not None else None
    playback_started = False
    segment_index = 0
    answer_started_at = time.perf_counter()

    def queue_speech(segment: str) -> None:
        nonlocal playback_started, segment_index
        segment_index += 1
        queued_ms = observability.elapsed_ms(answer_started_at)
        observability.event(
            "tts.segment_queued",
            duration_ms=queued_ms,
            segment_index=segment_index,
            segment_chars=_speakable_chars(segment),
            first=not playback_started,
        )
        if not playback_started:
            # 兼容现有日志查询，同时新增更准确的 segment 事件。
            observability.event(
                "tts.first_sentence_queued",
                duration_ms=queued_ms,
                sentence_chars=len(segment),
            )
            terminal.speaking()
            playback_started = True
        if on_speech_queued is not None:
            on_speech_queued(segment)
        assert speaker is not None
        speaker.say(segment)

    stream_options = {"recall": recall}
    # 兼容测试替身和第三方 Chat adapter：只有 Agency 真正给出本轮指令时
    # 才使用新增参数；传统 ANSWER 路径的调用形状保持不变。
    if response_instruction:
        stream_options["response_instruction"] = response_instruction
    if private:
        stream_options["private"] = True
    try:
        for token in chat.ask_stream(text, cancel, **stream_options):
            terminal.answer_chunk(token)
            parts.append(token)
            if on_token is not None:
                on_token(token)
            if speech_buffer is not None:
                for segment in speech_buffer.feed(token):
                    queue_speech(segment)
    finally:
        # 网络错误、服务端 5xx 或取消都必须清掉“正在想…”占位。
        terminal.answer_end()
    if chat.last_stop_reason == "repetition":
        terminal.notice("检测到模型重复生成，已自动截断")
    if speech_buffer is not None:
        for segment in speech_buffer.flush():
            queue_speech(segment)
    return "".join(parts).strip()


def _answer_impl(
    chat: llm.Chat,
    text: str,
    cancel: threading.Event | None = None,
    on_token: Callable[[str], None] | None = None,
    pcm: bytes | None = None,
    on_speech_queued: Callable[[str], None] | None = None,
) -> str:
    cancel = cancel or threading.Event()
    memory_allowed = _memory_persistence_allowed
    policy_started_at = time.perf_counter()
    decision = _current_response_decision(text)
    response_instruction = getattr(decision, "instruction", "") if decision else ""
    mode = getattr(getattr(decision, "mode", None), "value", "answer")
    if not memory_allowed:
        # 秘密模式有自己的叙事节奏：不能被通用 Agency 压成十八字短回复、
        # 随机沉默或突然转题。制止词仍交给秘密人格与本轮策略立即收住。
        mode = "answer"
        response_instruction = _SECRET_ATMOSPHERE_INSTRUCTION
    observability.event(
        "agency.policy.decision",
        duration_ms=observability.elapsed_ms(policy_started_at),
        mode=mode,
        reason=getattr(decision, "reason", "disabled") if decision else "disabled",
    )
    if mode == "silence":
        chat.record_silence(text, private=not memory_allowed)
        terminal.intentional_silence()
        return ""
    # 一段回答内锁定同一份 TTS 指令；中途换 mood 不影响本句。
    # Provider 未注册时（emotion 关闭）回退到 config.MLX_TTS_INSTRUCT。
    instruct = _current_tts_instruct(private=not memory_allowed)
    # 本轮检索到的相关记忆；临时插入，不进 system prompt / 不进 history。
    recall_started_at = time.perf_counter()
    recall = _current_recall(text)
    observability.event(
        "memory.recall.complete",
        duration_ms=observability.elapsed_ms(recall_started_at),
        hit=bool(recall),
        query_chars=len(text),
    )
    # 提交 utterance PCM 给 VoiceStateService（不阻塞）
    voice_ref = _current_voice_ref(pcm)
    if config.TTS_ENABLED and not config.TTS_WHOLE_ANSWER:
        with tts.StreamingSpeaker(
            cancel, terminal.audio_level, instruct=instruct
        ) as speaker:
            answer = _print_answer(
                chat, text, speaker, cancel, on_token, recall=recall,
                response_instruction=response_instruction,
                private=not memory_allowed,
                on_speech_queued=on_speech_queued,
            )
    else:
        answer = _print_answer(
            chat, text, cancel=cancel, on_token=on_token, recall=recall,
            response_instruction=response_instruction,
            private=not memory_allowed,
            on_speech_queued=on_speech_queued,
        )
        if config.TTS_ENABLED and answer and not cancel.is_set():
            # commit 07+ fix:whole-answer 也调 on_speech_queued,
            # 让 floor.agent_text 有回声参考
            if on_speech_queued is not None:
                on_speech_queued(answer)
            try:
                terminal.speaking()
                tts.speak(answer, cancel, terminal.audio_level, instruct=instruct)
            except Exception as e:
                if not cancel.is_set():
                    observability.exception(
                        "tts.error",
                        "TTS 合成或播放失败",
                        backend=config.TTS_BACKEND,
                    )
                    terminal.notice(f"TTS 失败: {e}")
    if answer and not cancel.is_set():
        reflection_options = {"voice_ref": voice_ref}
        if not memory_allowed:
            reflection_options["memory_allowed"] = False
        reflection.record_turn(text, answer, **reflection_options)
    return answer


def _answer(
    chat: llm.Chat,
    text: str,
    cancel: threading.Event | None = None,
    on_token: Callable[[str], None] | None = None,
    pcm: bytes | None = None,
    on_speech_queued: Callable[[str], None] | None = None,
    *,
    turn_id: str | None = None,
) -> str:
    """为一轮回答绑定 turn_id，并记录端到端耗时。"""
    effective_turn_id = turn_id or observability.current_turn_id()
    if effective_turn_id == "-":
        effective_turn_id = observability.new_turn_id()
    answer_started_at = time.perf_counter()
    with observability.bind_turn(effective_turn_id):
        observability.event(
            "answer.start",
            input_chars=len(text),
            history_messages=len(getattr(chat, "messages", ())),
            tts_whole_answer=config.TTS_WHOLE_ANSWER,
        )
        try:
            answer = _answer_impl(
                chat,
                text,
                cancel,
                on_token,
                pcm,
                on_speech_queued,
            )
        except Exception:
            observability.exception("answer.error", "主对话回答失败")
            # answer_start() 已把仪表盘切到 thinking；失败时恢复可交互状态，
            # 避免用户看到永久“正在思考”。
            terminal.listening()
            raise
        observability.event(
            "answer.complete",
            duration_ms=observability.elapsed_ms(answer_started_at),
            output_chars=len(answer),
            cancelled=bool(cancel is not None and cancel.is_set()),
            stop_reason=chat.last_stop_reason or "complete",
        )
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
    started_at = time.perf_counter()
    try:
        text = asr.transcribe(pcm)
    except Exception as e:
        observability.exception("asr.error", "语音识别失败")
        terminal.notice(f"识别失败: {e}，继续聆听")
        return None
    observability.event(
        "asr.complete",
        duration_ms=observability.elapsed_ms(started_at),
        audio_ms=round(len(pcm) / (config.SAMPLE_RATE * 2) * 1000, 2),
        text_chars=len(text or ""),
        usable=_usable_transcript(text),
    )
    return text if _usable_transcript(text) else None


def _answer_interruptibly(
    chat: llm.Chat,
    text: str,
    listener,
    pcm: bytes | None = None,
    *,
    turn_id: str | None = None,
) -> str | None:
    """后台回答，前台同时确认插话；返回已确认的下一句用户文本。"""
    cancel = threading.Event()
    answer_parts: list[str] = []
    outcome: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

    def run_answer() -> None:
        try:
            _answer(
                chat,
                text,
                cancel,
                answer_parts.append,
                pcm=pcm,
                turn_id=turn_id,
            )
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


def _show_final(text: str, *, interrupted: bool = False) -> None:
    reflection.user_activity()
    if interrupted:
        terminal.user_text(text, interrupted=True)
    else:
        terminal.user_text(text)


def _answer_half_duplex(
    chat: llm.Chat,
    mic,
    text: str,
    pcm: bytes | None = None,
    *,
    turn_id: str | None = None,
) -> None:
    mic.pause()
    try:
        _answer(chat, text, pcm=pcm, turn_id=turn_id)
    except Exception as e:
        observability.exception("answer.half_duplex.error", "半双工回答失败")
        terminal.notice(f"LLM 调用失败: {e}")
    finally:
        mic.reset_vad()
        mic.flush()
        mic.resume()
    terminal.listening()


def _run_sherpa_half_duplex_mic(chat: llm.Chat, audio) -> None:
    """PCM 持续送入在线模型；partial 只展示，endpoint final 才触发回答。"""
    terminal.model_loading(True)
    model_started_at = time.perf_counter()
    session = (
        asr.VadGatedSherpaStream()
        if config.SHERPA_VAD_GATE_ENABLED
        else asr.SherpaStream()
    )
    observability.event(
        "asr.model_ready",
        duration_ms=observability.elapsed_ms(model_started_at),
        mode="half_duplex",
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
                turn_id = observability.new_turn_id()
                observability.event(
                    "asr.final",
                    turn_id=turn_id,
                    text_chars=len(text),
                    audio_ms=(
                        round(len(update.pcm) / (config.SAMPLE_RATE * 2) * 1000, 2)
                        if update.pcm
                        else 0.0
                    ),
                    mode="half_duplex",
                    disposition="user",
                )
                _show_final(text)
                _answer_half_duplex(
                    chat,
                    mic,
                    text,
                    pcm=update.pcm,
                    turn_id=turn_id,
                )
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
            was_interruption = pending_text is not None
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
            turn_id = observability.new_turn_id()
            observability.event(
                "asr.final",
                turn_id=turn_id,
                text_chars=len(text),
                audio_ms=(
                    round(len(pcm) / (config.SAMPLE_RATE * 2) * 1000, 2)
                    if pcm
                    else 0.0
                ),
                mode="barge_in",
                disposition="interrupt" if was_interruption else "user",
            )
            reflection.user_activity()
            terminal.user_text(text, interrupted=was_interruption)
            try:
                pending_text = _answer_interruptibly(
                    chat,
                    text,
                    listener,
                    pcm=pcm,
                    turn_id=turn_id,
                )
            except Exception as e:
                observability.exception("answer.barge_in.error", "插话模式回答失败")
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
    from .interaction.floor import (
        FloorManager,
        UserFinalDisposition,
        UserPartialDisposition,
    )

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

    # AEC 后仍可能有低能量播放残差。把实时播放状态交给采集门控，在残差
    # 进入 VAD/ASR 前先变成静音；真人靠近麦克风说话时仍能立即打开门控。
    model_started_at = time.perf_counter()
    listener = duplex.DuplexListener(
        queue_maxsize=64,
        playback_active=(
            (lambda: audio_io.playback_active)
            if audio_io is not None
            else None
        ),
    )
    observability.event(
        "asr.model_ready",
        duration_ms=observability.elapsed_ms(model_started_at),
        mode="full_duplex",
    )

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
    speech_started_at: float | None = None
    try:
        for event in listener.events():
            if event.kind == duplex.DuplexEventKind.SPEECH_START:
                speech_started_at = time.perf_counter()
                observability.event("asr.speech_start", mode="full_duplex")
                # 用户开始说话 → 通知 FloorManager
                floor.user_start()
                # 注意:SPEECH_START 不等于打断。用户可能只是发出一个简短
                # 声音(清嗓/backchannel),真正的打断应由 PARTIAL 决定。
                # 所以这里不 cancel。 (commit 07+ fix)
                continue

            if event.kind == duplex.DuplexEventKind.PARTIAL:
                # FloorManager 决定是否打断
                should_interrupt = floor.user_partial(
                    event.text,
                    near_end=event.near_end,
                )
                partial_disposition = floor.last_partial_disposition

                # 回声和证据不足的 1~2 字 partial 不进入 UI，也不应被记作
                # 用户活动；否则即使 FINAL 被过滤，界面仍会像自说自话。
                if partial_disposition in (
                    UserPartialDisposition.ECHO,
                    UserPartialDisposition.HOLD,
                ):
                    continue

                if partial_disposition == UserPartialDisposition.BACKCHANNEL:
                    reflection.user_activity()
                    terminal.notice(f"…{event.text.strip()}")
                    continue

                _show_partial(event.text)
                if should_interrupt:
                    answer_state = _current_answer_state()
                    if answer_state is not None and not answer_state.cancel.is_set():
                        answer_state.cancel.set()
                        observability.event(
                            "interrupt.confirmed",
                            partial_chars=len(event.text),
                            source="partial",
                        )
                        terminal.interrupted(event.text)
                        # commit 07+ fix:立即清空 Swift 端已排队播放 buffer
                        if audio_io is not None:
                            try:
                                audio_io.flush_playback()
                            except Exception:
                                pass
                continue

            if event.kind == duplex.DuplexEventKind.FINAL:
                text = event.text if _usable_transcript(event.text) else None
                pcm_peak, pcm_rms = _pcm_energy(event.pcm)
                # 被 ASR 策略过滤的 hallucination 等同空 FINAL，绝不能让它
                # 把 FloorManager 从 AGENT_SPEAKING 推到 IDLE 并清空回声参考。
                cleaned = text.strip() if text else ""
                disposition = floor.user_final(
                    cleaned,
                    pcm=event.pcm,
                    near_end=event.near_end,
                    voice_rms=pcm_rms,
                )

                # commit 07+ fix:按 disposition 分流
                if disposition == UserFinalDisposition.ECHO:
                    # Serena 自己的回声 → 不触发新 answer
                    observability.event(
                        "asr.final.filtered",
                        mode="full_duplex",
                        disposition="echo",
                        text_chars=len(cleaned),
                        pcm_peak=pcm_peak,
                        pcm_rms=pcm_rms,
                    )
                    if config.DUPLEX_DEBUG:
                        terminal.notice(f"[echo-final] {cleaned}")
                    continue

                if disposition == UserFinalDisposition.BACKCHANNEL:
                    # 用户说"嗯/好的",不打断,不 spawn
                    observability.event(
                        "asr.final.filtered",
                        mode="full_duplex",
                        disposition="backchannel",
                        text_chars=len(cleaned),
                        pcm_peak=pcm_peak,
                        pcm_rms=pcm_rms,
                    )
                    if cleaned:
                        terminal.notice(f"…{cleaned}")
                    continue

                if disposition == UserFinalDisposition.IGNORED:
                    observability.event(
                        "asr.final.filtered",
                        mode="full_duplex",
                        disposition="ignored",
                        text_chars=len(cleaned),
                        pcm_peak=pcm_peak,
                        pcm_rms=pcm_rms,
                    )
                    continue

                # 真用户输入(USER 或 INTERRUPT)
                if not text:
                    continue

                turn_id = observability.new_turn_id()
                observability.event(
                    "asr.final",
                    turn_id=turn_id,
                    duration_ms=(
                        observability.elapsed_ms(speech_started_at)
                        if speech_started_at is not None
                        else None
                    ),
                    text_chars=len(text),
                    audio_ms=(
                        round(len(event.pcm) / (config.SAMPLE_RATE * 2) * 1000, 2)
                        if event.pcm
                        else 0.0
                    ),
                    mode="full_duplex",
                    disposition=disposition.value,
                    pcm_peak=pcm_peak,
                    pcm_rms=pcm_rms,
                )
                speech_started_at = None

                if disposition == UserFinalDisposition.INTERRUPT:
                    # partial 阶段没来得及打断,但最终仍需要 cancel
                    answer_state = _current_answer_state()
                    if answer_state is not None and not answer_state.cancel.is_set():
                        answer_state.cancel.set()
                        terminal.interrupted(cleaned)
                        if audio_io is not None:
                            try:
                                audio_io.flush_playback()
                            except Exception:
                                pass

                _show_final(
                    text,
                    interrupted=(
                        disposition == UserFinalDisposition.INTERRUPT
                    ),
                )
                wait_started_at = time.perf_counter()
                _wait_answer_done(timeout_s=0.5)
                observability.event(
                    "answer.previous_wait.complete",
                    turn_id=turn_id,
                    duration_ms=observability.elapsed_ms(wait_started_at),
                )
                _spawn_answer(
                    chat,
                    text,
                    pcm=event.pcm,
                    floor=floor,
                    audio_io=audio_io,
                    turn_id=turn_id,
                )
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
                observability.exception(
                    "shutdown.audio_io.error",
                    "关闭会话 AudioIO 失败",
                )


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


def _spawn_answer(
    chat: llm.Chat,
    text: str,
    pcm: bytes | None = None,
    floor=None,
    audio_io=None,
    turn_id: str | None = None,
) -> None:
    """后台跑 ``_answer``,cancel 可由 ``_current_answer_state().cancel`` 触发。

    如果传了 ``floor``(FloorManager),在 answer 生命周期内调
    ``agent_start/agent_chunk/agent_end``,让 ``agent_text`` 持续更新,
    供 ``is_probable_echo`` 做回声判定(避免 agent 自己的话被当成用户插话)。

    commit 07+ fix:
    - 用 ``on_speech_queued`` 替代 ``on_token`` 作为回声参考(LLM token
      可能还没送到 TTS 就中断了,用实际送 TTS 的文本更准确)
    - ``agent_end`` 延迟到 ``audio_io.wait_playback_drained()`` 之后
      (保证扬声器真正播完前,``agent_text`` 仍用于回声判定)
    """
    state = _AnswerState()
    effective_turn_id = turn_id or observability.new_turn_id()
    with _active_answer_lock:
        global _active_answer_state
        _active_answer_state = state

    def _run() -> None:
        with observability.bind_turn(effective_turn_id):
            if floor is not None:
                floor.agent_start()
            try:
                # 用 on_speech_queued 替代 on_token:实际送 TTS 的文本才是回声参考
                on_speech_queued = None
                if floor is not None:
                    def _on_speech_queued(sent: str) -> None:
                        floor.agent_chunk(sent)
                    on_speech_queued = _on_speech_queued
                _answer(
                    chat,
                    text,
                    state.cancel,
                    pcm=pcm,
                    on_speech_queued=on_speech_queued,
                    turn_id=effective_turn_id,
                )
            except BaseException as e:
                state.error = e
                observability.exception(
                    "answer.worker.error",
                    "双工回答线程失败",
                )
            finally:
                # commit 07+ fix:等到扬声器真正播完再 agent_end
                if audio_io is not None and not state.cancel.is_set():
                    drain_started_at = time.perf_counter()
                    try:
                        drained = audio_io.wait_playback_drained(timeout=15.0)
                        observability.event(
                            "audio.playback_drained",
                            duration_ms=observability.elapsed_ms(drain_started_at),
                            drained=drained,
                        )
                    except Exception:
                        observability.exception(
                            "audio.playback_drain.error",
                            "等待播放队列排空失败",
                        )
                if floor is not None:
                    floor.agent_end()
                restore_listening = False
                with _active_answer_lock:
                    global _active_answer_state
                    # 只清掉自己;防止被新 answer 覆盖后误清
                    if _active_answer_state is state:
                        _active_answer_state = None
                        restore_listening = not state.cancel.is_set()
                # full-duplex 没有外层 finally 帮它恢复 UI；必须等真正播放
                # 排空后再切回聆听。被打断或已被新回答替代时绝不能抢状态。
                try:
                    if restore_listening:
                        terminal.listening()
                finally:
                    state.done.set()

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
