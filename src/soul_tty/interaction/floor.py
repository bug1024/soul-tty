"""FloorManager:谁拥有麦克风(USER vs AGENT)。

设计目标:
- 把"是否打断"从 conversation 里抽出来,变成可单独测试的状态机。
- 不直接调 LLM / TTS / Mic,只暴露 ``should_interrupt()`` / ``take_floor()`` /
  ``release_floor()`` 这种纯决策接口,让上层(_run_duplex_mic)负责执行。

状态:
- IDLE:双方都没说话
- USER_SPEAKING:用户正在说话(从 DuplexListener SPEECH_START)
- AGENT_SPEAKING:Agent 正在说话(_answer 开始时)
- INTERRUPTED:用户打断了 Agent,但 Agent 还在收尾(cancel 后状态短暂停留)

决策:
- USER partial 出现 + 当前 AGENT_SPEAKING + 与已播放 transcript 不像回声
  → should_interrupt = True
- USER partial 是简短肯定词(backchannel,如"嗯""好的")→ 不打断,但记下来
  供 agent 下一轮参考
- USER final 出现:
  * 回声 → ``UserFinalDisposition.ECHO``,不改变 state/agent_text
  * backchannel → ``UserFinalDisposition.BACKCHANNEL``
  * 真插话 → ``UserFinalDisposition.INTERRUPT`` / ``USER``
- agent_end 后保留 ``_recent_agent_text`` 一段时间(grace period),
  避免房间尾声被当成用户输入。
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable
from typing import Optional

from .. import config
from .echo import (
    is_probable_echo as _is_probable_echo,
    normalize_speech_text,
)


# commit 11:backchannel 候选词 —— 中文里常见的"边听边回应"短词。
# 必须是 1~3 个字符,且不被认为是 echo。
BACKCHANNEL_WORDS: frozenset[str] = frozenset(
    {
        "嗯", "啊", "哦", "噢", "诶", "欸",
        "好的", "好", "是", "对", "对呀",
        "嗯嗯", "啊啊", "哦哦", "嗯哼",
    }
)


def is_backchannel(text: str) -> bool:
    """判断 ``text`` 是否像 backchannel(短肯定词,不该打断)。

    启发式:strip 后长度 ≤ 3 且命中 ``BACKCHANNEL_WORDS``。
    长度更长("嗯,听起来不错")会被当成完整插话,正常走 interrupt 路径。
    """
    cleaned = text.strip("()（）。.!！?？~～, \t\n")
    if not cleaned or len(cleaned) > 3:
        return False
    return cleaned in BACKCHANNEL_WORDS


class FloorState(str, enum.Enum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    AGENT_SPEAKING = "agent_speaking"
    INTERRUPTED = "interrupted"


class UserFinalDisposition(str, enum.Enum):
    """``user_final()`` 返回的分类结果(commit 07+ fix)。

    - ECHO: final 文本是 agent 回声,不改变 state,不清 agent_text
    - BACKCHANNEL: 用户说"嗯/好的",不打断,记下来供参考
    - IGNORED: 空文本或 ASR 已过滤结果，不改变 agent 状态
    - USER: 用户真插话,但之前没有 partial 打断(新 turn)
    - INTERRUPT: 用户真插话,且之前已通过 partial 触发打断
    """

    ECHO = "echo"
    BACKCHANNEL = "backchannel"
    IGNORED = "ignored"
    USER = "user"
    INTERRUPT = "interrupt"


class UserPartialDisposition(str, enum.Enum):
    """最近一条 partial 的判定，供上层决定是否展示及打断。"""

    USER = "user"
    ECHO = "echo"
    BACKCHANNEL = "backchannel"
    HOLD = "hold"
    INTERRUPT = "interrupt"


_EXPLICIT_INTERRUPT_PREFIXES = (
    "停",
    "不是",
    "不对",
    "换一个",
    "换个话题",
)

# 外放与近端人声重叠时，ASR 常在制止词前后混入 Serena 的几个字。这些
# 强制停止短语允许句中命中；相较之下“不是/不对/换一个”仍只允许前缀，
# 避免普通回答里的同名片段误触。
_EXPLICIT_INTERRUPT_CONTAINS = (
    "停下",
    "停止",
    "暂停",
    "先停",
    "听下",  # Paraformer 常把“停下”识别成同音词
    "别说",
    "别讲",
    "先别说",
    "不要说",
    "不要讲",
    "别再说",
    "不用说",
    "不说了",
    "够了",
    "可以了",
    "闭嘴",
    "打住",
    "等一下",
    "等等",
)

_SINGLE_CHAR_WAKE_WORDS = frozenset({"喂", "嗨"})


def is_explicit_interrupt(text: str) -> bool:
    """短文本只有明确的制止词才允许在 partial 阶段立即打断。"""
    cleaned = normalize_speech_text(text)
    return any(
        phrase in cleaned for phrase in _EXPLICIT_INTERRUPT_CONTAINS
    ) or any(cleaned.startswith(prefix) for prefix in _EXPLICIT_INTERRUPT_PREFIXES)


class FloorManager:
    """Floor 状态机:在 USER / AGENT 之间切换,决定是否打断。"""

    def __init__(
        self,
        echo_similarity: float | None = None,
        on_interrupt: Callable[[str], None] | None = None,
        backchannel_enabled: bool | None = None,
        partial_min_chars: int | None = None,
        partial_confirmations: int | None = None,
        natural_interrupt_enabled: bool | None = None,
        strong_interrupt_rms: float | None = None,
        strong_interrupt_min_chars: int | None = None,
    ) -> None:
        self._echo_similarity = (
            config.DUPLEX_ECHO_SIMILARITY
            if echo_similarity is None
            else echo_similarity
        )
        self._on_interrupt = on_interrupt
        self._backchannel_enabled = (
            backchannel_enabled
            if backchannel_enabled is not None
            else config.BACKCHANNEL_ENABLED
        )
        self._partial_min_chars = max(
            1,
            config.DUPLEX_PARTIAL_MIN_CHARS
            if partial_min_chars is None
            else partial_min_chars,
        )
        self._partial_confirmations = max(
            1,
            config.DUPLEX_PARTIAL_CONFIRMATIONS
            if partial_confirmations is None
            else partial_confirmations,
        )
        self._natural_interrupt_enabled = (
            config.DUPLEX_NATURAL_INTERRUPT_ENABLED
            if natural_interrupt_enabled is None
            else natural_interrupt_enabled
        )
        self._strong_interrupt_rms = max(
            0.0,
            config.DUPLEX_STRONG_INTERRUPT_RMS
            if strong_interrupt_rms is None
            else strong_interrupt_rms,
        )
        self._strong_interrupt_min_chars = max(
            1,
            config.DUPLEX_STRONG_INTERRUPT_MIN_CHARS
            if strong_interrupt_min_chars is None
            else strong_interrupt_min_chars,
        )
        self._lock = threading.Lock()
        self._state = FloorState.IDLE
        self._agent_text = ""
        self._last_interrupt_text: str | None = None
        self._pending_backchannel: str | None = None
        # 回声 grace period:agent_end 后保留最近文本一段时间
        self._recent_agent_text: str = ""
        self._agent_ended_at: float = 0.0
        # 一次 VAD utterance 固定一份播放参考。即使 FINAL 到达时 grace 已过，
        # 仍能用 SPEECH_START/partial 时的参考判断同一段残余回声。
        self._utterance_agent_text: str = ""
        self._last_partial_disposition = UserPartialDisposition.HOLD
        self._partial_candidate = ""
        self._partial_candidate_updates = 0
        # cancel 后 answer worker 可能比 ASR FINAL 更早调用 agent_end()。
        # 按 utterance 保存打断事实，避免 FINAL 被误判成新的普通输入。
        self._utterance_interrupted = False
        # 记录本段话是否在 Agent 仍持有话权时开始。普通接话若只与播放尾部
        # 轻微重叠，不立即打断；等 agent_end 后 FINAL 到达时可作为新轮提交。
        self._utterance_started_during_agent = False

    # ── 查询 ────────────────────────────────────────────────────────

    @property
    def state(self) -> FloorState:
        with self._lock:
            return self._state

    @property
    def agent_text(self) -> str:
        with self._lock:
            return self._agent_text

    @property
    def last_interrupt(self) -> str | None:
        with self._lock:
            return self._last_interrupt_text

    @property
    def pending_backchannel(self) -> str | None:
        """最近一次 backchannel(用户说了但没打断),agent 下一轮可参考。"""
        with self._lock:
            return self._pending_backchannel

    @property
    def last_partial_disposition(self) -> UserPartialDisposition:
        with self._lock:
            return self._last_partial_disposition

    def take_backchannel(self) -> str | None:
        """读走 pending_backchannel 并清空(避免重复注入)。"""
        with self._lock:
            val = self._pending_backchannel
            self._pending_backchannel = None
            return val

    def _effective_agent_text(self) -> str:
        """返回当前用于回声判定的文本。

        agent 正在说话 → 用 ``_agent_text``。
        agent 刚说完(grace period 内)→ 用 ``_recent_agent_text``。
        grace period 已过 → 空串(可以当作新用户输入)。
        """
        if self._agent_text:
            return self._agent_text
        if self._recent_agent_text:
            elapsed = time.monotonic() - self._agent_ended_at
            if elapsed < config.DUPLEX_ECHO_GRACE_MS / 1000:
                return self._recent_agent_text
        return ""

    def _utterance_echo_reference(self) -> str:
        """返回本次 utterance 的稳定播放参考，并吸收新增播放文本。"""
        current = self._effective_agent_text()
        if len(current) > len(self._utterance_agent_text):
            self._utterance_agent_text = current
        return self._utterance_agent_text

    # ── 状态变更 ────────────────────────────────────────────────────

    def user_start(self) -> None:
        """DuplexListener SPEECH_START 时调。

        修复(commit 07+):SPEECH_START 不等于打断。用户可能只是轻微出声
        (清嗓/backchannel),打断决策应由 ``user_partial`` 做。
        所以这里不再把 AGENT_SPEAKING → INTERRUPTED。
        """
        with self._lock:
            self._utterance_started_during_agent = (
                self._state == FloorState.AGENT_SPEAKING
            )
            self._utterance_agent_text = self._effective_agent_text()
            self._last_partial_disposition = UserPartialDisposition.HOLD
            self._partial_candidate = ""
            self._partial_candidate_updates = 0
            self._utterance_interrupted = False
            if self._state == FloorState.IDLE:
                self._state = FloorState.USER_SPEAKING
            # AGENT_SPEAKING 时保持原状态(不漏掉下一步的 user_partial 决策)

    def user_partial(self, text: str, *, near_end: bool = True) -> bool:
        """DuplexListener PARTIAL 时调。返回是否应当打断 agent。

        副作用:真正决定打断时,state 从 AGENT_SPEAKING → INTERRUPTED。
        backchannel:如果当前 AGENT_SPEAKING 且文本是 backchannel,
        不打断但记下来(供 agent 下一轮参考)。
        回声过滤:第一优先级——先判断回声,再判断 state。
        """
        with self._lock:
            if not text:
                self._last_partial_disposition = UserPartialDisposition.HOLD
                return False

            # 回声判断第一优先级；reference 固定到本次 utterance，避免
            # partial 在 grace 内、FINAL 却在 grace 外时发生自激。
            spoken = self._utterance_echo_reference()
            if spoken and _is_probable_echo(text, spoken, self._echo_similarity):
                self._last_partial_disposition = UserPartialDisposition.ECHO
                return False

            if self._state != FloorState.AGENT_SPEAKING:
                self._last_partial_disposition = UserPartialDisposition.USER
                return False

            if self._backchannel_enabled and is_backchannel(text):
                self._pending_backchannel = text.strip()
                self._last_partial_disposition = UserPartialDisposition.BACKCHANNEL
                return False

            normalized = normalize_speech_text(text)
            explicit = is_explicit_interrupt(text)

            if (
                self._backchannel_enabled
                and not explicit
                and not self._natural_interrupt_enabled
            ):
                self._last_partial_disposition = UserPartialDisposition.HOLD
                return False

            # 播放残差可以进入 ASR，以便低音量“停下”仍能被识别；但普通
            # 自然插话必须同时具备持续近端人声证据，不能只凭错误文本打断。
            if not near_end and not explicit:
                self._last_partial_disposition = UserPartialDisposition.HOLD
                return False

            # 中文 ASR 常先吐出 1~2 个字。此时既无法可靠做模糊回声判断，
            # 也不足以证明真人插话；明确的“停/等等/不对”等命令除外。
            if (
                self._backchannel_enabled
                and len(normalized) < self._partial_min_chars
                and not explicit
            ):
                self._last_partial_disposition = UserPartialDisposition.HOLD
                return False

            # 明确制止词走低延迟快路径。一般自然插话则要求 ASR 连续给出
            # 累积式 partial（如“我想问”→“我想问另一个问题”），避免单个
            # AEC 残差猜词立刻抢走话权。
            if self._backchannel_enabled and not explicit:
                if self._partial_candidate and (
                    normalized.startswith(self._partial_candidate)
                    or self._partial_candidate.startswith(normalized)
                ):
                    self._partial_candidate_updates += 1
                else:
                    self._partial_candidate_updates = 1
                self._partial_candidate = normalized
                if self._partial_candidate_updates < self._partial_confirmations:
                    self._last_partial_disposition = UserPartialDisposition.HOLD
                    return False

            self._state = FloorState.INTERRUPTED
            self._utterance_interrupted = True
            self._last_partial_disposition = UserPartialDisposition.INTERRUPT
            return True

    def user_final(
        self,
        text: str,
        pcm: bytes | None = None,
        *,
        near_end: bool = True,
        voice_rms: float = 0.0,
    ) -> UserFinalDisposition:
        """DuplexListener FINAL 时调。返回分类结果(不再只返回 bool)。

        修复(commit 07+):
        - 回声判断不依赖 ``AGENT_SPEAKING`` state(grace period 也生效)
        - 回声 final → ECHO,不改 state/不清 agent_text
        - backchannel → BACKCHANNEL
        - 真打断 → INTERRUPT(如果之前已 INTERRUPTED)或 USER
        """
        cleaned = text.strip() if text else ""
        with self._lock:
            # 空/已过滤 FINAL 不能抢走话权，更不能清掉 agent echo reference。
            if not cleaned:
                if self._state in (FloorState.USER_SPEAKING, FloorState.INTERRUPTED):
                    self._state = FloorState.IDLE
                self._utterance_interrupted = False
                self._utterance_agent_text = ""
                self._last_partial_disposition = UserPartialDisposition.HOLD
                self._partial_candidate = ""
                self._partial_candidate_updates = 0
                return UserFinalDisposition.IGNORED

            # 使用 utterance 固定参考，而不是仅依赖此刻是否仍在 grace 内。
            spoken = self._utterance_echo_reference()
            if cleaned and spoken and _is_probable_echo(cleaned, spoken, self._echo_similarity):
                # 如果 grace period 导致 IDLE→USER_SPEAKING,恢复 IDLE
                if self._state in (FloorState.USER_SPEAKING, FloorState.INTERRUPTED):
                    self._state = FloorState.IDLE
                self._utterance_interrupted = False
                self._utterance_agent_text = ""
                self._last_partial_disposition = UserPartialDisposition.ECHO
                self._partial_candidate = ""
                self._partial_candidate_updates = 0
                return UserFinalDisposition.ECHO

            # 2) backchannel
            if (
                self._state == FloorState.AGENT_SPEAKING
                and self._backchannel_enabled
                and is_backchannel(cleaned)
            ):
                self._pending_backchannel = cleaned
                self._utterance_interrupted = False
                self._utterance_agent_text = ""
                self._last_partial_disposition = UserPartialDisposition.BACKCHANNEL
                self._partial_candidate = ""
                self._partial_candidate_updates = 0
                return UserFinalDisposition.BACKCHANNEL

            # 待机时孤立的“嗯/啊/哦”通常是环境噪声触发后的 ASR 猜词，且
            # 本身没有足够语义开启一轮对话。真实用户可继续说完整内容；
            # Agent 说话期间的同类输入已在上方作为 backchannel 处理。
            if (
                self._state == FloorState.USER_SPEAKING
                and self._backchannel_enabled
                and is_backchannel(cleaned)
            ):
                self._state = FloorState.IDLE
                self._utterance_interrupted = False
                self._utterance_agent_text = ""
                self._last_partial_disposition = UserPartialDisposition.HOLD
                self._partial_candidate = ""
                self._partial_candidate_updates = 0
                return UserFinalDisposition.IGNORED

            # 待机时的单个汉字（实测底噪会偶发被猜成“这”）信息量不足，
            # 不应凭空启动 LLM/TTS；保留“喂/嗨”作为自然唤起词。
            normalized_final = normalize_speech_text(cleaned)

            # AEC 残差旁路只服务于明确制止词。没有持续近端人声证据且
            # utterance 起始于当前/最近一次播放时，其他文本一律不创建轮次。
            explicit_final = is_explicit_interrupt(cleaned)
            strong_interrupt = (
                self._utterance_started_during_agent
                and self._state == FloorState.AGENT_SPEAKING
                and near_end
                and voice_rms >= self._strong_interrupt_rms
                and len(normalized_final) >= self._strong_interrupt_min_chars
            )
            # 普通话语在播放期间开始时不抢话权。但如果播放已经自然结束，
            # 且这段话有持续近端人声和足够语义，就把它作为“尾部重叠接话”
            # 正常提交，避免用户看见识别结果却必须重说。
            deferred_user = (
                self._utterance_started_during_agent
                and self._state != FloorState.AGENT_SPEAKING
                and near_end
                and len(normalized_final) >= 4
            )
            if (
                spoken
                and self._backchannel_enabled
                and not explicit_final
                and self._utterance_started_during_agent
                and not deferred_user
                and not strong_interrupt
                and (
                    not self._natural_interrupt_enabled
                    or not near_end
                )
            ):
                if self._state in (FloorState.USER_SPEAKING, FloorState.INTERRUPTED):
                    self._state = FloorState.IDLE
                self._utterance_interrupted = False
                self._utterance_agent_text = ""
                self._last_partial_disposition = UserPartialDisposition.HOLD
                self._partial_candidate = ""
                self._partial_candidate_updates = 0
                return UserFinalDisposition.IGNORED

            if (
                self._state == FloorState.USER_SPEAKING
                and len(normalized_final) == 1
                and normalized_final not in _SINGLE_CHAR_WAKE_WORDS
            ):
                self._state = FloorState.IDLE
                self._utterance_interrupted = False
                self._utterance_agent_text = ""
                self._last_partial_disposition = UserPartialDisposition.HOLD
                self._partial_candidate = ""
                self._partial_candidate_updates = 0
                return UserFinalDisposition.IGNORED

            # 播放期间如果此前没有足够稳定的 partial，也不是明确制止词，
            # 单独一个 FINAL 更像 AEC 残差的 ASR 猜词。忽略它且保持 Agent
            # 的话权；正常（Agent 未说话时）的用户 FINAL 不受此规则影响。
            if (
                self._state == FloorState.AGENT_SPEAKING
                and self._backchannel_enabled
                and not is_explicit_interrupt(cleaned)
                and not strong_interrupt
                and self._partial_candidate_updates < self._partial_confirmations
            ):
                self._utterance_agent_text = ""
                self._last_partial_disposition = UserPartialDisposition.HOLD
                self._partial_candidate = ""
                self._partial_candidate_updates = 0
                return UserFinalDisposition.IGNORED

            # 3) 真插话:partial 阶段没来得及触发,但 final 到达时仍 AGENT_SPEAKING
            if self._state == FloorState.AGENT_SPEAKING:
                self._state = FloorState.INTERRUPTED

            was_interrupted = (
                self._utterance_interrupted
                or self._state == FloorState.INTERRUPTED
            )

            if was_interrupted and cleaned:
                self._last_interrupt_text = cleaned
                if self._on_interrupt is not None:
                    try:
                        self._on_interrupt(cleaned)
                    except Exception:
                        pass

            if cleaned:
                self._pending_backchannel = None

            self._state = FloorState.IDLE
            self._utterance_interrupted = False
            self._utterance_started_during_agent = False
            self._agent_text = ""
            self._utterance_agent_text = ""
            self._last_partial_disposition = UserPartialDisposition.USER
            self._partial_candidate = ""
            self._partial_candidate_updates = 0

            return (
                UserFinalDisposition.INTERRUPT
                if was_interrupted
                else UserFinalDisposition.USER
            )

    def agent_start(self) -> None:
        with self._lock:
            if self._state == FloorState.IDLE:
                self._state = FloorState.AGENT_SPEAKING
            self._agent_text = ""

    def agent_chunk(self, text: str) -> None:
        with self._lock:
            self._agent_text += text

    def agent_end(self) -> None:
        """Agent 逻辑回答结束,但扬声器可能还在播。

        保留 ``_recent_agent_text`` 用于回声 grace period。
        """
        with self._lock:
            if self._state in (FloorState.AGENT_SPEAKING, FloorState.INTERRUPTED):
                self._state = FloorState.IDLE
            # 保留最近文本用于回声 grace period
            if self._agent_text:
                self._recent_agent_text = self._agent_text
                self._agent_ended_at = time.monotonic()
            self._agent_text = ""

    def reset(self) -> None:
        with self._lock:
            self._state = FloorState.IDLE
            self._agent_text = ""
            self._recent_agent_text = ""
            self._agent_ended_at = 0.0
            self._utterance_agent_text = ""
            self._last_partial_disposition = UserPartialDisposition.HOLD
            self._partial_candidate = ""
            self._partial_candidate_updates = 0
            self._pending_backchannel = None
            self._utterance_interrupted = False
            self._utterance_started_during_agent = False
