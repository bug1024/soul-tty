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
from .echo import is_probable_echo as _is_probable_echo


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
    - USER: 用户真插话,但之前没有 partial 打断(新 turn)
    - INTERRUPT: 用户真插话,且之前已通过 partial 触发打断
    """

    ECHO = "echo"
    BACKCHANNEL = "backchannel"
    USER = "user"
    INTERRUPT = "interrupt"


class FloorManager:
    """Floor 状态机:在 USER / AGENT 之间切换,决定是否打断。"""

    def __init__(
        self,
        echo_similarity: float | None = None,
        on_interrupt: Callable[[str], None] | None = None,
        backchannel_enabled: bool | None = None,
    ) -> None:
        self._echo_similarity = echo_similarity or config.DUPLEX_ECHO_SIMILARITY
        self._on_interrupt = on_interrupt
        self._backchannel_enabled = (
            backchannel_enabled
            if backchannel_enabled is not None
            else config.BACKCHANNEL_ENABLED
        )
        self._lock = threading.Lock()
        self._state = FloorState.IDLE
        self._agent_text = ""
        self._last_interrupt_text: str | None = None
        self._pending_backchannel: str | None = None
        # 回声 grace period:agent_end 后保留最近文本一段时间
        self._recent_agent_text: str = ""
        self._agent_ended_at: float = 0.0

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

    # ── 状态变更 ────────────────────────────────────────────────────

    def user_start(self) -> None:
        """DuplexListener SPEECH_START 时调。

        修复(commit 07+):SPEECH_START 不等于打断。用户可能只是轻微出声
        (清嗓/backchannel),打断决策应由 ``user_partial`` 做。
        所以这里不再把 AGENT_SPEAKING → INTERRUPTED。
        """
        with self._lock:
            if self._state == FloorState.IDLE:
                self._state = FloorState.USER_SPEAKING
            # AGENT_SPEAKING 时保持原状态(不漏掉下一步的 user_partial 决策)

    def user_partial(self, text: str) -> bool:
        """DuplexListener PARTIAL 时调。返回是否应当打断 agent。

        副作用:真正决定打断时,state 从 AGENT_SPEAKING → INTERRUPTED。
        backchannel:如果当前 AGENT_SPEAKING 且文本是 backchannel,
        不打断但记下来(供 agent 下一轮参考)。
        回声过滤:heard 与 agent 已播放文本相似度 ≥ 阈值 → 不打断。
        """
        with self._lock:
            if self._state != FloorState.AGENT_SPEAKING:
                return False
            if not text:
                return False
            spoken = self._effective_agent_text()
            if _is_probable_echo(text, spoken):
                return False
            if self._backchannel_enabled and is_backchannel(text):
                self._pending_backchannel = text.strip()
                return False
            self._state = FloorState.INTERRUPTED
            return True

    def user_final(self, text: str, pcm: bytes | None = None) -> UserFinalDisposition:
        """DuplexListener FINAL 时调。返回分类结果(不再只返回 bool)。

        修复(commit 07+):
        - 回声 final → ECHO,不改变 state/agent_text
        - backchannel → BACKCHANNEL
        - 真打断 → INTERRUPT(如果之前已 INTERRUPTED)或 USER
        - 关键区别:ECHO 不会清空 agent_text/state,避免后续回声被误判
        """
        cleaned = text.strip() if text else ""
        with self._lock:
            agent_active = self._state in (
                FloorState.AGENT_SPEAKING,
                FloorState.INTERRUPTED,
            )

            # 1) Agent 仍在说,且这条 final 是回声
            if agent_active and cleaned and self._effective_agent_text():
                spoken = self._effective_agent_text()
                if _is_probable_echo(cleaned, spoken):
                    # 不改 state,不清 agent_text
                    return UserFinalDisposition.ECHO

            # 2) backchannel
            if (
                self._state == FloorState.AGENT_SPEAKING
                and self._backchannel_enabled
                and is_backchannel(cleaned)
            ):
                self._pending_backchannel = cleaned
                # 真 backchannel 在 final 阶段也不改 state(agent 继续说话)
                return UserFinalDisposition.BACKCHANNEL

            # 3) 真插话:partial 阶段没来得及触发,但 final 到达时仍 AGENT_SPEAKING
            if self._state == FloorState.AGENT_SPEAKING:
                self._state = FloorState.INTERRUPTED

            was_interrupted = self._state == FloorState.INTERRUPTED

            if was_interrupted and cleaned:
                self._last_interrupt_text = cleaned
                if self._on_interrupt is not None:
                    try:
                        self._on_interrupt(cleaned)
                    except Exception:
                        pass

            # 真插话 final 清掉 backchannel
            if cleaned:
                self._pending_backchannel = None

            self._state = FloorState.IDLE
            self._agent_text = ""

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
            self._pending_backchannel = None