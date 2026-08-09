"""在 Conversation Brain 之前决定「这一轮要不要、以及怎样回应」。"""

from __future__ import annotations

import enum
import random
import re
from dataclasses import dataclass

from .state import AgencyState


class ResponseMode(str, enum.Enum):
    ANSWER = "answer"
    SHORT_REPLY = "short_reply"
    SILENCE = "silence"
    CHANGE_TOPIC = "change_topic"
    ASK = "ask"


@dataclass(frozen=True)
class ResponseDecision:
    mode: ResponseMode
    reason: str
    instruction: str = ""


_EXPLICIT_REQUEST = re.compile(
    r"(?:请|帮我|告诉我|解释|分析|怎么|怎样|为什么|是什么|能不能|可以不可以|"
    r"讲(?:个|一个)?|说(?:说|一下)|查(?:一下)?|修复|停(?:止|下|一?下|吧)?|别说|回答)"
)
_NEEDS_CARE = re.compile(
    r"(?:难过|伤心|害怕|焦虑|痛苦|不舒服|生病|救命|危险|撑不住|不想活|出事)"
)
_SAFE_SILENCE = re.compile(
    r"^(?:serena|塞雷娜|嗯+|哦+|唔+|好吧|没事|算了|随便|就这样|……+|\.\.\.+)[吗嘛呢呀啊~～。！？?!]*$",
    re.IGNORECASE,
)
_NAME_PING = re.compile(
    r"^(?:serena|塞雷娜)[吗嘛呢呀啊~～。！？?!]*$",
    re.IGNORECASE,
)


def _instruction(mode: ResponseMode, thought: str = "") -> str:
    if mode is ResponseMode.SHORT_REPLY:
        return (
            "[Response Policy]\n本轮只用一句自然的短句回应，不解释为什么简短，"
            "不超过十八个汉字。"
        )
    if mode is ResponseMode.ASK:
        return (
            "[Response Policy]\n本轮不要展开长回答；"
            "先自然承接，再问一个真正关心的短问题。"
        )
    if mode is ResponseMode.CHANGE_TOPIC:
        topic = f" 可以自然转向这个仍未结束的念头：{thought}" if thought else ""
        return (
            "[Response Policy]\n本轮先用很短的一句承接用户，再自然转向另一个话题；"
            "不要解释你在转移话题。" + topic
        )
    return ""


class ResponsePolicy:
    """保守的本地决策器。

    明确问题、请求、制止词和需要关怀的表达永远不会被沉默或转题。随机性只在
    低风险闲聊里生效，并受最小轮数、连续沉默上限和 Need 阈值约束。
    """

    def __init__(
        self,
        *,
        silence_rate: float = 0.10,
        change_topic_rate: float = 0.08,
        ask_rate: float = 0.12,
        min_turns_before_silence: int = 6,
        rng: random.Random | None = None,
    ) -> None:
        self.silence_rate = max(0.0, min(1.0, silence_rate))
        self.change_topic_rate = max(0.0, min(1.0, change_topic_rate))
        self.ask_rate = max(0.0, min(1.0, ask_rate))
        self.min_turns_before_silence = max(0, min_turns_before_silence)
        self.rng = rng or random.Random()

    def decide(
        self,
        state: AgencyState,
        user_text: str,
        *,
        relationship_level: str = "",
        session_turn_count: int | None = None,
    ) -> ResponseDecision:
        text = user_text.strip()
        name_ping = bool(_NAME_PING.fullmatch(text))
        protected = bool(
            not text
            or (("?" in text or "？" in text) and not name_ping)
            or _EXPLICIT_REQUEST.search(text)
            or _NEEDS_CARE.search(text)
        )
        if protected:
            return ResponseDecision(ResponseMode.ANSWER, "explicit_or_sensitive")

        low_talk = state.desire_to_talk < 0.30
        strong_solitude = state.solitude_need > 0.72
        turns_this_session = (
            state.turn_count if session_turn_count is None else session_turn_count
        )
        can_silence = (
            turns_this_session >= self.min_turns_before_silence
            and state.consecutive_silences == 0
            and relationship_level
            in {"familiar", "companion", "close", "bonded"}
            and low_talk
            and strong_solitude
            and bool(_SAFE_SILENCE.fullmatch(text))
        )
        if can_silence and self.rng.random() < self.silence_rate:
            return ResponseDecision(ResponseMode.SILENCE, "low_talk_safe_silence")

        if low_talk or state.social_energy < 0.28:
            mode = ResponseMode.SHORT_REPLY
            return ResponseDecision(mode, "low_social_energy", _instruction(mode))

        if (
            state.unresolved_thoughts
            and turns_this_session >= 3
            and relationship_level
            in {"familiar", "companion", "close", "bonded"}
            and self.rng.random() < self.change_topic_rate
        ):
            mode = ResponseMode.CHANGE_TOPIC
            return ResponseDecision(
                mode,
                "unresolved_thought",
                _instruction(mode, state.unresolved_thoughts[0]),
            )

        # 很短的陈述偶尔换成追问，让 Conversation Brain 不总是抢着输出结论。
        if len(text) <= 12 and self.rng.random() < self.ask_rate:
            mode = ResponseMode.ASK
            return ResponseDecision(mode, "brief_statement", _instruction(mode))

        return ResponseDecision(ResponseMode.ANSWER, "default")
