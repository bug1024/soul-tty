"""在 Conversation Brain 之前决定「这一轮要不要、以及怎样参与」。"""

from __future__ import annotations

import enum
import random
import re
from dataclasses import dataclass

from .state import AgencyState


class ResponseMode(str, enum.Enum):
    ANSWER = "answer"
    ANSWER_AND_LEAD = "answer_and_lead"
    SELF_EXPRESS = "self_express"
    SHORT_REPLY = "short_reply"
    SILENCE = "silence"
    CHANGE_TOPIC = "change_topic"
    ASK = "ask"


@dataclass(frozen=True)
class ResponseDecision:
    mode: ResponseMode
    reason: str
    instruction: str = ""
    protected: bool = False


_TASK_REQUEST = re.compile(
    r"(?:请问|告诉我)|"
    r"(?:请|帮我|麻烦|替我|给我).{0,12}"
    r"(?:写|查|找|搜|分析|解释|总结|翻译|修复|检查|安装|配置|运行|执行|"
    r"生成|实现|做|讲|说|列|整理|打开|关闭|停止|回答)|"
    r"(?:分析|解释|总结|翻译|修复|检查|查询|搜索|安装|配置|运行|执行|生成|"
    r"实现|写一?|做一?|讲(?:个|一个)?|查(?:一下)?|停(?:止|下|一?下|吧)?|"
    r"别说|回答)(?:一下|这个|这段|代码|文件|问题|内容|吧|$)"
)
_NEEDS_CARE = re.compile(
    r"(?:难过|伤心|害怕|焦虑|痛苦|不舒服|生病|救命|危险|撑不住|不想活|出事)"
)
_STOP_REQUEST = re.compile(
    r"(?:停下|停止|别说了?|不用说了?|闭嘴|先别说|不要继续|到此为止)"
)
_SAFE_SILENCE = re.compile(
    r"^(?:serena|塞雷娜|嗯+|哦+|唔+|好吧|没事|算了|随便|就这样|……+|\.\.\.+)[吗嘛呢呀啊~～。！？?!]*$",
    re.IGNORECASE,
)
_SOCIAL_QUESTION = re.compile(
    r"^(?:serena|塞雷娜|你)",
    re.IGNORECASE,
)
_KNOWLEDGE_OR_OPERATION = re.compile(
    r"(?:如何|怎么|怎样|为什么|为何|是什么|多少|哪(?:个|里|些)|能不能|"
    r"可以不可以|是否可以)"
)


def _is_task_request(text: str) -> bool:
    if _TASK_REQUEST.search(text):
        return True
    # “Python 怎么读取文件”属于知识/操作请求；“你今天怎么样”属于社交问题。
    return bool(
        _KNOWLEDGE_OR_OPERATION.search(text)
        and not _SOCIAL_QUESTION.search(text)
    )


def _instruction(mode: ResponseMode, thought: str = "") -> str:
    if mode is ResponseMode.ANSWER_AND_LEAD:
        return (
            "[Response Policy]\n先简短回应用户真正关心的部分，然后主动拿走下一步"
            "话语权：表达一个你自己的想法、偏好或此刻真正想聊的方向。可以问一个"
            "问题，但不要套用‘你呢’，也不要为了主动而强行反问。"
        )
    if mode is ResponseMode.SELF_EXPRESS:
        return (
            "[Response Policy]\n本轮不要把完整回答用户最后一句当成中心任务。"
            "从那句话引发的真实感受、想法或偏好出发，自然表达你自己；可以不提问，"
            "不要解释你为什么改变回应方式。"
        )
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


def _low_energy_initiative_instruction() -> str:
    return (
        "[Response Policy]\n你现在话不多，但不要继续只承接用户最后一句。"
        "用一到两句自然口语，说出你此刻的一点感受、偏好或想做的事，给对话一个"
        "由你带来的新落点。不要为了推进而提问，不要解释规则，也不要刻意变长。"
    )


class ResponsePolicy:
    """本地参与策略。

    明确任务、制止词和需要关怀的表达永远正常回答。普通社交问题不再天然拥有
    ANSWER 特权；低风险闲聊由 Need、主动性债务和对话节奏共同决定。
    """

    def __init__(
        self,
        *,
        silence_rate: float = 0.10,
        change_topic_rate: float = 0.08,
        ask_rate: float = 0.12,
        answer_and_lead_rate: float = 0.30,
        self_express_rate: float = 0.15,
        min_turns_before_silence: int = 6,
        initiative_debt_threshold: float = 0.50,
        max_passive_answers: int = 2,
        rng: random.Random | None = None,
    ) -> None:
        self.silence_rate = max(0.0, min(1.0, silence_rate))
        self.change_topic_rate = max(0.0, min(1.0, change_topic_rate))
        self.ask_rate = max(0.0, min(1.0, ask_rate))
        self.answer_and_lead_rate = max(0.0, min(1.0, answer_and_lead_rate))
        self.self_express_rate = max(0.0, min(1.0, self_express_rate))
        self.min_turns_before_silence = max(0, min_turns_before_silence)
        self.initiative_debt_threshold = max(
            0.0, min(1.0, initiative_debt_threshold)
        )
        self.max_passive_answers = max(1, max_passive_answers)
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
        protected = bool(
            not text
            or _is_task_request(text)
            or _STOP_REQUEST.search(text)
            or _NEEDS_CARE.search(text)
        )
        if protected:
            return ResponseDecision(
                ResponseMode.ANSWER,
                "protected_task_or_sensitive",
                protected=True,
            )

        low_talk = state.desire_to_talk < 0.30
        strong_solitude = state.solitude_need > 0.72
        turns_this_session = (
            state.turn_count if session_turn_count is None else session_turn_count
        )
        can_silence = (
            turns_this_session >= self.min_turns_before_silence
            and state.consecutive_silences == 0
            and relationship_level in {"familiar", "companion", "close", "bonded"}
            and low_talk
            and strong_solitude
            and bool(_SAFE_SILENCE.fullmatch(text))
        )
        if can_silence and self.rng.random() < self.silence_rate:
            return ResponseDecision(ResponseMode.SILENCE, "low_talk_safe_silence")

        low_energy = low_talk or state.social_energy < 0.28
        passive_limit_reached = (
            state.passive_answer_streak >= self.max_passive_answers
            or state.initiative_debt >= self.initiative_debt_threshold
        )
        if low_energy and passive_limit_reached:
            # 能量低表示话少，不表示永远只能接球。用克制的自我表达打破
            # “用户追问 -> 一句回应 -> 用户继续追问”的稳定死区。
            mode = ResponseMode.SELF_EXPRESS
            return ResponseDecision(
                mode,
                "low_energy_initiative",
                _low_energy_initiative_instruction(),
            )

        if low_energy:
            mode = ResponseMode.SHORT_REPLY
            return ResponseDecision(mode, "low_social_energy", _instruction(mode))

        can_take_initiative = (
            state.desire_to_talk > 0.45 and state.social_energy > 0.35
        )
        if (
            can_take_initiative
            and state.passive_answer_streak >= self.max_passive_answers
        ):
            mode = ResponseMode.ANSWER_AND_LEAD
            return ResponseDecision(
                mode,
                "passive_answer_limit",
                _instruction(mode),
            )
        if (
            can_take_initiative
            and state.initiative_debt >= self.initiative_debt_threshold
        ):
            mode = (
                ResponseMode.ANSWER_AND_LEAD
                if self.rng.random() < 0.65
                else ResponseMode.SELF_EXPRESS
            )
            return ResponseDecision(mode, "initiative_debt", _instruction(mode))

        # 普通闲聊使用一次加权抽样；没有 unresolved thought 时，转题份额
        # 自然回落到最后的 ANSWER。
        roll = self.rng.random()
        if roll < self.answer_and_lead_rate:
            mode = ResponseMode.ANSWER_AND_LEAD
            return ResponseDecision(mode, "ordinary_conversation", _instruction(mode))
        roll -= self.answer_and_lead_rate
        if roll < self.self_express_rate:
            mode = ResponseMode.SELF_EXPRESS
            return ResponseDecision(mode, "ordinary_conversation", _instruction(mode))
        roll -= self.self_express_rate
        if roll < self.ask_rate:
            mode = ResponseMode.ASK
            return ResponseDecision(mode, "ordinary_conversation", _instruction(mode))
        roll -= self.ask_rate
        if (
            state.unresolved_thoughts
            and turns_this_session >= 3
            and relationship_level in {"familiar", "companion", "close", "bonded"}
            and roll < self.change_topic_rate
        ):
            mode = ResponseMode.CHANGE_TOPIC
            return ResponseDecision(
                mode,
                "unresolved_thought",
                _instruction(mode, state.unresolved_thoughts[0]),
            )

        return ResponseDecision(ResponseMode.ANSWER, "default")
