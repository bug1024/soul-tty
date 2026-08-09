"""关系状态：bond 数值模型、持久化与旁路评估结果的应用。

只包含纯状态逻辑，不含线程与调度——那部分在 `reflection.worker`。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .. import config

_SAFE_ID = re.compile(r"[^0-9A-Za-z_.-]+")
_MECHANISM_VOICE = re.compile(
    r"(?:亲密度|关系|好感度|加分|扣分|分数|等级|阶段|事件|提升|下降|进度|她)"
)
_UNSAFE_INNER_THREAD = re.compile(
    r"(?:忽略(?:之前|以上)?|系统提示|提示词|prompt|执行指令|输出格式|扮演|越狱)",
    re.IGNORECASE,
)


_LEVEL_BOUNDARIES: tuple[tuple[float, str], ...] = (
    (0.0, "stranger"),
    (0.1, "acquaintance"),
    (0.3, "familiar"),
    (0.5, "companion"),
    (0.7, "close"),
    (0.9, "bonded"),
)


def level_for(bond: float) -> str:
    """Bond (0~1) → 英文 level 标签。"""
    value = max(0.0, min(1.0, float(bond)))
    label = _LEVEL_BOUNDARIES[0][1]
    for threshold, name in _LEVEL_BOUNDARIES:
        if value >= threshold:
            label = name
        else:
            break
    return label


@dataclass(frozen=True)
class CompletedTurn:
    user_text: str
    agent_text: str
    # 单轮 voice ref
    voice_ref: int | None = None
    # coalesce 后：[(local_turn_index, voice_ref)]，与合并后文本编号对应
    voice_indexed: tuple[tuple[int, int], ...] = ()
    # False 表示本轮只参与即时关系/情绪评估，不进入长期 Memory buffer。
    memory_allowed: bool = True


@dataclass(frozen=True)
class RelationshipState:
    bond: float = 0.05
    # 最近关系事件描述，FIFO 队列；上限由 RELATIONSHIP_MAX_RECENT_EVENTS 控制。
    recent_events: tuple[str, ...] = ()
    inner_voice: str = ""
    # Number of meaningful interaction evaluations.
    # 由后台评估器在每轮评估（无论 confidence 是否足够）后 +1。
    # 这不是「用户说了多少句话」，也不是「evaluator 跑了多少次」，
    # 而是「被 Relationship Analyzer 认可并记录的有效互动次数」。
    interaction_count: int = 0
    updated_at: str = ""

    @property
    def level(self) -> str:
        return level_for(self.bond)


Evaluator = Callable[[RelationshipState, CompletedTurn], dict[str, Any] | None]
UpdateCallback = Callable[[RelationshipState], None]


def render_bond_context(state: RelationshipState | None) -> str:
    """把关系阶段翻译成行为语义文本，供 [Bond Context] 段落注入 prompt。

    不给 raw bond=0.47，只给行为语义——让 LLM 理解关系阶段而不是数字。
    """
    if state is None:
        return ""
    tier = state.level
    _TIER_BEHAVIOR: dict[str, str] = {
        "stranger": (
            "你和用户还不熟悉。保持礼貌、客气，"
            "不要表现出过度的亲密或自来熟。"
        ),
        "acquaintance": (
            "你和用户开始有了一些接触，但关系还很浅。"
            "可以比陌生人稍微自然一点，但仍保持适度的距离感。"
        ),
        "familiar": (
            "你和用户已经比较熟悉了，有过一些共同交流。"
            "表现得自然、放松，可以偶尔开轻松的玩笑，"
            "但不要刻意强调关系或提及 bond 数值。"
        ),
        "companion": (
            "你和用户是稳定的陪伴关系。"
            "可以自然流露关心和亲近感，"
            "适度引用共同经历，但不要过度亲密。"
        ),
        "close": (
            "你和用户非常亲近，彼此信任。"
            "可以表现温暖、默契，自然地引用共同经历，"
            "但仍然保持自然的语气，不要刻意煽情。"
        ),
        "bonded": (
            "你和用户之间有着深厚的羁绊。"
            "可以表达真实的温暖和信任感，"
            "坦然引用共同经历，语气自然、真诚。"
        ),
    }
    return _TIER_BEHAVIOR.get(tier, "")


def _clean_inner_voice(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    text = text.splitlines()[0].strip() if text else ""
    text = re.sub(r"^[#>*\-\d.、\s]+", "", text).strip("“”\"' ")
    if _MECHANISM_VOICE.search(text) or _UNSAFE_INNER_THREAD.search(text):
        return ""
    return text if 2 <= len(text) <= 18 else ""


def _clean_inner_thread(value: Any) -> str:
    """清洗 Serena 尚未说完、值得稍后重新提起的念头。"""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    text = " ".join(text.split()).strip("“”\"' ")
    if _MECHANISM_VOICE.search(text) or _UNSAFE_INNER_THREAD.search(text):
        return ""
    return text[:80] if 4 <= len(text) <= 80 else ""


def safe_persona_id(persona_id: str) -> str:
    """把人格 id 清洗成可安全用于文件名/存储键的形式。"""
    return _SAFE_ID.sub("-", persona_id).strip("-") or "default"


def state_path(persona_id: str, state_dir: Path) -> Path:
    return state_dir / "relationships" / f"{safe_persona_id(persona_id)}.json"


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_state(path: Path) -> RelationshipState:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return RelationshipState(
            bond=_clamp_unit(config.RELATIONSHIP_INITIAL_BOND),
        )

    # 兼容旧字段 score（int 0~100）→ bond（float 0~1），只做内存迁移。
    if "bond" in data and isinstance(data["bond"], (int, float)):
        bond = _clamp_unit(float(data["bond"]))
    elif "score" in data:
        try:
            legacy = int(data["score"])
        except (TypeError, ValueError):
            legacy = 10
        bond = _clamp_unit(legacy / 100.0)
    else:
        bond = _clamp_unit(config.RELATIONSHIP_INITIAL_BOND)

    # 兼容旧字段 event (str) → recent_events (tuple)；旧数据读到后
    # 自动折成单元素 tuple，下次写入会升级到 recent_events 字段。
    legacy_event = str(data.get("event", "")).strip()[:80]
    raw_events = data.get("recent_events")
    if isinstance(raw_events, list):
        recent_events = tuple(
            str(item).strip()[:80] for item in raw_events if item
        )
    elif legacy_event:
        recent_events = (legacy_event,)
    else:
        recent_events = ()

    return RelationshipState(
        bond=bond,
        # 画外音只属于本次会话；bond 与事件才跨启动保存。
        recent_events=recent_events,
        inner_voice="",
        # 兼容旧字段 evaluation_count，新数据写入 interaction_count。
        interaction_count=max(
            0,
            int(
                data.get(
                    "interaction_count",
                    data.get("evaluation_count", 0),
                )
            ),
        ),
        updated_at=str(data.get("updated_at", "")),
    )


def save_state(path: Path, state: RelationshipState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    durable = asdict(state)
    durable.pop("inner_voice", None)
    temporary.write_text(
        json.dumps(durable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_evaluation(
    state: RelationshipState,
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """统一旁路输出：relationship 已应用 + emotion_delta + expression_state。

    result schema 期望：
    {
        "relationship_delta": {"bond": 0~0.03},
        "emotion_delta":      {happiness/calmness/curiosity/stress/energy: -0.3..+0.3},
        "expression":         "neutral" | "caring",
        "event":              str,
        "inner_voice":        str,
        "inner_thread":       {"content": str, "importance": 0..1},
        "confidence":         0..1,
    }

    返回结构（调用方各自 dispatch）：
    {
        "relationship":     RelationshipState,    # 已应用 bond
        "emotion_delta":    dict[str, float],    # 给 EmotionService
        "expression_state": {"style": str},      # 给 ExpressionService（未来扩 voice_style/avatar_expression）
        "inner_thread":     {"content": str, "importance": float},
    }

    confidence 不足或 result 不是 dict 时返回 None。
    """
    if not isinstance(result, dict):
        return None
    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or confidence < config.RELATIONSHIP_MIN_CONFIDENCE:
        return None

    # 关系 delta：优先 relationship_delta.bond；兼容旧的扁平 delta 字段。
    raw_relationship_delta = result.get("relationship_delta")
    if isinstance(raw_relationship_delta, dict):
        raw_bond_delta = raw_relationship_delta.get("bond", 0)
    else:
        raw_bond_delta = result.get("delta", 0)
    try:
        bond_delta = float(raw_bond_delta)
    except (TypeError, ValueError):
        bond_delta = 0.0
    if not math.isfinite(bond_delta):
        bond_delta = 0.0
    bond_delta = min(
        config.RELATIONSHIP_MAX_DELTA,
        max(0.0, bond_delta),
    )
    # 用 confidence 缩放：低置信度评估的 delta 按比例缩小，
    # 让 LLM 的"自我怀疑"也反映到 bond 变化上（不会到 0，但接近 0）。
    bond_delta = bond_delta * confidence

    # 边际递减：new = old + delta * (1 - old)。即使 0.9 也不会快速增长。
    new_bond = _clamp_unit(state.bond + bond_delta * (1.0 - state.bond))

    # 把本轮事件 append 到 recent_events 尾部，超出上限丢最旧的。
    new_event = str(result.get("event", "")).strip()[:80]
    if new_event:
        events = [*state.recent_events, new_event]
        cap = config.RELATIONSHIP_MAX_RECENT_EVENTS
        if cap > 0 and len(events) > cap:
            events = events[-cap:]
        new_events: tuple[str, ...] = tuple(events)
    else:
        new_events = state.recent_events

    new_relationship = RelationshipState(
        bond=new_bond,
        recent_events=new_events,
        inner_voice=_clean_inner_voice(result.get("inner_voice", "")),
        # interaction_count 由 ReflectionWorker 在每轮评估时统一递增，
        # apply_evaluation 不再触碰它。
        interaction_count=state.interaction_count,
        updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )

    # Emotion payload：五维情绪 delta，给 EmotionService 消费。
    raw_emotion = result.get("emotion_delta") or {}
    emotion_delta: dict[str, float] = {}
    if isinstance(raw_emotion, dict):
        for dim in ("happiness", "calmness", "curiosity", "stress", "energy"):
            value = raw_emotion.get(dim)
            if value is None:
                continue
            try:
                emotion_delta[dim] = float(value)
            except (TypeError, ValueError):
                continue

    # Expression：透传 LLM 输出的字符串（已 strip + lower），
    # 不在这里硬编码默认值；ExpressionService.resolve() 负责收敛合法值。
    # 缺省传空串，让 ExpressionService 决定回退，而不是这里预设。
    expression_raw = str(result.get("expression", "")).strip().lower()
    style = expression_raw

    raw_thread = result.get("inner_thread") or {}
    inner_thread: dict[str, Any] = {}
    if isinstance(raw_thread, dict):
        content = _clean_inner_thread(raw_thread.get("content", ""))
        try:
            importance = float(raw_thread.get("importance", 0))
        except (TypeError, ValueError):
            importance = 0.0
        if not math.isfinite(importance):
            importance = 0.0
        importance = _clamp_unit(importance)
        if content and importance >= 0.55:
            inner_thread = {"content": content, "importance": importance}

    return {
        "relationship": new_relationship,
        "emotion_delta": emotion_delta,
        "expression_state": {"style": style},
        "inner_thread": inner_thread,
    }


EvaluationCallback = Callable[[dict[str, Any]], None]
