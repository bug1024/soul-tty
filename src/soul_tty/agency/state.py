"""Serena 的持续内在倾向。

这些值不是向用户展示的游戏数值，也不直接等同于 Emotion。Emotion 描述
「此刻感受」，AgencyState 描述「此刻更倾向于怎样参与交流」。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


def _unit(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, min(1.0, parsed))


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class AgencyState:
    """跨会话保存的内在倾向；social 与 solitude 刻意不互为反值。"""

    mood: str = "calm"
    social_energy: float = 0.68
    desire_to_talk: float = 0.62
    desire_for_company: float = 0.56
    solitude_need: float = 0.22
    attention: str = "present"
    unresolved_thoughts: tuple[str, ...] = ()
    turn_count: int = 0
    consecutive_silences: int = 0
    last_mode: str = "answer"
    last_interaction_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["unresolved_thoughts"] = list(self.unresolved_thoughts)
        data["schema_version"] = 1
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgencyState":
        raw_threads = data.get("unresolved_thoughts", ())
        threads = (
            tuple(
                str(item).strip()[:120]
                for item in raw_threads
                if str(item).strip()
            )[:8]
            if isinstance(raw_threads, (list, tuple))
            else ()
        )
        attention = str(data.get("attention", "present")).strip().lower()
        if attention not in {"present", "focused", "wandering", "inward"}:
            attention = "present"
        return cls(
            mood=str(data.get("mood", "calm")).strip().lower() or "calm",
            social_energy=_unit(data.get("social_energy"), 0.68),
            desire_to_talk=_unit(data.get("desire_to_talk"), 0.62),
            desire_for_company=_unit(data.get("desire_for_company"), 0.56),
            solitude_need=_unit(data.get("solitude_need"), 0.22),
            attention=attention,
            unresolved_thoughts=threads,
            turn_count=_nonnegative_int(data.get("turn_count", 0)),
            consecutive_silences=_nonnegative_int(
                data.get("consecutive_silences", 0)
            ),
            last_mode=str(data.get("last_mode", "answer")) or "answer",
            last_interaction_at=str(data.get("last_interaction_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )


def load_state(path: Path) -> AgencyState:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return AgencyState()
    return AgencyState.from_dict(payload) if isinstance(payload, dict) else AgencyState()


def save_state(path: Path, state: AgencyState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def continuity_update(
    state: AgencyState,
    *,
    now: datetime | None = None,
) -> AgencyState:
    """按离线时长做一次克制的状态延续，不虚构离线经历。"""

    current = now or datetime.now().astimezone()
    previous_text = state.last_interaction_at or state.updated_at
    try:
        previous = datetime.fromisoformat(previous_text)
        if previous.tzinfo is None:
            previous = previous.replace(tzinfo=current.tzinfo)
        offline_hours = max(0.0, (current - previous).total_seconds() / 3600)
    except (TypeError, ValueError):
        offline_hours = 0.0

    # 休息会恢复社交能量；较久未见会提高陪伴倾向，但不会无限增长。
    recovery = min(1.0, offline_hours / 8.0)
    social_energy = state.social_energy + (0.72 - state.social_energy) * recovery
    desire_to_talk = (
        state.desire_to_talk + (0.60 - state.desire_to_talk) * recovery
    )
    desire_for_company = min(
        0.84,
        state.desire_for_company + min(0.18, offline_hours * 0.01),
    )
    solitude_need = state.solitude_need + (0.24 - state.solitude_need) * recovery
    return replace(
        state,
        social_energy=_unit(social_energy, 0.68),
        desire_to_talk=_unit(desire_to_talk, 0.62),
        desire_for_company=_unit(desire_for_company, 0.56),
        solitude_need=_unit(solitude_need, 0.22),
        attention="present" if offline_hours >= 0.25 else state.attention,
        updated_at=current.isoformat(timespec="seconds"),
    )


def evolve_after_decision(
    state: AgencyState,
    *,
    mode: str,
    mood: str,
    user_text: str,
) -> AgencyState:
    """一次互动后更新 Need；只做小幅连续变化，避免随机人格跳变。"""

    mode = mode.lower()
    mood = (mood or state.mood or "calm").lower()
    cost = {
        "answer": 0.020,
        "short_reply": 0.010,
        "ask": 0.016,
        "change_topic": 0.018,
        "silence": -0.006,
    }.get(mode, 0.015)
    energy = _unit(state.social_energy - cost, state.social_energy)

    mood_talk = {
        "excited": 0.10,
        "curious": 0.08,
        "happy": 0.04,
        "calm": 0.0,
        "sad": -0.10,
        "tired": -0.16,
        "numb": -0.22,
    }.get(mood, 0.0)
    talk_target = _unit(0.18 + energy * 0.62 + mood_talk, 0.56)
    desire_to_talk = state.desire_to_talk * 0.82 + talk_target * 0.18

    # 用户已经在场时，陪伴需求得到一点满足；它和独处需求仍可同时偏高。
    company = _unit(state.desire_for_company - 0.018, state.desire_for_company)
    solitude_target = _unit(0.12 + (1.0 - energy) * 0.62, 0.22)
    solitude = state.solitude_need * 0.86 + solitude_target * 0.14

    compact = user_text.strip()
    if any(word in compact for word in ("陪你", "想你", "来看你", "晚安")):
        company = _unit(company - 0.025, company)
        desire_to_talk = _unit(desire_to_talk + 0.025, desire_to_talk)

    attention = "focused" if any(mark in compact for mark in "?？") else "present"
    if mood in {"tired", "numb"} and energy < 0.30:
        attention = "inward"

    now = _now_iso()
    return replace(
        state,
        mood=mood,
        social_energy=_unit(energy, state.social_energy),
        desire_to_talk=_unit(desire_to_talk, state.desire_to_talk),
        desire_for_company=_unit(company, state.desire_for_company),
        solitude_need=_unit(solitude, state.solitude_need),
        attention=attention,
        turn_count=state.turn_count + 1,
        consecutive_silences=(
            state.consecutive_silences + 1 if mode == "silence" else 0
        ),
        last_mode=mode,
        last_interaction_at=now,
        updated_at=now,
    )
