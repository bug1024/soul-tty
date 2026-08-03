"""角色在场感：只记录启动节奏，不保存任何对话内容。"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import config

_SAFE_ID = re.compile(r"[^0-9A-Za-z_.-]+")


@dataclass(frozen=True)
class LaunchContext:
    repeat_launch: bool = False
    special_greeting: bool = False
    interval_s: float | None = None
    launch_count: int = 1


def _state_path(persona_id: str, state_dir: Path) -> Path:
    safe_id = _SAFE_ID.sub("-", persona_id).strip("-") or "default"
    return state_dir / "presence" / f"{safe_id}.json"


def record_launch(
    persona_id: str,
    *,
    state_dir: Path | None = None,
    now: datetime | None = None,
    random_value: float | None = None,
) -> LaunchContext:
    """原子记录本次启动，并返回可供欢迎语使用的轻量上下文。"""
    path = _state_path(persona_id, state_dir or config.SOUL_TTY_STATE_DIR)
    now = now or datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()

    previous: datetime | None = None
    launch_count = 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        launch_count = max(0, int(data.get("launch_count", 0)))
        value = data.get("last_started_at")
        if isinstance(value, str) and value:
            previous = datetime.fromisoformat(value)
            if previous.tzinfo is None:
                previous = previous.astimezone()
    except (OSError, ValueError, TypeError):
        pass

    interval_s = (
        max(0.0, (now - previous).total_seconds()) if previous is not None else None
    )
    repeat_launch = (
        interval_s is not None
        and interval_s <= config.PRESENCE_REPEAT_LAUNCH_WINDOW_S
    )
    probability = min(1.0, max(0.0, config.PRESENCE_SPECIAL_GREETING_RATE))
    draw = random.random() if random_value is None else random_value
    context = LaunchContext(
        repeat_launch=repeat_launch,
        special_greeting=draw < probability,
        interval_s=interval_s,
        launch_count=launch_count + 1,
    )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "last_started_at": now.isoformat(timespec="seconds"),
                    "launch_count": context.launch_count,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        # 在场感状态不可写时静默降级，绝不阻塞主流程。
        pass
    return context
