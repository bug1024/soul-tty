"""五维情绪值数据结构 + 本地 JSON 持久化。"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

_SAFE_ID = re.compile(r"[^0-9A-Za-z_.-]+")

DIMENSIONS = ("happiness", "calmness", "curiosity", "stress", "energy")

DEFAULT_BASELINE_VALUES = {
    "happiness": 0.65,
    "calmness": 0.75,
    "curiosity": 0.70,
    "stress": 0.20,
    "energy": 0.75,
}


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class EmotionVector:
    happiness: float
    calmness: float
    curiosity: float
    stress: float
    energy: float

    def __post_init__(self) -> None:
        for dim in DIMENSIONS:
            value = getattr(self, dim)
            if not 0.0 <= value <= 1.0:
                object.__setattr__(self, dim, _clamp_unit(value))

    def to_dict(self) -> dict[str, float]:
        return {dim: getattr(self, dim) for dim in DIMENSIONS}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "EmotionVector":
        kwargs = {
            dim: float(data.get(dim, DEFAULT_BASELINE_VALUES[dim]))
            for dim in DIMENSIONS
        }
        return cls(**kwargs)


DEFAULT_BASELINE = EmotionVector(**DEFAULT_BASELINE_VALUES)


def _state_dir(base_dir: Path) -> Path:
    return base_dir / "emotion"


def _persona_path(persona_id: str, base_dir: Path) -> Path:
    safe_id = _SAFE_ID.sub("-", persona_id).strip("-") or "default"
    return _state_dir(base_dir) / f"{safe_id}.json"


def _runtime_path(base_dir: Path) -> Path:
    return base_dir / "runtime.json"


def load_emotion_state(persona_id: str, base_dir: Path) -> dict | None:
    """读取 emotion/{persona_id}.json；不存在或解析失败返回 None。"""
    path = _persona_path(persona_id, base_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_emotion_state(
    persona_id: str,
    base_dir: Path,
    session_id: str,
    baseline: EmotionVector,
    emotion: EmotionVector,
    updated_at: str,
) -> None:
    """写入 emotion/{persona_id}.json（原子替换）。"""
    path = _persona_path(persona_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "baseline": baseline.to_dict(),
        "emotion": emotion.to_dict(),
        "updated_at": updated_at,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_runtime(base_dir: Path) -> int:
    """读取 runtime.json 的 total_sessions；不存在或解析失败返回 0。"""
    path = _runtime_path(base_dir)
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return max(0, int(data.get("total_sessions", 0)))
    except (OSError, ValueError, TypeError):
        return 0


def save_runtime(base_dir: Path, total_sessions: int) -> None:
    """写入 runtime.json（原子替换）。"""
    path = _runtime_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"total_sessions": total_sessions}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def new_session_id() -> str:
    return str(uuid.uuid4())