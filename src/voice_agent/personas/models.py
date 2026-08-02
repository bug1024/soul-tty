"""人格配置的数据模型与校验。"""

from dataclasses import dataclass, replace
from typing import Any


def _text(data: dict[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"persona.{key} 必须是字符串")
    return value.strip()


@dataclass(frozen=True)
class Personality:
    system_prompt: str
    greeting: str
    farewell: str
    speaking_style: str = ""


@dataclass(frozen=True)
class Voice:
    backend: str = "mlx"
    voice: str = "Serena"
    instruct: str = ""


@dataclass(frozen=True)
class Avatar:
    idle: str
    listening: str = ""
    thinking: str = ""
    speaking: str = ""
    speaking_closed: str = ""
    speaking_half: str = ""
    speaking_open: str = ""
    renderer: str = "auto"
    width: int = 26

    def for_state(self, state: str) -> str:
        value = getattr(self, state, "")
        if value:
            return value
        if state.startswith("speaking_"):
            return self.speaking or self.idle
        return self.idle


@dataclass(frozen=True)
class Appearance:
    symbol: str = "moon"
    primary_color: str = "#c084fc"
    secondary_color: str = "#67e8f9"
    accent_color: str = "#fb7185"
    avatar: Avatar | None = None


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    display_name: str
    tagline: str
    personality: Personality
    voice: Voice
    appearance: Appearance

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        if not isinstance(data, dict):
            raise ValueError("persona 文件根节点必须是对象")
        personality_data = data.get("personality", {})
        voice_data = data.get("voice", {})
        appearance_data = data.get("appearance", {})
        if not all(
            isinstance(item, dict)
            for item in (personality_data, voice_data, appearance_data)
        ):
            raise ValueError("personality、voice、appearance 必须是对象")

        persona_id = _text(data, "id")
        name = _text(data, "name")
        display_name = _text(data, "display_name", name)
        system_prompt = _text(personality_data, "system_prompt")
        if not persona_id or not name or not display_name or not system_prompt:
            raise ValueError("persona 必须包含 id、name、display_name 和 system_prompt")

        backend = _text(voice_data, "backend", "mlx").lower()
        if backend not in {"mlx", "macos"}:
            raise ValueError("persona.voice.backend 只支持 mlx 或 macos")

        avatar_data = appearance_data.get("avatar")
        avatar = None
        if avatar_data is not None:
            if not isinstance(avatar_data, dict):
                raise ValueError("persona.appearance.avatar 必须是对象")
            idle = _text(avatar_data, "idle")
            if not idle:
                raise ValueError("persona.appearance.avatar.idle 不能为空")
            renderer = _text(avatar_data, "renderer", "auto").lower()
            if renderer not in {"auto", "pixels", "symbols", "off"}:
                raise ValueError(
                    "persona.appearance.avatar.renderer 只支持 auto/pixels/symbols/off"
                )
            width = avatar_data.get("width", 26)
            if not isinstance(width, int) or not 12 <= width <= 48:
                raise ValueError("persona.appearance.avatar.width 必须是 12-48 的整数")
            avatar = Avatar(
                idle=idle,
                listening=_text(avatar_data, "listening"),
                thinking=_text(avatar_data, "thinking"),
                speaking=_text(avatar_data, "speaking"),
                speaking_closed=_text(avatar_data, "speaking_closed"),
                speaking_half=_text(avatar_data, "speaking_half"),
                speaking_open=_text(avatar_data, "speaking_open"),
                renderer=renderer,
                width=width,
            )

        return cls(
            id=persona_id,
            name=name,
            display_name=display_name,
            tagline=_text(data, "tagline"),
            personality=Personality(
                system_prompt=system_prompt,
                greeting=_text(personality_data, "greeting"),
                farewell=_text(personality_data, "farewell", "再见。"),
                speaking_style=_text(personality_data, "speaking_style"),
            ),
            voice=Voice(
                backend=backend,
                voice=_text(voice_data, "voice", "Serena"),
                instruct=_text(voice_data, "instruct"),
            ),
            appearance=Appearance(
                symbol=_text(appearance_data, "symbol", "moon"),
                primary_color=_text(
                    appearance_data, "primary_color", "#c084fc"
                ),
                secondary_color=_text(
                    appearance_data, "secondary_color", "#67e8f9"
                ),
                accent_color=_text(appearance_data, "accent_color", "#fb7185"),
                avatar=avatar,
            ),
        )

    def renamed(self, display_name: str) -> "Persona":
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("Agent 名字不能为空")
        return replace(self, display_name=display_name)
