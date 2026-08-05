"""人格配置的数据模型与校验。"""

from dataclasses import dataclass, replace
from typing import Any


def _text(data: dict[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"persona.{key} 必须是字符串")
    return value.strip()


def _texts(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key, ())
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"persona.{key} 必须是字符串列表")
    return tuple(item.strip() for item in value if item.strip())


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
class AvatarOutfit:
    id: str
    label: str
    idle: str
    description: str = ""
    switch_greetings: tuple[str, ...] = ()
    listening: str = ""
    thinking: str = ""
    speaking: str = ""
    speaking_closed: str = ""
    speaking_half: str = ""
    speaking_open: str = ""
    mode: str = "companion"  # companion | focused | late_night

    def for_state(self, state: str) -> str:
        value = getattr(self, state, "")
        if value:
            return value
        if state.startswith("speaking_"):
            return self.speaking or self.idle
        return self.idle


@dataclass(frozen=True)
class Avatar:
    outfits: tuple[AvatarOutfit, ...]
    selected_outfit: str = "default"
    renderer: str = "auto"
    width: int = 26

    @property
    def outfit(self) -> AvatarOutfit:
        for outfit in self.outfits:
            if outfit.id == self.selected_outfit:
                return outfit
        raise ValueError(f"头像套装不存在: {self.selected_outfit}")

    def for_state(self, state: str) -> str:
        return self.outfit.for_state(state)

    def wearing(self, outfit_id: str) -> "Avatar":
        outfit_id = outfit_id.strip()
        available = {outfit.id for outfit in self.outfits}
        if outfit_id not in available:
            choices = ", ".join(outfit.id for outfit in self.outfits)
            raise ValueError(f"头像套装 {outfit_id} 不存在（可用: {choices}）")
        return replace(self, selected_outfit=outfit_id)


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
            renderer = _text(avatar_data, "renderer", "auto").lower()
            if renderer not in {"auto", "pixels", "symbols", "off"}:
                raise ValueError(
                    "persona.appearance.avatar.renderer 只支持 auto/pixels/symbols/off"
                )
            width = avatar_data.get("width", 26)
            if not isinstance(width, int) or not 12 <= width <= 48:
                raise ValueError("persona.appearance.avatar.width 必须是 12-48 的整数")
            outfits_data = avatar_data.get("outfits")
            if outfits_data is None:
                outfits_data = {"default": avatar_data}
            if not isinstance(outfits_data, dict) or not outfits_data:
                raise ValueError("persona.appearance.avatar.outfits 必须是非空对象")
            outfits: list[AvatarOutfit] = []
            for outfit_id, outfit_data in outfits_data.items():
                if not isinstance(outfit_id, str) or not outfit_id.strip():
                    raise ValueError("avatar.outfits 的套装 id 必须是非空字符串")
                if not isinstance(outfit_data, dict):
                    raise ValueError(f"avatar.outfits.{outfit_id} 必须是对象")
                idle = _text(outfit_data, "idle")
                if not idle:
                    raise ValueError(f"avatar.outfits.{outfit_id}.idle 不能为空")
                outfits.append(
                    AvatarOutfit(
                        id=outfit_id.strip(),
                        label=_text(outfit_data, "label", outfit_id),
                        idle=idle,
                        description=_text(outfit_data, "description"),
                        switch_greetings=_texts(
                            outfit_data, "switch_greetings"
                        ),
                        listening=_text(outfit_data, "listening"),
                        thinking=_text(outfit_data, "thinking"),
                        speaking=_text(outfit_data, "speaking"),
                        speaking_closed=_text(outfit_data, "speaking_closed"),
                        speaking_half=_text(outfit_data, "speaking_half"),
                        speaking_open=_text(outfit_data, "speaking_open"),
                        mode=_text(outfit_data, "mode", "companion"),
                    )
                )
            selected_outfit = _text(
                avatar_data, "default_outfit", outfits[0].id
            )
            if selected_outfit not in {outfit.id for outfit in outfits}:
                raise ValueError(
                    f"persona.appearance.avatar.default_outfit 不存在: {selected_outfit}"
                )
            avatar = Avatar(
                outfits=tuple(outfits),
                selected_outfit=selected_outfit,
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

    def wearing(self, outfit_id: str) -> "Persona":
        avatar = self.appearance.avatar
        if avatar is None:
            raise ValueError(f"人格 {self.id} 没有可切换的头像套装")
        return replace(
            self,
            appearance=replace(self.appearance, avatar=avatar.wearing(outfit_id)),
        )
