"""从 YAML 加载人格，并把角色偏好应用到运行配置。"""

import os
from dataclasses import replace
from pathlib import Path

import yaml

from .. import config
from .models import Avatar, AvatarOutfit, Persona

_PROJECT_PERSONA_DIR = Path(__file__).resolve().parents[3] / "personas"


def persona_directories() -> list[Path]:
    directories: list[Path] = []
    custom = os.environ.get("SOUL_TTY_PERSONA_DIR")
    if custom:
        directories.append(Path(custom).expanduser())
    directories.append(_PROJECT_PERSONA_DIR)
    return directories


def _load_file(path: Path) -> Persona:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        persona = Persona.from_dict(data)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise RuntimeError(f"人格配置加载失败 {path}: {exc}") from exc
    avatar = persona.appearance.avatar
    if avatar is not None:

        def resolved(value: str) -> str:
            if not value:
                return ""
            asset = Path(value).expanduser()
            if not asset.is_absolute():
                asset = path.parent / asset
            return str(asset.resolve())

        outfits = tuple(
            AvatarOutfit(
                id=outfit.id,
                label=outfit.label,
                idle=resolved(outfit.idle),
                description=outfit.description,
                switch_greetings=outfit.switch_greetings,
                listening=resolved(outfit.listening),
                thinking=resolved(outfit.thinking),
                speaking=resolved(outfit.speaking),
                speaking_closed=resolved(outfit.speaking_closed),
                speaking_half=resolved(outfit.speaking_half),
                speaking_open=resolved(outfit.speaking_open),
            )
            for outfit in avatar.outfits
        )
        avatar = Avatar(
            outfits=outfits,
            selected_outfit=avatar.selected_outfit,
            renderer=avatar.renderer,
            width=avatar.width,
        )
        persona = replace(
            persona,
            appearance=replace(persona.appearance, avatar=avatar),
        )
    return persona


def load_persona(identifier: str) -> Persona:
    """按 id 或 YAML 路径加载人格；自定义目录优先于内置目录。"""
    candidate = Path(identifier).expanduser()
    if candidate.suffix in {".yaml", ".yml"} or candidate.parent != Path("."):
        if not candidate.is_file():
            raise RuntimeError(f"人格配置不存在: {candidate}")
        return _load_file(candidate)

    for directory in persona_directories():
        for suffix in (".yaml", ".yml"):
            path = directory / f"{identifier}{suffix}"
            if path.is_file():
                return _load_file(path)
    choices = ", ".join(persona.id for persona in available_personas()) or "无"
    raise RuntimeError(f"未找到人格 {identifier}（可用: {choices}）")


def available_personas() -> list[Persona]:
    """返回所有有效人格；同 id 时自定义目录覆盖内置目录。"""
    found: dict[str, Persona] = {}
    for directory in persona_directories():
        if not directory.is_dir():
            continue
        for path in sorted((*directory.glob("*.yaml"), *directory.glob("*.yml"))):
            try:
                persona = _load_file(path)
            except RuntimeError:
                continue
            found.setdefault(persona.id, persona)
    return sorted(found.values(), key=lambda persona: persona.id)


def apply_persona(persona: Persona) -> None:
    """应用角色默认值，同时保留环境变量的最高优先级。"""
    if "SYSTEM_PROMPT" not in os.environ:
        config.SYSTEM_PROMPT = (
            f"你的名字是“{persona.display_name}”。\n"
            f"{persona.personality.system_prompt}"
        )
    if "TTS_BACKEND" not in os.environ:
        config.TTS_BACKEND = persona.voice.backend
    if "MLX_TTS_VOICE" not in os.environ and persona.voice.voice:
        config.MLX_TTS_VOICE = persona.voice.voice
    if "MLX_TTS_INSTRUCT" not in os.environ:
        config.MLX_TTS_INSTRUCT = persona.voice.instruct
