"""从 YAML 加载人格，并把角色偏好应用到运行配置。"""

import os
from dataclasses import replace
from pathlib import Path

import yaml

from .. import config

# 模式 → system prompt 修饰符
_MODE_MODIFIERS: dict[str, str] = {
    "companion": (
        "你处于陪伴模式：允许自然闲聊和适度调侃，语气俏皮温暖，"
        "回答保持正常长度（2-3句），话题不受限制。"
    ),
    "focused": (
        "你处于专注模式：减少寒暄，直奔主题，回答更短（1-2句），"
        "优先协助任务，克制情绪化表达，语气友善但精简。"
    ),
    "late_night": (
        "你处于夜间模式：表达更松弛，允许聊私密话题，"
        "语气柔和低沉，像深夜轻声交谈，控制在2-3句。"
    ),
}


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
                mode=outfit.mode,
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


def apply_persona(persona: Persona, emotion_service=None) -> None:
    """应用角色默认值，同时保留环境变量的最高优先级。

    emotion_service 不为 None 时，追加 [Emotion Context] 段落。
    """
    if "SYSTEM_PROMPT" not in os.environ:
        # 中文排版用全角引号；用 \u 转义避免与 f-string 自身引号冲突。
        base = f"你的名字是“{persona.display_name}”。\n{persona.personality.system_prompt}"
        avatar = persona.appearance.avatar
        mode = avatar.outfit.mode if avatar else "companion"
        modifier = _MODE_MODIFIERS.get(mode, _MODE_MODIFIERS["companion"])
        sections = [base, modifier]
        if emotion_service is not None:
            sections.append("[Emotion Context]\n" + emotion_service.snapshot().context_text)
        config.SYSTEM_PROMPT = "\n\n".join(sections)
    if "TTS_BACKEND" not in os.environ:
        config.TTS_BACKEND = persona.voice.backend
    if "MLX_TTS_VOICE" not in os.environ and persona.voice.voice:
        config.MLX_TTS_VOICE = persona.voice.voice
    if "MLX_TTS_INSTRUCT" not in os.environ:
        config.MLX_TTS_INSTRUCT = persona.voice.instruct
