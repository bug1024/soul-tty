"""Soul TTY（终端之魂）命令行入口。"""

import argparse
import os
import sys
import threading
import time

from . import config, conversation, presence, relationship
from .clients import llm
from .personas import apply_persona, available_personas, load_persona
from .personas.models import AvatarOutfit
from .ui import terminal


def _tts_description() -> str | None:
    if not config.TTS_ENABLED:
        return None
    if config.TTS_BACKEND == "macos":
        return f"macOS {config.MACOS_VOICE} (低延迟)"
    return f"{config.MLX_TTS_URL} (MLX Qwen3-TTS / {config.MLX_TTS_VOICE})"


def main() -> None:
    parser = argparse.ArgumentParser(prog="soul-tty", description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=["personas", "outfits"],
        help="列出可用人格或当前人格的头像套装",
    )
    parser.add_argument(
        "--persona",
        default=os.environ.get("SOUL_TTY_PERSONA", "serena"),
        help="人格 id 或 YAML 文件路径（默认: serena）",
    )
    parser.add_argument(
        "--name",
        default=os.environ.get("AGENT_NAME"),
        help="临时覆盖 Agent 显示名称",
    )
    parser.add_argument(
        "--outfit",
        default=os.environ.get("SOUL_TTY_OUTFIT"),
        help="启动时选择头像套装，例如 default、late-night、work",
    )
    parser.add_argument("--file", help="用 WAV 文件测 ASR->LLM 链路")
    parser.add_argument("--text", help="跳过 ASR 直测 LLM（含 TTS 播放）")
    args = parser.parse_args()

    if args.command == "personas":
        for persona in available_personas():
            print(f"{persona.id:<12} {persona.display_name}  {persona.tagline}")
        return

    try:
        persona = load_persona(args.persona)
        if args.command == "outfits":
            avatar = persona.appearance.avatar
            if avatar is None:
                print(f"{persona.display_name} 没有头像套装")
                return
            selected = args.outfit or avatar.selected_outfit
            avatar.wearing(selected)
            for outfit in avatar.outfits:
                marker = " (当前)" if outfit.id == selected else ""
                print(f"{outfit.id:<12} {outfit.label}{marker}")
            return
        if args.outfit:
            persona = persona.wearing(args.outfit)
        if args.name:
            persona = persona.renamed(args.name)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    apply_persona(persona)
    terminal.configure(persona)

    try:
        llm.start_conversation()
    except RuntimeError as exc:
        parser.error(str(exc))

    try:
        model = llm.pick_model()
    except Exception as exc:
        print(f"LLM 服务不可用: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    launch_context = (
        presence.LaunchContext()
        if args.text or args.file
        else presence.record_launch(persona.id)
    )
    terminal.configure_presence(launch_context)

    relationship_state = None
    relationship_service = None
    if config.RELATIONSHIP_ENABLED:

        def evaluate(
            state: relationship.RelationshipState,
            turn: relationship.CompletedTurn,
        ) -> dict | None:
            return llm.evaluate_relationship(
                model,
                persona.display_name,
                state.score,
                state.tier,
                state.mood,
                turn.user_text,
                turn.agent_text,
            )

        relationship_service = relationship.RelationshipService(
            persona.id,
            evaluate,
            terminal.update_relationship,
        )
        relationship_state = relationship_service.state
        terminal.configure_relationship(
            relationship_state.score,
            relationship_state.tier,
            relationship_state.mood,
            relationship_state.inner_voice,
        )
        relationship.install(relationship_service)
        relationship_service.start()
    else:
        terminal.configure_relationship()

    outfit_greeting_generator = None
    if config.LLM_GREETING_ENABLED:

        def outfit_greeting_generator(outfit: AvatarOutfit) -> str | None:
            state = (
                relationship_service.state
                if relationship_service is not None
                else None
            )
            return llm.generate_outfit_greeting(
                model,
                persona.display_name,
                terminal.day_period(),
                outfit.label,
                outfit.description,
                relationship_tier=state.tier if state is not None else "",
                mood=state.mood if state is not None else "calm",
            )

    terminal.configure_outfit_greetings(outfit_greeting_generator)

    dashboard_started = terminal.splash(
        model=model,
        tts=_tts_description(),
    )
    if (
        dashboard_started
        and config.LLM_GREETING_ENABLED
    ):
        period = terminal.day_period()
        initial_outfit = (
            persona.appearance.avatar.selected_outfit
            if persona.appearance.avatar is not None
            else None
        )

        def refresh_greeting() -> None:
            try:
                greeting = llm.generate_greeting(
                    model,
                    persona.display_name,
                    period,
                    relationship_tier=(
                        relationship_state.tier if relationship_state else ""
                    ),
                    repeat_launch=launch_context.repeat_launch,
                    special=launch_context.special_greeting,
                )
            except Exception:
                return
            if greeting:
                terminal.update_greeting(greeting, outfit_id=initial_outfit)

        threading.Thread(target=refresh_greeting, daemon=True).start()
    if dashboard_started:
        idle_generator = None
        if config.LLM_IDLE_EMOTION_ENABLED:
            last_idle_llm_at = 0.0

            def idle_generator() -> str | None:
                nonlocal last_idle_llm_at
                now = time.monotonic()
                if (
                    now - last_idle_llm_at
                    < config.LLM_IDLE_EMOTION_MIN_INTERVAL_S
                ):
                    return None
                last_idle_llm_at = now
                return llm.generate_idle_emotion(
                    model,
                    persona.display_name,
                    terminal.day_period(),
                    relationship_tier=(
                        relationship_service.state.tier
                        if relationship_service is not None
                        else ""
                    ),
                    mood=(
                        relationship_service.state.mood
                        if relationship_service is not None
                        else "calm"
                    ),
                )

        terminal.start_idle_emotions(idle_generator)
    chat = llm.Chat(model)
    try:
        if args.text:
            conversation.answer_text(chat, args.text)
        elif args.file:
            conversation.run_file(chat, args.file)
        else:
            conversation.run_microphone(chat)
    except KeyboardInterrupt:
        terminal.goodbye()
    finally:
        relationship.close()
        terminal.close()
