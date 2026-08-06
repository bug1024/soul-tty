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


def _on_emotion_update(emotion_service, snap) -> None:
    """EmotionService 可视化与 prompt 热更新共用一份快照。"""
    terminal.update_emotion(snap)
    if snap.should_update_prompt:
        emit_emotion_update(emotion_service, snap)


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

    # Initialize EmotionService
    emotion_service = None
    if config.EMOTION_ENABLED:
        from .emotion import EmotionService
        from .emotion.state import load_runtime, save_runtime
        from .conversation import emit_emotion_update

        existing = load_runtime(config.SOUL_TTY_STATE_DIR)
        save_runtime(config.SOUL_TTY_STATE_DIR, existing + 1)

        baseline = persona.personality.mood_baseline
        emotion_service = EmotionService(
            persona_id=persona.id,
            baseline=baseline,
            state_dir=config.SOUL_TTY_STATE_DIR,
            jitter=0.1,
            ema_rate=config.EMOTION_EMA_RATE,
            delta_cap=config.EMOTION_DELTA_CAP,
            decay_rate=config.EMOTION_DECAY_RATE,
            intensity_update_threshold=config.EMOTION_PROMPT_UPDATE_INTENSITY,
            on_update=lambda snap: (
                _on_emotion_update(emotion_service, snap)
            ),
            decay_interval_s=config.EMOTION_DECAY_INTERVAL_S,
            idle_threshold_s=config.EMOTION_IDLE_THRESHOLD_S,
        )
        apply_persona(persona, emotion_service=emotion_service)
        emotion_service.start_decay_thread()
        terminal.configure_emotion(emotion_service)

    try:
        main_model = llm.pick_model(config.LLM_URL, config.LLM_MODEL)
    except Exception as exc:
        print(f"主 LLM 服务不可用: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # 辅助 LLM 默认走主 LLM（同 URL/同 model）；只有用户显式配置了不同的
    # AUX_LLM_URL 才单独去 auto-discover 一个独立模型，避免无谓请求。
    aux_model = main_model
    aux_url = config._resolve_aux_url()
    aux_model_name = config._resolve_aux_model()
    if aux_url != config.LLM_URL or aux_model_name != config.LLM_MODEL:
        try:
            aux_model = llm.pick_model(aux_url, aux_model_name)
        except Exception:
            aux_model = main_model

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
            # 当前 mood 由 EmotionService 持有；评估时把它的快照喂给 LLM
            # 当 prompt context，让打分参考 Soul 的真实情绪状态。
            current_mood = (
                emotion_service.snapshot().mood
                if emotion_service is not None
                else "calm"
            )
            return llm.evaluate_relationship(
                aux_model,
                persona.display_name,
                state.bond,
                state.level,
                current_mood,
                turn.user_text,
                turn.agent_text,
            )

        def on_relationship_update(state) -> None:
            current_mood = (
                emotion_service.snapshot().mood
                if emotion_service is not None
                else "calm"
            )
            terminal.update_relationship(state, mood=current_mood)

        def on_evaluation_result(payload: dict) -> None:
            """RelationshipService 把 apply_evaluation 的 payload 抛上来；这里分发到 emotion/expression。"""
            if emotion_service is None:
                return
            emotion_delta = payload.get("emotion_delta") or {}
            expression_state = payload.get("expression_state") or {}
            style = expression_state.get("style", "neutral")
            if not emotion_delta and style == "neutral":
                return
            try:
                # EmotionService.apply_delta 内部会把 expression_hint 转给
                # ExpressionService.resolve 完成合法性收敛。
                emotion_service.apply_delta(emotion_delta, expression_hint=style)
            except Exception:
                pass

        relationship_service = relationship.RelationshipService(
            persona.id,
            evaluate,
            on_relationship_update,
            on_evaluation_result,
        )
        relationship_state = relationship_service.state
        initial_mood = (
            emotion_service.snapshot().mood
            if emotion_service is not None
            else "calm"
        )
        terminal.configure_relationship(
            relationship_state.bond,
            relationship_state.level,
            initial_mood,
            relationship_state.inner_voice,
            relationship_state.session_count,
            relationship_state.event,
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
            current_mood = (
                emotion_service.snapshot().mood
                if emotion_service is not None
                else "calm"
            )
            return llm.generate_outfit_greeting(
                aux_model,
                persona.display_name,
                terminal.day_period(),
                outfit.label,
                outfit.description,
                relationship_tier=state.level if state is not None else "",
                mood=current_mood,
                expression=emotion_service.snapshot().expression if emotion_service is not None else "neutral",
            )

    terminal.configure_outfit_greetings(outfit_greeting_generator)

    dashboard_started = terminal.splash(
        model=main_model,
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
                    aux_model,
                    persona.display_name,
                    period,
                    relationship_tier=(
                        relationship_state.level if relationship_state else ""
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
                current_mood = (
                    emotion_service.snapshot().mood
                    if emotion_service is not None
                    else "calm"
                )
                return llm.generate_idle_emotion(
                    aux_model,
                    persona.display_name,
                    terminal.day_period(),
                    relationship_tier=(
                        relationship_service.state.level
                        if relationship_service is not None
                        else ""
                    ),
                    mood=current_mood,
                    expression=emotion_service.snapshot().expression if emotion_service is not None else "neutral",
                )

        terminal.start_idle_emotions(idle_generator)
    chat = llm.Chat(main_model)
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
        if emotion_service is not None:
            emotion_service.stop()
        terminal.close()
