"""Soul TTY（终端之魂）命令行入口。"""

import argparse
import os
import sys
import threading

from . import config, conversation
from .clients import llm
from .personas import apply_persona, available_personas, load_persona
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
        "command", nargs="?", choices=["personas"], help="列出可用人格"
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
    parser.add_argument("--file", help="用 WAV 文件测 ASR->LLM 链路")
    parser.add_argument("--text", help="跳过 ASR 直测 LLM（含 TTS 播放）")
    args = parser.parse_args()

    if args.command == "personas":
        for persona in available_personas():
            print(f"{persona.id:<12} {persona.display_name}  {persona.tagline}")
        return

    try:
        persona = load_persona(args.persona)
        if args.name:
            persona = persona.renamed(args.name)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    apply_persona(persona)
    terminal.configure(persona)

    try:
        model = llm.pick_model()
    except Exception as exc:
        print(f"LLM 服务不可用: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    dashboard_started = terminal.splash(
        model=model,
        tts=_tts_description(),
    )
    if dashboard_started and config.LLM_GREETING_ENABLED:
        period = terminal.day_period()

        def refresh_greeting() -> None:
            try:
                greeting = llm.generate_greeting(
                    model,
                    persona.display_name,
                    period,
                )
            except Exception:
                return
            if greeting:
                terminal.update_greeting(greeting)

        threading.Thread(target=refresh_greeting, daemon=True).start()
    if dashboard_started:
        idle_generator = None
        if config.LLM_IDLE_EMOTION_ENABLED:

            def idle_generator() -> str | None:
                return llm.generate_idle_emotion(
                    model,
                    persona.display_name,
                    terminal.day_period(),
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
        terminal.close()
