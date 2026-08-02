"""Rich 终端展示层；对话编排不依赖具体渲染实现。"""

import os
import re
import sys
import threading
import time
from dataclasses import dataclass

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import config
from ..personas.models import Persona
from . import avatar as avatar_ui

_console = Console(highlight=False)
_persona: Persona | None = None
_answer_pending = False
_answer_has_content = False
_dashboard: "Dashboard | None" = None

_STATE_LABELS = {
    "idle": "待机中",
    "listening": "正在聆听",
    "thinking": "正在思考",
    "speaking": "正在说话",
}


@dataclass(frozen=True)
class RuntimeDetails:
    model: str
    tts: str | None


class Dashboard:
    """固定终端画布；状态变化只重绘画布，不向滚动区追加角色图。"""

    def __init__(self, persona: Persona, runtime: RuntimeDetails) -> None:
        self.persona = persona
        self.runtime = runtime
        self.state = "idle"
        self.partial_text = ""
        self.messages: list[tuple[str, str]] = []
        self.answer_index: int | None = None
        self.mouth_frame = 1
        self._lip_animation_ready = False
        self._last_mouth_update = 0.0
        self._lock = threading.RLock()
        configured_renderer = os.environ.get(
            "VOICE_AGENT_AVATAR_RENDERER",
            persona.appearance.avatar.renderer
            if persona.appearance.avatar is not None
            else "off",
        ).lower()
        preferred_renderer = (
            "symbols" if configured_renderer == "symbols" else "pixels"
        )
        self.avatars = {
            state: avatar_ui.render_avatar(
                persona, state, True, renderer_override=preferred_renderer
            )
            for state in _STATE_LABELS
        }
        self.mouth_avatars = tuple(
            avatar_ui.render_avatar(
                persona, state, True, renderer_override=preferred_renderer
            )
            for state in (
                "speaking_closed",
                "speaking_half",
                "speaking_open",
            )
        )
        self.live = Live(
            self.render(),
            console=_console,
            screen=True,
            auto_refresh=False,
            transient=False,
        )

    def render(self) -> Group:
        avatar_render = self._active_avatar()
        avatar = avatar_render.symbols
        if avatar_render.native is not None and _console.width >= 82:
            width = self.persona.appearance.avatar.width
            height = max(8, width // 2)
            avatar = Text("\n".join(" " * width for _ in range(height)))
        header = _splash_panel(
            self.persona,
            self.runtime,
            3,
            avatar,
            state=self.state,
        )
        accent = self.persona.appearance.accent_color
        status = Text()
        status.append("  ◉ ", style=f"bold {accent}")
        status.append(_STATE_LABELS[self.state], style="bold")
        if self.partial_text:
            status.append(f"  {self.partial_text}", style="dim")
        elif self.state == "listening":
            status.append("  说话即可", style="dim")

        transcript = Text()
        visible = self.messages[-10:]
        for index, (role, content) in enumerate(visible):
            if role == "you":
                transcript.append("YOU", style="bold")
            elif role == "agent":
                transcript.append(
                    self.persona.display_name.upper(),
                    style=f"bold {self.persona.appearance.primary_color}",
                )
            else:
                transcript.append("·", style="dim")
            transcript.append(f"  {content}", style="dim" if role == "notice" else None)
            if index < len(visible) - 1:
                transcript.append("\n\n")
        if not visible:
            transcript.append("等待你的第一句话…", style="dim italic")

        body_width = min(110, max(44, _console.width - 4))
        body = Panel(
            transcript,
            border_style="dim",
            padding=(1, 2),
            width=body_width,
            title="[dim]对话[/dim]",
            title_align="left",
        )
        return Group(Align.left(header), Text(""), status, Text(""), Align.left(body))

    def _active_avatar(self) -> avatar_ui.AvatarRender:
        if (
            self.state == "speaking"
            and len(self.mouth_avatars) == 3
            and self.mouth_avatars[self.mouth_frame - 1].mode != "off"
        ):
            return self.mouth_avatars[self.mouth_frame - 1]
        return self.avatars.get(self.state) or self.avatars["idle"]

    def start(self) -> None:
        with self._lock:
            self.live.start(refresh=True)
            self._paint_native()

    def _paint_native(self) -> None:
        if _console.width < 82:
            return
        if self.state == "speaking" and avatar_ui.write_native_animation_at(
            self.mouth_avatars,
            _console.file,
            row=3,
            column=7,
        ):
            self._lip_animation_ready = True
            return
        self._lip_animation_ready = False
        render = self._active_avatar()
        if render is None or render.native is None:
            return
        # Panel: border(1) + top padding(1) => row 3;
        # border(1) + left padding(4) + table optical padding(1) => column 7.
        avatar_ui.write_native_at(
            render,
            _console.file,
            row=3,
            column=7,
        )

    def refresh(self, *, paint_avatar: bool = False) -> None:
        with self._lock:
            self.live.update(self.render(), refresh=True)
            if paint_avatar:
                self._paint_native()

    def set_state(self, state: str) -> None:
        if state not in _STATE_LABELS:
            return
        self.state = state
        self.mouth_frame = 1
        if state != "speaking":
            self._lip_animation_ready = False
        if state != "listening":
            self.partial_text = ""
        self.refresh(paint_avatar=True)

    def set_mouth_level(self, level: float) -> None:
        """将平滑播放音量映射为三段口型，并限制终端刷新频率。"""
        if self.state != "speaking" or not config.AVATAR_LIP_SYNC_ENABLED:
            return
        frame = 1 if level < 0.12 else 2 if level < 0.55 else 3
        if frame == self.mouth_frame:
            return
        now = time.monotonic()
        if frame != 1 and now - self._last_mouth_update < 0.075:
            return
        self.mouth_frame = frame
        self._last_mouth_update = now
        with self._lock:
            if self._lip_animation_ready:
                avatar_ui.select_native_animation_frame(_console.file, frame)
                return
            render = self._active_avatar()
            if render.native is not None and _console.width >= 82:
                avatar_ui.write_native_at(
                    render, _console.file, row=3, column=7
                )
            else:
                self.live.update(self.render(), refresh=True)

    def add(self, role: str, text: str) -> int:
        self.messages.append((role, text))
        self.refresh()
        return len(self.messages) - 1

    def update(self, index: int, text: str) -> None:
        role, _ = self.messages[index]
        self.messages[index] = (role, text)
        self.refresh()

    def stop(self) -> None:
        with self._lock:
            self.live.stop()


def configure(persona: Persona) -> None:
    global _persona
    _persona = persona


def _current() -> Persona:
    if _persona is None:
        raise RuntimeError("终端 UI 尚未配置人格")
    return _persona


def _short_model(model: str) -> str:
    name = model.removesuffix(".gguf")
    if match := re.match(r"^(Qwen[^-]*-[^-]+)", name, re.IGNORECASE):
        return match.group(1)
    for marker in ("-Q2_", "-Q3_", "-Q4_", "-Q5_", "-Q6_", "-Q8_"):
        if marker in name:
            name = name.split(marker, 1)[0]
            break
    return name if len(name) <= 30 else f"{name[:29]}…"


def _logo(persona: Persona, stage: int) -> Text:
    primary = persona.appearance.primary_color
    secondary = persona.appearance.secondary_color
    accent = persona.appearance.accent_color
    logos = {
        "moon": (
            ("╭──────────────╮", primary),
            ("│      ◌       │", secondary),
            ("│   ╱  │  ╲    │", primary),
            ("│  ╲  ◉  ╱     │", accent),
            ("╰──────────────╯", secondary),
        ),
        "orbit": (
            ("╭──────────────╮", primary),
            ("│    ╭────╮    │", secondary),
            ("│  ──┤ ◉  ├──  │", accent),
            ("│    ╰────╯    │", secondary),
            ("╰──────────────╯", primary),
        ),
    }
    lines = logos.get(persona.appearance.symbol, logos["moon"])
    text = Text()
    visible = len(lines) if stage >= 1 else 0
    for index, (line, style) in enumerate(lines):
        text.append(line if index < visible else " " * len(line), style=style)
        if index < len(lines) - 1:
            text.append("\n")
    return text


def _splash_panel(
    persona: Persona,
    runtime: RuntimeDetails,
    stage: int,
    avatar: Text | None = None,
    native_avatar: bool = False,
    state: str = "idle",
) -> Panel:
    primary = persona.appearance.primary_color
    secondary = persona.appearance.secondary_color
    accent = persona.appearance.accent_color

    title = Text(
        persona.display_name.upper() if stage >= 2 else "",
        style=f"bold {primary}",
    )
    tagline = Text(persona.tagline if stage >= 2 else "", style="dim")

    greeting = Text()
    if stage >= 3 and persona.personality.greeting:
        greeting.append(f"“{persona.personality.greeting}”", style="italic")

    status = Text()
    technology = Text()
    if stage >= 3:
        status.append("● ", style=f"bold {accent}")
        status.append(_STATE_LABELS.get(state, "角色已就绪"), style="bold")
        status.append(f"    {persona.voice.voice}    中文", style="dim")
        tts_name = (
            "Qwen3-TTS"
            if runtime.tts and "MLX" in runtime.tts
            else "macOS TTS"
            if runtime.tts
            else "文字模式"
        )
        technology.append(
            f"sherpa-onnx · {_short_model(runtime.model)} · {tts_name}",
            style=f"dim {secondary}",
        )

    details = Group(
        Align.center(title),
        Align.center(tagline),
        Text(""),
        Align.center(greeting),
        Text(""),
        Align.center(status),
        Align.center(technology),
    )
    wide_avatar = avatar is not None and _console.width >= 82
    if wide_avatar:
        content = Table.grid(expand=True, padding=(0, 2))
        content.add_column(width=persona.appearance.avatar.width + 2)
        content.add_column(ratio=1)
        content.add_row(Align.center(avatar, vertical="middle"), details)
    elif avatar is not None:
        content = Group(Align.center(avatar), Text(""), details)
    elif native_avatar:
        content = details
    else:
        content = Group(
            Align.center(_logo(persona, stage)),
            Text(""),
            details,
        )
    panel_width = (
        min(110, _console.width - 4)
        if wide_avatar
        else min(64, max(44, _console.width - 4))
    )
    return Panel(
        content,
        border_style=primary,
        padding=(1, 4),
        subtitle="[dim]直接说话 · Ctrl+C 退出[/dim]",
        width=panel_width,
    )


def splash(*, model: str, tts: str | None) -> None:
    """展示一次性角色开场；非交互输出直接打印最终帧。"""
    global _dashboard
    persona = _current()
    runtime = RuntimeDetails(model=model, tts=tts)
    dashboard_enabled = os.environ.get("VOICE_AGENT_DASHBOARD", "1") not in {
        "0", "false", "False"
    }
    if _console.is_terminal and _console.file is sys.stdout and dashboard_enabled:
        _dashboard = Dashboard(persona, runtime)
        _dashboard.start()
        return
    avatar = avatar_ui.render_avatar(persona, "idle", _console.is_terminal)
    native_avatar = avatar_ui.write_native(avatar, _console.file)
    animations = os.environ.get("VOICE_AGENT_ANIMATIONS", "1") not in {
        "0",
        "false",
        "False",
    }
    _console.print()
    if _console.is_terminal and animations:
        with Live(
            _splash_panel(
                persona, runtime, 1, avatar.symbols, native_avatar=native_avatar
            ),
            console=_console,
            auto_refresh=False,
            transient=False,
        ) as live:
            for stage in (1, 2, 3):
                live.update(
                    _splash_panel(
                        persona,
                        runtime,
                        stage,
                        avatar.symbols,
                        native_avatar=native_avatar,
                    ),
                    refresh=True,
                )
                if stage < 3:
                    time.sleep(0.09)
    else:
        _console.print(
            _splash_panel(
                persona, runtime, 3, avatar.symbols, native_avatar=native_avatar
            )
        )
    _console.print()


def _clear_current_line() -> None:
    """绕过 Rich 布局计算，可靠地回到行首并清除当前终端行。"""
    if not _console.is_terminal:
        return
    _console.file.write("\r\033[2K")
    _console.file.flush()


def model_loading(started: bool) -> None:
    if _dashboard is not None:
        _dashboard.set_state("thinking" if started else "idle")
        return
    color = _current().appearance.secondary_color
    if started:
        _console.print(f"  [{color}]◌[/{color}] 正在唤醒语音识别…", end="")
    else:
        if _console.is_terminal:
            _clear_current_line()
        else:
            _console.print()
        _console.print("  [green]●[/green] 语音识别已就绪")


def listening(initial: bool = False) -> None:
    if _dashboard is not None:
        _dashboard.set_state("listening")
        return
    accent = _current().appearance.accent_color
    suffix = "  说话即可" if initial else ""
    _console.print(
        f"\n  [{accent}]◉[/{accent}] [bold]正在聆听[/bold]{suffix}  "
        f"[{accent}]▁▂▃▅▃▂▁[/{accent}]"
    )


def partial(text: str) -> None:
    if _dashboard is not None:
        _dashboard.partial_text = text
        _dashboard.refresh()
        return
    if sys.stdout.isatty():
        secondary = _current().appearance.secondary_color
        _clear_current_line()
        _console.print(
            Text.from_markup(
                f"  [{secondary}]◉[/{secondary}] [dim]{text}[/dim]"
            ),
            end="",
        )


def user_text(text: str) -> None:
    if _dashboard is not None:
        _dashboard.partial_text = ""
        _dashboard.add("you", text)
        return
    if sys.stdout.isatty():
        _clear_current_line()
    _console.print(f"\n  [bold]YOU[/bold]  {text}")


def answer_start() -> None:
    global _answer_pending, _answer_has_content
    persona = _current()
    _answer_pending = True
    _answer_has_content = False
    if _dashboard is not None:
        _dashboard.messages.append(
            ("agent", f"{persona.display_name} 正在想…")
        )
        _dashboard.answer_index = len(_dashboard.messages) - 1
        # 合并为一次刷新，并让高清头像在文本画布之后绘制。
        _dashboard.set_state("thinking")
        return
    _console.print(
        f"\n  [{persona.appearance.primary_color}]✦[/{persona.appearance.primary_color}] "
        f"[dim]{persona.display_name} 正在想…[/dim]",
        end="",
    )


def answer_chunk(text: str) -> None:
    global _answer_pending, _answer_has_content
    if _dashboard is not None:
        if _dashboard.answer_index is None:
            _dashboard.answer_index = _dashboard.add("agent", "")
        old = _dashboard.messages[_dashboard.answer_index][1]
        if _answer_pending:
            old = ""
            _answer_pending = False
        _answer_has_content = True
        _dashboard.update(_dashboard.answer_index, old + text)
        return
    if _answer_pending:
        persona = _current()
        if _console.is_terminal:
            _clear_current_line()
        else:
            _console.print()
        _console.print(
            f"  [bold {persona.appearance.primary_color}]"
            f"{persona.display_name.upper()}[/bold {persona.appearance.primary_color}]\n  ",
            end="",
        )
        _answer_pending = False
    _answer_has_content = True
    _console.print(text, end="", markup=False, soft_wrap=True)


def answer_end() -> None:
    global _answer_pending
    if _dashboard is not None:
        if _answer_pending and _dashboard.answer_index is not None:
            # 取消或无输出时移除“正在想”，避免留下永远进行中的假状态。
            _dashboard.messages.pop(_dashboard.answer_index)
        _answer_pending = False
        _dashboard.answer_index = None
        _dashboard.refresh()
        return
    if _answer_pending:
        if _console.is_terminal:
            _clear_current_line()
        else:
            _console.print()
        _answer_pending = False
    if _answer_has_content:
        _console.print("\n")


def speaking() -> None:
    if _dashboard is not None:
        _dashboard.set_state("speaking")
        return
    accent = _current().appearance.accent_color
    _console.print(f"  [{accent}]≋[/{accent}] [dim]正在说话[/dim]")


def mouth_level(level: float) -> None:
    if _dashboard is not None:
        _dashboard.set_mouth_level(level)


def notice(text: str) -> None:
    if _dashboard is not None:
        _dashboard.add("notice", text)
        return
    _console.print(f"  [dim]· {text}[/dim]")


def warning(text: str) -> None:
    if _dashboard is not None:
        _dashboard.add("notice", f"⚠ {text}")
        return
    _console.print(f"  [yellow]⚠[/yellow]  {text}")


def interrupted(text: str) -> None:
    if _dashboard is not None:
        _dashboard.add("you", f"[打断] {text}")
        _dashboard.set_state("thinking")
        return
    _console.print(f"\n  [yellow]Ⅱ[/yellow] 已打断  [bold]YOU[/bold]  {text}")


def recognized(text: str) -> None:
    if _dashboard is not None:
        _dashboard.add("you", text)
        return
    _console.print(f"  [bold]识别结果[/bold]  {text}")


def goodbye() -> None:
    persona = _current()
    close()
    _console.print(
        f"\n  [{persona.appearance.primary_color}]◌[/{persona.appearance.primary_color}] "
        f"{persona.personality.farewell}\n"
    )


def close() -> None:
    """恢复 alternate screen；可重复调用。"""
    global _dashboard
    if _dashboard is not None:
        _dashboard.stop()
        _dashboard = None
