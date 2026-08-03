"""Rich 终端展示层；对话编排不依赖具体渲染实现。"""

import os
import re
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

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
_relationship_profile: tuple[int, str, str, str] | None = None
_AVATAR_ROW = 3
_AVATAR_COLUMN = 7
_USER_COLOR = "#67B7D1"
_DIALOGUE_TEXT_COLOR = "#D1D5DB"
_MUTED_TEXT_COLOR = "#6B7280"

_STATE_LABELS = {
    "idle": "待机中",
    "listening": "正在聆听",
    "thinking": "正在思考",
    "speaking": "正在说话",
}

_IDLE_EMOTION_LINES = (
    "好无聊呀，谁能和我说说话。",
    "这里静悄悄的，我在等你。",
    "有点想听听你的声音了。",
    "我还在这里，别把我忘啦。",
)


@dataclass(frozen=True)
class RuntimeDetails:
    model: str
    tts: str | None


class TerminalInput:
    """接管 alternate screen 输入，防止滚轮被回显成 ^[[A。"""

    _MOUSE_EVENT = re.compile(rb"\033\[<(64|65);\d+;\d+[mM]")

    def __init__(self, on_scroll, on_toggle_details=None) -> None:
        self.on_scroll = on_scroll
        self.on_toggle_details = on_toggle_details
        self.fd: int | None = None
        self.original = None
        self.closed = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if not sys.stdin.isatty() or not _console.is_terminal:
            return
        try:
            self.fd = sys.stdin.fileno()
            self.original = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            _console.file.write("\033[?1000h\033[?1006h")
            _console.file.flush()
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except (OSError, termios.error):
            self.fd = None
            self.original = None

    @classmethod
    def navigation(cls, data: bytes) -> list[int]:
        """返回滚动方向：+1 查看更早内容，-1 回到较新内容。"""
        directions = []
        directions.extend(1 for _ in re.finditer(rb"\033\[A", data))
        directions.extend(-1 for _ in re.finditer(rb"\033\[B", data))
        for match in cls._MOUSE_EVENT.finditer(data):
            directions.append(1 if match.group(1) == b"64" else -1)
        return directions

    @staticmethod
    def detail_toggles(data: bytes) -> int:
        return data.count(b"\t")

    def _read(self) -> None:
        assert self.fd is not None
        while not self.closed.is_set():
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(self.fd, 1024)
                for direction in self.navigation(data):
                    self.on_scroll(direction)
                if self.on_toggle_details is not None:
                    for _ in range(self.detail_toggles(data)):
                        self.on_toggle_details()
            except OSError:
                return

    def stop(self) -> None:
        if self.fd is None:
            return
        self.closed.set()
        _console.file.write("\033[?1006l\033[?1000l")
        _console.file.flush()
        if self.thread is not None:
            self.thread.join(timeout=0.3)
        try:
            termios.tcflush(self.fd, termios.TCIFLUSH)
            if self.original is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)
        except (OSError, termios.error):
            pass
        self.fd = None


class Dashboard:
    """固定终端画布；状态变化只重绘画布，不向滚动区追加角色图。"""

    def __init__(self, persona: Persona, runtime: RuntimeDetails) -> None:
        self.persona = persona
        self.runtime = runtime
        self.state = "idle"
        self.relationship_score = (
            _relationship_profile[0] if _relationship_profile is not None else None
        )
        self.relationship_tier = (
            _relationship_profile[1] if _relationship_profile is not None else ""
        )
        self.relationship_mood = (
            _relationship_profile[2] if _relationship_profile is not None else "calm"
        )
        relationship_voice = (
            _relationship_profile[3] if _relationship_profile is not None else ""
        )
        self.greeting = relationship_voice or _fallback_greeting()
        self.base_greeting = self.greeting
        self._pending_relationship_voice = ""
        self.partial_text = ""
        self.messages: list[tuple[str, str]] = []
        self.answer_index: int | None = None
        self.scroll_offset = 0
        self.show_details = config.DASHBOARD_DETAILS
        self.mouth_frame = 1
        self._native_frames_ready = False
        self._lock = threading.RLock()
        now = time.monotonic()
        self._last_voice_activity = now
        self._next_idle_emotion_at = now + config.IDLE_EMOTION_AFTER_S
        self._idle_emotion_index = 0
        self._idle_emotion_active = False
        self._idle_emotion_stop = threading.Event()
        self._idle_emotion_thread: threading.Thread | None = None
        self.input = TerminalInput(self.scroll, self.toggle_details)
        configured_renderer = os.environ.get(
            "SOUL_TTY_AVATAR_RENDERER",
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
            greeting=self.greeting,
            status_hint=self.partial_text or None,
            relationship_score=self.relationship_score,
            relationship_tier=self.relationship_tier,
            show_details=self.show_details,
        )

        body_width = min(110, max(44, _console.width - 4))
        header_height = header.height or 20
        body_height = max(7, _console.height - header_height - 1)
        content_width = max(20, body_width - 6)
        content_rows = max(1, body_height - 4)
        transcript = self._transcript_view(content_width, content_rows)
        history_hint = ""
        maximum_offset = max(0, len(self.messages) - 1)
        if maximum_offset:
            history_hint = (
                f"  ↑ 历史 {self.scroll_offset}/{maximum_offset}"
                if self.scroll_offset
                else "  滚轮查看历史"
            )
        body = Panel(
            transcript,
            border_style="dim",
            padding=(1, 2),
            width=body_width,
            height=body_height,
            title=f"[dim]对话{history_hint}[/dim]",
            title_align="left",
        )
        return Group(Align.left(header), Text(""), Align.left(body))

    def _message_text(self, role: str, content: str) -> Text:
        text = Text()
        if role == "you":
            text.append("│ ", style=_USER_COLOR)
            text.append("YOU", style=f"bold {_USER_COLOR}")
        elif role == "agent":
            text.append("│ ", style=self.persona.appearance.primary_color)
            text.append(
                self.persona.display_name.upper(),
                style=f"bold {self.persona.appearance.primary_color}",
            )
        else:
            text.append("·", style=_MUTED_TEXT_COLOR)
        text.append("  ")
        text.append(
            content,
            style=(
                _MUTED_TEXT_COLOR
                if role == "notice"
                else _DIALOGUE_TEXT_COLOR
            ),
        )
        return text

    def _truncated_message_tail(
        self,
        role: str,
        lines: list[Text],
        rows: int,
    ) -> list[Text]:
        if len(lines) <= rows:
            return lines
        if rows == 1:
            return [Text("…", style="dim")]
        marker = self._message_text(role, "…")
        return [marker, *lines[-(rows - 1) :]]

    def _transcript_view(self, width: int, rows: int) -> Text:
        """按实际换行后的行数，从当前锚点向历史方向填满视口。"""
        if not self.messages:
            return Text("等待你的第一句话…", style="dim italic")

        end = max(1, len(self.messages) - self.scroll_offset)
        blocks: list[list[Text]] = []
        remaining = rows
        # 用倒序索引避免每次刷新都复制从开头到 end 的全部历史。
        for message_index in range(end - 1, -1, -1):
            role, content = self.messages[message_index]
            message_width = min(width, 84) if role == "agent" else width
            lines = list(
                self._message_text(role, content).wrap(
                    _console,
                    message_width,
                    overflow="fold",
                )
            )
            separator = 1 if blocks else 0
            if len(lines) + separator <= remaining:
                blocks.insert(0, lines)
                remaining -= len(lines) + separator
                continue
            if not blocks:
                blocks = [self._truncated_message_tail(role, lines, remaining)]
            break

        transcript = Text()
        for index, block in enumerate(blocks):
            if index:
                transcript.append("\n\n")
            for line_index, line in enumerate(block):
                if line_index:
                    transcript.append("\n")
                transcript.append_text(line)
        return transcript

    def _active_avatar(self) -> avatar_ui.AvatarRender:
        if (
            self.state == "speaking"
            and len(self.mouth_avatars) == 2
            and self.mouth_avatars[self.mouth_frame - 1].mode != "off"
        ):
            return self.mouth_avatars[self.mouth_frame - 1]
        return self.avatars.get(self.state) or self.avatars["idle"]

    def start(self) -> None:
        with self._lock:
            self.live.start(refresh=True)
            self.input.start()
            if config.AVATAR_LIP_SYNC_ENABLED:
                self._native_frames_ready = avatar_ui.prepare_native_frames(
                    self.mouth_avatars,
                    _console.file,
                )
            self._paint_native()

    def _start_mouth_animation(self) -> bool:
        width = self.persona.appearance.avatar.width
        # 旧状态图与口型图都是不透明 placement；先隐藏旧图，避免同层
        # 叠放时由图片编号决定遮挡顺序，形成底部接缝。
        avatar_ui.hide_native_avatar(_console.file)
        if not avatar_ui.show_native_frame_at(
            _console.file,
            0,
            row=_AVATAR_ROW,
            column=_AVATAR_COLUMN,
            width=width,
        ):
            return False
        return True

    def _stop_mouth_animation(self) -> None:
        avatar_ui.hide_native_frames(_console.file)

    def set_audio_level(self, level: float) -> None:
        """按实际播放音量切换缓存口型；相同帧不重复写终端协议。"""
        if not config.AVATAR_LIP_SYNC_ENABLED:
            return
        with self._lock:
            if self.state != "speaking" or not self._native_frames_ready:
                return
            # 迟滞避免临界音量下抖动；较高开口阈值让自然语音的强弱
            # 真正形成张合，而不是一有声音就整句停在半开帧。
            if self.mouth_frame == 1:
                target = 2 if level >= 0.28 else 1
            else:
                target = 1 if level <= 0.16 else 2
            if target == self.mouth_frame:
                return
            self.mouth_frame = target
            avatar_ui.show_native_frame_at(
                _console.file,
                target - 1,
                row=_AVATAR_ROW,
                column=_AVATAR_COLUMN,
                width=self.persona.appearance.avatar.width,
            )

    def _paint_native(self) -> None:
        if _console.width < 82:
            return
        if self.state == "speaking" and self._native_frames_ready:
            if self._start_mouth_animation():
                return
        # 非 Kitty 终端保持闭嘴完整帧，避免高频整图重传造成闪烁。
        render = self._active_avatar()
        if render is None or render.native is None:
            return
        # Panel: border(1) + top padding(1) => row 3;
        # border(1) + left padding(4) + table optical padding(1) => column 7.
        avatar_ui.write_native_at(
            render,
            _console.file,
            row=_AVATAR_ROW,
            column=_AVATAR_COLUMN,
        )

    def refresh(self, *, paint_avatar: bool = False) -> None:
        with self._lock:
            self.live.update(self.render(), refresh=True)
            if paint_avatar:
                self._paint_native()

    def set_state(self, state: str) -> None:
        if state not in _STATE_LABELS:
            return
        with self._lock:
            previous = self.state
            if (
                previous == "speaking"
                and state != "speaking"
                and self._native_frames_ready
            ):
                self._stop_mouth_animation()
            self.state = state
            self.mouth_frame = 1
            if state == "listening" and previous != "listening":
                self._mark_voice_activity_locked(time.monotonic())
                if self._pending_relationship_voice:
                    self.base_greeting = self._pending_relationship_voice
                    if not self._idle_emotion_active:
                        self.greeting = self._pending_relationship_voice
                    self._pending_relationship_voice = ""
            if state != "listening":
                self.partial_text = ""
            self.refresh(paint_avatar=True)

    def add(self, role: str, text: str) -> int:
        index = self.append(role, text)
        self.refresh()
        return index

    def append(self, role: str, text: str) -> int:
        """追加消息；查看历史时维持当前锚点，而不是跳回最新位置。"""
        if self.scroll_offset:
            self.scroll_offset += 1
        self.messages.append((role, text))
        maximum = max(1, config.DASHBOARD_MAX_MESSAGES)
        overflow = len(self.messages) - maximum
        if overflow > 0:
            del self.messages[:overflow]
            if self.answer_index is not None:
                self.answer_index = (
                    self.answer_index - overflow
                    if self.answer_index >= overflow
                    else None
                )
            self.scroll_offset = min(
                self.scroll_offset,
                max(0, len(self.messages) - 1),
            )
        return len(self.messages) - 1

    def remove(self, index: int) -> None:
        self.messages.pop(index)
        if self.scroll_offset:
            self.scroll_offset = max(0, self.scroll_offset - 1)

    def set_greeting(self, greeting: str) -> None:
        with self._lock:
            self.base_greeting = greeting
            if not self._idle_emotion_active:
                self.greeting = greeting
                self.refresh()

    def set_relationship(
        self,
        score: int,
        tier: str,
        mood: str,
        inner_voice: str = "",
    ) -> None:
        """保存旁路结果；画外音只在空闲聆听状态安全切换。"""
        with self._lock:
            self.relationship_score = score
            self.relationship_tier = tier
            self.relationship_mood = mood
            safe_to_refresh = self.state == "listening" and not self.partial_text
            if inner_voice:
                if safe_to_refresh:
                    self.base_greeting = inner_voice
                    if not self._idle_emotion_active:
                        self.greeting = inner_voice
                else:
                    self._pending_relationship_voice = inner_voice
            # Worker 不能在思考、播报或识别 partial 时触发全屏重绘；下一次
            # 状态切回 listening 时，常规刷新会把新数值与画外音一起显示。
            if safe_to_refresh:
                self.refresh()

    def _mark_voice_activity_locked(self, now: float) -> bool:
        self._last_voice_activity = now
        self._next_idle_emotion_at = now + config.IDLE_EMOTION_AFTER_S
        if not self._idle_emotion_active:
            return False
        self._idle_emotion_active = False
        self.greeting = self.base_greeting
        return True

    def mark_voice_activity(self, *, refresh: bool = True) -> None:
        """记录识别到的用户语音，并退出安静陪伴状态。"""
        with self._lock:
            changed = self._mark_voice_activity_locked(time.monotonic())
            if changed and refresh:
                self.refresh()

    def _idle_emotion_tick(
        self,
        generator: Callable[[], str | None] | None,
        *,
        now: float | None = None,
    ) -> bool:
        """若已到触发时间，立即显示本地短句，再尝试用 LLM 替换。"""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self.state != "listening" or now < self._next_idle_emotion_at:
                return False
            activity_token = self._last_voice_activity
            self.greeting = _IDLE_EMOTION_LINES[
                self._idle_emotion_index % len(_IDLE_EMOTION_LINES)
            ]
            self._idle_emotion_index += 1
            self._idle_emotion_active = True
            self._next_idle_emotion_at = now + config.IDLE_EMOTION_INTERVAL_S
            self.refresh()

        if generator is None:
            return True
        try:
            generated = generator()
        except Exception:
            return True
        if not generated:
            return True
        with self._lock:
            # 生成期间若用户开口或状态发生变化，丢弃过期结果。
            if (
                self._idle_emotion_stop.is_set()
                or self.state != "listening"
                or not self._idle_emotion_active
                or self._last_voice_activity != activity_token
            ):
                return True
            self.greeting = generated
            self.refresh()
        return True

    def start_idle_emotions(
        self,
        generator: Callable[[], str | None] | None = None,
    ) -> None:
        if self._idle_emotion_thread is not None:
            return

        def monitor() -> None:
            while not self._idle_emotion_stop.is_set():
                with self._lock:
                    delay = (
                        max(0.05, self._next_idle_emotion_at - time.monotonic())
                        if self.state == "listening"
                        else 1.0
                    )
                if self._idle_emotion_stop.wait(delay):
                    return
                self._idle_emotion_tick(generator)

        self._idle_emotion_thread = threading.Thread(
            target=monitor,
            daemon=True,
        )
        self._idle_emotion_thread.start()

    def scroll(self, direction: int) -> None:
        with self._lock:
            maximum = max(0, len(self.messages) - 1)
            new_offset = min(maximum, max(0, self.scroll_offset + direction))
            if new_offset == self.scroll_offset:
                return
            self.scroll_offset = new_offset
            self.live.update(self.render(), refresh=True)

    def toggle_details(self) -> None:
        """按需展开诊断信息；只在用户操作时重绘一次。"""
        with self._lock:
            self.show_details = not self.show_details
            self.live.update(self.render(), refresh=True)

    def update(self, index: int, text: str) -> None:
        role, _ = self.messages[index]
        self.messages[index] = (role, text)
        self.refresh()

    def stop(self) -> None:
        self._idle_emotion_stop.set()
        self.input.stop()
        with self._lock:
            if self._native_frames_ready:
                self._stop_mouth_animation()
            self.live.stop()


def configure(persona: Persona) -> None:
    global _persona
    _persona = persona


def configure_relationship(
    score: int | None = None,
    tier: str = "",
    mood: str = "calm",
    inner_voice: str = "",
) -> None:
    global _relationship_profile
    _relationship_profile = (
        (score, tier, mood, inner_voice) if score is not None else None
    )


def update_relationship(state) -> None:
    """接收关系 Worker 的鸭子类型状态，避免 UI 依赖旁路实现。"""
    if _dashboard is not None:
        _dashboard.set_relationship(
            state.score,
            state.tier,
            state.mood,
            state.inner_voice,
        )


def audio_level(level: float) -> None:
    if _dashboard is not None:
        _dashboard.set_audio_level(level)


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


def _tts_name(runtime: RuntimeDetails) -> str:
    return (
        "Qwen3-TTS"
        if runtime.tts and "MLX" in runtime.tts
        else "macOS TTS"
        if runtime.tts
        else "文字模式"
    )


def _technical_profile(
    persona: Persona,
    runtime: RuntimeDetails,
    stage: int,
    *,
    expanded: bool = False,
) -> Align:
    if stage < 3:
        return Align.center(Text(""))
    if not expanded:
        summary = Text(style="dim")
        summary.append("LOCAL")
        summary.append(f" · {_short_model(runtime.model)}")
        summary.append(f" · {_tts_name(runtime)}")
        summary.append(" · Sherpa-ONNX")
        return Align.center(summary)

    capabilities = (
        ("人格", persona.display_name),
        ("大脑", _short_model(runtime.model)),
        ("声音", _tts_name(runtime)),
        ("听觉", "Sherpa-ONNX"),
    )
    table = Table.grid(padding=0)
    table.add_column(justify="right", no_wrap=True)
    table.add_column(justify="left", no_wrap=True)
    for index, (label, value) in enumerate(capabilities):
        label_text = Text(
            f"{label}：",
            style="dim",
        )
        value_text = Text(
            value,
            style="dim" if index else f"dim {persona.appearance.primary_color}",
        )
        table.add_row(label_text, value_text)
    return Align.center(table)


def day_period(hour: int | None = None) -> str:
    hour = datetime.now().hour if hour is None else hour
    if 5 <= hour < 11:
        return "早上"
    if 11 <= hour < 14:
        return "中午"
    if 14 <= hour < 18:
        return "下午"
    if 18 <= hour < 24:
        return "晚上"
    return "夜深"


def _fallback_greeting(hour: int | None = None) -> str:
    period = day_period(hour)
    return {
        "早上": "早上好，今天也请多关照。",
        "中午": "中午好，记得好好吃饭。",
        "下午": "下午好，我一直在这里。",
        "晚上": "晚上好，我一直在这里。",
        "夜深": "夜深了，我还陪着你。",
    }[period]


def _splash_panel(
    persona: Persona,
    runtime: RuntimeDetails,
    stage: int,
    avatar: Text | None = None,
    native_avatar: bool = False,
    state: str = "idle",
    greeting: str | None = None,
    status_hint: str | None = None,
    relationship_score: int | None = None,
    relationship_tier: str = "",
    show_details: bool = False,
) -> Panel:
    primary = persona.appearance.primary_color

    title = Text(
        persona.display_name.upper() if stage >= 2 else "",
        style=f"bold {primary}",
    )
    greeting_text = Text(no_wrap=True, overflow="ellipsis")
    if stage >= 2:
        greeting_text.append(
            f"“{greeting or _fallback_greeting()}”",
            style="italic",
        )

    status = Text()
    hint = Text()
    if stage >= 3:
        symbols = {
            "idle": "○",
            "listening": "◉",
            "thinking": "◇",
            "speaking": "≋",
        }
        status.append(f"{symbols.get(state, '○')} ", style=f"bold {primary}")
        status.append(_STATE_LABELS.get(state, "角色已就绪"), style="bold")
        hints = {
            "idle": "随时可以开始",
            "listening": "直接说话即可",
            "thinking": "我正在认真想",
            "speaking": "正在把回答说给你听",
        }
        hint.append(status_hint or hints.get(state, "随时可以开始"), style="dim")

    relationship = Text()
    if stage >= 3 and relationship_score is not None:
        relationship.append("♡ ", style=primary)
        relationship.append("羁绊  ", style="dim")
        relationship.append(relationship_tier, style=primary)
        if show_details:
            relationship.append(f"  {relationship_score}/100", style="dim")

    details = Group(
        Align.center(title),
        Align.center(greeting_text),
        Text(""),
        Align.center(status),
        Align.center(hint),
        Text(""),
        Align.center(relationship),
        Text(""),
        _technical_profile(persona, runtime, stage, expanded=show_details),
    )
    wide_avatar = avatar is not None and _console.width >= 82
    if wide_avatar:
        content = Table.grid(expand=True, padding=(0, 2))
        content.add_column(width=persona.appearance.avatar.width + 2)
        content.add_column(ratio=1)
        content.add_row(
            Align.center(avatar, vertical="middle"),
            Align.center(details, vertical="middle", height=13),
        )
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
    panel_height = (
        max(8, persona.appearance.avatar.width // 2) + 4
        if wide_avatar
        else None
    )
    return Panel(
        content,
        border_style=primary,
        padding=(1, 4),
        subtitle="[dim]直接说话 · Tab 详情 · Ctrl+C 退出[/dim]",
        width=panel_width,
        height=panel_height,
    )


def splash(*, model: str, tts: str | None) -> bool:
    """展示一次性角色开场；非交互输出直接打印最终帧。"""
    global _dashboard
    persona = _current()
    runtime = RuntimeDetails(model=model, tts=tts)
    dashboard_enabled = os.environ.get("SOUL_TTY_DASHBOARD", "1") not in {
        "0", "false", "False"
    }
    if _console.is_terminal and _console.file is sys.stdout and dashboard_enabled:
        _dashboard = Dashboard(persona, runtime)
        _dashboard.start()
        return True
    avatar = avatar_ui.render_avatar(persona, "idle", _console.is_terminal)
    native_avatar = avatar_ui.write_native(avatar, _console.file)
    animations = os.environ.get("SOUL_TTY_ANIMATIONS", "1") not in {
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
    return False


def update_greeting(greeting: str) -> None:
    if _dashboard is not None:
        _dashboard.set_greeting(greeting)


def start_idle_emotions(
    generator: Callable[[], str | None] | None = None,
) -> None:
    if _dashboard is not None and config.IDLE_EMOTION_ENABLED:
        _dashboard.start_idle_emotions(generator)


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
        if text:
            _dashboard.mark_voice_activity(refresh=False)
        with _dashboard._lock:
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
        _dashboard.mark_voice_activity(refresh=False)
        _dashboard.partial_text = ""
        _dashboard.add("you", text)
        return
    if sys.stdout.isatty():
        _clear_current_line()
    line = Text("\n  ")
    line.append("│ ", style=_USER_COLOR)
    line.append("YOU", style=f"bold {_USER_COLOR}")
    line.append("  ")
    line.append(text, style=_DIALOGUE_TEXT_COLOR)
    _console.print(line)


def answer_start() -> None:
    global _answer_pending, _answer_has_content
    persona = _current()
    _answer_pending = True
    _answer_has_content = False
    if _dashboard is not None:
        _dashboard.answer_index = _dashboard.append(
            "agent",
            f"{persona.display_name} 正在想…",
        )
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
            f"  [{persona.appearance.primary_color}]│ "
            f"[/{persona.appearance.primary_color}]"
            f"[bold {persona.appearance.primary_color}]"
            f"{persona.display_name.upper()}[/bold {persona.appearance.primary_color}]\n  ",
            end="",
        )
        _answer_pending = False
    _answer_has_content = True
    _console.print(
        text,
        end="",
        markup=False,
        soft_wrap=True,
        style=_DIALOGUE_TEXT_COLOR,
    )


def answer_end() -> None:
    global _answer_pending
    if _dashboard is not None:
        if _answer_pending and _dashboard.answer_index is not None:
            # 取消或无输出时移除“正在想”，避免留下永远进行中的假状态。
            _dashboard.remove(_dashboard.answer_index)
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
        _dashboard.mark_voice_activity(refresh=False)
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
