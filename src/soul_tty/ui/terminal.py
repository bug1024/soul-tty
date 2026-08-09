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
from .. import conversation
from ..personas.models import AvatarOutfit, Persona
from ..presence import LaunchContext
from . import avatar as avatar_ui

_console = Console(highlight=False)
_persona: Persona | None = None
_answer_pending = False
_answer_has_content = False
_dashboard: "Dashboard | None" = None
_relationship_profile: tuple[float, str, str, str, int, str] | None = None
_emotion_service = None
_agency_profile = None
_memory_profile = None
_reflection_service = None
_launch_context = LaunchContext()
_outfit_greeting_generator: Callable[[AvatarOutfit], str | None] | None = None
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

_MODE_ORDER = ("companion", "focused", "late_night")
_MODE_LABELS: dict[str, str] = {
    "companion": "陪伴模式",
    "focused": "专注模式",
    "late_night": "夜间模式",
}

# 情绪 Mood → 中文标签。贴近口语，不抢戏。
_MOOD_LABELS: dict[str, str] = {
    "numb": "麻木",
    "tired": "疲惫",
    "sad": "低落",
    "excited": "雀跃",
    "curious": "好奇",
    "happy": "愉悦",
    "calm": "平静",
}

# 用户声音情绪感知 → 克制的中文描述（不是"情绪标签"，是"听起来像什么"）
_VOICE_EMOTION_LABELS: dict[str, str] = {
    "happy": "语气愉悦",
    "sad": "语气偏低落",
    "angry": "语气偏激动",
    "neutral": "语气平稳",
    "surprise": "语气意外",
    "fear": "语气紧张",
    "disgust": "语气排斥",
    "unknown": "—",
}

_VOICE_EVENT_LABELS: dict[str, str] = {
    "speech": "说话",
    "laughter": "轻笑",
    "crying": "哭腔",
    "cough": "咳嗽",
    "sneeze": "打喷嚏",
    "applause": "掌声",
}

# 5 维情绪值 → 短中文标签（详情行使用）。
_EMOTION_DIM_LABELS: dict[str, str] = {
    "happiness": "愉悦",
    "calmness": "平静",
    "curiosity": "好奇",
    "stress": "压力",
    "energy": "活力",
}

# 关系 level 英文 → 中文展示。HUD 内部仍用英文做语义键，仅显示层翻译。
_RELATIONSHIP_LEVEL_ZH: dict[str, str] = {
    "stranger": "初识阶段",
    "acquaintance": "相识阶段",
    "familiar": "熟悉阶段",
    "companion": "信任阶段",
    "close": "默契阶段",
    "bonded": "深度连接",
}

_IDLE_EMOTION_LINES = (
    "好无聊呀，谁能和我说说话。",
    "这里静悄悄的，我在等你。",
    "有点想听听你的声音了。",
    "我还在这里，别把我忘啦。",
)
_IDLE_PRESENCE_HINTS = (
    "她安静地等你开口",
    "似乎并不急着说话",
    "今天更想听你说",
    "等你先打破沉默",
)


@dataclass(frozen=True)
class RuntimeDetails:
    model: str
    tts: str | None


class TerminalInput:
    """接管 alternate screen 输入，防止滚轮被回显成 ^[[A。"""

    _MOUSE_EVENT = re.compile(rb"\033\[<(64|65);\d+;\d+[mM]")
    _CSI_EVENT = re.compile(rb"\033\[[0-?]*[ -/]*[@-~]")

    def __init__(
        self,
        on_scroll,
        on_toggle_details=None,
        on_cycle_mode=None,
        on_secret_mode=None,
    ) -> None:
        self.on_scroll = on_scroll
        self.on_toggle_details = on_toggle_details
        self.on_cycle_mode = on_cycle_mode
        self.on_secret_mode = on_secret_mode
        self.fd: int | None = None
        self.original = None
        self.closed = threading.Event()
        self.thread: threading.Thread | None = None
        self._pending_input = b""
        self._secret_pending = b""
        self._secret_pending_at = 0.0

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

    @classmethod
    def outfit_toggles(cls, data: bytes) -> int:
        """统计裸 `0` 按键，忽略所有 CSI 键盘与鼠标转义序列。"""
        plain = cls._CSI_EVENT.sub(b"", data)
        return plain.count(b"0")

    @classmethod
    def secret_toggles(
        cls,
        data: bytes,
        pending: bytes = b"",
    ) -> tuple[int, bytes]:
        """识别隐藏 `1`→`8` 序列；鼠标/CSI 坐标永远不参与匹配。"""
        plain = cls._CSI_EVENT.sub(b"", data)
        combined = pending + plain
        return combined.count(b"18"), (b"1" if combined.endswith(b"1") else b"")

    @staticmethod
    def split_incomplete_escape(data: bytes) -> tuple[bytes, bytes]:
        """保留末尾未收完的 CSI 序列，避免其中的坐标被当成按键。"""
        start = data.rfind(b"\033")
        if start < 0:
            return data, b""
        suffix = data[start:]
        if suffix == b"\033" or suffix == b"\033[":
            return data[:start], suffix
        if suffix.startswith(b"\033[") and not any(
            0x40 <= byte <= 0x7E for byte in suffix[2:]
        ):
            return data[:start], suffix
        return data, b""

    def _read(self) -> None:
        assert self.fd is not None
        while not self.closed.is_set():
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.1)
                if not ready:
                    continue
                chunk = self._pending_input + os.read(self.fd, 1024)
                data, self._pending_input = self.split_incomplete_escape(chunk)
                for direction in self.navigation(data):
                    self.on_scroll(direction)
                if self.on_toggle_details is not None:
                    for _ in range(self.detail_toggles(data)):
                        self.on_toggle_details()
                if self.on_cycle_mode is not None:
                    for _ in range(self.outfit_toggles(data)):
                        self.on_cycle_mode()
                if self.on_secret_mode is not None:
                    if (
                        self._secret_pending
                        and time.monotonic() - self._secret_pending_at > 1.5
                    ):
                        self._secret_pending = b""
                    toggles, self._secret_pending = self.secret_toggles(
                        data,
                        self._secret_pending,
                    )
                    self._secret_pending_at = (
                        time.monotonic() if self._secret_pending else 0.0
                    )
                    for _ in range(toggles):
                        self.on_secret_mode()
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
        self.relationship_bond = (
            _relationship_profile[0] if _relationship_profile is not None else None
        )
        self.relationship_level = (
            _relationship_profile[1] if _relationship_profile is not None else ""
        )
        self.relationship_mood = (
            _relationship_profile[2] if _relationship_profile is not None else "calm"
        )
        relationship_voice = (
            _relationship_profile[3] if _relationship_profile is not None else ""
        )
        self.relationship_interaction_count = (
            _relationship_profile[4] if _relationship_profile is not None else 0
        )
        self.relationship_recent_events: tuple[str, ...] = (
            _relationship_profile[5] if _relationship_profile is not None else ()
        )
        # 情绪系统：初值从 EmotionService 快照拉一次，
        # 后续由 set_emotion 接收 on_update 回调热更新。
        self.emotion_mood = "calm"
        self.emotion_intensity = 0.0
        self.emotion_expression = "neutral"
        self.emotion_vector = None
        if _emotion_service is not None:
            try:
                initial = _emotion_service.snapshot()
            except Exception:
                initial = None
            if initial is not None:
                self.emotion_mood = initial.mood
                self.emotion_intensity = initial.intensity
                self.emotion_expression = initial.expression
                if initial.emotion is not None:
                    self.emotion_vector = initial.emotion
        # 声音感知：最近一次用户语气观察，由 set_voice_observation 更新。
        self.voice_emotion = ""
        self.voice_event = ""
        self.voice_language = ""
        self._voice_observed_at: float | None = None  # TTL 过期判定
        self.greeting = relationship_voice or _fallback_greeting(
            tier=self.relationship_level,
            repeat_launch=_launch_context.repeat_launch,
            special=_launch_context.special_greeting,
        )
        self.base_greeting = self.greeting
        self._pending_relationship_voice = ""

        # Presence Panel 只保存业务服务的真实快照；None 表示对应能力未启用。
        self.agency_state = _agency_profile
        self.memory_presence = _memory_profile
        self.reflection_service = _reflection_service

        self.partial_text = ""
        self.presence_hint = ""
        self.messages: list[tuple[str, str]] = []
        self.answer_index: int | None = None
        self.interrupt_index: int | None = None
        self.scroll_offset = 0
        self.show_details = config.DASHBOARD_DETAILS
        self.dev_mode = False
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
        self._outfit_switch_index = 0
        self._outfit_greeting_serial = 0
        self._mode_switch_index = 0
        self._secret_return_outfit = "default"
        self._outfit_greeting_timer: threading.Timer | None = None
        self._outfit_greeting_generator = _outfit_greeting_generator
        self.input = TerminalInput(
            self.scroll,
            self.toggle_details,
            self.cycle_mode,
            self.toggle_secret_mode,
        )
        configured_renderer = os.environ.get(
            "SOUL_TTY_AVATAR_RENDERER",
            persona.appearance.avatar.renderer
            if persona.appearance.avatar is not None
            else "off",
        ).lower()
        preferred_renderer = (
            "symbols" if configured_renderer == "symbols" else "pixels"
        )
        self._preferred_avatar_renderer = preferred_renderer
        self.avatars, self.mouth_avatars = self._load_avatar_set(persona)
        self.mode = (
            persona.appearance.avatar.outfit.mode
            if persona.appearance.avatar
            else "companion"
        )
        if persona.appearance.avatar is not None:
            conversation.set_memory_persistence_allowed(
                persona.appearance.avatar.outfit.memory_enabled
            )
        self.live = Live(
            self.render(),
            console=_console,
            screen=True,
            auto_refresh=False,
            transient=False,
        )

    def _load_avatar_set(
        self,
        persona: Persona,
    ) -> tuple[
        dict[str, avatar_ui.AvatarRender],
        tuple[avatar_ui.AvatarRender, ...],
    ]:
        """只转换当前套装；同一路径的多个状态共享一次渲染结果。"""
        avatar = persona.appearance.avatar
        cache: dict[str, avatar_ui.AvatarRender] = {}

        def render(state: str) -> avatar_ui.AvatarRender:
            key = avatar.for_state(state) if avatar is not None else state
            if key not in cache:
                cache[key] = avatar_ui.render_avatar(
                    persona,
                    state,
                    True,
                    renderer_override=self._preferred_avatar_renderer,
                )
            return cache[key]

        avatars = {state: render(state) for state in _STATE_LABELS}
        mouths = tuple(
            render(state) for state in ("speaking_closed", "speaking_half")
        )
        return avatars, mouths

    def _local_outfit_greeting(self, outfit: AvatarOutfit) -> str:
        lines = outfit.switch_greetings
        if not lines:
            return f"换成{outfit.label}，感觉也不错。"
        line = lines[self._outfit_switch_index % len(lines)]
        self._outfit_switch_index += 1
        return line

    def _schedule_outfit_greeting(
        self,
        outfit: AvatarOutfit,
        serial: int,
    ) -> None:
        generator = self._outfit_greeting_generator
        if generator is None:
            return

        def generate() -> None:
            try:
                greeting = generator(outfit)
            except Exception:
                return
            if not greeting:
                return
            with self._lock:
                avatar = self.persona.appearance.avatar
                if (
                    self._idle_emotion_stop.is_set()
                    or serial != self._outfit_greeting_serial
                    or avatar is None
                    or avatar.selected_outfit != outfit.id
                ):
                    return
                self.base_greeting = greeting
                if not self._idle_emotion_active:
                    self.greeting = greeting
                    self.refresh()

        timer = threading.Timer(0.25, generate)
        timer.daemon = True
        with self._lock:
            if self._outfit_greeting_timer is not None:
                self._outfit_greeting_timer.cancel()
            self._outfit_greeting_timer = timer
        timer.start()

    def _switch_outfit(self, outfit_id: str) -> None:
        with self._lock:
            persona = self.persona.wearing(outfit_id)
            avatars, mouths = self._load_avatar_set(persona)

            if self._native_frames_ready:
                self._stop_mouth_animation()
            avatar_ui.hide_native_avatar(_console.file)
            self.persona = persona
            _persona = persona
            self.avatars = avatars
            self.mouth_avatars = mouths
            self.mouth_frame = 1
            self._native_frames_ready = False
            if config.AVATAR_LIP_SYNC_ENABLED:
                self._native_frames_ready = avatar_ui.prepare_native_frames(
                    self.mouth_avatars,
                    _console.file,
                )

            outfit = persona.appearance.avatar.outfit
            self.mode = outfit.mode
            conversation.set_memory_persistence_allowed(
                outfit.memory_enabled
            )

            # 重新 apply_persona 更新 config.SYSTEM_PROMPT（包含新 mode 修饰符）
            from ..personas import apply_persona

            apply_persona(self.persona)

            # 热更新活跃 Chat 的 system prompt
            chat = conversation._active_chat
            if chat is not None:
                chat.update_system_prompt(config.SYSTEM_PROMPT)
                if outfit.memory_enabled:
                    chat.clear_private_history()

            self._mark_voice_activity_locked(time.monotonic())
            greeting = self._local_outfit_greeting(outfit)
            self.base_greeting = greeting
            self.greeting = greeting
            self.presence_hint = ""
            self._outfit_greeting_serial += 1
            serial = self._outfit_greeting_serial
            self.refresh(paint_avatar=True)

        self._schedule_outfit_greeting(outfit, serial)

    def cycle_mode(self) -> None:
        """只在三个公开模式间循环；秘密模式不属于普通换装列表。"""
        with self._lock:
            avatar = self.persona.appearance.avatar
            if avatar is None:
                return
            current_outfit = avatar.outfit
            if current_outfit.hidden:
                next_outfit_id = self._secret_return_outfit
            else:
                try:
                    idx = _MODE_ORDER.index(current_outfit.mode)
                except ValueError:
                    idx = -1
                next_mode = _MODE_ORDER[(idx + 1) % len(_MODE_ORDER)]
                next_outfit_id = next(
                    (
                        outfit.id
                        for outfit in avatar.outfits
                        if outfit.mode == next_mode and not outfit.hidden
                    ),
                    "",
                )
        if next_outfit_id:
            self._switch_outfit(next_outfit_id)

    def toggle_secret_mode(self) -> None:
        """隐藏序列触发；再次输入同一序列恢复进入前的公开套装。"""
        with self._lock:
            avatar = self.persona.appearance.avatar
            if avatar is None:
                return
            current = avatar.outfit
            if current.hidden:
                target = self._secret_return_outfit
            else:
                target = next(
                    (outfit.id for outfit in avatar.outfits if outfit.hidden),
                    "",
                )
                if target:
                    self._secret_return_outfit = current.id
        if target:
            self._switch_outfit(target)

    def cycle_outfit(self) -> None:
        """兼容旧调用名称；换装现在同时切换对应行为模式。"""
        self.cycle_mode()

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
            status_hint=self.partial_text or self.presence_hint or None,
            relationship_score=self.relationship_bond,
            relationship_tier=self.relationship_level,
            show_details=self.show_details,
            quiet_presence=self._idle_emotion_active,
            emotion_mood=self.emotion_mood,
            emotion_intensity=self.emotion_intensity,
            emotion_expression=self.emotion_expression,
            emotion_vector=self.emotion_vector,
            relationship_interaction_count=self.relationship_interaction_count,
            relationship_recent_events=self.relationship_recent_events,
            voice_emotion=self.voice_emotion,
            voice_event=self.voice_event,
            voice_language=self.voice_language,
            voice_observed_at=self._voice_observed_at,
            dev_mode=self.dev_mode,
            agency_state=self.agency_state,
            memory_presence=self.memory_presence,
            reflection_service=self.reflection_service,
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
            if self.interrupt_index is not None:
                self.interrupt_index = (
                    self.interrupt_index - overflow
                    if self.interrupt_index >= overflow
                    else None
                )
            self.scroll_offset = min(
                self.scroll_offset,
                max(0, len(self.messages) - 1),
            )
        return len(self.messages) - 1

    def remove(self, index: int) -> None:
        self.messages.pop(index)
        for attr in ("answer_index", "interrupt_index"):
            current = getattr(self, attr)
            if current is None:
                continue
            if current == index:
                setattr(self, attr, None)
            elif current > index:
                setattr(self, attr, current - 1)
        if self.scroll_offset:
            self.scroll_offset = max(0, self.scroll_offset - 1)

    def set_greeting(
        self,
        greeting: str,
        *,
        outfit_id: str | None = None,
    ) -> None:
        with self._lock:
            avatar = self.persona.appearance.avatar
            if (
                outfit_id is not None
                and avatar is not None
                and avatar.selected_outfit != outfit_id
            ):
                return
            self.base_greeting = greeting
            if not self._idle_emotion_active:
                self.greeting = greeting
                self.refresh()

    def set_relationship(
        self,
        bond: float,
        level: str,
        mood: str,
        inner_voice: str = "",
        *,
        interaction_count: int = 0,
        recent_events: tuple[str, ...] = (),
    ) -> None:
        """保存旁路结果；画外音只在空闲聆听状态安全切换。"""
        with self._lock:
            self.relationship_bond = bond
            self.relationship_level = level
            self.relationship_mood = mood
            self.relationship_interaction_count = interaction_count
            self.relationship_recent_events = recent_events
            if self._idle_emotion_active:
                self.presence_hint = _idle_presence_hint(
                    mood,
                    self._idle_emotion_index - 1,
                )
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

    def set_emotion(self, snap) -> None:
        """保存 EmotionSnapshot；只有安全窗口才触发重绘。

        `snap` 具备以下鸭子类型字段：mood / intensity / expression / emotion。
        """
        with self._lock:
            self.emotion_mood = snap.mood
            self.emotion_intensity = snap.intensity
            self.emotion_expression = snap.expression
            self.emotion_vector = getattr(snap, "emotion", None)
            safe_to_refresh = self.state == "listening" and not self.partial_text
            if safe_to_refresh:
                self.refresh()

    def set_voice_observation(self, emotion: str, event: str, language: str) -> None:
        """保存最近一次用户语气观察（含时间戳）；安全窗口内触发重绘。"""
        with self._lock:
            self.voice_emotion = emotion
            self.voice_event = event
            self.voice_language = language
            self._voice_observed_at = time.monotonic()
            safe_to_refresh = self.state == "listening" and not self.partial_text
            if safe_to_refresh:
                self.refresh()

    def _mark_voice_activity_locked(self, now: float) -> bool:
        self._last_voice_activity = now
        self._next_idle_emotion_at = now + config.IDLE_EMOTION_AFTER_S
        if not self._idle_emotion_active:
            return False
        self._idle_emotion_active = False
        self.presence_hint = ""
        self.greeting = self.base_greeting
        return True

    def mark_voice_activity(self, *, refresh: bool = True) -> None:
        """记录识别到的用户语音，并退出安静陪伴状态。"""
        with self._lock:
            changed = self._mark_voice_activity_locked(time.monotonic())
            if changed and refresh:
                self.refresh()

    def set_intentional_silence(self) -> None:
        """表现「听见了但选择不说」，区别于卡死或请求失败。"""
        with self._lock:
            self.state = "listening"
            self.partial_text = ""
            self._idle_emotion_active = True
            self.presence_hint = "她听见了，只是没有急着开口"
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
            index = self._idle_emotion_index
            self.greeting = _idle_emotion_line(self.relationship_mood, index)
            self.presence_hint = _idle_presence_hint(self.relationship_mood, index)
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
        """Tab 在欢迎区、状态页、开发页之间循环。"""
        with self._lock:
            if not self.show_details:
                self.show_details = True
                self.dev_mode = False
            elif not self.dev_mode:
                self.dev_mode = True
            else:
                self.show_details = False
                self.dev_mode = False
            self.live.update(self.render(), refresh=True)

    def set_agency(self, state) -> None:
        """保存 Agency 快照；下一次正常状态刷新时一起呈现。"""
        with self._lock:
            self.agency_state = state

    def set_memory_presence(self, presence) -> None:
        """保存长期记忆摘要；旁路更新不单独触发全屏重绘。"""
        with self._lock:
            self.memory_presence = presence

    def update(self, index: int, text: str) -> None:
        role, _ = self.messages[index]
        self.messages[index] = (role, text)
        self.refresh()

    def stop(self) -> None:
        self._idle_emotion_stop.set()
        if self._outfit_greeting_timer is not None:
            self._outfit_greeting_timer.cancel()
        self.input.stop()
        with self._lock:
            if self._native_frames_ready:
                self._stop_mouth_animation()
            self.live.stop()


def configure(persona: Persona) -> None:
    global _persona
    _persona = persona


def configure_outfit_greetings(
    generator: Callable[[AvatarOutfit], str | None] | None = None,
) -> None:
    """配置非阻塞换装台词生成器；本地短句始终可独立工作。"""
    global _outfit_greeting_generator
    _outfit_greeting_generator = generator


def configure_relationship(
    bond: float | None = None,
    level: str = "",
    mood: str = "calm",
    inner_voice: str = "",
    interaction_count: int = 0,
    recent_events: tuple[str, ...] = (),
) -> None:
    global _relationship_profile
    _relationship_profile = (
        (bond, level, mood, inner_voice, interaction_count, recent_events)
        if bond is not None
        else None
    )


def configure_emotion(service) -> None:
    """把 EmotionService 句柄交给 UI；splash 时会取一次初值。"""
    global _emotion_service
    _emotion_service = service


def configure_agency(state=None) -> None:
    """配置 Agency 初始快照；UI 不持有 AgencyService。"""
    global _agency_profile
    _agency_profile = state


def configure_memory(presence=None) -> None:
    """配置 Memory 初始摘要；None 表示能力未启用或不可用。"""
    global _memory_profile
    _memory_profile = presence


def configure_reflection(service=None) -> None:
    """开发详情只读观察 Reflection 队列，不参与其调度。"""
    global _reflection_service
    _reflection_service = service


def configure_presence(context: LaunchContext | None = None) -> None:
    global _launch_context
    _launch_context = context or LaunchContext()


def update_relationship(state, *, mood: str | None = None) -> None:
    """接收关系 Worker 的鸭子类型状态，避免 UI 依赖旁路实现。

    B 方案后关系状态不再持有 mood；调用方在每次旁路评估完成后，
    把当前 EmotionService 的 mood 显式传进来，dashboard 仍能据此
    渲染 idle presence hint。
    """
    if _dashboard is None:
        return
    resolved_mood = mood
    if resolved_mood is None and _emotion_service is not None:
        try:
            resolved_mood = _emotion_service.snapshot().mood
        except Exception:
            resolved_mood = "calm"
    if resolved_mood is None:
        resolved_mood = "calm"
    _dashboard.set_relationship(
        state.bond,
        state.level,
        resolved_mood,
        state.inner_voice,
        interaction_count=state.interaction_count,
        recent_events=state.recent_events,
    )


def update_emotion(snap) -> None:
    """接收 EmotionService 的快照；非 listening 状态下会被 dashboard 延迟刷新。"""
    if _dashboard is not None:
        _dashboard.set_emotion(snap)


def update_agency(state) -> None:
    global _agency_profile
    _agency_profile = state
    if _dashboard is not None:
        _dashboard.set_agency(state)


def update_memory(presence) -> None:
    global _memory_profile
    _memory_profile = presence
    if _dashboard is not None:
        _dashboard.set_memory_presence(presence)


def update_voice_observation(
    emotion: str, event: str, language: str
) -> None:
    """接收 VoiceStateService 的最新观察。"""
    if _dashboard is not None:
        _dashboard.set_voice_observation(emotion, event, language)


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


def _is_close_tier(tier: str) -> bool:
    """亲密阶段里"已经建立关系"那一档的判定。

    兼容旧的中文 tier 和新英文 level；中文 key 用 UI 翻译后的标签。
    """
    return tier in {
        "亲近", "默契", "灵魂共鸣",
        "熟悉", "相熟", "挚友",  # 新 UI 中文标签
        "familiar", "companion", "close", "bonded",
    }


def _fallback_greeting(
    hour: int | None = None,
    *,
    tier: str = "",
    repeat_launch: bool = False,
    special: bool = False,
) -> str:
    period = day_period(hour)
    close = _is_close_tier(tier)
    if special:
        return (
            "先别急，安静一会儿也可以。"
            if close
            else "今天让我先问一个问题，好吗？"
        )
    if repeat_launch:
        return "这么快就回来了？" if close else "欢迎回来。"
    greetings = {
        "早上": (
            "早上好，今天开始得很早。",
            "这么早？看来今天有事要做。",
        ),
        "中午": ("你好，想聊些什么？", "来了，今天想从哪里开始？"),
        "下午": ("你好，想聊些什么？", "来了，今天想从哪里开始？"),
        "晚上": ("晚上好。", "晚上好，今天过得怎么样？"),
        "夜深": ("已经很晚了。", "我就知道，这个时间你还没睡。"),
    }
    return greetings[period][1 if close else 0]


def _emotion_detail_text(vector) -> Text:
    """5 维情绪值 → 单行 dim 文本，给 Tab 详情行使用。"""
    text = Text(style="dim")
    if vector is None:
        return text
    parts: list[str] = []
    for dim in ("happiness", "calmness", "curiosity", "stress", "energy"):
        label = _EMOTION_DIM_LABELS[dim]
        raw = getattr(vector, dim, None)
        if raw is None and isinstance(vector, dict):
            raw = vector.get(dim)
        if raw is None:
            continue
        try:
            pct = int(round(float(raw) * 100))
        except (TypeError, ValueError):
            continue
        parts.append(f"{label} {pct}")
    if not parts:
        return text
    text.append("  " + " · ".join(parts))
    return text


_RELATIONSHIP_EVENT_DISPLAY_MAX = 40


def _relationship_detail_text(
    interaction_count: int,
    recent_events: tuple[str, ...],
) -> Text:
    """亲密详情行：关系事件次数 + 最近事件，给 Tab 详情行使用。

    默认只显示最近一次；超过 1 条时把更早的事件用「此前：N 次」汇总。
    最近事件最多显示 40 字（事件存储上限是 80 字），极长事件由 Rich
    的 overflow="ellipsis" 自动折叠成省略号，避免单行被拉爆。
    """
    text = Text(style="dim", no_wrap=True, overflow="ellipsis")
    bits: list[str] = []
    if interaction_count > 0:
        bits.append(f"共 {interaction_count} 次互动")
    if recent_events:
        last_event = recent_events[-1]
        truncated = last_event[:_RELATIONSHIP_EVENT_DISPLAY_MAX]
        suffix = "…" if len(last_event) > _RELATIONSHIP_EVENT_DISPLAY_MAX else ""
        bits.append(f"上次：{truncated}{suffix}")
        if len(recent_events) > 1:
            bits.append(f"此前：{len(recent_events) - 1} 次")
    if not bits:
        bits.append("初次见面")
    text.append("  " + " · ".join(bits))
    return text


def _voice_detail_text(
    voice_emotion: str,
    voice_event: str,
    voice_language: str,
    voice_observed_at: float | None = None,
) -> Text:
    """声音感知详情行：用户最近一句话的语气/事件/语言，给 Tab 详情行使用。

    只在有可显示的数据时构建文本；全空时返回空 Text() 不占行。
    超过 UI_TTL 则不显示。
    """
    if voice_observed_at is not None:
        if time.monotonic() - voice_observed_at > config.VOICE_STATE_UI_TTL_S:
            return Text()
    emotion_label = _VOICE_EMOTION_LABELS.get(voice_emotion, "")
    event_label = _VOICE_EVENT_LABELS.get(voice_event, "")
    bits = [b for b in (emotion_label, event_label, voice_language) if b]
    if not bits:
        return Text()
    text = Text(style="dim", no_wrap=True, overflow="ellipsis")
    text.append("◎ ", style="cyan")
    text.append("感知  ", style="dim")
    text.append(" · ".join(bits), style="cyan")
    return text


def _idle_emotion_line(mood: str, index: int) -> str:
    mood_lines = {
        "happy": ("刚才聊得挺开心的。", "我还在等你继续呢。"),
        "shy": ("安静一下也挺好的。", "我还在这里哦。"),
        "warm": ("有点想再听听你的声音。", "陪你安静一会儿也好。"),
        "concerned": ("不用着急，我在这里。", "慢慢来就好。"),
        "upset": ("让我先安静一会儿。", "等你想说的时候再说吧。"),
    }
    lines = mood_lines.get(mood, _IDLE_EMOTION_LINES)
    return lines[index % len(lines)]


def _idle_presence_hint(mood: str, index: int) -> str:
    mood_hints = {
        "happy": "她似乎还带着一点笑意",
        "shy": "她安静地移开了视线",
        "warm": "她并不急着让你开口",
        "concerned": "她耐心地等你整理心情",
        "upset": "她暂时安静了下来",
    }
    return mood_hints.get(
        mood,
        _IDLE_PRESENCE_HINTS[index % len(_IDLE_PRESENCE_HINTS)],
    )


def _unit_percent(value) -> int | None:
    try:
        return max(0, min(100, int(round(float(value) * 100))))
    except (TypeError, ValueError):
        return None


def _presence_state_label(
    mode: str,
    runtime_state: str,
    quiet_presence: bool,
    agency_state,
) -> str:
    if runtime_state == "thinking":
        return "思考中"
    if mode == "focused":
        return "专注中"
    if quiet_presence:
        return "安静中"
    if agency_state is None:
        return "陪伴中"
    energy = _unit_percent(getattr(agency_state, "social_energy", None))
    talk = _unit_percent(getattr(agency_state, "desire_to_talk", None))
    attention = str(getattr(agency_state, "attention", ""))
    if energy is not None and energy <= 15:
        return "休息中"
    if talk is not None and talk >= 65:
        return "想聊天"
    if (talk is not None and talk < 30) or attention == "inward":
        return "安静中"
    return "陪伴中"


def _agency_tendency(mode: str, agency_state) -> str:
    if agency_state is None:
        return "主动性未启用"
    talk = _unit_percent(getattr(agency_state, "desire_to_talk", None))
    energy = _unit_percent(getattr(agency_state, "social_energy", None))
    solitude = _unit_percent(getattr(agency_state, "solitude_need", None))
    if mode == "focused":
        return "更想专心陪你"
    if energy is not None and energy <= 20:
        return "更想安静陪着你"
    if talk is not None and talk >= 65:
        return "有些话想和你说"
    if solitude is not None and solitude >= 65:
        return "今天更想听你说"
    return "愿意陪你聊聊"


def _visible_voice_state(
    emotion: str,
    event: str,
    language: str,
    observed_at: float | None,
) -> tuple[str, str, str]:
    if (
        observed_at is not None
        and time.monotonic() - observed_at > config.VOICE_STATE_UI_TTL_S
    ):
        return "", "", ""
    languages = {
        "zh": "中文",
        "yue": "粤语",
        "en": "英语",
        "ja": "日语",
        "ko": "韩语",
    }
    return (
        _VOICE_EMOTION_LABELS.get(emotion, ""),
        _VOICE_EVENT_LABELS.get(event, ""),
        languages.get(language, ""),
    )


def _presence_details(
    persona: Persona,
    *,
    mode: str,
    state: str,
    quiet_presence: bool,
    emotion_mood: str,
    emotion_vector,
    relationship_tier: str,
    voice_emotion: str,
    voice_event: str,
    voice_language: str,
    voice_observed_at: float | None,
    agency_state,
    memory_presence,
) -> Text:
    """固定十行以内的用户态 Presence Panel，适配头像的稳定高度。"""
    primary = persona.appearance.primary_color
    accent = persona.appearance.accent_color
    text = Text(no_wrap=True, overflow="ellipsis")
    text.append(f"{persona.display_name.upper()} 状态\n", style=f"bold {primary}")

    presence = _presence_state_label(mode, state, quiet_presence, agency_state)
    text.append("◈ 当前状态  ", style=f"bold {accent}")
    text.append(f"{presence}\n", style="bold")
    talk = _unit_percent(getattr(agency_state, "desire_to_talk", None))
    energy = _unit_percent(getattr(agency_state, "social_energy", None))
    if talk is None or energy is None:
        text.append("  主动性未启用\n", style=_MUTED_TEXT_COLOR)
    else:
        text.append(
            f"  想聊天程度 {talk}%  ·  精力状态 {energy}%\n",
            style=_MUTED_TEXT_COLOR,
        )

    text.append("◌ 情绪  ", style=f"bold {accent}")
    emotion_parts: list[str] = []
    if emotion_vector is not None:
        for key in ("happiness", "calmness", "curiosity", "stress", "energy"):
            value = _unit_percent(getattr(emotion_vector, key, None))
            if value is not None:
                emotion_parts.append(f"{_EMOTION_DIM_LABELS[key]} {value}")
    text.append(
        ("  ".join(emotion_parts) if emotion_parts else _MOOD_LABELS.get(emotion_mood, emotion_mood))
        + "\n",
        style="bold",
    )

    tier = _RELATIONSHIP_LEVEL_ZH.get(relationship_tier, relationship_tier) or "初识阶段"
    experience_count = int(getattr(memory_presence, "experience_count", 0) or 0)
    text.append("♡ 关系  ", style=f"bold {accent}")
    text.append(tier, style=f"bold {primary}")
    if experience_count:
        text.append(f"  ·  共同经历 {experience_count} 件", style=_MUTED_TEXT_COLOR)
    text.append("\n")

    memory_count = getattr(memory_presence, "count", None)
    text.append("◆ 记忆  ", style=f"bold {accent}")
    if memory_count is None:
        text.append("长期记忆未启用\n", style=_MUTED_TEXT_COLOR)
    elif memory_count:
        text.append(f"长期记忆 {memory_count} 条\n", style="bold")
    else:
        text.append("尚无长期记忆\n", style=_MUTED_TEXT_COLOR)
    recent = str(getattr(memory_presence, "recent_recall", "") or "").strip()
    if recent:
        text.append("  最近想起  ", style=_MUTED_TEXT_COLOR)
        text.append(f"{recent[:42]}{'…' if len(recent) > 42 else ''}\n", style="italic")

    sensed_emotion, sensed_event, sensed_language = _visible_voice_state(
        voice_emotion,
        voice_event,
        voice_language,
        voice_observed_at,
    )
    sensed = [item for item in (sensed_emotion, sensed_event, sensed_language) if item]
    text.append("♪ 感知  ", style=f"bold {accent}")
    text.append("  ·  ".join(sensed) if sensed else _STATE_LABELS.get(state, "等待声音"))
    text.append("\n")

    pending = len(getattr(agency_state, "unresolved_thoughts", ()) or ())
    text.append("◎ 内在状态  ", style=f"bold {accent}")
    text.append(_agency_tendency(mode, agency_state), style="bold")
    if pending:
        text.append(f"  ·  未完成话题 {pending} 个", style=_MUTED_TEXT_COLOR)
    return text


def _developer_details(
    persona: Persona,
    *,
    runtime: RuntimeDetails,
    emotion_vector,
    relationship_score: float | None,
    relationship_tier: str,
    emotion_intensity: float,
    emotion_expression: str,
    agency_state,
    memory_presence,
    reflection_service,
) -> Text:
    """固定高度的开发摘要；不展示隐藏思考内容。"""
    primary = persona.appearance.primary_color
    text = Text(no_wrap=True, overflow="ellipsis")
    text.append("开发详情\n", style=f"bold {primary}")
    text.append(
        f"运行组件  本地 · {_short_model(runtime.model)} · {_tts_name(runtime)} · Sherpa-ONNX\n",
        style=_MUTED_TEXT_COLOR,
    )
    text.append(f"情绪向量  {emotion_vector}\n", style=_MUTED_TEXT_COLOR)
    text.append(
        f"关系分数  {relationship_score if relationship_score is not None else '未启用'}"
        f"  ·  阶段 {relationship_tier or '未启用'}\n",
        style=_MUTED_TEXT_COLOR,
    )
    text.append(
        f"情绪强度  {emotion_intensity:.2f}  ·  表达 {emotion_expression}\n",
        style=_MUTED_TEXT_COLOR,
    )
    if agency_state is None:
        text.append("主动性  未启用\n", style=_MUTED_TEXT_COLOR)
    else:
        text.append(
            "主动性  "
            f"交流 {getattr(agency_state, 'desire_to_talk', 0):.2f}  ·  "
            f"精力 {getattr(agency_state, 'social_energy', 0):.2f}  ·  "
            f"独处 {getattr(agency_state, 'solitude_need', 0):.2f}\n",
            style=_MUTED_TEXT_COLOR,
        )
        text.append(
            "参与节奏  "
            f"债务 {getattr(agency_state, 'initiative_debt', 0):.2f}  ·  "
            f"连续被动 {getattr(agency_state, 'passive_answer_streak', 0)}  ·  "
            f"未完话题 {len(getattr(agency_state, 'unresolved_thoughts', ()) or ())}\n",
            style=_MUTED_TEXT_COLOR,
        )
    text.append(
        f"记忆  {getattr(memory_presence, 'count', 0) if memory_presence is not None else '未启用'} 条"
        f"  ·  最新编号 {getattr(memory_presence, 'latest_id', None) or '无'}\n",
        style=_MUTED_TEXT_COLOR,
    )
    queue_size = "未启用"
    buffer_size = "未启用"
    if reflection_service is not None:
        try:
            diagnostics = reflection_service.diagnostics()
            queue_size = diagnostics["queue_size"]
            buffer_size = diagnostics["memory_buffer_size"]
        except Exception:
            queue_size = buffer_size = "不可用"
    text.append(
        f"反思旁路  队列 {queue_size}  ·  记忆缓冲 {buffer_size}\n",
        style=_MUTED_TEXT_COLOR,
    )
    text.append("仅显示状态摘要，不展示内部推理。", style="dim italic")
    return text


def _splash_panel(
    persona: Persona,
    runtime: RuntimeDetails,
    stage: int,
    avatar: Text | None = None,
    native_avatar: bool = False,
    state: str = "idle",
    greeting: str | None = None,
    status_hint: str | None = None,
    relationship_score: float | None = None,
    relationship_tier: str = "",
    show_details: bool = False,
    dev_mode: bool = False,
    quiet_presence: bool = False,
    emotion_mood: str = "calm",
    emotion_intensity: float = 0.0,
    emotion_expression: str = "neutral",
    emotion_vector=None,
    relationship_interaction_count: int = 0,
    relationship_recent_events: tuple[str, ...] = (),
    voice_emotion: str = "",
    voice_event: str = "",
    voice_language: str = "",
    voice_observed_at: float | None = None,
    agency_state=None,
    memory_presence=None,
    reflection_service=None,
) -> Panel:
    primary = persona.appearance.primary_color

    # ── 标题 ──
    title = Text(
        persona.display_name.upper() if stage >= 2 else "",
        style=f"bold {primary}",
    )
    greeting_text = Text(no_wrap=True, overflow="ellipsis")
    if stage >= 2:
        greeting_text.append(
            f"“{greeting or _fallback_greeting()}”", style="italic",
        )

    # ── 状态行 ──
    status = Text()
    if stage >= 3:
        symbols = {"idle": "○", "listening": "◉", "thinking": "◇", "speaking": "≋"}
        status_label = _STATE_LABELS.get(state, "待机中")
        if quiet_presence and state == "listening":
            status_label = "安静陪伴"
        status.append(f"{symbols.get(state, '○')} {status_label}", style=f"bold {primary}")

    hint = Text(
        status_hint
        or {
            "idle": "随时可以开始",
            "listening": "直接说话即可",
            "thinking": "我正在认真想",
            "speaking": "正在把回答说给你听",
        }.get(state, "随时可以开始"),
        style="dim",
        no_wrap=True,
        overflow="ellipsis",
    )

    if show_details:
        detail_text = (
            _developer_details(
                persona,
                runtime=runtime,
                emotion_vector=emotion_vector,
                relationship_score=relationship_score,
                relationship_tier=relationship_tier,
                emotion_intensity=emotion_intensity,
                emotion_expression=emotion_expression,
                agency_state=agency_state,
                memory_presence=memory_presence,
                reflection_service=reflection_service,
            )
            if dev_mode
            else _presence_details(
                persona,
                mode=(
                    persona.appearance.avatar.outfit.mode
                    if persona.appearance.avatar is not None
                    else "companion"
                ),
                state=state,
                quiet_presence=quiet_presence,
                emotion_mood=emotion_mood,
                emotion_vector=emotion_vector,
                relationship_tier=relationship_tier,
                voice_emotion=voice_emotion,
                voice_event=voice_event,
                voice_language=voice_language,
                voice_observed_at=voice_observed_at,
                agency_state=agency_state,
                memory_presence=memory_presence,
            )
        )
        # 详情页替换欢迎文案，而不是在其下方不断追加内容；这样宽屏头像
        # 与右栏都稳定在同一高度，Tab 不会挤掉对话区。
        all_details = Align.center(detail_text, vertical="middle", height=13)
    else:
        emotion = Text()
        emotion.append("◌ ", style=primary)
        emotion.append("情绪  ", style="dim")
        emotion.append(
            _MOOD_LABELS.get(emotion_mood, emotion_mood), style=primary
        )
        relationship = Text()
        if relationship_score is not None:
            relationship.append("♡ ", style=primary)
            relationship.append("关系  ", style="dim")
            relationship.append(
                _RELATIONSHIP_LEVEL_ZH.get(relationship_tier, relationship_tier),
                style=primary,
            )
        all_details = Group(
            Align.center(title),
            Align.center(greeting_text),
            Text(""),
            Align.center(status),
            Align.center(hint),
            Text(""),
            Align.center(emotion),
            Align.center(relationship),
            Text(""),
            _technical_profile(persona, runtime, stage, expanded=False),
        )
    wide_avatar = avatar is not None and _console.width >= 82
    if wide_avatar:
        content = Table.grid(expand=True, padding=(0, 2))
        content.add_column(width=persona.appearance.avatar.width + 2)
        content.add_column(ratio=1)
        content.add_row(
            Align.center(avatar, vertical="middle"),
            Align.center(all_details, vertical="middle", height=13),
        )
    elif avatar is not None:
        content = Group(Align.center(avatar), Text(""), all_details)
    elif native_avatar:
        content = all_details
    else:
        content = Group(
            Align.center(_logo(persona, stage)),
            Text(""),
            all_details,
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
        subtitle="[dim]直接说话 · 0 换装 · Tab 状态/开发 · Ctrl+C 退出[/dim]",
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


def update_greeting(greeting: str, *, outfit_id: str | None = None) -> None:
    if _dashboard is not None:
        _dashboard.set_greeting(greeting, outfit_id=outfit_id)


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


def user_text(text: str, *, interrupted: bool = False) -> None:
    if _dashboard is not None:
        _dashboard.mark_voice_activity(refresh=False)
        _dashboard.partial_text = ""
        if interrupted and _dashboard.interrupt_index is not None:
            index = _dashboard.interrupt_index
            _dashboard.interrupt_index = None
            _dashboard.update(index, f"[打断] {text}")
            return
        # 上一次打断若没有可用 FINAL，不允许它影响下一条普通输入。
        _dashboard.interrupt_index = None
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
            _dashboard.remove(_dashboard.answer_index)
        _answer_pending = False
        _dashboard.answer_index = None
        # 此时只表示 LLM 文本流结束，TTS 队列可能仍在播放。聆听状态由
        # half/full duplex 的播放完成边界负责恢复，不能在这里提前闭嘴。
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


def intentional_silence() -> None:
    """Agency 主动选择 SILENCE；不伪装成空回答或系统错误。"""
    if _dashboard is not None:
        _dashboard.set_intentional_silence()
        return
    accent = _current().appearance.accent_color
    _console.print(f"  [{accent}]…[/{accent}] [dim]安静陪伴[/dim]")


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
        if _dashboard.interrupt_index is None:
            _dashboard.interrupt_index = _dashboard.add(
                "you",
                f"[打断] {text}",
            )
        else:
            _dashboard.update(
                _dashboard.interrupt_index,
                f"[打断] {text}",
            )
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
