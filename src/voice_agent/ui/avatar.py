"""Chafa 头像渲染：原生像素协议 -> ANSI 像素画 -> 无头像。"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text

from ..personas.models import Persona

_NATIVE_MARKERS = (b"\033_G", b"\033]1337;", b"\033Pq", b"\033P0;")
_KITTY_IMAGE_NUMBER = 0x564F4943  # "VOIC"，仅用于当前 alternate screen。


@dataclass(frozen=True)
class AvatarRender:
    native: bytes | None = None
    symbols: Text | None = None
    mode: str = "off"
    protocol: str | None = None


def _native_format() -> str | None:
    term = os.environ.get("TERM", "").lower()
    program = os.environ.get("TERM_PROGRAM", "").lower()
    if (
        os.environ.get("KITTY_WINDOW_ID")
        or os.environ.get("GHOSTTY_RESOURCES_DIR")
        or "kitty" in term
        or program in {"kitty", "ghostty"}
    ):
        return "kitty"
    if os.environ.get("WEZTERM_PANE") or program in {"iterm.app", "wezterm"}:
        return "iterm"
    return None


def _run_chafa(path: Path, width: int, output_format: str) -> bytes:
    command = [
        shutil.which("chafa") or "chafa",
        "--size",
        f"{width}x{max(8, width // 2)}",
        "--animate",
        "off",
        "--polite",
        "on",
        "--relative",
        "off",
        "--threads",
        "2",
        "--work",
        "5",
    ]
    command.extend(["--format", output_format, "--probe", "off"])
    if output_format == "symbols":
        command.extend(
            [
                "--colors",
                "full",
                "--symbols",
                "vhalf",
            ]
        )
    command.append(str(path))
    result = subprocess.run(
        command,
        capture_output=True,
        check=True,
        timeout=2,
    )
    return result.stdout


def _is_native(payload: bytes) -> bool:
    return any(marker in payload for marker in _NATIVE_MARKERS)


def render_avatar(
    persona: Persona,
    state: str,
    terminal: bool,
    renderer_override: str | None = None,
) -> AvatarRender:
    avatar = persona.appearance.avatar
    enabled = os.environ.get("VOICE_AGENT_AVATAR", "1") not in {
        "0",
        "false",
        "False",
    }
    if (
        not terminal
        or not enabled
        or os.environ.get("NO_COLOR")
        or avatar is None
    ):
        return AvatarRender()
    configured_renderer = os.environ.get(
        "VOICE_AGENT_AVATAR_RENDERER", avatar.renderer
    ).lower()
    # renderer_override 只选择布局所需的输出格式，不能绕过用户的关闭设置。
    if configured_renderer == "off":
        return AvatarRender()
    renderer = (renderer_override or configured_renderer).lower()
    if renderer not in {"auto", "pixels", "symbols", "off"}:
        renderer = avatar.renderer
    if renderer == "off":
        return AvatarRender()
    if shutil.which("chafa") is None:
        return AvatarRender()

    path = Path(avatar.for_state(state))
    if not path.is_file():
        path = Path(avatar.idle)
    if not path.is_file():
        return AvatarRender()

    try:
        native_format = None if renderer == "symbols" else _native_format()
        output_format = native_format or "symbols"
        payload = _run_chafa(path, avatar.width, output_format)
        if output_format != "symbols" and _is_native(payload):
            return AvatarRender(
                native=payload,
                mode="pixels",
                protocol=output_format,
            )
        if output_format == "symbols" and payload:
            return AvatarRender(
                symbols=Text.from_ansi(payload.decode("utf-8", errors="replace")),
                mode="symbols",
            )
    except (OSError, subprocess.SubprocessError):
        pass

    if renderer in {"auto", "pixels"}:
        try:
            payload = _run_chafa(path, avatar.width, "symbols")
            return AvatarRender(
                symbols=Text.from_ansi(payload.decode("utf-8", errors="replace")),
                mode="symbols",
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return AvatarRender()


def write_native(render: AvatarRender, output) -> bool:
    if render.native is None:
        return False
    buffer = getattr(output, "buffer", None)
    if buffer is None:
        return False
    flush = getattr(output, "flush", None)
    if flush is not None:
        flush()
    buffer.write(render.native)
    buffer.flush()
    return True


def write_native_at(
    render: AvatarRender,
    output,
    *,
    row: int,
    column: int,
) -> bool:
    """在 alternate screen 的固定单元格绘制图片，并恢复文本光标。"""
    if render.native is None or row < 1 or column < 1:
        return False
    buffer = getattr(output, "buffer", None)
    if buffer is None:
        return False
    flush = getattr(output, "flush", None)
    if flush is not None:
        flush()
    # Chafa 在 payload 末尾添加换行；固定定位时不需要它，否则可能滚动画布。
    payload = render.native.rstrip(b"\r\n")
    buffer.write(b"\0337")
    buffer.write(f"\033[{row};{column}H".encode("ascii"))
    if render.protocol == "kitty":
        # 删除 voice-agent 上一次放置的图像及数据，避免长时间对话累积显存。
        buffer.write(
            f"\033_Ga=d,d=N,I={_KITTY_IMAGE_NUMBER},q=2\033\\".encode("ascii")
        )
        payload = payload.replace(
            b"\033_Ga=T,",
            f"\033_Ga=T,I={_KITTY_IMAGE_NUMBER},".encode("ascii"),
            1,
        )
    buffer.write(payload)
    buffer.write(b"\0338")
    buffer.flush()
    return True


def _kitty_frame_payload(render: AvatarRender) -> bytes | None:
    """把 Chafa 的 Kitty 根图传输改写为同一图像的动画帧传输。"""
    if render.native is None or render.protocol != "kitty":
        return None
    payload = render.native.rstrip(b"\r\n")
    prefix = b"\033_G"
    start = payload.find(prefix)
    separator = payload.find(b";", start + len(prefix))
    if start < 0 or separator < 0:
        return None
    controls = payload[start + len(prefix) : separator].split(b",")
    kept = []
    for control in controls:
        key = control.split(b"=", 1)[0]
        if key not in {b"a", b"c", b"r", b"i", b"I", b"p", b"z"}:
            kept.append(control)
    header = b",".join(
        [b"a=f", f"I={_KITTY_IMAGE_NUMBER}".encode("ascii"), *kept]
    )
    return payload[: start + len(prefix)] + header + payload[separator:]


def write_native_animation_at(
    frames: tuple[AvatarRender, AvatarRender, AvatarRender],
    output,
    *,
    row: int,
    column: int,
) -> bool:
    """一次上传闭/半开/张嘴三帧；后续只需发送很小的选帧指令。"""
    if not all(frame.protocol == "kitty" for frame in frames):
        return False
    if not write_native_at(frames[0], output, row=row, column=column):
        return False
    extra = [_kitty_frame_payload(frame) for frame in frames[1:]]
    if any(payload is None for payload in extra):
        return False
    buffer = getattr(output, "buffer", None)
    if buffer is None:
        return False
    for payload in extra:
        buffer.write(payload)
    buffer.flush()
    return True


def select_native_animation_frame(output, frame: int) -> bool:
    """选择 Kitty 动画帧：1=闭嘴，2=半开，3=张嘴。"""
    if frame not in {1, 2, 3}:
        return False
    buffer = getattr(output, "buffer", None)
    if buffer is None:
        return False
    buffer.write(
        f"\033_Ga=a,I={_KITTY_IMAGE_NUMBER},c={frame},q=2\033\\".encode("ascii")
    )
    buffer.flush()
    return True
