"""Chafa 头像渲染：原生像素协议 -> ANSI 像素画 -> 无头像。"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text

from ..personas.models import Persona

_NATIVE_MARKERS = (b"\033_G", b"\033]1337;", b"\033Pq", b"\033P0;")
_KITTY_IMAGE_NUMBER = 0x564F4943  # "VOIC"，仅用于当前 alternate screen。
_KITTY_IMAGE_PLACEMENT_ID = 1
_KITTY_SPEAKING_IMAGE_NUMBER = 0x5350454B  # "SPEK"
_KITTY_SPEAKING_FRAME_NUMBERS = tuple(
    _KITTY_SPEAKING_IMAGE_NUMBER + index for index in range(2)
)
_KITTY_SPEAKING_PLACEMENT_ID = 1
_KITTY_APC = re.compile(rb"\033_G(.*?)\033\\", re.DOTALL)


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
    enabled = os.environ.get("SOUL_TTY_AVATAR", "1") not in {
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
        "SOUL_TTY_AVATAR_RENDERER", avatar.renderer
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
        path = Path(avatar.for_state("idle"))
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
    if render.protocol == "kitty":
        size = _kitty_display_size(render)
        hidden = _rewrite_kitty_hidden_payload(
            render,
            image_number=_KITTY_IMAGE_NUMBER,
        )
        if size is not None and hidden is not None:
            width, _ = size
            # 静态与说话状态统一采用隐藏传输 + 固定 placement，避免
            # Ghostty 对 a=T 和 a=p 使用不同的显示几何。
            buffer.write(
                f"\033_Ga=d,d=N,I={_KITTY_IMAGE_NUMBER},q=2\033\\".encode(
                    "ascii"
                )
            )
            buffer.write(hidden)
            buffer.write(f"\033[{row};{column}H".encode("ascii"))
            buffer.write(
                (
                    f"\033_Ga=p,I={_KITTY_IMAGE_NUMBER},"
                    f"p={_KITTY_IMAGE_PLACEMENT_ID},c={width},"
                    f"C=1,q=2\033\\"
                ).encode("ascii")
            )
            buffer.write(b"\0338")
            buffer.flush()
            return True
    buffer.write(f"\033[{row};{column}H".encode("ascii"))
    buffer.write(payload)
    buffer.write(b"\0338")
    buffer.flush()
    return True


def hide_native_avatar(output) -> bool:
    """隐藏当前普通状态头像，保留图片数据直到下一次状态重绘。"""
    buffer = getattr(output, "buffer", None)
    if buffer is None:
        return False
    buffer.write(
        (
            f"\033_Ga=d,d=n,I={_KITTY_IMAGE_NUMBER},"
            f"p={_KITTY_IMAGE_PLACEMENT_ID},q=2\033\\"
        ).encode("ascii")
    )
    buffer.flush()
    return True


def _kitty_display_size(render: AvatarRender) -> tuple[int, int] | None:
    if render.native is None or render.protocol != "kitty":
        return None
    match = _KITTY_APC.search(render.native)
    if match is None:
        return None
    controls = match.group(1).partition(b";")[0]
    values = dict(
        part.split(b"=", 1) for part in controls.split(b",") if b"=" in part
    )
    try:
        width, height = int(values[b"c"]), int(values[b"r"])
    except (KeyError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _rewrite_kitty_hidden_payload(
    render: AvatarRender,
    *,
    image_number: int,
) -> bytes | None:
    """把 Chafa 的直接显示分块改写为带稳定编号的隐藏图片。"""
    if render.native is None or render.protocol != "kitty":
        return None
    first = True

    def rewrite(match: re.Match[bytes]) -> bytes:
        nonlocal first
        body = match.group(1)
        controls, separator, data = body.partition(b";")
        parts = controls.split(b",") if controls else []
        if first:
            first = False
            kept = []
            for part in parts:
                key = part.split(b"=", 1)[0]
                if key not in {b"a", b"c", b"r", b"i", b"I", b"p", b"z", b"X"}:
                    kept.append(part)
            prefix = [
                b"a=t",
                f"I={image_number}".encode("ascii"),
            ]
            controls = b",".join([*prefix, *kept])
        rewritten = controls + (separator + data if separator else b"")
        return b"\033_G" + rewritten + b"\033\\"

    payload, count = _KITTY_APC.subn(rewrite, render.native.rstrip(b"\r\n"))
    return payload if count and not first else None


def prepare_native_frames(
    frames: tuple[AvatarRender, AvatarRender],
    output,
) -> bool:
    """缓存两张干净的普通 Kitty 图片；不创建会改变显示几何的动画对象。"""
    payloads = [
        _rewrite_kitty_hidden_payload(
            frame,
            image_number=number,
        )
        for frame, number in zip(frames, _KITTY_SPEAKING_FRAME_NUMBERS)
    ]
    if any(payload is None for payload in payloads):
        return False
    buffer = getattr(output, "buffer", None)
    if buffer is None:
        return False
    flush = getattr(output, "flush", None)
    if flush is not None:
        flush()
    for number, payload in zip(_KITTY_SPEAKING_FRAME_NUMBERS, payloads):
        buffer.write(f"\033_Ga=d,d=N,I={number},q=2\033\\".encode("ascii"))
        buffer.write(payload)
    buffer.flush()
    return True


def show_native_frame_at(
    output,
    frame: int,
    *,
    row: int,
    column: int,
    width: int,
) -> bool:
    """切换缓存图片；只限定宽度，由终端按源图比例计算高度。"""
    if frame not in range(len(_KITTY_SPEAKING_FRAME_NUMBERS)):
        return False
    if min(row, column, width) < 1:
        return False
    buffer = getattr(output, "buffer", None)
    if buffer is None:
        return False
    placement = _KITTY_SPEAKING_PLACEMENT_ID
    for number in _KITTY_SPEAKING_FRAME_NUMBERS:
        buffer.write(
            f"\033_Ga=d,d=n,I={number},p={placement},q=2\033\\".encode(
                "ascii"
            )
        )
    number = _KITTY_SPEAKING_FRAME_NUMBERS[frame]
    buffer.write(b"\0337")
    buffer.write(f"\033[{row};{column}H".encode("ascii"))
    buffer.write(
        (
            f"\033_Ga=p,I={number},p={placement},c={width},"
            "C=1,q=2\033\\"
        ).encode("ascii")
    )
    buffer.write(b"\0338")
    buffer.flush()
    return True


def hide_native_frames(output) -> bool:
    """隐藏客户端口型帧，保留缓存数据供下一轮说话复用。"""
    buffer = getattr(output, "buffer", None)
    if buffer is None:
        return False
    placement = _KITTY_SPEAKING_PLACEMENT_ID
    for number in _KITTY_SPEAKING_FRAME_NUMBERS:
        buffer.write(
            f"\033_Ga=d,d=n,I={number},p={placement},q=2\033\\".encode(
                "ascii"
            )
        )
    buffer.flush()
    return True
