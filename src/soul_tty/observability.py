"""Soul TTY 结构化运行日志与轻量性能埋点。

日志默认写入 JSONL 文件，绝不写交互终端。每条记录自动携带 session_id、
turn_id 和线程名；业务代码只需调用 ``event()``，无需关心 formatter。
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import logging.handlers
import threading
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import config

_SESSION_ID = f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
_TURN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "soul_tty_turn_id", default="-"
)
_TURN_SEQUENCE = 0
_TURN_LOCK = threading.Lock()
_CONFIGURED_PATH: Path | None = None
_LOGGER = logging.getLogger("soul_tty")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event_name", "log"),
            "session_id": _SESSION_ID,
            "turn_id": getattr(record, "turn_id", None) or _TURN_ID.get(),
            "thread": record.threadName,
            "message": record.getMessage(),
        }
        fields = getattr(record, "event_fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure() -> Path | None:
    """配置 root file handler；幂等，返回实际日志路径。"""
    global _CONFIGURED_PATH
    if not config.SOUL_TTY_LOG_ENABLED:
        return None
    if _CONFIGURED_PATH is not None:
        return _CONFIGURED_PATH

    path = config.SOUL_TTY_LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=config.SOUL_TTY_LOG_MAX_BYTES,
        backupCount=config.SOUL_TTY_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(_JsonFormatter())
    handler.setLevel(getattr(logging, config.SOUL_TTY_LOG_LEVEL, logging.INFO))

    root = logging.getLogger()
    root.setLevel(min(root.level or logging.WARNING, handler.level))
    root.addHandler(handler)
    # 性能节点由 Soul TTY 自己记录；依赖库的成功请求会制造大量重复 INFO。
    # 保留它们的 warning/error 即可。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        _LOGGER.error(
            "后台线程未捕获异常",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            extra={
                "event_name": "error.thread_uncaught",
                "event_fields": {"failed_thread": args.thread.name},
            },
        )

    threading.excepthook = thread_hook
    _CONFIGURED_PATH = path
    return path


def session_id() -> str:
    return _SESSION_ID


def current_turn_id() -> str:
    return _TURN_ID.get()


def new_turn_id() -> str:
    global _TURN_SEQUENCE
    with _TURN_LOCK:
        _TURN_SEQUENCE += 1
        return f"turn-{_TURN_SEQUENCE:04d}"


@contextlib.contextmanager
def bind_turn(turn_id: str) -> Iterator[None]:
    token = _TURN_ID.set(turn_id)
    try:
        yield
    finally:
        _TURN_ID.reset(token)


def event(
    name: str,
    *,
    level: int = logging.INFO,
    message: str | None = None,
    turn_id: str | None = None,
    **fields: Any,
) -> None:
    """写一条结构化事件；字段应避免包含完整对话和密钥。"""
    _LOGGER.log(
        level,
        message or name,
        extra={
            "event_name": name,
            "event_fields": fields,
            "turn_id": turn_id or _TURN_ID.get(),
        },
    )


def exception(name: str, message: str, **fields: Any) -> None:
    _LOGGER.exception(
        message,
        extra={
            "event_name": name,
            "event_fields": fields,
            "turn_id": _TURN_ID.get(),
        },
    )


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
