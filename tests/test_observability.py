"""结构化日志：文件输出、上下文关联与异常字段。"""

from __future__ import annotations

import json
import logging
import threading

from soul_tty import config, observability


def test_jsonl_log_contains_session_turn_warning_and_exception(tmp_path, monkeypatch):
    log_file = tmp_path / "soul-tty.jsonl"
    monkeypatch.setattr(config, "SOUL_TTY_LOG_ENABLED", True)
    monkeypatch.setattr(config, "SOUL_TTY_LOG_FILE", log_file)
    monkeypatch.setattr(config, "SOUL_TTY_LOG_LEVEL", "INFO")
    monkeypatch.setattr(config, "SOUL_TTY_LOG_MAX_BYTES", 1024 * 1024)
    monkeypatch.setattr(config, "SOUL_TTY_LOG_BACKUP_COUNT", 1)
    monkeypatch.setattr(observability, "_CONFIGURED_PATH", None)

    root = logging.getLogger()
    before = list(root.handlers)
    previous_thread_hook = threading.excepthook
    try:
        assert observability.configure() == log_file
        with observability.bind_turn("turn-test"):
            observability.event("llm.first_token", duration_ms=123.45)
            logging.getLogger("soul_tty.test").warning("测试 warning")
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                observability.exception("test.error", "测试 error")

        for handler in root.handlers:
            handler.flush()
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()
        threading.excepthook = previous_thread_hook
        monkeypatch.setattr(observability, "_CONFIGURED_PATH", None)

    rows = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "llm.first_token",
        "log",
        "test.error",
    ]
    assert all(row["session_id"] == observability.session_id() for row in rows)
    assert all(row["turn_id"] == "turn-test" for row in rows)
    assert rows[0]["duration_ms"] == 123.45
    assert rows[1]["level"] == "WARNING"
    assert "RuntimeError: boom" in rows[2]["exception"]


def test_turn_ids_are_monotonic_and_context_is_restored():
    before = observability.current_turn_id()
    first = observability.new_turn_id()
    second = observability.new_turn_id()
    assert int(second.split("-")[-1]) == int(first.split("-")[-1]) + 1

    with observability.bind_turn("turn-local"):
        assert observability.current_turn_id() == "turn-local"
    assert observability.current_turn_id() == before
