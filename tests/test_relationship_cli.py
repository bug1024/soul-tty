"""soul-tty relationship 子命令测试。"""

import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from soul_tty import config
from soul_tty.reflection.cli import run_relationship
from soul_tty.reflection.relationship import (
    RelationshipState,
    load_state,
    save_state,
    state_path,
)


def test_show_reads_current_persona_state():
    with TemporaryDirectory() as tmp:
        state_root = Path(tmp)
        path = state_path("serena", state_root)
        save_state(
            path,
            RelationshipState(
                bond=0.72,
                interaction_count=8,
                recent_events=("一起调试语音",),
            ),
        )
        output = io.StringIO()
        with patch("soul_tty.reflection.cli.config.SOUL_TTY_STATE_DIR", state_root):
            with redirect_stdout(output):
                rc = run_relationship(["show"], persona_id="serena")

        assert rc == 0
        assert "0.72" in output.getvalue()
        assert "8" in output.getvalue()
        assert "一起调试语音" in output.getvalue()


def test_clear_yes_resets_only_selected_persona():
    with TemporaryDirectory() as tmp:
        state_root = Path(tmp)
        serena_path = state_path("serena", state_root)
        other_path = state_path("luna", state_root)
        save_state(serena_path, RelationshipState(bond=0.82, interaction_count=12))
        save_state(other_path, RelationshipState(bond=0.55, interaction_count=4))

        with patch("soul_tty.reflection.cli.config.SOUL_TTY_STATE_DIR", state_root):
            with patch("builtins.input", return_value="y"):
                rc = run_relationship(["clear"], persona_id="serena")

        assert rc == 0
        assert not serena_path.exists()
        assert other_path.exists()
        assert load_state(serena_path).bond == config.RELATIONSHIP_INITIAL_BOND


def test_clear_no_keeps_relationship_state():
    with TemporaryDirectory() as tmp:
        state_root = Path(tmp)
        path = state_path("serena", state_root)
        save_state(path, RelationshipState(bond=0.82, interaction_count=12))

        with patch("soul_tty.reflection.cli.config.SOUL_TTY_STATE_DIR", state_root):
            with patch("builtins.input", return_value="n"):
                rc = run_relationship(["clear"], persona_id="serena")

        assert rc == 0
        assert path.exists()
        assert load_state(path).bond == 0.82


def test_clear_empty_state_is_idempotent():
    with TemporaryDirectory() as tmp:
        state_root = Path(tmp)
        with patch("soul_tty.reflection.cli.config.SOUL_TTY_STATE_DIR", state_root):
            with patch("builtins.input") as prompt:
                rc = run_relationship(["clear"], persona_id="serena")

        assert rc == 0
        prompt.assert_not_called()
