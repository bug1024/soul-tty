import io
import unittest
from unittest.mock import patch

from rich.cells import cell_len
from rich.console import Console

from voice_agent.personas import load_persona
from voice_agent.ui import terminal


class TerminalUITests(unittest.TestCase):
    def tearDown(self):
        terminal._dashboard = None

    def test_splash_name_and_tagline_are_independently_centered(self):
        output = io.StringIO()
        console = Console(
            file=output,
            width=80,
            color_system=None,
            force_terminal=False,
        )
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B-Q4_K_M.gguf", tts=None)
        with patch.object(terminal, "_console", console):
            console.print(terminal._splash_panel(persona, runtime, 3))

        lines = output.getvalue().splitlines()
        name_line = next(line for line in lines if "SERENA" in line)
        tagline_line = next(line for line in lines if persona.tagline in line)

        def center(line: str, text: str) -> float:
            before = line.split(text, 1)[0]
            return cell_len(before) + cell_len(text) / 2

        self.assertAlmostEqual(
            center(name_line, "SERENA"),
            center(tagline_line, persona.tagline),
            delta=1,
        )

    def test_answer_header_returns_to_fixed_left_indent(self):
        output = io.StringIO()
        console = Console(
            file=output,
            width=80,
            color_system=None,
            force_terminal=True,
        )
        terminal.configure(load_persona("serena"))
        with patch.object(terminal, "_console", console):
            terminal.answer_start()
            terminal.answer_chunk("你好")

        after_clear = output.getvalue().rsplit("\033[2K", 1)[-1]
        self.assertTrue(after_clear.startswith("  SERENA\n  你好"))

    def test_dashboard_loads_and_switches_all_avatar_states(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        calls = []

        def render(persona, state, terminal_enabled, renderer_override=None):
            calls.append((state, renderer_override))
            return terminal.avatar_ui.AvatarRender(symbols=terminal.Text(state))

        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.avatar_ui, "render_avatar", side_effect=render),
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard.set_state("listening")
            dashboard.set_state("thinking")
            dashboard.set_state("speaking")

        self.assertEqual(
            calls,
            [
                *((state, "pixels") for state in terminal._STATE_LABELS),
                ("speaking_closed", "pixels"),
                ("speaking_half", "pixels"),
                ("speaking_open", "pixels"),
            ],
        )
        self.assertEqual(dashboard.state, "speaking")
        self.assertEqual(
            dashboard.avatars["listening"].symbols.plain,
            "listening",
        )

    def test_dashboard_paints_native_avatar_in_reserved_card_area(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")

        def render(persona, state, terminal_enabled, renderer_override=None):
            return terminal.avatar_ui.AvatarRender(
                native=f"native-{state}".encode(),
                mode="pixels",
                protocol="kitty",
            )

        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.avatar_ui, "render_avatar", side_effect=render),
            patch.object(terminal.avatar_ui, "write_native_at") as paint,
            patch.object(
                terminal.avatar_ui,
                "write_native_animation_at",
                return_value=True,
            ) as paint_animation,
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard.set_state("speaking")

        paint_animation.assert_called_once_with(
            dashboard.mouth_avatars,
            console.file,
            row=3,
            column=7,
        )
        paint.assert_not_called()

    def test_dashboard_maps_audio_level_to_preloaded_native_frame(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")

        def render(persona, state, terminal_enabled, renderer_override=None):
            return terminal.avatar_ui.AvatarRender(
                native=state.encode(), mode="pixels", protocol="kitty"
            )

        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.avatar_ui, "render_avatar", side_effect=render),
            patch.object(terminal.avatar_ui, "write_native_animation_at", return_value=True),
            patch.object(terminal.avatar_ui, "select_native_animation_frame") as select,
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard.set_state("speaking")
            dashboard.set_mouth_level(1.0)

        select.assert_called_once_with(console.file, 3)
        self.assertEqual(dashboard.mouth_frame, 3)


if __name__ == "__main__":
    unittest.main()
