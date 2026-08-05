import io
import unittest
from unittest.mock import patch

from rich.cells import cell_len
from rich.console import Console

from soul_tty.personas import load_persona
from soul_tty.presence import LaunchContext
from soul_tty.ui import terminal


class TerminalUITests(unittest.TestCase):
    def tearDown(self):
        terminal._dashboard = None
        terminal._relationship_profile = None
        terminal._launch_context = LaunchContext()
        terminal._outfit_greeting_generator = None

    def test_splash_has_emotion_status_and_lightweight_technical_layers(self):
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
            console.print(
                terminal._splash_panel(
                    persona,
                    runtime,
                    3,
                    state="listening",
                    greeting="早上好，今天也请多关照。",
                )
            )

        lines = output.getvalue().splitlines()
        expected = (
            "SERENA",
            "早上好，今天也请多关照。",
            "◉ 正在聆听",
            "直接说话即可",
            "0 换装",
            "LOCAL · Qwen3.5-9B · 文字模式 · Sherpa-ONNX",
        )
        for text in expected:
            self.assertTrue(any(text in line for line in lines), text)
        self.assertFalse(any("人格：" in line for line in lines))
        self.assertFalse(any("sherpa-onnx ·" in line for line in lines))

    def test_greeting_fallback_tracks_the_local_time_period(self):
        self.assertEqual(terminal.day_period(8), "早上")
        self.assertEqual(terminal.day_period(12), "中午")
        self.assertEqual(terminal.day_period(16), "下午")
        self.assertEqual(terminal.day_period(21), "晚上")
        self.assertEqual(terminal.day_period(2), "夜深")
        self.assertIn("早上", terminal._fallback_greeting(8))
        self.assertEqual(terminal._fallback_greeting(2), "已经很晚了。")

    def test_greeting_fallback_combines_time_bond_and_launch_rhythm(self):
        self.assertEqual(
            terminal._fallback_greeting(7, tier="灵魂共鸣"),
            "这么早？看来今天有事要做。",
        )
        self.assertEqual(
            terminal._fallback_greeting(
                15,
                tier="灵魂共鸣",
                repeat_launch=True,
            ),
            "这么快就回来了？",
        )
        self.assertEqual(
            terminal._fallback_greeting(15, tier="初识", special=True),
            "今天让我先问一个问题，好吗？",
        )

    def test_dashboard_uses_configured_launch_context_without_waiting_for_llm(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        terminal.configure_relationship(90, "灵魂共鸣", "calm")
        terminal.configure_presence(LaunchContext(repeat_launch=True))

        with patch.object(terminal, "_console", console):
            dashboard = terminal.Dashboard(persona, runtime)

        self.assertEqual(dashboard.greeting, "这么快就回来了？")

    def test_idle_emotion_appears_and_voice_activity_restores_greeting(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")

        def render(persona, state, terminal_enabled, renderer_override=None):
            return terminal.avatar_ui.AvatarRender(symbols=terminal.Text(state))

        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.avatar_ui, "render_avatar", side_effect=render),
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard.state = "listening"
            dashboard.set_greeting("晚上好，我一直在这里。")
            dashboard._next_idle_emotion_at = 0

            triggered = dashboard._idle_emotion_tick(
                lambda: "有点想听听你的声音了。",
                now=10,
            )
            self.assertTrue(triggered)
            self.assertEqual(dashboard.greeting, "有点想听听你的声音了。")
            self.assertTrue(dashboard._idle_emotion_active)
            self.assertTrue(dashboard.presence_hint)
            console.print(dashboard.render())
            self.assertIn("安静陪伴", output.getvalue())

            dashboard.mark_voice_activity()

        self.assertEqual(dashboard.greeting, "晚上好，我一直在这里。")
        self.assertFalse(dashboard._idle_emotion_active)
        self.assertEqual(dashboard.presence_hint, "")

    def test_idle_llm_result_is_discarded_if_user_speaks_while_generating(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")

        def render(persona, state, terminal_enabled, renderer_override=None):
            return terminal.avatar_ui.AvatarRender(symbols=terminal.Text(state))

        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.avatar_ui, "render_avatar", side_effect=render),
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard.state = "listening"
            dashboard.set_greeting("基础欢迎语。")
            dashboard._next_idle_emotion_at = 0

            def generate_after_speech():
                dashboard.mark_voice_activity()
                return "这是已经过期的短句。"

            dashboard._idle_emotion_tick(generate_after_speech, now=10)

        self.assertEqual(dashboard.greeting, "基础欢迎语。")
        self.assertFalse(dashboard._idle_emotion_active)

    def test_wide_splash_has_a_fixed_seventeen_row_height(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        avatar = terminal.Text("\n".join(" " * 26 for _ in range(13)))
        with patch.object(terminal, "_console", console):
            panels = [
                terminal._splash_panel(
                    persona,
                    runtime,
                    3,
                    avatar,
                    state=state,
                    greeting="这是一句故意非常非常非常非常非常长的欢迎语",
                )
                for state in terminal._STATE_LABELS
            ]

        self.assertTrue(all(panel.height == 17 for panel in panels))

    def test_expanded_technical_profile_uses_one_shared_colon_column(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with patch.object(terminal, "_console", console):
            console.print(
                terminal._technical_profile(persona, runtime, 3, expanded=True)
            )

        lines = [line for line in output.getvalue().splitlines() if "：" in line]
        colon_columns = [cell_len(line.split("：", 1)[0]) for line in lines]
        self.assertEqual(len(lines), 4)
        self.assertEqual(len(set(colon_columns)), 1)

    def test_splash_hides_exact_relationship_score_by_default(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with patch.object(terminal, "_console", console):
            console.print(
                terminal._splash_panel(
                    persona,
                    runtime,
                    3,
                    relationship_score=0.47,
                    relationship_tier="亲近",
                )
            )

        self.assertIn("♡ 亲密  亲近", output.getvalue())
        self.assertNotIn("47", output.getvalue())

    def test_splash_reveals_exact_score_and_emotion_dimensions_in_details(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        from soul_tty.emotion.state import EmotionVector

        vector = EmotionVector(
            happiness=0.82,
            calmness=0.66,
            curiosity=0.71,
            stress=0.18,
            energy=0.74,
        )
        with patch.object(terminal, "_console", console):
            console.print(
                terminal._splash_panel(
                    persona,
                    runtime,
                    3,
                    relationship_score=0.47,
                    relationship_tier="亲近",
                    show_details=True,
                    emotion_mood="excited",
                    emotion_intensity=0.75,
                    emotion_vector=vector,
                    relationship_session_count=8,
                    relationship_event="共同玩笑",
                )
            )

        rendered = output.getvalue()
        self.assertIn("♡ 亲密  亲近  47/100", rendered)
        # 5 维情绪数值按 NN 形式展开
        self.assertIn("愉悦 82", rendered)
        self.assertIn("平静 66", rendered)
        self.assertIn("好奇 71", rendered)
        self.assertIn("压力 18", rendered)
        self.assertIn("活力 74", rendered)
        # 亲密进度：关系事件次数 + 上次事件
        self.assertIn("共 8 次关系事件", rendered)
        self.assertIn("上次：共同玩笑", rendered)
        # Tab 详情模式下不再展开技术栈
        self.assertNotIn("人格：Serena", rendered)
        self.assertNotIn("听觉：Sherpa-ONNX", rendered)

    def test_relationship_voice_waits_until_dashboard_returns_to_listening(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")

        def render(persona, state, terminal_enabled, renderer_override=None):
            return terminal.avatar_ui.AvatarRender(symbols=terminal.Text(state))

        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.avatar_ui, "render_avatar", side_effect=render),
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard.state = "thinking"
            original = dashboard.greeting

            dashboard.set_relationship(
                0.47,
                "亲近",
                "warm",
                "被你惦记着真好。",
            )

            self.assertEqual(dashboard.greeting, original)
            self.assertEqual(
                dashboard._pending_relationship_voice,
                "被你惦记着真好。",
            )

            dashboard.set_state("listening")

        self.assertEqual(dashboard.greeting, "被你惦记着真好。")
        self.assertEqual(dashboard.relationship_bond, 0.47)
        self.assertEqual(dashboard._pending_relationship_voice, "")

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
        self.assertTrue(after_clear.startswith("  │ SERENA\n  你好"))

    def test_dialogue_roles_use_distinct_accents_and_neutral_body_text(self):
        output = io.StringIO()
        console = Console(file=output, width=120, color_system="truecolor")
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with patch.object(terminal, "_console", console):
            dashboard = terminal.Dashboard(persona, runtime)
            user = dashboard._message_text("you", "你好")
            agent = dashboard._message_text("agent", "晚上好")

        user_label = user.get_style_at_offset(console, user.plain.index("YOU"))
        user_body = user.get_style_at_offset(console, user.plain.index("你好"))
        agent_label = agent.get_style_at_offset(console, agent.plain.index("SERENA"))
        agent_body = agent.get_style_at_offset(console, agent.plain.index("晚上好"))
        self.assertEqual(user_label.color.get_truecolor(), (103, 183, 209))
        self.assertEqual(agent_label.color.get_truecolor(), (192, 132, 252))
        self.assertEqual(user_body.color.get_truecolor(), (209, 213, 219))
        self.assertEqual(agent_body.color.get_truecolor(), (209, 213, 219))

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
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard.set_state("speaking")

        paint.assert_called_once_with(
            dashboard.mouth_avatars[0],
            console.file,
            row=3,
            column=7,
        )

    def test_dashboard_uses_cached_static_frames_for_speaking_state(self):
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
            patch.object(terminal.avatar_ui, "write_native_at") as paint,
            patch.object(
                terminal.Dashboard, "_start_mouth_animation", return_value=True
            ) as start,
            patch.object(terminal.Dashboard, "_stop_mouth_animation") as stop,
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard._native_frames_ready = True
            dashboard.set_state("speaking")
            paint.assert_not_called()
            dashboard.set_state("listening")

        start.assert_called_once_with()
        paint.assert_called_once_with(
            dashboard.avatars["listening"], console.file, row=3, column=7
        )
        stop.assert_called_once_with()

    def test_dashboard_preloads_two_clean_static_mouth_frames_on_start(self):
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
            patch.object(
                terminal.avatar_ui, "prepare_native_frames", return_value=True
            ) as prepare,
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.start = lambda **kwargs: None
            dashboard.input.start = lambda: None
            dashboard._paint_native = lambda: None
            dashboard.start()

        prepare.assert_called_once_with(
            dashboard.mouth_avatars,
            console.file,
        )
        self.assertTrue(dashboard._native_frames_ready)

    def test_speaking_hides_the_previous_state_before_first_mouth_frame(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")

        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.avatar_ui, "hide_native_avatar") as hide,
            patch.object(
                terminal.avatar_ui, "show_native_frame_at", return_value=False
            ) as show,
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            self.assertFalse(dashboard._start_mouth_animation())

        hide.assert_called_once_with(console.file)
        show.assert_called_once()

    def test_mouth_only_changes_when_pcm_level_crosses_the_threshold(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.avatar_ui, "show_native_frame_at") as show,
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.state = "speaking"
            dashboard._native_frames_ready = True
            dashboard.mouth_frame = 1

            dashboard.set_audio_level(0.7)
            dashboard.set_audio_level(0.8)
            dashboard.set_audio_level(0.0)

        self.assertEqual(show.call_count, 2)
        self.assertEqual(show.call_args_list[0].args[1], 1)
        self.assertEqual(show.call_args_list[1].args[1], 0)

    def test_terminal_input_converts_wheel_and_arrow_keys_to_navigation(self):
        data = b"\x1b[<64;20;10M\x1b[<65;20;10M\x1b[A\x1b[B"
        self.assertEqual(terminal.TerminalInput.navigation(data), [1, -1, 1, -1])
        self.assertEqual(terminal.TerminalInput.detail_toggles(b"\t\t"), 2)
        self.assertEqual(terminal.TerminalInput.outfit_toggles(data), 0)
        self.assertEqual(terminal.TerminalInput.outfit_toggles(b"00"), 2)

    def test_mouse_click_coordinates_never_trigger_outfit_switch(self):
        click = b"\x1b[<0;120;30M\x1b[<0;120;30m"
        right_click = b"\x1b[<2;100;40M\x1b[<2;100;40m"

        self.assertEqual(terminal.TerminalInput.outfit_toggles(click), 0)
        self.assertEqual(terminal.TerminalInput.outfit_toggles(right_click), 0)

    def test_split_mouse_escape_is_buffered_until_complete(self):
        complete, pending = terminal.TerminalInput.split_incomplete_escape(
            b"\x1b[<0;120;"
        )
        self.assertEqual(complete, b"")
        self.assertEqual(pending, b"\x1b[<0;120;")

        complete, pending = terminal.TerminalInput.split_incomplete_escape(
            pending + b"30M0"
        )
        self.assertEqual(pending, b"")
        self.assertEqual(terminal.TerminalInput.outfit_toggles(complete), 1)

    def test_zero_cycles_mode_and_loads_only_its_unique_frames(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        paths = []

        def render(persona, state, terminal_enabled, renderer_override=None):
            path = persona.appearance.avatar.for_state(state)
            paths.append(path)
            return terminal.avatar_ui.AvatarRender(symbols=terminal.Text(state))

        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.avatar_ui, "render_avatar", side_effect=render),
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard._next_idle_emotion_at = 0
            paths.clear()
            dashboard.cycle_outfit()
            dashboard.set_greeting("过期的默认装台词", outfit_id="default")

        self.assertEqual(
            dashboard.persona.appearance.avatar.selected_outfit,
            "work",
        )
        self.assertEqual(len(paths), 2)
        self.assertTrue(all("/work/" in path for path in paths))
        self.assertEqual(dashboard.greeting, "工作模式，今天从哪里开始？")
        self.assertGreater(
            dashboard._next_idle_emotion_at,
            dashboard._last_voice_activity,
        )

    def test_dashboard_details_toggle_is_explicit_and_reversible(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.config, "DASHBOARD_DETAILS", False),
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            self.assertFalse(dashboard.show_details)
            dashboard.toggle_details()
            self.assertTrue(dashboard.show_details)
            dashboard.toggle_details()
            self.assertFalse(dashboard.show_details)

    def test_dashboard_scrolls_conversation_inside_fixed_viewport(self):
        output = io.StringIO()
        console = Console(file=output, width=120, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with patch.object(terminal, "_console", console):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard.messages = [("you", str(index)) for index in range(15)]
            dashboard.scroll(1)
            self.assertEqual(dashboard.scroll_offset, 1)
            dashboard.scroll(-1)
            self.assertEqual(dashboard.scroll_offset, 0)

    def test_dashboard_transcript_always_contains_the_latest_messages(self):
        output = io.StringIO()
        console = Console(file=output, width=120, height=30, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with patch.object(terminal, "_console", console):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.messages = [("you", f"消息 {index}") for index in range(20)]
            transcript = dashboard._transcript_view(width=40, rows=5).plain

        self.assertIn("消息 19", transcript)
        self.assertIn("消息 18", transcript)
        self.assertNotIn("消息 0\n", transcript)

    def test_dashboard_shows_the_tail_of_one_answer_larger_than_viewport(self):
        output = io.StringIO()
        console = Console(file=output, width=120, height=30, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with patch.object(terminal, "_console", console):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.messages = [
                ("agent", "第一行\n第二行\n第三行\n第四行\n最新结尾"),
            ]
            transcript = dashboard._transcript_view(width=40, rows=3).plain

        self.assertTrue(transcript.startswith("│ SERENA  …"))
        self.assertIn("最新结尾", transcript)
        self.assertNotIn("第一行", transcript)

    def test_new_messages_preserve_history_anchor_until_user_returns_to_latest(self):
        output = io.StringIO()
        console = Console(file=output, width=120, height=30, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with patch.object(terminal, "_console", console):
            dashboard = terminal.Dashboard(persona, runtime)
            dashboard.live.update = lambda *args, **kwargs: None
            dashboard.messages = [("you", str(index)) for index in range(15)]
            dashboard.scroll(1)
            dashboard.add("agent", "新消息")
            self.assertEqual(dashboard.scroll_offset, 2)
            self.assertNotIn(
                "新消息",
                dashboard._transcript_view(width=40, rows=5).plain,
            )

            dashboard.scroll(-1)
            dashboard.scroll(-1)
            self.assertEqual(dashboard.scroll_offset, 0)
            self.assertIn(
                "新消息",
                dashboard._transcript_view(width=40, rows=5).plain,
            )

    def test_dashboard_bounds_long_running_visual_history(self):
        output = io.StringIO()
        console = Console(file=output, width=120, height=30, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with (
            patch.object(terminal, "_console", console),
            patch.object(terminal.config, "DASHBOARD_MAX_MESSAGES", 3),
        ):
            dashboard = terminal.Dashboard(persona, runtime)
            for index in range(5):
                dashboard.append("you", f"消息 {index}")

        self.assertEqual(
            dashboard.messages,
            [("you", "消息 2"), ("you", "消息 3"), ("you", "消息 4")],
        )

    def test_splash_renders_relationship_level_in_chinese(self):
        """亲密 level 内部用英文键，HUD 展示层翻译为中文。"""
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        cases = [
            ("stranger", "初识"),
            ("acquaintance", "相熟"),
            ("familiar", "熟悉"),
            ("companion", "亲近"),
            ("close", "默契"),
            ("bonded", "挚友"),
        ]
        for english, chinese in cases:
            output = io.StringIO()
            console = Console(file=output, width=120, force_terminal=False)
            with patch.object(terminal, "_console", console):
                console.print(
                    terminal._splash_panel(
                        persona,
                        runtime,
                        3,
                        relationship_score=0.5,
                        relationship_tier=english,
                    )
                )
            self.assertIn(
                f"♡ 亲密  {chinese}",
                output.getvalue(),
                f"english={english!r} should render as {chinese!r}",
            )
        from soul_tty.emotion.service import EmotionSnapshot

        output = io.StringIO()
        console = Console(file=output, width=120, height=30, force_terminal=False)
        persona = load_persona("serena")
        runtime = terminal.RuntimeDetails(model="Qwen3.5-9B.gguf", tts="MLX")
        with patch.object(terminal, "_console", console):
            dashboard = terminal.Dashboard(persona, runtime)
            # 模拟 listening 安全窗口
            dashboard.state = "listening"
            dashboard.partial_text = ""
            snap = EmotionSnapshot(
                baseline=None,
                emotion=None,
                mood="excited",
                intensity=0.75,
                expression="caring",
                should_update_prompt=False,
                context_text="",
            )
            dashboard.set_emotion(snap)
        self.assertEqual(dashboard.emotion_mood, "excited")
        self.assertEqual(dashboard.emotion_intensity, 0.75)
        self.assertEqual(dashboard.emotion_expression, "caring")


if __name__ == "__main__":
    unittest.main()
