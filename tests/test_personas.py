import os
import unittest
from pathlib import Path
from unittest.mock import patch

from soul_tty import config
from soul_tty.personas import apply_persona, available_personas, load_persona


class PersonaTests(unittest.TestCase):
    def test_lists_builtin_personas(self):
        self.assertEqual(
            {persona.id for persona in available_personas()},
            {"assistant", "serena"},
        )

    def test_loads_and_renames_serena(self):
        persona = load_persona("serena").renamed("小夜")
        self.assertEqual(persona.display_name, "小夜")
        self.assertEqual(persona.voice.voice, "Serena")
        self.assertIn("温柔、聪明", persona.personality.system_prompt)
        self.assertIsNotNone(persona.appearance.avatar)
        for state in ("idle", "listening", "thinking", "speaking"):
            self.assertTrue(
                Path(persona.appearance.avatar.for_state(state)).is_file()
            )
        for mouth in ("speaking_closed", "speaking_half", "speaking_open"):
            self.assertTrue(Path(persona.appearance.avatar.for_state(mouth)).is_file())

    def test_selects_serena_outfit_without_changing_other_persona_data(self):
        persona = load_persona("serena")
        self.assertEqual(
            [outfit.id for outfit in persona.appearance.avatar.outfits],
            ["default", "late-night", "work"],
        )

        work = persona.wearing("work")

        self.assertEqual(work.display_name, "Serena")
        self.assertEqual(work.appearance.avatar.selected_outfit, "work")
        self.assertIn("/work/", work.appearance.avatar.for_state("idle"))
        self.assertIn("/work/", work.appearance.avatar.for_state("speaking_half"))
        self.assertIn("编程", work.appearance.avatar.outfit.description)
        self.assertTrue(work.appearance.avatar.outfit.switch_greetings)

    def test_rejects_unknown_outfit(self):
        with self.assertRaisesRegex(
            ValueError, "可用: default, late-night, work"
        ):
            load_persona("serena").wearing("missing")

    def test_applies_persona_defaults(self):
        persona = load_persona("assistant")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(config, "SYSTEM_PROMPT", "old"),
            patch.object(config, "TTS_BACKEND", "old"),
            patch.object(config, "MLX_TTS_VOICE", "old"),
            patch.object(config, "MLX_TTS_INSTRUCT", "old"),
        ):
            apply_persona(persona)
            self.assertEqual(
                config.SYSTEM_PROMPT,
                f"你的名字是“小助理”。\n{persona.personality.system_prompt}",
            )
            self.assertEqual(config.TTS_BACKEND, "mlx")
            self.assertEqual(config.MLX_TTS_VOICE, "Serena")
            self.assertEqual(
                config.MLX_TTS_INSTRUCT, "用清晰、自然、从容的语气说"
            )

    def test_environment_keeps_priority_over_persona(self):
        persona = load_persona("serena")
        with (
            patch.dict(
                os.environ,
                {
                    "SYSTEM_PROMPT": "环境人格",
                    "TTS_BACKEND": "macos",
                    "MLX_TTS_VOICE": "Vivian",
                    "MLX_TTS_INSTRUCT": "环境语气",
                },
                clear=True,
            ),
            patch.object(config, "SYSTEM_PROMPT", "环境人格"),
            patch.object(config, "TTS_BACKEND", "macos"),
            patch.object(config, "MLX_TTS_VOICE", "Vivian"),
            patch.object(config, "MLX_TTS_INSTRUCT", "环境语气"),
        ):
            apply_persona(persona)
            self.assertEqual(config.SYSTEM_PROMPT, "环境人格")
            self.assertEqual(config.TTS_BACKEND, "macos")
            self.assertEqual(config.MLX_TTS_VOICE, "Vivian")
            self.assertEqual(config.MLX_TTS_INSTRUCT, "环境语气")


if __name__ == "__main__":
    unittest.main()
