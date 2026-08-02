import os
import unittest
from unittest.mock import patch

from voice_agent.personas import load_persona
from voice_agent.ui import avatar


class AvatarRendererTests(unittest.TestCase):
    def test_non_terminal_does_not_emit_avatar_protocols(self):
        render = avatar.render_avatar(load_persona("serena"), "idle", False)
        self.assertEqual(render.mode, "off")

    def test_unknown_terminal_uses_symbol_fallback_without_probe(self):
        payload = b"\x1b[38;2;1;2;3m\xe2\x96\x80\x1b[0m\n"
        with (
            patch.dict(os.environ, {"TERM": "dumb"}, clear=True),
            patch("voice_agent.ui.avatar.shutil.which", return_value="/bin/chafa"),
            patch("voice_agent.ui.avatar._run_chafa", return_value=payload) as run,
        ):
            render = avatar.render_avatar(load_persona("serena"), "idle", True)
        self.assertEqual(render.mode, "symbols")
        self.assertEqual(run.call_args.args[2], "symbols")
        self.assertIsNotNone(render.symbols)

    def test_kitty_terminal_selects_native_pixels(self):
        payload = b"\x1b_Gf=100;payload\x1b\\"
        with (
            patch.dict(os.environ, {"KITTY_WINDOW_ID": "1"}, clear=True),
            patch("voice_agent.ui.avatar.shutil.which", return_value="/bin/chafa"),
            patch("voice_agent.ui.avatar._run_chafa", return_value=payload) as run,
        ):
            render = avatar.render_avatar(load_persona("serena"), "speaking", True)
        self.assertEqual(render.mode, "pixels")
        self.assertEqual(render.native, payload)
        self.assertEqual(render.protocol, "kitty")
        self.assertEqual(run.call_args.args[2], "kitty")

    def test_native_image_can_be_painted_at_a_fixed_position(self):
        class Output:
            def __init__(self):
                import io

                self.buffer = io.BytesIO()

        output = Output()
        render = avatar.AvatarRender(
            native=b"\x1b_Gpayload\x1b\\\n",
            mode="pixels",
            protocol="kitty",
        )
        self.assertTrue(
            avatar.write_native_at(render, output, row=3, column=7)
        )
        self.assertEqual(
            output.buffer.getvalue(),
            b"\x1b7\x1b[3;7H"
            + f"\x1b_Ga=d,d=N,I={avatar._KITTY_IMAGE_NUMBER},q=2\x1b\\".encode()
            + b"\x1b_Gpayload\x1b\\\x1b8",
        )

    def test_chafa_kitty_transmission_gets_a_stable_image_number(self):
        class Output:
            def __init__(self):
                import io

                self.buffer = io.BytesIO()

        output = Output()
        render = avatar.AvatarRender(
            native=b"\x1b_Ga=T,f=32,m=0;payload\x1b\\\n",
            mode="pixels",
            protocol="kitty",
        )
        avatar.write_native_at(render, output, row=3, column=7)
        self.assertIn(
            f"a=T,I={avatar._KITTY_IMAGE_NUMBER},f=32".encode(),
            output.buffer.getvalue(),
        )

    def test_chafa_transmission_can_be_rewritten_as_animation_frame(self):
        render = avatar.AvatarRender(
            native=(
                b"\x1b_Ga=T,f=32,s=260,v=260,c=26,r=13,m=1,q=2;part1\x1b\\"
                b"\x1b_Gm=0;part2\x1b\\\n"
            ),
            mode="pixels",
            protocol="kitty",
        )
        payload = avatar._kitty_frame_payload(render)
        self.assertIn(
            f"a=f,I={avatar._KITTY_IMAGE_NUMBER},f=32,s=260,v=260,m=1,q=2".encode(),
            payload,
        )
        self.assertNotIn(b"c=26", payload)
        self.assertNotIn(b"r=13", payload)


if __name__ == "__main__":
    unittest.main()
