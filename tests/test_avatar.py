import base64
import os
import unittest
from unittest.mock import patch

from soul_tty.personas import load_persona
from soul_tty.ui import avatar


class AvatarRendererTests(unittest.TestCase):
    @staticmethod
    def _rgba_render(pixels: bytes, *, width: int = 2, height: int = 2):
        encoded = base64.b64encode(pixels)
        return avatar.AvatarRender(
            native=(
                f"\x1b_Ga=T,f=32,s={width},v={height},c=26,r=13,m=0,q=2;".encode()
                + encoded
                + b"\x1b\\\n"
            ),
            mode="pixels",
            protocol="kitty",
        )

    def test_non_terminal_does_not_emit_avatar_protocols(self):
        render = avatar.render_avatar(load_persona("serena"), "idle", False)
        self.assertEqual(render.mode, "off")

    def test_unknown_terminal_uses_symbol_fallback_without_probe(self):
        payload = b"\x1b[38;2;1;2;3m\xe2\x96\x80\x1b[0m\n"
        with (
            patch.dict(os.environ, {"TERM": "dumb"}, clear=True),
            patch("soul_tty.ui.avatar.shutil.which", return_value="/bin/chafa"),
            patch("soul_tty.ui.avatar._run_chafa", return_value=payload) as run,
        ):
            render = avatar.render_avatar(load_persona("serena"), "idle", True)
        self.assertEqual(render.mode, "symbols")
        self.assertEqual(run.call_args.args[2], "symbols")
        self.assertIsNotNone(render.symbols)

    def test_renderer_reads_only_the_selected_outfit_path(self):
        payload = b"avatar\n"
        persona = load_persona("serena").wearing("work")
        with (
            patch.dict(os.environ, {"TERM": "dumb"}, clear=True),
            patch("soul_tty.ui.avatar.shutil.which", return_value="/bin/chafa"),
            patch("soul_tty.ui.avatar._run_chafa", return_value=payload) as run,
        ):
            avatar.render_avatar(persona, "speaking_half", True)

        self.assertIn("/work/", str(run.call_args.args[0]))
        self.assertNotIn("/late-night/", str(run.call_args.args[0]))

    def test_kitty_terminal_selects_native_pixels(self):
        payload = b"\x1b_Gf=100;payload\x1b\\"
        with (
            patch.dict(os.environ, {"KITTY_WINDOW_ID": "1"}, clear=True),
            patch("soul_tty.ui.avatar.shutil.which", return_value="/bin/chafa"),
            patch("soul_tty.ui.avatar._run_chafa", return_value=payload) as run,
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
            b"\x1b7\x1b[3;7H\x1b_Gpayload\x1b\\\x1b8",
        )

    def test_chafa_kitty_transmission_uses_the_same_fixed_placement(self):
        class Output:
            def __init__(self):
                import io

                self.buffer = io.BytesIO()

        output = Output()
        render = avatar.AvatarRender(
            native=b"\x1b_Ga=T,f=32,s=260,v=260,c=26,r=13,m=0;payload\x1b\\\n",
            mode="pixels",
            protocol="kitty",
        )
        avatar.write_native_at(render, output, row=3, column=7)
        payload = output.buffer.getvalue()
        self.assertIn(
            f"a=t,I={avatar._KITTY_IMAGE_NUMBER},f=32".encode(),
            payload,
        )
        self.assertIn(
            f"a=p,I={avatar._KITTY_IMAGE_NUMBER},p=1,c=26,C=1".encode(),
            payload,
        )
        self.assertNotIn(b"p=1,c=26,r=13", payload)

    def test_cached_mouth_frames_use_normal_fixed_placements(self):
        class Output:
            def __init__(self):
                import io

                self.buffer = io.BytesIO()

        output = Output()
        pixels = bytes((0, 0, 0, 255)) * 4
        frames = (
            self._rgba_render(pixels),
            self._rgba_render(pixels[:12] + bytes((255, 0, 0, 255))),
        )
        self.assertTrue(avatar.prepare_native_frames(frames, output))
        self.assertTrue(
            avatar.show_native_frame_at(
                output,
                1,
                row=3,
                column=7,
                width=26,
            )
        )
        self.assertTrue(avatar.hide_native_frames(output))

        payload = output.buffer.getvalue()
        for number in avatar._KITTY_SPEAKING_FRAME_NUMBERS:
            self.assertIn(f"a=t,I={number},f=32".encode(), payload)
        selected = avatar._KITTY_SPEAKING_FRAME_NUMBERS[1]
        self.assertIn(
            f"a=p,I={selected},p=1,c=26,C=1".encode(),
            payload,
        )
        self.assertNotIn(b"p=1,c=26,r=13", payload)
        self.assertNotIn(b"a=f", payload)
        self.assertNotIn(b"a=a", payload)

    def test_static_avatar_can_be_hidden_before_showing_mouth_frames(self):
        class Output:
            def __init__(self):
                import io

                self.buffer = io.BytesIO()

        output = Output()
        self.assertTrue(avatar.hide_native_avatar(output))
        self.assertEqual(
            output.buffer.getvalue(),
            (
                f"\x1b_Ga=d,d=n,I={avatar._KITTY_IMAGE_NUMBER},"
                f"p={avatar._KITTY_IMAGE_PLACEMENT_ID},q=2\x1b\\"
            ).encode(),
        )

if __name__ == "__main__":
    unittest.main()
