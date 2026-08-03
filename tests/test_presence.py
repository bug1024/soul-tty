import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from soul_tty import config
from soul_tty.presence import record_launch


class LaunchPresenceTests(unittest.TestCase):
    def test_first_launch_records_only_timing_metadata(self):
        now = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
        with TemporaryDirectory() as directory, patch.object(
            config, "PRESENCE_SPECIAL_GREETING_RATE", 0.05
        ):
            context = record_launch(
                "serena",
                state_dir=Path(directory),
                now=now,
                random_value=0.9,
            )
            data = json.loads(
                (Path(directory) / "presence" / "serena.json").read_text()
            )

        self.assertFalse(context.repeat_launch)
        self.assertFalse(context.special_greeting)
        self.assertEqual(context.launch_count, 1)
        self.assertEqual(set(data), {"last_started_at", "launch_count"})

    def test_repeat_launch_and_low_frequency_opening_are_derived(self):
        first = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
        with (
            TemporaryDirectory() as directory,
            patch.object(config, "PRESENCE_REPEAT_LAUNCH_WINDOW_S", 600),
            patch.object(config, "PRESENCE_SPECIAL_GREETING_RATE", 0.05),
        ):
            root = Path(directory)
            record_launch(
                "serena", state_dir=root, now=first, random_value=0.9
            )
            context = record_launch(
                "serena",
                state_dir=root,
                now=first + timedelta(seconds=120),
                random_value=0.01,
            )

        self.assertTrue(context.repeat_launch)
        self.assertTrue(context.special_greeting)
        self.assertEqual(context.interval_s, 120)
        self.assertEqual(context.launch_count, 2)


if __name__ == "__main__":
    unittest.main()
