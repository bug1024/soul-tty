"""soul-tty memory 子命令：list / show / forget / clear。"""

import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from soul_tty import config
from soul_tty.memory.cli import run_memory
from soul_tty.memory.service import MemoryService


def _service_with(tmp_path) -> MemoryService:
    service = MemoryService(Path(tmp_path) / "memory.db")
    service.remember_many([
        {"type": "profile", "content": "用户是工程师", "importance": 0.9},
        {"type": "preference", "content": "喜欢结构化", "importance": 0.8},
        {
            "type": "experience",
            "content": "和 Serena 完成了 Emotion 系统",
            "importance": 0.85,
            "persona_id": "serena",
        },
    ])
    return service


def _run(tmp_path, args, service):
    out = io.StringIO()
    err = io.StringIO()
    with patch("soul_tty.memory.cli._resolve_service", return_value=service):
        with redirect_stdout(out):
            with redirect_stderr(err):
                rc = run_memory(list(args))
    return rc, out.getvalue(), err.getvalue()


# 兼容旧测试
def _run_stdout_only(tmp_path, args, service):
    rc, out, _ = _run(tmp_path, args, service)
    return rc, out


# 修复旧测试用 _run 但只收两个返回值——加一个两返回值版本
def _run_legacy(tmp_path, args, service):
    return _run_stdout_only(tmp_path, args, service)


class ListTests(unittest.TestCase):
    def test_list_shows_all_memories(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            rc, output, _err = _run(tmp, [], service)
            self.assertEqual(rc, 0)
            self.assertIn("用户是工程师", output)
            self.assertIn("喜欢结构化", output)
            self.assertIn("和 Serena 完成了 Emotion 系统", output)

    def test_list_groups_by_type(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            rc, output, _err = _run(tmp, ["list"], service)
            self.assertEqual(rc, 0)
            # 类型标题
            self.assertIn("用户画像", output)
            self.assertIn("交流偏好", output)
            self.assertIn("共同经历", output)

    def test_list_filters_by_type(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            rc, output, _err = _run(tmp, ["list", "--type", "profile"], service)
            self.assertEqual(rc, 0)
            self.assertIn("用户是工程师", output)
            self.assertNotIn("喜欢结构化", output)
            self.assertNotIn("和 Serena 完成了 Emotion 系统", output)

    def test_list_empty_db(self):
        with TemporaryDirectory() as tmp:
            service = MemoryService(Path(tmp) / "memory.db")
            rc, output, _err = _run(tmp, ["list"], service)
            self.assertEqual(rc, 0)
            self.assertIn("（无）", output)

    def test_bare_command_is_alias_for_list(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            rc1, out1, _ = _run(tmp, [], service)
            rc2, out2, _ = _run(tmp, ["list"], service)
            self.assertEqual(rc1, rc2)
            self.assertEqual(out1, out2)


class ShowTests(unittest.TestCase):
    def test_show_existing(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            row = service.list(types=("profile",))[0]
            rc, output, _err = _run(tmp, ["show", str(row.id)], service)
            self.assertEqual(rc, 0)
            self.assertIn(str(row.id), output)
            self.assertIn("用户是工程师", output)
            self.assertIn("profile", output)

    def test_show_missing_returns_error(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            rc, _out, err = _run(tmp, ["show", "9999"], service)
            self.assertNotEqual(rc, 0)
            self.assertIn("9999", err)


class ForgetTests(unittest.TestCase):
    def test_forget_existing(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            row = service.list()[0]
            rc, output, _err = _run(tmp, ["forget", str(row.id)], service)
            self.assertEqual(rc, 0)
            self.assertNotIn(row.content, service.render_resident_context() + "\n".join(
                m.content for m in service.list()
            ))

    def test_forget_missing_returns_error(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            rc, _out, _err = _run(tmp, ["forget", "9999"], service)
            self.assertNotEqual(rc, 0)


class ClearTests(unittest.TestCase):
    def test_clear_yes(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            out = io.StringIO()
            with patch("soul_tty.memory.cli._resolve_service", return_value=service):
                with redirect_stdout(out):
                    with patch("builtins.input", return_value="y"):
                        rc = run_memory(["clear"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(service.list()), 0)

    def test_clear_no(self):
        with TemporaryDirectory() as tmp:
            service = _service_with(tmp)
            with patch("soul_tty.memory.cli._resolve_service", return_value=service):
                with patch("builtins.input", return_value="n"):
                    rc = run_memory(["clear"])
            self.assertEqual(rc, 0)
            # 数据保留
            self.assertEqual(len(service.list()), 3)


class ErrorTests(unittest.TestCase):
    def test_unknown_subcommand(self):
        with TemporaryDirectory() as tmp:
            service = MemoryService(Path(tmp) / "memory.db")
            rc, _out, _err = _run(tmp, ["bogus"], service)
            self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
