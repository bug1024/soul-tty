"""soul-tty relationship 子命令：查看或清空当前人格的亲密状态。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .. import config
from .relationship import load_state, state_path


def _resolve_path(persona_id: str) -> Path:
    return state_path(persona_id, config.SOUL_TTY_STATE_DIR)


def _show(persona_id: str, path: Path) -> int:
    state = load_state(path)
    print(f"人格:     {persona_id}")
    print(f"亲密阶段: {state.level}")
    print(f"亲密值:   {state.bond:.2f}")
    print(f"互动次数: {state.interaction_count}")
    if state.recent_events:
        print("近期事件:")
        for event in state.recent_events:
            print(f"  - {event}")
    return 0


def _clear(persona_id: str, path: Path) -> int:
    if not path.exists():
        print(f"{persona_id} 的亲密状态已经是初始状态")
        return 0

    state = load_state(path)
    print(
        f"将重置 {persona_id} 的亲密状态："
        f"{state.level}（{state.bond:.2f}），"
        f"并清空 {state.interaction_count} 次互动记录和近期关系事件。"
    )
    print("长期记忆、情绪状态和其他人格不会受影响。")
    print("请先退出正在运行的 Soul-TTY，避免后台旁路把旧状态重新写回。")
    answer = input("确认清空？[y/N] ").strip().lower()
    if answer != "y":
        print("已取消")
        return 0

    try:
        path.unlink()
    except FileNotFoundError:
        pass
    print(f"已将 {persona_id} 的亲密状态恢复为初始值")
    print("下次启动 Soul-TTY 时生效。")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soul-tty relationship",
        description="当前人格的亲密状态管理入口",
    )
    sub = parser.add_subparsers(dest="subcommand")
    sub.add_parser("show", help="查看当前亲密状态")
    sub.add_parser("clear", help="清空亲密值与关系事件（需二次确认）")
    return parser


def run_relationship(args: Sequence[str], *, persona_id: str) -> int:
    """CLI 入口；裸 relationship 等同于 show。"""
    parser = _build_parser()
    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        return int(exc.code or 0)

    path = _resolve_path(persona_id)
    if parsed.subcommand in (None, "show"):
        return _show(persona_id, path)
    if parsed.subcommand == "clear":
        return _clear(persona_id, path)
    parser.error(f"未知子命令: {parsed.subcommand}")
    return 2
