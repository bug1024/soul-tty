"""soul-tty memory 子命令：list / show / forget / clear。

本模块只负责 CLI 表面：参数解析、表格输出、确认 prompt。
DB 操作全部走 MemoryService，业务规则在 service 层。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .. import config
from ..reflection.relationship import safe_persona_id
from .service import MemoryService


def _resolve_service() -> MemoryService:
    """CLI 走自己的进程，独立连接 DB；与运行时服务互不影响。"""
    return MemoryService(config.MEMORY_DB_PATH)


def _print_memory(memory) -> None:
    persona_part = (
        f"persona={memory.persona_id}  " if memory.persona_id else ""
    )
    print(
        f"  [{memory.id}] {memory.type_label}  {persona_part}"
        f"importance={memory.importance:.2f}  "
        f"({memory.created_at[:10]})"
    )
    print(f"        {memory.content}")


def _list(service: MemoryService, types: tuple[str, ...] | None) -> int:
    rows = service.list(types=types)
    if not rows:
        print("（无）")
        return 0
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row.type, []).append(row)
    for type_name, items in grouped.items():
        head = items[0].type_label if items else type_name
        print(f"{head}（{len(items)} 条）")
        for item in items:
            _print_memory(item)
        print()
    return 0


def _show(service: MemoryService, memory_id: int) -> int:
    memory = service.get(memory_id)
    if memory is None:
        print(f"未找到记忆 {memory_id}", file=sys.stderr)
        return 1
    print(f"ID:        {memory.id}")
    print(f"Scope:     {memory.scope}")
    if memory.persona_id:
        print(f"Persona:   {memory.persona_id}")
    print(f"Type:      {memory.type}（{memory.type_label}）")
    print(f"Importance: {memory.importance:.2f}")
    print(f"Source:    {memory.source}")
    print(f"Created:   {memory.created_at}")
    print(f"Updated:   {memory.updated_at}")
    print()
    print(memory.content)
    return 0


def _forget(service: MemoryService, memory_id: int) -> int:
    if service.delete(memory_id):
        print(f"已删除记忆 {memory_id}")
        print("注意：运行中的 Soul-TTY 需重启后刷新常驻用户上下文。")
        return 0
    print(f"未找到记忆 {memory_id}", file=sys.stderr)
    return 1


def _clear(service: MemoryService) -> int:
    count = len(service.list())
    if count == 0:
        print("数据库已经是空的")
        return 0
    print(f"将删除全部 {count} 条记忆，此操作不可撤销。")
    answer = input("确认清空？[y/N] ").strip().lower()
    if answer != "y":
        print("已取消")
        return 0
    removed = service.clear()
    print(f"已清空 {removed} 条记忆")
    print("注意：运行中的 Soul-TTY 需重启后刷新常驻用户上下文。")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soul-tty memory",
        description="Memory 调试与管理入口",
    )
    sub = parser.add_subparsers(dest="subcommand")

    list_p = sub.add_parser("list", help="列出全部记忆")
    list_p.add_argument(
        "--type",
        choices=("profile", "preference", "experience"),
        help="按类型过滤",
    )

    show_p = sub.add_parser("show", help="查看单条记忆")
    show_p.add_argument("id", type=int, help="记忆 ID")

    forget_p = sub.add_parser("forget", help="删除单条记忆")
    forget_p.add_argument("id", type=int, help="记忆 ID")

    sub.add_parser("clear", help="清空全部记忆（需二次确认）")
    return parser


def run_memory(args: Sequence[str]) -> int:
    """CLI 入口。返回 0 成功，非 0 失败。"""
    parser = _build_parser()
    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        return int(exc.code or 0)

    service = _resolve_service()
    if parsed.subcommand is None:
        # 裸 `memory` 等同于 list
        return _list(service, None)
    if parsed.subcommand == "list":
        return _list(service, (parsed.type,) if parsed.type else None)
    if parsed.subcommand == "show":
        return _show(service, parsed.id)
    if parsed.subcommand == "forget":
        return _forget(service, parsed.id)
    if parsed.subcommand == "clear":
        return _clear(service)
    parser.error(f"未知子命令: {parsed.subcommand}")
    return 2
