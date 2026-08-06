"""记忆抽取：把多轮对话合并成一次 LLM 调用，落库到 MemoryService。

设计上与 `relationship.py:evaluate_relationship` 对称：
单次 LLM 调用、JSON 响应、由 `_parse_json_object` 清洗、输出 schema
稳定。但本模块不持有任何线程或状态，调度由 ReflectionWorker 负责。
"""

from __future__ import annotations

from ..clients.llm import extract_memories
from .service import MemoryService
from ..reflection.relationship import CompletedTurn


def _coalesce(turns: list[CompletedTurn]) -> CompletedTurn:
    """与 ReflectionWorker 同样的合并策略。"""
    if len(turns) == 1:
        return turns[0]
    return CompletedTurn(
        "\n".join(
            f"第{index}轮：{turn.user_text}"
            for index, turn in enumerate(turns, 1)
        ),
        "\n".join(
            f"第{index}轮：{turn.agent_text}"
            for index, turn in enumerate(turns, 1)
        ),
    )


def extract_from_turns(
    service: MemoryService,
    *,
    persona_id: str,
    display_name: str,
    model: str,
    turns: list[CompletedTurn],
    known_facts: list[dict] | None = None,
) -> bool:
    """从一段对话里抽取记忆并写入 service。

    返回 True 当且仅当至少一条记忆成功落库。返回 False 不代表失败——
    也可能是「本轮没有值得保存的内容」（recall 反馈链路需要这个信号
    决定要不要 hot-update system prompt）。
    """
    if not service.available or not turns:
        return False
    if known_facts is None:
        known_facts = service.known_facts(persona_id=persona_id)
    combined = _coalesce(turns)
    try:
        result = extract_memories(
            model=model,
            display_name=display_name,
            known_facts=known_facts,
            user_text=combined.user_text,
            agent_text=combined.agent_text,
        )
    except Exception:
        return False
    if not isinstance(result, dict):
        return False
    accepted = service.remember_many(
        result.get("memories", []),
        persona_id=persona_id,
    )
    return accepted > 0
