"""记忆抽取：把多轮对话合并成一次 LLM 调用，落库到 MemoryService。

设计上与 `relationship.py:evaluate_relationship` 对称：
单次 LLM 调用、JSON 响应、由 `_parse_json_object` 清洗、输出 schema
稳定。但本模块不持有任何线程或状态，调度由 ReflectionWorker 负责。
"""

from __future__ import annotations

from ..clients.llm import extract_memories
from .models import ExtractionStatus
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
) -> ExtractionStatus:
    """从一段对话里抽取记忆并写入 service。

    返回 ExtractionStatus：
    - FAILED：LLM 调用失败或响应无法解析，buffer 保留。
    - NO_CHANGE：处理成功但没有新增记忆，buffer 应 ack。
    - UPDATED：至少一条新记忆落库，buffer 应 ack 且刷新 prompt。
    """
    if not service.available or not turns:
        return ExtractionStatus.FAILED
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
        return ExtractionStatus.FAILED
    if result is None:
        return ExtractionStatus.FAILED
    accepted = service.remember_many(
        result.get("memories", []),
        persona_id=persona_id,
    )
    if accepted > 0:
        return ExtractionStatus.UPDATED
    return ExtractionStatus.NO_CHANGE
