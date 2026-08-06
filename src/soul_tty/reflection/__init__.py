"""反思旁路：主对话只投递事件，后台在空闲窗口独立推理并持久化状态。

对外保持与旧 `soul_tty.relationship` 一致的调用方式：

    from . import reflection
    reflection.record_turn(user_text, agent_text)
"""

from .relationship import (
    CompletedTurn,
    EvaluationCallback,
    Evaluator,
    RelationshipState,
    UpdateCallback,
    apply_evaluation,
    level_for,
    load_state,
    safe_persona_id,
    save_state,
    state_path,
)
from .worker import (
    ReflectionWorker,
    close,
    install,
    record_turn,
    user_activity,
)

__all__ = [
    "CompletedTurn",
    "EvaluationCallback",
    "Evaluator",
    "ReflectionWorker",
    "RelationshipState",
    "UpdateCallback",
    "apply_evaluation",
    "close",
    "install",
    "level_for",
    "load_state",
    "record_turn",
    "safe_persona_id",
    "save_state",
    "state_path",
    "user_activity",
]
