"""亲密成长旁路：主对话只投递事件，后台 LLM 独立评估并持久化。"""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import config

_STOP = object()
_SAFE_ID = re.compile(r"[^0-9A-Za-z_.-]+")
_MOODS = {"calm", "happy", "shy", "concerned", "upset", "warm"}
_MECHANISM_VOICE = re.compile(
    r"(?:亲密度|关系|好感度|加分|扣分|分数|等级|阶段|事件|提升|下降|进度|她)"
)


def tier_for(score: int) -> str:
    if score < 15:
        return "初识"
    if score < 35:
        return "熟悉"
    if score < 60:
        return "亲近"
    if score < 85:
        return "默契"
    return "灵魂共鸣"


@dataclass(frozen=True)
class CompletedTurn:
    user_text: str
    agent_text: str


@dataclass(frozen=True)
class RelationshipState:
    score: int = 10
    mood: str = "calm"
    event: str = ""
    inner_voice: str = ""
    session_count: int = 0
    updated_at: str = ""

    @property
    def tier(self) -> str:
        return tier_for(self.score)


Evaluator = Callable[[RelationshipState, CompletedTurn], dict[str, Any] | None]
UpdateCallback = Callable[[RelationshipState], None]


def _clean_inner_voice(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    text = text.splitlines()[0].strip() if text else ""
    text = re.sub(r"^[#>*\-\d.、\s]+", "", text).strip("“”\"' ")
    if _MECHANISM_VOICE.search(text):
        return ""
    return text if 2 <= len(text) <= 18 else ""


def _state_path(persona_id: str, state_dir: Path) -> Path:
    safe_id = _SAFE_ID.sub("-", persona_id).strip("-") or "default"
    return state_dir / "relationships" / f"{safe_id}.json"


def load_state(path: Path) -> RelationshipState:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        score = min(100, max(0, int(data.get("score", 10))))
        return RelationshipState(
            score=score,
            # 情绪与画外音只属于本次会话；关系分数和阶段才跨启动保存。
            mood="calm",
            event=str(data.get("event", ""))[:80],
            inner_voice="",
            session_count=max(0, int(data.get("session_count", 0))),
            updated_at=str(data.get("updated_at", "")),
        )
    except (OSError, ValueError, TypeError):
        return RelationshipState(score=config.RELATIONSHIP_INITIAL_SCORE)


def save_state(path: Path, state: RelationshipState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    durable = asdict(state)
    durable.pop("mood", None)
    durable.pop("inner_voice", None)
    temporary.write_text(
        json.dumps(durable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_evaluation(
    state: RelationshipState,
    result: dict[str, Any] | None,
) -> RelationshipState | None:
    if not isinstance(result, dict):
        return None
    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or confidence < config.RELATIONSHIP_MIN_CONFIDENCE:
        return None
    try:
        delta = int(result.get("delta", 0))
    except (TypeError, ValueError):
        delta = 0
    delta = min(
        config.RELATIONSHIP_MAX_DELTA,
        max(-config.RELATIONSHIP_MAX_DELTA, delta),
    )
    mood = str(result.get("mood", state.mood))
    if mood not in _MOODS:
        mood = state.mood
    return RelationshipState(
        score=min(100, max(0, state.score + delta)),
        mood=mood,
        event=str(result.get("event", ""))[:80],
        inner_voice=_clean_inner_voice(result.get("inner_voice", "")),
        session_count=state.session_count + 1,
        updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


class RelationshipService:
    """单 Worker 有界旁路；队列满或评估失败时静默降级。"""

    def __init__(
        self,
        persona_id: str,
        evaluator: Evaluator,
        on_update: UpdateCallback | None = None,
        *,
        state_dir: Path | None = None,
        queue_size: int | None = None,
        idle_delay_s: float | None = None,
        min_interval_s: float | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.on_update = on_update
        self.path = _state_path(
            persona_id,
            state_dir or config.SOUL_TTY_STATE_DIR,
        )
        self.state = load_state(self.path)
        self.queue: queue.Queue[CompletedTurn | object] = queue.Queue(
            maxsize=queue_size or config.RELATIONSHIP_QUEUE_SIZE
        )
        self.idle_delay_s = (
            config.RELATIONSHIP_IDLE_DELAY_S
            if idle_delay_s is None
            else idle_delay_s
        )
        self.min_interval_s = (
            config.RELATIONSHIP_MIN_INTERVAL_S
            if min_interval_s is None
            else min_interval_s
        )
        self._last_activity = time.monotonic()
        self._last_evaluation = 0.0
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="soul-tty-relationship",
            daemon=True,
        )
        self._thread.start()

    def submit(self, user_text: str, agent_text: str) -> bool:
        if not user_text.strip() or not agent_text.strip() or self._stop.is_set():
            return False
        turn = CompletedTurn(user_text.strip(), agent_text.strip())
        with self._lock:
            # 从完整回答结束开始等待空闲窗口，避免立刻与下一轮主对话争抢模型。
            self._last_activity = time.monotonic()
        try:
            self.queue.put_nowait(turn)
            return True
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                return False
            try:
                self.queue.put_nowait(turn)
                return True
            except queue.Full:
                return False

    def user_activity(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()

    def _wait_for_idle(self) -> bool:
        while not self._stop.is_set():
            with self._lock:
                remaining = self.idle_delay_s - (
                    time.monotonic() - self._last_activity
                )
            if remaining <= 0:
                return True
            if self._stop.wait(min(remaining, 0.1)):
                return False
        return False

    def _wait_for_evaluation_slot(self) -> bool:
        """限制旁路推理频率，并在等待后重新确认用户仍处于空闲。"""
        remaining = self.min_interval_s - (
            time.monotonic() - self._last_evaluation
        )
        if remaining > 0 and self._stop.wait(remaining):
            return False
        return self._wait_for_idle()

    def _coalesce_pending(self, first: CompletedTurn) -> CompletedTurn:
        """把冷却窗口内积累的多轮合并成一次 LLM 关系评估。"""
        turns = [first]
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(item, CompletedTurn):
                    turns.append(item)
            finally:
                self.queue.task_done()
        if len(turns) == 1:
            return first
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

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self.queue.get()
            try:
                if item is _STOP or not isinstance(item, CompletedTurn):
                    return
                if not self._wait_for_idle():
                    return
                if not self._wait_for_evaluation_slot():
                    return
                item = self._coalesce_pending(item)
                with self._lock:
                    current = self.state
                self._last_evaluation = time.monotonic()
                try:
                    result = self.evaluator(current, item)
                except Exception:
                    continue
                if self._stop.is_set():
                    return
                updated = apply_evaluation(current, result)
                if updated is None:
                    continue
                try:
                    save_state(self.path, updated)
                except OSError:
                    continue
                with self._lock:
                    self.state = updated
                if self.on_update is not None:
                    try:
                        self.on_update(updated)
                    except Exception:
                        pass
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=0.2)


_service: RelationshipService | None = None


def install(service: RelationshipService | None) -> None:
    global _service
    _service = service


def record_turn(user_text: str, agent_text: str) -> bool:
    return _service.submit(user_text, agent_text) if _service is not None else False


def user_activity() -> None:
    if _service is not None:
        _service.user_activity()


def close() -> None:
    global _service
    if _service is not None:
        _service.stop()
        _service = None
