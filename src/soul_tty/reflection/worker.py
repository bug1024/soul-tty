"""反思旁路 Worker：主对话只投递事件，后台按空闲窗口独立推理。

Worker 只负责调度（队列、空闲门控、限频、合并），具体推理任务由外部注入：
- relationship evaluator：本轮互动对关系/情绪/表达的影响
- memory extractor（后续接入）：本轮互动里值得长期保存的事实

「共享调度，不共享推理」——两类反思共用一次空闲窗口，但各自独立 prompt。
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from .. import config
from .relationship import (
    CompletedTurn,
    EvaluationCallback,
    Evaluator,
    RelationshipState,
    UpdateCallback,
    apply_evaluation,
    load_state,
    save_state,
    state_path,
)

_STOP = object()

# 记忆抽取器签名：消费一批 turn，返回是否至少落库一条。
# 失败/未抽取时返回 False，worker 据此保留 buffer 不动。
MemoryExtractor = Callable[[list[CompletedTurn]], bool]
# 抽取完成后触发，签名无参；通常用于让 system prompt 重新渲染常驻段。
OnMemoryUpdated = Callable[[], None]


class ReflectionWorker:
    """单 Worker 有界旁路；只管 bond 持久化，emotion/expression 透传给上层协调器。"""

    def __init__(
        self,
        persona_id: str,
        evaluator: Evaluator,
        on_update: UpdateCallback | None = None,
        on_evaluation: EvaluationCallback | None = None,
        memory_extractor: MemoryExtractor | None = None,
        on_memory_updated: OnMemoryUpdated | None = None,
        *,
        state_dir: Path | None = None,
        queue_size: int | None = None,
        idle_delay_s: float | None = None,
        min_interval_s: float | None = None,
        memory_min_interval_s: float | None = None,
        memory_buffer_turns: int | None = None,
        memory_min_text_chars: int | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.on_update = on_update
        self.on_evaluation = on_evaluation
        self.memory_extractor = memory_extractor
        self.on_memory_updated = on_memory_updated
        self.path = state_path(
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
        self.memory_min_interval_s = (
            config.MEMORY_MIN_INTERVAL_S
            if memory_min_interval_s is None
            else memory_min_interval_s
        )
        self.memory_min_text_chars = (
            config.MEMORY_MIN_TEXT_CHARS
            if memory_min_text_chars is None
            else memory_min_text_chars
        )
        # 记忆抽取的待办轮次；与关系评估的 queue 解耦——queue 溢出丢轮
        # 可以接受，记忆不能丢。每条 turn 配一个递增 seq。
        # maxlen 天然防止无限增长。
        self._memory_buffer: deque[tuple[int, CompletedTurn]] = deque(
            maxlen=memory_buffer_turns or config.MEMORY_BUFFER_TURNS
        )
        self._memory_seq = 0
        self._last_memory_at = 0.0
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
            name="soul-tty-reflection",
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
            # 记忆 buffer：无条件追加；deque 在 maxlen 满了之后静默丢最旧
            self._memory_seq += 1
            self._memory_buffer.append((self._memory_seq, turn))
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
                # 每轮评估 +1：无论 confidence 够不够，HUD 上的「互动次数」计数都该往上走。
                incremented = RelationshipState(
                    bond=current.bond,
                    recent_events=current.recent_events,
                    inner_voice=current.inner_voice,
                    interaction_count=current.interaction_count + 1,
                    updated_at=datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                )
                updated_payload = apply_evaluation(incremented, result)
                final_state = (
                    updated_payload["relationship"]
                    if updated_payload is not None
                    else incremented
                )
                try:
                    save_state(self.path, final_state)
                except OSError:
                    continue
                with self._lock:
                    self.state = final_state
                if self.on_update is not None:
                    try:
                        self.on_update(final_state)
                    except Exception:
                        pass
                # 把 apply_evaluation 的完整 payload 抛给上层协调器；
                # emotion / expression 的应用都不在 ReflectionWorker 的职责里。
                if updated_payload is not None and self.on_evaluation is not None:
                    try:
                        self.on_evaluation(updated_payload)
                    except Exception:
                        pass
                # 关系评估完成后串行跑记忆抽取；共享 idle 窗口，两次独立推理。
                # 失败/未抽取都返回 False，buffer 不动，下次连带重试。
                if self.memory_extractor is not None:
                    self._maybe_extract_memory()
            finally:
                self.queue.task_done()

    def _memory_due(self) -> bool:
        """是否到了抽取时机。

        两条独立条件：
        1. 距上次抽取 ≥ memory_min_interval_s（默认 120s）
        2. buffer 里未抽取的用户文本累计 ≥ memory_min_text_chars

        任意一条不满足就跳过——这避免「每轮都问 LLM」以及「嗯/好的也发 LLM」。
        """
        with self._lock:
            if time.monotonic() - self._last_memory_at < self.memory_min_interval_s:
                return False
            if not self._memory_buffer:
                return False
            pending_chars = sum(
                len(turn.user_text) for _, turn in self._memory_buffer
            )
            return pending_chars >= self.memory_min_text_chars

    def _drain_memory_buffer(self) -> tuple[list[CompletedTurn], int]:
        """取走 buffer 中所有 turn 与最大 seq。返回 (turns, max_seq)。

        不真正清空 buffer——LLM 推理期间可能有新 turn 进来，
        ack 由调用方按 max_seq 决定保留哪些。
        """
        with self._lock:
            if not self._memory_buffer:
                return [], 0
            turns = [turn for _, turn in self._memory_buffer]
            max_seq = max(seq for seq, _ in self._memory_buffer)
            return turns, max_seq

    def _ack_memory_extracted(self, max_seq: int) -> int:
        """成功抽取后，移除 seq ≤ max_seq 的条目；返回实际移除条数。"""
        with self._lock:
            cap = self._memory_buffer.maxlen
            kept = deque(
                (seq, turn)
                for seq, turn in self._memory_buffer
                if seq > max_seq
            )
            # deque 的 maxlen 在构造时设定，不可写——必须新建
            new_buffer: deque[tuple[int, CompletedTurn]] = deque(
                kept, maxlen=cap
            )
            removed = len(self._memory_buffer) - len(new_buffer)
            self._memory_buffer = new_buffer
            return removed

    def _maybe_extract_memory(self) -> None:
        if not self._memory_due():
            return
        turns, max_seq = self._drain_memory_buffer()
        if not turns or self.memory_extractor is None:
            return
        try:
            landed = bool(self.memory_extractor(turns))
        except Exception:
            return  # 失败时 buffer 不动
        with self._lock:
            self._last_memory_at = time.monotonic()
        if landed:
            self._ack_memory_extracted(max_seq)
            if self.on_memory_updated is not None:
                try:
                    self.on_memory_updated()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop.set()
        try:
            self.queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        self._stop.set()
        try:
            self.queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=0.2)


_service: ReflectionWorker | None = None


def install(service: ReflectionWorker | None) -> None:
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
