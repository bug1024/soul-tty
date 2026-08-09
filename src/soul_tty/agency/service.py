"""Agency 顶层服务：决策在内存完成，状态写盘走阻塞队列旁路。"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from .. import observability
from .policy import ResponseDecision, ResponsePolicy
from .state import (
    AgencyState,
    continuity_update,
    evolve_after_decision,
    load_state,
    save_state,
)


class AgencyService:
    def __init__(
        self,
        state_path: Path,
        *,
        policy: ResponsePolicy | None = None,
    ) -> None:
        self.state_path = state_path
        self.policy = policy or ResponsePolicy()
        self._lock = threading.RLock()
        self._state = continuity_update(load_state(state_path))
        self._session_turns = 0
        self._writes: queue.Queue[AgencyState | None] = queue.Queue(maxsize=1)
        self._writer = threading.Thread(
            target=self._write_loop,
            name="soul-tty-agency-state",
            daemon=True,
        )
        self._writer.start()
        self._schedule_save(self._state)

    @property
    def state(self) -> AgencyState:
        with self._lock:
            return self._state

    def decide(
        self,
        user_text: str,
        *,
        mood: str = "calm",
        relationship_level: str = "",
    ) -> ResponseDecision:
        with self._lock:
            # 同一进程内长时间安静也算 continuity，轻微恢复社交能量。
            self._state = continuity_update(self._state)
            decision = self.policy.decide(
                self._state,
                user_text,
                relationship_level=relationship_level,
                session_turn_count=self._session_turns,
            )
            self._state = evolve_after_decision(
                self._state,
                mode=decision.mode.value,
                mood=mood,
                user_text=user_text,
            )
            snapshot = self._state
            self._session_turns += 1
        self._schedule_save(snapshot)
        return decision

    def _schedule_save(self, state: AgencyState) -> None:
        try:
            self._writes.put_nowait(state)
            return
        except queue.Full:
            pass
        try:
            self._writes.get_nowait()
        except queue.Empty:
            pass
        try:
            self._writes.put_nowait(state)
        except queue.Full:
            pass

    def _write_loop(self) -> None:
        while True:
            state = self._writes.get()
            if state is None:
                return
            try:
                save_state(self.state_path, state)
            except OSError:
                observability.exception(
                    "agency.state.save_error",
                    "Agency 状态写入失败，本轮继续使用内存状态",
                )

    def close(self) -> None:
        # 先停 writer，再同步写最终快照；避免一个较旧的排队快照在最终写后
        # 才落盘，把最新状态覆盖回去。
        try:
            self._writes.get_nowait()
        except queue.Empty:
            pass
        try:
            self._writes.put_nowait(None)
        except queue.Full:
            return
        self._writer.join(timeout=1.0)
        try:
            save_state(self.state_path, self.state)
        except OSError:
            observability.exception(
                "agency.state.save_error",
                "Agency 最终状态写入失败",
            )
