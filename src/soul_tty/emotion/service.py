"""EmotionService：情绪系统顶层协调（状态、应用 delta、idle decay、节流更新）。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import analyzer, resolver, updater
from .prompt_builder import build_emotion_context
from .state import (
    DEFAULT_BASELINE,
    EmotionVector,
    new_session_id,
    save_emotion_state,
)


@dataclass(frozen=True)
class EmotionSnapshot:
    baseline: EmotionVector
    emotion: EmotionVector
    mood: str
    intensity: float
    expression: str
    should_update_prompt: bool
    context_text: str


UpdateHook = Callable[[EmotionSnapshot], None]


class EmotionService:
    """单进程持有；惰性开启 idle decay 线程。"""

    def __init__(
        self,
        persona_id: str,
        baseline: EmotionVector | None = None,
        *,
        state_dir: Path | None = None,
        jitter: float = 0.1,
        seed: int | None = None,
        ema_rate: float = 0.2,
        delta_cap: float = 0.3,
        decay_rate: float = 0.05,
        intensity_update_threshold: float = 0.1,
        on_update: UpdateHook | None = None,
        decay_interval_s: float = 300.0,
        idle_threshold_s: float = 300.0,
    ) -> None:
        self.persona_id = persona_id
        self.baseline = baseline if baseline is not None else DEFAULT_BASELINE
        self.state_dir = state_dir
        self._ema_rate = ema_rate
        self._delta_cap = delta_cap
        self._decay_rate = decay_rate
        self._intensity_threshold = intensity_update_threshold
        self._on_update = on_update
        self._decay_interval_s = decay_interval_s
        self._idle_threshold_s = idle_threshold_s
        self._session_id = new_session_id()
        # 启动扰动
        self._emotion = updater.perturb_baseline(
            self.baseline, jitter=jitter, seed=seed
        )
        self._expression = "neutral"
        self._prev_mood, self._prev_intensity = resolver.resolve_mood(self._emotion)
        # spec §6.4：mood / intensity / expression 任一变化都触发 prompt 热更新。
        self._prev_expression = "neutral"
        self._lock = threading.RLock()
        self._last_activity = time.monotonic()
        self._stop = threading.Event()
        self._decay_thread: threading.Thread | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    def snapshot(self) -> EmotionSnapshot:
        with self._lock:
            mood, intensity = resolver.resolve_mood(self._emotion)
            should_update = self._should_update(mood, intensity, self._expression)
            return EmotionSnapshot(
                baseline=self.baseline,
                emotion=self._emotion,
                mood=mood,
                intensity=intensity,
                expression=self._expression,
                should_update_prompt=should_update,
                context_text=build_emotion_context(
                    mood, intensity, expression=self._expression
                ),
            )

    def apply_delta(
        self,
        delta: dict[str, float] | object,
        *,
        expression_hint: str = "neutral",
    ) -> EmotionSnapshot:
        cleaned = analyzer.parse_emotion_delta(delta, delta_cap=self._delta_cap)
        with self._lock:
            new_emotion = updater.apply_delta(
                self._emotion,
                cleaned,
                rate=self._ema_rate,
                delta_cap=self._delta_cap,
            )
            self._emotion = new_emotion
            hint = (expression_hint or "").strip().lower()
            self._expression = hint if hint in ("neutral", "caring") else "neutral"
            self._last_activity = time.monotonic()
            snap = self.snapshot()
            self._persist(snap)
            if self._on_update is not None:
                try:
                    self._on_update(snap)
                except Exception:
                    pass
            self._prev_mood = snap.mood
            self._prev_intensity = snap.intensity
            self._prev_expression = snap.expression
            return snap

    def _should_update(self, mood: str, intensity: float, expression: str) -> bool:
        """spec §6.4：mood 切换 / |Δintensity|>阈值 / expression 切换 任一即触发。"""
        if mood != self._prev_mood:
            return True
        if abs(intensity - self._prev_intensity) > self._intensity_threshold:
            return True
        if expression != self._prev_expression:
            return True
        return False

    def _persist(self, snap: EmotionSnapshot) -> None:
        if self.state_dir is None:
            return
        try:
            save_emotion_state(
                self.persona_id,
                self.state_dir,
                self._session_id,
                snap.baseline,
                snap.emotion,
                datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        except OSError:
            pass

    def user_activity(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()

    def start_decay_thread(self) -> None:
        if self._decay_thread is not None:
            return
        self._decay_thread = threading.Thread(
            target=self._run_decay, name="soul-tty-emotion-decay", daemon=True
        )
        self._decay_thread.start()

    def _run_decay(self) -> None:
        while not self._stop.wait(self._decay_interval_s):
            with self._lock:
                idle = time.monotonic() - self._last_activity
                if idle < self._idle_threshold_s:
                    continue
                self._emotion = updater.apply_decay(
                    self._emotion, self.baseline, rate=self._decay_rate
                )
            snap = self.snapshot()
            self._persist(snap)
            if self._on_update is not None:
                try:
                    self._on_update(snap)
                except Exception:
                    pass
            self._prev_mood = snap.mood
            self._prev_intensity = snap.intensity
            self._prev_expression = snap.expression

    def stop(self) -> None:
        self._stop.set()
        if self._decay_thread is not None:
            self._decay_thread.join(timeout=0.2)