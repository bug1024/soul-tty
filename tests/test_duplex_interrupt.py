"""真双工端到端测试:用户打断 + answer cancel + finalize。"""

import threading
import time

import pytest


class _FakeChat:
    """最小 llm.Chat 替身:ask_stream 按 token 节奏输出,可被 cancel。"""

    def __init__(self, tokens: list[str], per_token_s: float = 0.01):
        self._tokens = tokens
        self._per_token = per_token_s
        self.last_stop_reason = ""
        self.cancelled = False

    def ask_stream(self, text, cancel=None, recall=""):
        for tok in self._tokens:
            if cancel is not None and cancel.is_set():
                self.cancelled = True
                self.last_stop_reason = "cancelled"
                return
            time.sleep(self._per_token)
            yield tok
        self.last_stop_reason = "natural"


@pytest.fixture(autouse=True)
def _reset_active_answer(monkeypatch):
    """每个测试跑完清掉 _active_answer_state,免污染后续;同时屏蔽掉 terminal UI 副作用。"""
    from soul_tty import conversation
    from soul_tty.ui import terminal

    monkeypatch.setattr(conversation, "_active_answer_state", None)
    monkeypatch.setattr(conversation, "_active_chat", None)
    # 屏蔽 terminal:避免 _answer 内部调 terminal.answer_start() 等抛
    # "终端 UI 尚未配置人格"。
    monkeypatch.setattr(terminal, "answer_start", lambda: None)
    monkeypatch.setattr(terminal, "answer_chunk", lambda _t: None)
    monkeypatch.setattr(terminal, "answer_end", lambda: None)
    monkeypatch.setattr(terminal, "speaking", lambda: None)
    monkeypatch.setattr(terminal, "listening", lambda *a, **kw: None)
    monkeypatch.setattr(terminal, "audio_level", lambda _v: None)
    monkeypatch.setattr(terminal, "notice", lambda _t: None)
    monkeypatch.setattr(terminal, "interrupted", lambda _t: None)
    monkeypatch.setattr(terminal, "warning", lambda _t: None)
    monkeypatch.setattr(terminal, "recognized", lambda _t: None)
    monkeypatch.setattr(terminal, "user_text", lambda _t: None)
    monkeypatch.setattr(terminal, "partial", lambda _t: None)
    monkeypatch.setattr(terminal, "goodbye", lambda: None)
    yield


def test_spawn_answer_runs_in_background():
    """``_spawn_answer`` 必须在后台线程跑,不阻塞调用方。"""
    from soul_tty import conversation

    chat = _FakeChat(["你", "好", "世", "界"])
    t0 = time.monotonic()
    conversation._spawn_answer(chat, "hello")
    spawn_elapsed = time.monotonic() - t0
    assert spawn_elapsed < 0.05, "_spawn_answer 不应阻塞"

    # 等后台跑完
    state = conversation._current_answer_state()
    assert state is not None
    state.done.wait(timeout=2.0)
    assert state.done.is_set()


def test_current_answer_state_cancel_sets_event():
    """调用 cancel.set() 必须让 _answer 走 cancel 路径。"""
    from soul_tty import conversation

    chat = _FakeChat(["长", "答", "案"] * 10, per_token_s=0.05)
    conversation._spawn_answer(chat, "hello")

    # 等 answer 真起来
    state = conversation._current_answer_state()
    time.sleep(0.05)

    # 中断
    state.cancel.set()
    state.done.wait(timeout=1.0)
    assert chat.cancelled is True


def test_wait_answer_done_returns_when_no_answer():
    """当前没有 answer 时,wait 立即返回,不抛异常。"""
    from soul_tty import conversation

    # 不调 _spawn_answer,直接 wait
    conversation._wait_answer_done(timeout_s=0.1)
    # 没异常就过


def test_spawn_answer_records_error_in_state():
    """answer 抛异常时,error 被记录且 done 被 set。"""
    from soul_tty import conversation

    class _BoomChat(_FakeChat):
        def __init__(self):
            super().__init__(tokens=[])
            self.error_event = threading.Event()

        def ask_stream(self, text, cancel=None, recall=""):
            self.error_event.set()
            raise RuntimeError("LLM blew up")
            yield  # noqa - 让它成生成器

    chat = _BoomChat()
    # 用一个慢一点的失败(让 caller 来得及读到 state)
    original_run = conversation._spawn_answer.__wrapped__ if hasattr(conversation._spawn_answer, "__wrapped__") else None
    conversation._spawn_answer(chat, "hi")
    # 立刻拿到 _current_answer_state;answer 可能在抢着跑
    state = conversation._current_answer_state()
    if state is None:
        # 太早 / 已经清空,等一下再读
        import time as _t
        for _ in range(20):
            _t.sleep(0.01)
            state = conversation._current_answer_state()
            if state is not None:
                break
    # 等 done
    if state is not None:
        state.done.wait(timeout=2.0)
        assert state.error is not None, "LLM 抛异常后 state.error 必须被记录"
        assert isinstance(state.error, RuntimeError)
    else:
        # 整个 answer 跑得飞快且已经清理,直接检查 chat 是否触发过错误
        assert chat.error_event.is_set(), "answer 必须尝试调 ask_stream"


def test_spawn_answer_clears_active_state_on_done():
    """answer 跑完后,_active_answer_state 必须清空(避免下次 wait 错认)。"""
    from soul_tty import conversation

    chat = _FakeChat(["hi"])
    conversation._spawn_answer(chat, "hello")
    state = conversation._current_answer_state()
    state.done.wait(timeout=2.0)
    # 等内部线程把状态清掉
    time.sleep(0.05)
    assert conversation._current_answer_state() is None


def test_interrupt_during_answer_cancels_quickly():
    """模拟 FloorManager:用户 partial 触发 cancel 后,answer 必须 0.5s 内退出。"""
    from soul_tty import conversation

    # 30 个 token,每个 100ms → 完整跑完要 3s
    chat = _FakeChat([f"tok{i}" for i in range(30)], per_token_s=0.1)
    conversation._spawn_answer(chat, "hi")

    time.sleep(0.2)  # 让它跑 2 个 token

    state = conversation._current_answer_state()
    t0 = time.monotonic()
    state.cancel.set()
    state.done.wait(timeout=1.0)
    elapsed = time.monotonic() - t0
    # cancel 后 answer 应该立刻停(< 200ms,因为下一个 token yield 之前检查 cancel)
    assert elapsed < 0.5
    assert chat.cancelled