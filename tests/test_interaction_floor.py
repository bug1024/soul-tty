"""FloorManager 状态机测试。"""

import threading

import pytest

from soul_tty.interaction.floor import FloorManager, FloorState


def test_initial_state_is_idle():
    fm = FloorManager()
    assert fm.state == FloorState.IDLE


def test_agent_take_floor():
    fm = FloorManager()
    fm.agent_start()
    assert fm.state == FloorState.AGENT_SPEAKING
    fm.agent_end()
    assert fm.state == FloorState.IDLE


def test_user_start_takes_floor():
    fm = FloorManager()
    fm.user_start()
    assert fm.state == FloorState.USER_SPEAKING


def test_user_during_agent_interrupts():
    """Agent 在说话时用户开口 → 立即切到 INTERRUPTED。"""
    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("你好世界")
    fm.user_start()
    assert fm.state == FloorState.INTERRUPTED


def test_user_partial_does_not_interrupt_when_idle():
    fm = FloorManager()
    assert fm.user_partial("你好") is False


def test_user_partial_interrupts_agent():
    """用户 partial 与已播放 agent 文本不同 → 打断。"""
    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("今天天气真不错")
    # 完全不同的内容
    assert fm.user_partial("明天会下雨") is True
    assert fm.state == FloorState.INTERRUPTED


def test_user_partial_echo_does_not_interrupt():
    """用户 partial 是 agent 的回声 → 不打断。"""
    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("今天天气真不错")
    # 完整复述 → 命中 _is_probable_echo
    assert fm.user_partial("今天天气真不错") is False
    assert fm.state == FloorState.AGENT_SPEAKING


def test_user_final_returns_to_idle():
    fm = FloorManager()
    fm.user_start()
    fm.user_final("你好")
    assert fm.state == FloorState.IDLE


def test_user_final_records_interrupt_text():
    fm = FloorManager()
    fm.agent_start()
    fm.user_start()  # 进入 INTERRUPTED
    interrupted = fm.user_final("打断的话")
    assert interrupted is True
    assert fm.last_interrupt == "打断的话"
    assert fm.state == FloorState.IDLE


def test_user_final_no_interrupt_returns_false():
    """用户在没有 agent 时说话 → 不算打断。"""
    fm = FloorManager()
    fm.user_start()
    assert fm.user_final("你好") is False


def test_on_interrupt_callback_fires():
    fired = []
    fm = FloorManager(on_interrupt=lambda t: fired.append(t))
    fm.agent_start()
    fm.user_start()
    fm.user_final("打断的话")
    assert fired == ["打断的话"]


def test_on_interrupt_callback_exception_swallowed():
    """callback 抛异常不影响 state 推进。"""
    def bad(_):
        raise RuntimeError("oops")

    fm = FloorManager(on_interrupt=bad)
    fm.agent_start()
    fm.user_start()
    fm.user_final("打断的话")
    assert fm.state == FloorState.IDLE


def test_agent_chunk_accumulates():
    """agent_text 跨多次 chunk 累加,用于回声判定。"""
    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("你好")
    fm.agent_chunk("世界")
    assert fm.agent_text == "你好世界"


def test_agent_end_clears_text():
    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("你好")
    fm.agent_end()
    assert fm.agent_text == ""


def test_reset_brings_to_idle():
    fm = FloorManager()
    fm.agent_start()
    fm.user_start()  # INTERRUPTED
    fm.reset()
    assert fm.state == FloorState.IDLE


def test_user_partial_after_interrupt_does_not_double_interrupt():
    """已经 INTERRUPTED 后再收 partial → 不再触发新的中断事件(state 已变)。"""
    fm = FloorManager()
    fm.agent_start()
    fm.user_start()  # INTERRUPTED
    # 此时 state == INTERRUPTED,不是 AGENT_SPEAKING → 不返回 True
    assert fm.user_partial("更多") is False


def test_thread_safety_state_changes():
    """并发 user/agent 调用不能破坏 state 一致性。"""
    fm = FloorManager()
    errors = []

    def worker(tag):
        try:
            for _ in range(1000):
                if tag == "u":
                    fm.user_start()
                    fm.user_final("x")
                else:
                    fm.agent_start()
                    fm.agent_chunk(".")
                    fm.agent_end()
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=("u",)),
        threading.Thread(target=worker, args=("a",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # 最终状态必须是合法的
    assert fm.state in (
        FloorState.IDLE,
        FloorState.AGENT_SPEAKING,
        FloorState.USER_SPEAKING,
        FloorState.INTERRUPTED,
    )


def test_user_start_during_user_speaking_is_idempotent():
    """重复 user_start 不应当把 state 弄到非法值。"""
    fm = FloorManager()
    fm.user_start()
    fm.user_start()
    assert fm.state == FloorState.USER_SPEAKING


def test_agent_start_during_user_speaking_does_not_take_floor():
    """Agent 不应在用户说话时抢 floor(交给上层 cancel-on-interrupt 决策)。"""
    fm = FloorManager()
    fm.user_start()
    fm.agent_start()
    assert fm.state == FloorState.USER_SPEAKING


def test_echo_similarity_override():
    """构造时传入 echo_similarity 必须能覆盖 config 默认值。"""
    fm = FloorManager(echo_similarity=0.99)
    assert fm._echo_similarity == 0.99