"""Backchannel（轻量肯定词）决策测试。

Backchannel = 用户在 agent 说话时插一句"嗯/好的/是",不算打断,只记下来。
见 ``soul_tty.interaction.floor.is_backchannel`` / ``FloorManager.user_partial``。
"""

from soul_tty.interaction import (
    BACKCHANNEL_WORDS,
    FloorManager,
    FloorState,
    is_backchannel,
)


# ── is_backchannel 纯函数测试 ──────────────────────────────────────


def test_is_backchannel_recognizes_common_affirmations():
    for word in ("嗯", "啊", "好的", "对", "嗯嗯", "哦"):
        assert is_backchannel(word), f"{word!r} 应当被识别为 backchannel"


def test_is_backchannel_rejects_long_text():
    """长度 > 3 的句子就算包含肯定词也不算 backchannel。"""
    assert not is_backchannel("嗯听起来不错")
    assert not is_backchannel("好的我去看看")
    assert not is_backchannel("嗯嗯这个问题很有趣")


def test_is_backchannel_rejects_empty():
    assert not is_backchannel("")
    assert not is_backchannel("   ")
    assert not is_backchannel("。。")


def test_is_backchannel_strips_punctuation():
    """标点/空白先剥掉,只比对核心字符。"""
    assert is_backchannel("嗯。")
    assert is_backchannel("好的!")
    assert is_backchannel("嗯?")
    assert is_backchannel("  对  ")


def test_is_backchannel_rejects_unknown_words():
    """不在白名单里的词不算 backchannel。"""
    assert not is_backchannel("你好")
    assert not is_backchannel("不是")
    assert not is_backchannel("再见")


def test_backchannel_words_is_frozenset():
    assert isinstance(BACKCHANNEL_WORDS, frozenset)


# ── FloorManager 集成测试 ──────────────────────────────────────


def _make_manager(backchannel_enabled=True) -> FloorManager:
    return FloorManager(backchannel_enabled=backchannel_enabled)


def test_backchannel_does_not_interrupt():
    """backchannel 不算打断:state 保持 AGENT_SPEAKING,返回 False。"""
    fm = _make_manager()
    fm.agent_start()
    assert fm.state == FloorState.AGENT_SPEAKING
    interrupted = fm.user_partial("嗯")
    assert interrupted is False
    assert fm.state == FloorState.AGENT_SPEAKING  # 没变


def test_backchannel_records_pending():
    """backchannel 记到 pending_backchannel。"""
    fm = _make_manager()
    fm.agent_start()
    fm.user_partial("好的")
    assert fm.pending_backchannel == "好的"


def test_take_backchannel_clears():
    """take_backchannel 取走并清空。"""
    fm = _make_manager()
    fm.agent_start()
    fm.user_partial("嗯")
    assert fm.take_backchannel() == "嗯"
    assert fm.take_backchannel() is None  # 二次取为空
    assert fm.pending_backchannel is None


def test_backchannel_when_disabled_interrupts():
    """BACKCHANNEL_ENABLED=False 时,'嗯' 走 interrupt 路径。"""
    fm = _make_manager(backchannel_enabled=False)
    fm.agent_start()
    interrupted = fm.user_partial("嗯")
    assert interrupted is True
    assert fm.state == FloorState.INTERRUPTED


def test_long_text_still_interrupts_even_with_backchannel_on():
    """'嗯但是我觉得' 这种长文本不受 backchannel 白名单影响,正常打断。"""
    fm = _make_manager()
    fm.agent_start()
    interrupted = fm.user_partial("嗯但是我觉得应该")
    assert interrupted is True
    assert fm.state == FloorState.INTERRUPTED


def test_backchannel_only_in_agent_speaking():
    """IDLE 时 backchannel 也不打断(本来 partial 也不该打断)。"""
    fm = _make_manager()
    # 不 agent_start,state == IDLE
    interrupted = fm.user_partial("嗯")
    assert interrupted is False
    # 没 agent_start,所以也没记 pending
    assert fm.pending_backchannel is None


def test_real_interrupt_after_backchannel():
    """先 backchannel,再真插话:backchannel 应被清,真打断生效。"""
    fm = _make_manager()
    fm.agent_start()
    fm.user_partial("嗯")  # backchannel
    assert fm.pending_backchannel == "嗯"

    # 真打断
    fm.user_partial("等一下这个问题不对")
    assert fm.state == FloorState.INTERRUPTED

    # 真 final 后 pending 被清
    was_interrupted = fm.user_final("等一下这个问题不对", pcm=b"\x00\x00" * 16)
    assert was_interrupted is True
    assert fm.pending_backchannel is None


def test_backchannel_user_final_clears_pending():
    """用户说完一句 final(非空)后,pending backchannel 一定被清。"""
    fm = _make_manager()
    fm.agent_start()
    fm.user_partial("嗯")
    assert fm.pending_backchannel == "嗯"

    # 句尾 final 带文本(虽然没有真打断)
    fm.user_final("好的", pcm=b"\x00\x00" * 16)
    assert fm.pending_backchannel is None


def test_consecutive_backchannels_only_keep_last():
    """连续多个 backchannel,只保留最新的(避免污染下一轮)。"""
    fm = _make_manager()
    fm.agent_start()
    fm.user_partial("嗯")
    fm.user_partial("嗯嗯")
    fm.user_partial("好的")
    assert fm.pending_backchannel == "好的"