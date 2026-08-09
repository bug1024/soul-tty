"""FloorManager + backchannel 测试。

commit 07+ 修复:
- ``user_start()`` 不再将 AGENT_SPEAKING → INTERRUPTED(SPEECH_START 不等于打断)。
- 打断决策由 ``user_partial()`` 做:非回声 + 非 backchannel 的文本才触发。
"""

import threading

from soul_tty.interaction import (
    UserFinalDisposition,
    UserPartialDisposition,
    BACKCHANNEL_WORDS,
    FloorManager,
    FloorState,
    is_explicit_interrupt,
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


# ── FloorManager 状态机测试 ──────────────────────────────────────


def test_initial_state_is_idle():
    fm = FloorManager()
    assert fm.state == FloorState.IDLE


def test_user_start_transitions_to_user_speaking():
    fm = FloorManager()
    fm.user_start()
    assert fm.state == FloorState.USER_SPEAKING


def test_user_start_during_agent_stays_agent_speaking():
    """修复(commit 07+):SPEECH_START 不等于打断,user_start 不改变 AGENT_SPEAKING 状态。"""
    fm = FloorManager()
    fm.agent_start()
    fm.user_start()
    assert fm.state == FloorState.AGENT_SPEAKING


def test_user_partial_during_agent_triggers_interrupt():
    """非回声、非 backchannel 的 partial → INTERRUPTED。"""
    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("你好世界")
    assert fm.user_partial("等一下") is True
    assert fm.state == FloorState.INTERRUPTED


def test_user_partial_echo_does_not_interrupt():
    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("今天天气真不错")
    # 完整复述 → 命中 _is_probable_echo
    assert fm.user_partial("今天天气真不错") is False
    assert fm.state == FloorState.AGENT_SPEAKING


def test_two_character_exact_echo_does_not_interrupt():
    """中文短 partial 无法做模糊比较，但精确命中播放文本时仍应过滤。"""
    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("你好，今天想聊些什么？")
    assert fm.user_partial("你好") is False
    assert fm.last_partial_disposition == UserPartialDisposition.ECHO
    assert fm.state == FloorState.AGENT_SPEAKING


def test_short_ambiguous_partial_waits_for_more_evidence():
    """非回声的 1~2 字 partial 不应立刻抢走话权。"""
    fm = FloorManager(partial_min_chars=3)
    fm.agent_start()
    fm.agent_chunk("我正在解释音频处理流程")
    assert fm.user_partial("可以") is False
    assert fm.last_partial_disposition == UserPartialDisposition.HOLD
    assert fm.state == FloorState.AGENT_SPEAKING


def test_short_explicit_command_interrupts_immediately():
    fm = FloorManager(partial_min_chars=3)
    fm.agent_start()
    fm.agent_chunk("我正在解释音频处理流程")
    assert is_explicit_interrupt("停！")
    assert fm.user_partial("停") is True
    assert fm.last_partial_disposition == UserPartialDisposition.INTERRUPT
    assert fm.state == FloorState.INTERRUPTED


def test_explicit_stop_survives_asr_leading_noise():
    assert is_explicit_interrupt("呀好啦停下别说了")
    # 容易出现在正常句子中的纠正词仍不能做句中触发。
    assert not is_explicit_interrupt("我觉得这不是问题")


def test_explicit_interrupt_does_not_require_near_end_gate():
    """低音量明确制止词必须能绕过声学门控。"""
    fm = FloorManager(partial_min_chars=3)
    fm.agent_start()

    assert fm.user_partial("停下，别说了", near_end=False) is True
    assert fm.last_partial_disposition == UserPartialDisposition.INTERRUPT
    assert fm.user_final(
        "停下，别说了",
        near_end=False,
    ) == UserFinalDisposition.INTERRUPT


def test_non_explicit_aec_residual_without_near_end_is_ignored():
    """没有近端声学证据的普通错误文本不得触发自激。"""
    fm = FloorManager(partial_confirmations=1)
    fm.agent_start()
    fm.agent_chunk("今天想聊些什么？")

    assert fm.user_partial("呀想想了", near_end=False) is False
    assert fm.last_partial_disposition == UserPartialDisposition.HOLD
    assert fm.user_final(
        "呀想想了",
        near_end=False,
    ) == UserFinalDisposition.IGNORED


def test_tail_overlap_user_is_committed_after_agent_finishes():
    """用户在播放尾部开始说话，等播放结束后不应要求重说。"""
    fm = FloorManager(natural_interrupt_enabled=False)
    fm.agent_start()
    fm.agent_chunk("Serena 正在播放回答的最后一句")
    fm.user_start()
    assert fm.user_partial("现在测试语音打断功能", near_end=True) is False

    # 用户还在说时，播放队列自然排空。
    fm.agent_end()
    assert fm.user_final(
        "现在测试语音打断功能",
        near_end=True,
    ) == UserFinalDisposition.USER


def test_non_explicit_user_final_is_ignored_while_agent_still_speaking():
    fm = FloorManager(natural_interrupt_enabled=False)
    fm.agent_start()
    fm.agent_chunk("Serena 仍在播放")
    fm.user_start()

    assert fm.user_final(
        "现在测试语音打断功能",
        near_end=True,
    ) == UserFinalDisposition.IGNORED


def test_strong_near_end_final_interrupts_when_stop_words_are_garbled():
    """制止词被 ASR 识别坏时，高能近端整句仍应在 FINAL 阶段叫停。"""
    fm = FloorManager(
        natural_interrupt_enabled=False,
        strong_interrupt_rms=0.020,
        strong_interrupt_min_chars=3,
    )
    fm.agent_start()
    fm.agent_chunk("Serena 仍在播放一段较长回答")
    fm.user_start()

    assert fm.user_final(
        "我你来来下吧",
        near_end=True,
        voice_rms=0.081,
    ) == UserFinalDisposition.INTERRUPT


def test_low_energy_non_explicit_final_remains_ignored():
    fm = FloorManager(
        natural_interrupt_enabled=False,
        strong_interrupt_rms=0.020,
        strong_interrupt_min_chars=3,
    )
    fm.agent_start()
    fm.agent_chunk("Serena 仍在播放一段较长回答")
    fm.user_start()

    assert fm.user_final(
        "混响猜词片段",
        near_end=True,
        voice_rms=0.007,
    ) == UserFinalDisposition.IGNORED


def test_user_started_after_agent_end_is_not_blocked_by_echo_grace():
    """播放结束后立即接话，即使仍在文本 grace 内也应正常提交。"""
    fm = FloorManager(natural_interrupt_enabled=False)
    fm.agent_start()
    fm.agent_chunk("刚刚播放完的句子")
    fm.agent_end()
    fm.user_start()

    assert fm.user_final(
        "我们继续测试吧",
        near_end=False,
    ) == UserFinalDisposition.USER


def test_natural_interrupt_requires_two_cumulative_partials():
    """播放期间的一次孤立猜词不足以证明真人插话。"""
    fm = FloorManager(
        partial_min_chars=3,
        partial_confirmations=2,
        natural_interrupt_enabled=True,
    )
    fm.agent_start()
    fm.agent_chunk("我正在继续说话")
    assert fm.user_partial("我想问个") is False
    assert fm.last_partial_disposition == UserPartialDisposition.HOLD
    assert fm.user_partial("我想问个别的问题") is True
    assert fm.state == FloorState.INTERRUPTED


def test_unconfirmed_final_during_agent_is_ignored():
    """只有一个短 partial + FINAL 的 AEC 残差不能开启新一轮回答。"""
    fm = FloorManager(partial_min_chars=3, partial_confirmations=2)
    fm.agent_start()
    fm.agent_chunk("Serena 正在外放一段回答")
    fm.user_start()
    assert fm.user_partial("不能") is False
    assert fm.user_final("不能") == UserFinalDisposition.IGNORED
    assert fm.state == FloorState.AGENT_SPEAKING


def test_idle_single_backchannel_does_not_start_conversation():
    """待机噪声被识别成“嗯”时不能自己开启一轮对话。"""
    fm = FloorManager()
    fm.user_start()
    assert fm.user_final("嗯") == UserFinalDisposition.IGNORED
    assert fm.state == FloorState.IDLE


def test_idle_noise_like_single_character_is_ignored():
    fm = FloorManager()
    fm.user_start()
    assert fm.user_final("这") == UserFinalDisposition.IGNORED
    assert fm.state == FloorState.IDLE


def test_idle_single_character_wake_word_is_allowed():
    fm = FloorManager()
    fm.user_start()
    assert fm.user_final("喂") == UserFinalDisposition.USER
    assert fm.state == FloorState.IDLE


def test_echo_reference_survives_until_same_utterance_final(monkeypatch):
    """SPEECH_START 在 grace 内时，晚到的 FINAL 不能变成新用户 turn。"""
    from soul_tty import config
    from soul_tty.interaction import floor as floor_module

    now = [10.0]
    monkeypatch.setattr(floor_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(config, "DUPLEX_ECHO_GRACE_MS", 800)

    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("这段声音来自 Serena 的扬声器")
    fm.agent_end()
    fm.user_start()  # 仍在 grace 内，固定本 utterance 的播放参考
    now[0] = 12.0  # 模拟 ASR FINAL 到达时 grace 已过

    assert (
        fm.user_final("这段声音来自 Serena 的扬声器")
        == UserFinalDisposition.ECHO
    )
    assert fm.state == FloorState.IDLE


def test_fuzzy_echo_matches_middle_of_long_playback():
    """长回答中间片段带少量同音错字，也不应触发打断。"""
    fm = FloorManager(echo_similarity=0.72)
    fm.agent_start()
    fm.agent_chunk("先说第一点，我们需要把音频链路统一起来，然后再检查回声过滤。")
    assert fm.user_partial("音频链路同意起来") is False
    assert fm.last_partial_disposition == UserPartialDisposition.ECHO


def test_user_final_returns_to_idle():
    fm = FloorManager()
    fm.user_start()
    fm.user_final("你好")
    assert fm.state == FloorState.IDLE


def test_empty_final_does_not_drop_agent_floor_or_echo_reference():
    """噪声/被过滤 hallucination 的空 FINAL 不能拆掉正在播放的防线。"""
    fm = FloorManager()
    fm.agent_start()
    fm.agent_chunk("Serena 仍然正在说这句话")
    fm.user_start()

    assert fm.user_final("") == UserFinalDisposition.IGNORED
    assert fm.state == FloorState.AGENT_SPEAKING
    assert fm.agent_text == "Serena 仍然正在说这句话"
    assert fm.user_partial("仍然正在说这句话") is False
    assert fm.last_partial_disposition == UserPartialDisposition.ECHO


def test_user_final_records_interrupt_text():
    """user_partial 打断后,user_final 应记录打断文本。"""
    fm = FloorManager()
    fm.agent_start()
    fm.user_partial("等一下")  # → INTERRUPTED
    interrupted = fm.user_final("等一下")
    assert interrupted == UserFinalDisposition.INTERRUPT
    assert fm.last_interrupt == "等一下"
    assert fm.state == FloorState.IDLE


def test_interrupt_survives_agent_end_before_asr_final():
    """取消 answer 的 worker 先结束时，晚到 FINAL 仍归入同一次打断。"""
    fm = FloorManager(partial_confirmations=1)
    fm.agent_start()
    fm.agent_chunk("我还在说一段很长的回答")
    fm.user_start()

    assert fm.user_partial("停下来") is True
    fm.agent_end()  # cancel 后 answer worker 比 ASR FINAL 更早结束

    assert fm.user_final("停下来吧") == UserFinalDisposition.INTERRUPT
    assert fm.last_interrupt == "停下来吧"


def test_user_final_no_interrupt_returns_false():
    """用户在没有 agent 时说话 → 不算打断。"""
    fm = FloorManager()
    fm.user_start()
    assert fm.user_final("你好") == UserFinalDisposition.USER


def test_on_interrupt_callback_fires_via_partial():
    """打断回调必须通过 user_partial 触发,而非 user_start。"""
    fired = []
    fm = FloorManager(on_interrupt=lambda t: fired.append(t))
    fm.agent_start()
    fm.user_partial("等一下")  # → INTERRUPTED
    fm.user_final("等一下")
    assert fired == ["等一下"]


def test_on_interrupt_callback_exception_swallowed():
    """callback 抛异常不影响 state 推进。"""
    def bad(_):
        raise RuntimeError("oops")

    fm = FloorManager(on_interrupt=bad)
    fm.agent_start()
    fm.user_partial("等一下")  # → INTERRUPTED
    fm.user_final("等一下")
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
    fm.user_partial("等一下")  # → INTERRUPTED
    fm.reset()
    assert fm.state == FloorState.IDLE


def test_user_partial_after_interrupt_does_not_double_interrupt():
    """已经 INTERRUPTED 后再收 partial → 不再触发新的中断事件(state 已变)。"""
    fm = FloorManager()
    fm.agent_start()
    fm.user_partial("等一下")  # → INTERRUPTED
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
    assert fm.state == FloorState.USER_SPEAKING
    fm.user_start()
    assert fm.state == FloorState.USER_SPEAKING


def test_user_partial_idle_returns_false():
    """IDLE 时收到 partial → 无论如何都不打断(没有 agent 可打断)。"""
    fm = FloorManager()
    assert fm.user_partial("你好") is False
    assert fm.state == FloorState.IDLE


def test_user_partial_empty_text_returns_false():
    fm = FloorManager()
    fm.agent_start()
    assert fm.user_partial("") is False
    assert fm.state == FloorState.AGENT_SPEAKING


def test_agent_start_during_user_speaking_does_not_change_state():
    """用户正在说话时 agent_start 不应覆盖 USER_SPEAKING。"""
    fm = FloorManager()
    fm.user_start()
    fm.agent_start()
    assert fm.state == FloorState.USER_SPEAKING


# ── Backchannel 集成测试 ──────────────────────────────────────


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
    """长自然插话经连续 partial 确认后正常打断。"""
    fm = FloorManager(
        backchannel_enabled=True,
        natural_interrupt_enabled=True,
    )
    fm.agent_start()
    assert fm.user_partial("嗯但是我觉得") is False
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
    assert was_interrupted == UserFinalDisposition.INTERRUPT
    assert fm.pending_backchannel is None


def test_backchannel_user_final_clears_pending():
    """用户说一句 backchannel 式的 final → 更新 pending_backchannel,不打断。"""
    fm = _make_manager()
    fm.agent_start()
    fm.user_partial("嗯")
    assert fm.pending_backchannel == "嗯"

    # 句尾 final 也是 backchannel("好的") → 更新 pending,不打断
    disp = fm.user_final("好的", pcm=b"\x00\x00" * 16)
    assert disp == UserFinalDisposition.BACKCHANNEL
    assert fm.pending_backchannel == "好的"


def test_consecutive_backchannels_only_keep_last():
    """连续多个 backchannel,只保留最新的(避免污染下一轮)。"""
    fm = _make_manager()
    fm.agent_start()
    fm.user_partial("嗯")
    fm.user_partial("嗯嗯")
    fm.user_partial("好的")
    assert fm.pending_backchannel == "好的"
