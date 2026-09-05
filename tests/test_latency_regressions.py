"""Regression coverage for UI burst coalescing and interruptible playback."""
import io
import threading
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from soul_tty.audio import tts
from soul_tty.personas import load_persona
from soul_tty.ui import terminal


def make_dashboard(console):
    with patch.object(terminal, '_console', console), patch.object(
        terminal.avatar_ui, 'render_avatar',
        return_value=terminal.avatar_ui.AvatarRender(mode='off'),
    ):
        return terminal.Dashboard(
            load_persona('serena'), terminal.RuntimeDetails(model='local', tts='MLX'),
        )


def test_streaming_burst_keeps_tail_and_cancels_stale_refresh():
    console = Console(file=io.StringIO(), width=100, height=40)
    dashboard = make_dashboard(console)
    dashboard.live = Mock()
    with patch.object(terminal, '_console', console), patch.object(terminal.threading, 'Timer') as timer:
        index = dashboard.append('agent', '')
        for n in range(1, 101):
            dashboard.update(index, '字' * n, deferred=True)
        assert timer.call_count == 1
        dashboard.live.update.assert_not_called()
        # Simulate the trailing timer when no further token arrives.
        timer.call_args.args[1]()
        assert dashboard.messages[index][1] == '字' * 100
        dashboard.live.update.assert_called_once()
        dashboard.update(index, '最后一个字', deferred=True)
        stale_callback = timer.call_args.args[1]
        dashboard.refresh()  # answer_end flush
        count = dashboard.live.update.call_count
        stale_callback()
        assert dashboard.live.update.call_count == count
        dashboard.update(index, '关闭前', deferred=True)
        stale_callback = timer.call_args.args[1]
        dashboard.stop()
        stale_callback()
        assert dashboard.live.update.call_count == count


@pytest.mark.parametrize('width,height', [(40, 16), (60, 24), (100, 24), (120, 40)])
def test_dashboard_keeps_latest_dialogue_inside_viewport(width, height):
    console = Console(file=io.StringIO(), width=width, height=height)
    dashboard = make_dashboard(console)
    dashboard.append('you', '今天怎么样？')
    dashboard.append('agent', '我在这里。')
    with patch.object(terminal, '_console', console):
        lines = console.render_lines(dashboard.render(), console.options, pad=False)
    assert len(lines) <= height
    assert '我在这里。' in ''.join(segment.text for line in lines for segment in line)


def test_playback_failure_releases_full_synthesis_queue(monkeypatch):
    monkeypatch.setattr(tts.config, 'TTS_BACKEND', 'mlx')
    monkeypatch.setattr(tts, 'synthesize_mlx_semantic_segment',
                        lambda *a, **kw: iter([b'\x00\x01' * 16] * 100))
    audio_io = Mock()
    audio_io.write_playback.side_effect = RuntimeError('device disconnected')
    speaker = tts.StreamingSpeaker(audio_io=audio_io)
    speaker.__enter__()
    speaker.say('播放失败测试')
    speaker._sent_q.put(tts._SENTINEL)
    speaker._synth_t.join(timeout=1)
    speaker._play_t.join(timeout=1)
    assert not speaker._synth_t.is_alive()
    assert not speaker._play_t.is_alive()
    assert speaker._playback_failed.is_set()
    assert not speaker._cancel.is_set()  # Audio failure must not cancel the text answer.


def test_cancel_wakes_empty_playback_queue_without_sentinel(monkeypatch):
    monkeypatch.setattr(tts.config, 'TTS_BACKEND', 'mlx')
    speaker = tts.StreamingSpeaker(audio_io=Mock())
    speaker._play_t.start()
    speaker._cancel.set()
    speaker._play_t.join(timeout=0.5)
    assert not speaker._play_t.is_alive()


def test_cancel_stops_inside_large_pcm_chunk():
    cancel = threading.Event()
    stream = Mock()
    stream.write.side_effect = lambda pcm: cancel.set()
    tts._write_metered_pcm(stream, b'\x00\x01' * 48000, Mock(), cancel)
    stream.write.assert_called_once()
