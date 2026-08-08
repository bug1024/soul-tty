"""Mic frame listener fan-out tests (commit 01).

不依赖 sounddevice 真硬件——Mic 用 ``MagicMock`` 替换 ``_stream``,
直接调 ``_callback`` 触发 fan-out 路径。
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from soul_tty.audio import capture


@pytest.fixture
def mic(monkeypatch):
    """绕过 sd.RawInputStream,直接持有 MagicMock stream。"""
    monkeypatch.setattr(capture.sd, "RawInputStream", MagicMock())
    return capture.Mic()


def _drive_callback(mic, frame_bytes: bytes, count: int = 1) -> None:
    """模拟 sounddevice 回调:连续触发 ``_callback`` 几次。"""
    indata = frame_bytes
    for _ in range(count):
        mic._callback(indata, len(indata) // 2, None, None)


def test_mic_add_and_remove_frame_listener(mic):
    """注册 listener 后,回调里能收到 PCM;注销后再也不收到。"""
    received: list[tuple[bytes, int]] = []
    listener = lambda pcm, sr: received.append((pcm, sr))
    mic.add_frame_listener(listener)

    frame = b"\x01\x02" * 480
    _drive_callback(mic, frame, count=2)
    assert len(received) == 2
    assert received[0][0] == frame
    assert received[0][1] == capture.config.SAMPLE_RATE

    mic.remove_frame_listener(listener)
    _drive_callback(mic, frame, count=1)
    assert len(received) == 2, "listener 注销后不应再被调用"


def test_mic_listener_receives_exact_pcm(mic):
    """listener 收到的 PCM 必须与 sounddevice 传入的 ``bytes(indata)`` 完全一致。"""
    received: list[bytes] = []
    mic.add_frame_listener(lambda pcm, sr: received.append(pcm))

    payload = bytes(range(256)) * 4  # 1024 字节
    _drive_callback(mic, payload)
    assert received == [payload]


def test_mic_listener_exception_does_not_break_callback(mic):
    """单个 listener 抛异常不能影响 sounddevice 回调或主 _q。"""
    received: list[bytes] = []
    bad = lambda pcm, sr: (_ for _ in ()).throw(RuntimeError("boom"))
    good = lambda pcm, sr: received.append(pcm)
    mic.add_frame_listener(bad)
    mic.add_frame_listener(good)

    frame = b"\x00\x01" * 480
    _drive_callback(mic, frame, count=3)

    # good 仍然每次都收到,bad 没把后续帧吞掉
    assert received == [frame] * 3
    # _q 也照常累积
    assert mic._q.qsize() == 3


def test_mic_listener_remove_during_callback_is_safe(mic):
    """listener 在自己被调用时注销自己,不能影响后续 listener。"""
    seen_other: list[bytes] = []
    self_ref: dict = {}

    def self_removing(pcm, sr):
        mic.remove_frame_listener(self_ref["fn"])

    def other(pcm, sr):
        seen_other.append(pcm)

    other_ref = other
    mic.add_frame_listener(self_removing)
    mic.add_frame_listener(other_ref)
    self_ref["fn"] = self_removing

    frame = b"\x00\x00" * 480
    _drive_callback(mic, frame, count=3)

    # self_removing 在第一次回调时把自己注销,所以 frame 2/3 不会
    # 再被调用;但 other 每次都该被调用
    assert seen_other == [frame] * 3


def test_mic_main_q_unchanged_by_listeners(mic):
    """frame listener 与主 _q 完全独立;listener bug 不影响 frames() 输出。"""
    received: list[bytes] = []
    mic.add_frame_listener(lambda pcm, sr: received.append(pcm))

    frame = b"\x00\x03" * 480
    _drive_callback(mic, frame, count=5)

    # 主 _q 保留 5 帧
    drained = []
    for _ in range(5):
        drained.append(mic._q.get_nowait())
    assert drained == [frame] * 5
    # listener 也收到 5 帧
    assert received == [frame] * 5


def test_mic_listener_concurrent_add_remove(mic):
    """多线程并发 add/remove listener,所有 listener 注册期内都被调过。"""
    calls_per_listener: dict[int, int] = {}
    counter = 0
    lock = threading.Lock()

    def make_listener():
        idx = len(calls_per_listener)
        calls_per_listener[idx] = 0

        def fn(pcm, sr):
            with lock:
                calls_per_listener[idx] += 1

        return fn, idx

    listeners = []
    for _ in range(20):
        fn, _ = make_listener()
        mic.add_frame_listener(fn)
        listeners.append(fn)

    frame = b"\x00\x00" * 480

    def driver():
        for _ in range(50):
            _drive_callback(mic, frame, count=1)
            time.sleep(0.001)

    def remover():
        for fn in listeners[::2]:
            time.sleep(0.005)
            mic.remove_frame_listener(fn)

    t1 = threading.Thread(target=driver)
    t2 = threading.Thread(target=remover)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # 没被移除的 listener 一定收到 50 次;被移除的至多收到 50 次
    for i, fn in enumerate(listeners):
        if fn in mic._frame_listeners:
            assert calls_per_listener[i] == 50
        # 已注销的不再做断言(被注销时机不确定)