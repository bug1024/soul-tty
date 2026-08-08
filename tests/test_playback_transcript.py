"""PlaybackTranscript 测试。"""

import threading

from soul_tty.interaction import PlaybackTranscript


def test_empty_played_text():
    pt = PlaybackTranscript()
    assert pt.played_text() == ""
    assert len(pt) == 0


def test_add_concatenates_in_order():
    pt = PlaybackTranscript()
    pt.add("你好")
    pt.add("世界")
    assert pt.played_text() == "你好世界"


def test_empty_string_skipped():
    pt = PlaybackTranscript()
    pt.add("")
    pt.add("hello")
    assert pt.played_text() == "hello"
    # len 计有效 chunks;空串被跳过
    assert len(pt) == 1


def test_clear_resets():
    pt = PlaybackTranscript()
    pt.add("foo")
    pt.add("bar")
    pt.clear()
    assert pt.played_text() == ""
    assert len(pt) == 0
    pt.add("baz")
    assert pt.played_text() == "baz"


def test_thread_safety():
    pt = PlaybackTranscript()

    def writer(tag: str) -> None:
        for i in range(1000):
            pt.add(f"{tag}{i}")

    threads = [
        threading.Thread(target=writer, args=("a",)),
        threading.Thread(target=writer, args=("b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    text = pt.played_text()
    # 不丢不重:每个 tag 各 1000 个,顺序无关紧要。
    # (总长度依赖数字宽度:1-9 是 1 位,10-99 是 2 位,...,900-999 是 3 位)
    a_count = sum(1 for i in range(1000) if f"a{i}" in text)
    b_count = sum(1 for i in range(1000) if f"b{i}" in text)
    assert a_count == 1000
    assert b_count == 1000
    assert text.count("a") >= 1000  # 每个 "aN" 至少含 1 个 'a'
    assert text.count("b") >= 1000


def test_played_text_returns_snapshot():
    """played_text 返回拼接结果,后续 add 不影响已读取的字符串。"""
    pt = PlaybackTranscript()
    pt.add("hello")
    snapshot = pt.played_text()
    pt.add(" world")
    assert snapshot == "hello"
    assert pt.played_text() == "hello world"