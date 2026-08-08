"""interaction 包:FloorManager / PlaybackTranscript / backchannel 决策层。

commit 06+ 引入;与 audio / ui / conversation 解耦,只依赖纯函数与
state 转换,便于单独测试。
"""

from .floor import (
    BACKCHANNEL_WORDS,
    FloorManager,
    FloorState,
    UserFinalDisposition,
    is_backchannel,
)
from .playback_transcript import PlaybackTranscript

__all__ = [
    "BACKCHANNEL_WORDS",
    "FloorManager",
    "FloorState",
    "PlaybackTranscript",
    "UserFinalDisposition",
    "is_backchannel",
]