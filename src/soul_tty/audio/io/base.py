"""AudioIO Protocol:把"采集 + 播放 + 增益"统一成一组方法。

设计要点:
- 抽象层只描述能力,不绑实现(sounddevice / AVAudioEngine / future)。
- ``add_capture_listener`` 与 ``Mic.add_frame_listener`` 同形
  (``Callable[[bytes, int], None]``),便于 commit 04 切换时复用。
- ``write_playback`` 接受 sample_rate,允许同一个 backend 处理
  24 kHz TTS 与 16 kHz ASR 回放(commit 03 阶段 backend 仍可能
  拒绝非常用采样率)。
"""

from __future__ import annotations

from typing import Callable, Protocol


# 单帧回调签名:(int16 mono PCM, sample_rate)
CaptureListener = Callable[[bytes, int], None]


class AudioIO(Protocol):
    """音频 I/O 后端协议。"""

    def start(self) -> None:
        """打开硬件 + 启动后台 fan-out 线程。"""
        ...

    def stop(self) -> None:
        """关闭后台线程 + 释放硬件。幂等。"""
        ...

    def write_playback(self, pcm: bytes, sample_rate: int) -> None:
        """把 int16 mono PCM 推到扬声器。

        同步等待:调用方负责提供已切好句的 PCM 块,内部通常
        用一个独立播放线程消费。
        """
        ...

    def add_capture_listener(self, listener: CaptureListener) -> None:
        """注册一个采集帧回调,必须非阻塞(≤1ms 工作量)。"""
        ...

    def remove_capture_listener(self, listener: CaptureListener) -> None:
        """注销采集帧回调。"""
        ...

    def set_playback_gain(self, value: float) -> None:
        """设置线性播放增益。``value == 0`` 静音,``value < 0`` 抛错。"""
        ...

    def get_playback_gain(self) -> float:
        """读取当前播放增益。默认 1.0。"""
        ...

    def flush_playback(self) -> None:
        """立即清空已排队但尚未播放的 PCM(打断时调用)。

        默认 no-op;macos_voice 后端会发 PLAYBACK_FLUSH 到 Swift helper,
        清空 AVAudioPlayerNode 的已调度 buffer。
        """
        ...

    def wait_playback_drained(self, timeout: float | None = None) -> bool:
        """等待扬声器真正播完当前所有已排队的 PCM。

        Returns:
            True = 已排空, False = 超时。
        """
        return True

    @property
    def playback_active(self) -> bool:
        """是否有尚未播放完的 PCM(用于 agent_end 的时机判断)。"""
        return False