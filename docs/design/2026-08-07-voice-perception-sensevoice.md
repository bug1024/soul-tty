# Voice Perception: SenseVoiceSmall 异步旁路

## 设计目标

在现有 ASR（Streaming Paraformer）之外增加一路异步声音感知，不替换、不阻塞主对话链路。

**核心原则：** ASR 回答"你说了什么"，Voice Perception 回答"你是怎么说的"，Reflection 决定"这意味着什么"。

## 架构

```text
                           User
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
      Realtime Perception         Voice Perception
          ASR · text             tone · event
               │                       │
               ▼                       │
       Conversation Brain              │
          Realtime                     │
               │                       │
               └───────────┬───────────┘
                           ▼
                    Reflection Brain
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
            Bond        Emotion       Memory
```

## 数据模型

```python
@dataclass(frozen=True)
class VoiceObservation:
    emotion: str    # happy / sad / angry / neutral / unknown
    event: str      # Speech / Laughter / Crying / Cough / ...
    language: str   # zh / en / ja / ...
    duration_ms: int
```

## 关键设计决策

1. **SenseVoiceSmall 不替换 Paraformer** — 两路独立，前者是实时 ASR，后者是离线感知
2. **VoiceObservation 不是状态** — 不叫 UserEmotionState，避免误认为"用户真实心理"
3. **PCM tap 复用现有 VAD 切句** — `TranscriptUpdate.pcm` 暴露完整 utterance，不做第二轮 VAD
4. **VoiceRef** — `submit()` 返回递增 ID，解决异步结果和对话轮次对应问题
5. **Reflection 才消费，不等** — evaluator 中只读缓存，不阻塞
6. **声音情绪是弱证据** — 不直接修改 Bond/Memory，不暴露 emotion 标签给用户
7. **模型懒加载** — 不阻塞启动，第一句话 submit 后才 load
8. **默认关闭** — `VOICE_STATE_ENABLED=0`，因模型 228MB 非仓库自带

## 声音情绪影响规则

| 状态 | 能否影响 | 规则 |
|------|---------|------|
| Emotion | ✅ | 通过 Reflection 间接影响 Serena 情绪 |
| Bond | ⚠️ 很弱 | 不得仅凭情绪变化 Bond |
| Memory | ❌ | 不保存"用户刚才很悲伤"等临时状态 |

## 实现顺序

1. **Commit 1** — `voice_state.py`: VoiceObservation / SenseVoice load+decode / queue / cache / TTL
2. **Commit 2** — `asr.py`: TranscriptUpdate.pcm 暴露 / PCM 积累 / 不变量测试
3. **Commit 3** — `reflection/`: VoiceRef / CompletedTurn 绑定 / coalesce / evaluator Voice Context
4. **Commit 4** — 产品化: config / .env.example / README / debug dashboard / benchmark

## 参考

- SenseVoiceSmall: https://github.com/QwenAudio/SenseVoice
- sherpa-onnx SenseVoice: https://github.com/k2-fsa/sherpa-onnx
- 模型: sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17 (~228MB)