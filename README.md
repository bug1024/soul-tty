# Soul-TTY · 终端之魂

> **Most AI companions try to bring AI into a virtual world. Soul-TTY explores the opposite: bringing a small digital life into the developer's world — the terminal.**
>
> **Soul-TTY is a local AI companion and agent architecture experiment — real-time voice in your terminal, backed by a persistent personality, an emotion state, and a relationship that evolves.**
>
> **Soul-TTY 是一个属于技术人的本地 AI 伙伴，也是一个探索状态化 Agent 架构的开源项目：实时语音、人格、情绪状态与关系成长，全部跑在你自己的机器上。**

麦克风 → 流式识别 → 流式对话 → 流式语音，整条链路没有云依赖。
`uv run soul-tty` 之后，你对着终端说话，她就醒了。

<p align="center">
  <img src="assets/screenshots/serena-0.png" alt="Soul-TTY 运行截图：启动后的对话界面" width="900">
</p>

**换装不只是换图，也是在切换陪伴模式。** 每套装映射一种行为语气，按 `0` 循环切换：

<table>
  <tr>
    <td align="center" width="33%"><img src="assets/screenshots/serena-3.png" alt="默认装 companion"></td>
    <td align="center" width="33%"><img src="assets/screenshots/serena-2.png" alt="深夜装 late_night"></td>
    <td align="center" width="33%"><img src="assets/screenshots/serena-1.png" alt="工作装 focused"></td>
  </tr>
  <tr>
    <td align="center"><b>默认装</b> · <code>companion</code><br><sub>标准陪伴形态，情绪中性、活力在线</sub></td>
    <td align="center"><b>深夜装</b> · <code>late_night</code><br><sub>情绪基线压低、节奏放缓</sub></td>
    <td align="center"><b>工作装</b> · <code>focused</code><br><sub>警觉度上升、好奇度拉高、不跑题</sub></td>
  </tr>
</table>

---

## 1. What is Soul-TTY

Soul-TTY 把两个意向并置：

- **Soul** — 一个有情绪、有温度、会成长的角色。她知道现在是深夜，知道你已经问过几次同一个问题，知道你说"工作装"的时候其实想专注。
- **TTY** — 没有窗口、没有按钮、没有花哨动画。**一个终端就够了。**

市面上不缺 AI 陪伴项目，但它们大多跑在浏览器里，依赖 Live2D 和 Electron 窗口，目标是创造一个看得见的虚拟角色。Soul-TTY 选择了相反的方向——不是把 AI 请进一个虚拟世界，而是在开发者的世界里安置一个小小的数字生命。你的伙伴就住在 `$SHELL` 里，和你的 git、vim、`ls` 待在同一个地方。

这不是技术妥协，而是技术人独有的浪漫。终端不是缺少 UI——终端本身就是一种身份表达。

区别于纯对话式方案，Soul-TTY 在对话历史之外显式维护 Emotion、Bond 和 Memory 三类持续演化的内部状态——情绪是五维连续向量，关系是跨启动持久化的标量，两者共同决定她下一句话怎么说、用什么语气说。

Soul-TTY 的不同不在于第一次提出 Emotion / Bond / Memory 这些概念——一些优秀的陪伴项目已经开始考虑类似的维度。真正的区别在于：**把三者显式拆成不同时间尺度的状态，并与实时对话主路径解耦。**

- **情绪不是标签。** Soul-TTY 没有"开心/难过/生气"这种瞬时表情标签，而是一个五维连续向量（happiness / calmness / curiosity / stress / energy），有平滑、有惯性、会随时间自然衰减。情绪不是用来控制 Live2D 表情的，而是用来决定她怎么说话、用什么语气。
- **Bond 不是游戏数值。** 没有"送礼 +10 好感度"的机制。Bond 是一个边际递减的连续梯度，只在真实的关心和共同经历中缓慢增长，越往后越难涨。它不是用来解锁功能的进度条。
- **Memory 不只是 RAG。** 不是简单的"对话→向量化→召回"流水线。Memory 分三层（画像/偏好/经历），各自有不同的作用域和注入时机——画像和偏好常驻 prompt，经历只在用户主动提起时按需检索。
- **三个状态各司其职，不混在一起。** Memory 回答"我们经历过什么"，Bond 回答"我们关系有多深"，Emotion 回答"我现在是什么状态"。三者时间尺度不同、存储位置不同、注入 LLM 的时机不同。

因此这个项目同时是两件事：

- 一个**可以每天打开的终端语音伙伴**；
- 一个**状态化本地 Agent 的架构实验**：实时对话与状态演化分离、状态与表达分离、全部本地运行。

---

## 2. Features

| 能力 | 状态 | 说明 |
|---|---|---|
| **实时语音对话** | ✅ | 麦克风 → sherpa-onnx 流式 ASR → llama.cpp 流式 LLM → MLX Qwen3-TTS 流式语音 |
| **五维情绪体系** | ✅ | happiness / calmness / curiosity / stress / energy 连续向量，平滑演化 + 空闲自然衰减 |
| **Bond 羁绊系统** | ✅ | 跨启动持久化的关系标量，边际递减增长；后台旁路评估，不阻塞对话 |
| **三套装 × 三种模式** | ✅ | companion（默认）/ late_night（深夜）/ focused（工作），每套装映射一种行为语气 |
| **本地人格系统** | ✅ | YAML 描述人格、台词、形象、TTS 语气；不锁死，可热加载 |
| **动态口型** | ✅ | 闭嘴 / 半开双缓存图，由实际播放 PCM 音量驱动切换 |
| **动态台词** | ✅ | 启动欢迎、换装语、长时间安静后的陪伴短句均由 LLM 实时生成，超时回退本地短句 |
| **会话记忆** | ✅ | 画像 / 偏好 / 经历三层分离，异步抽取 + 常驻段 + 按需召回 |
| **语音感知 SenseVoice** | ✅ | SenseVoiceSmall 异步感知用户语气与声学事件，作为弱证据进入 Reflection Brain，辅助 Serena 的情绪与表达演化；最近一次感知在 Tab 详情中查看 |
| **日期时间感知** | ✅ | 启动时写入当前日期星期，每分钟热更新；Serena 知道"今天是星期几" |
| **全双工语音打断** | ✅ | 实验开关 `DUPLEX_ENABLED=1`：partial 流、agent 期间插话打断、backchannel 短肯定词不打断；AEC 走 macos_voice 后端（AVAudioEngine voice-processing），其他平台建议佩戴耳机 |

---

## 3. Design Philosophy

> **Soul-TTY separates what an agent says from what an agent becomes.**
>
> **Soul-TTY 将"即时回复"和"持续成长"拆分，让 Agent 不只是生成文本，而是拥有持续演化的内部状态。**

这一句是整个项目的核心，展开为四条原则：

### Real-time first

主对话链路永远优先。任何状态计算、持久化、旁路评估都不能阻塞回复，旁路失败也不影响对话继续。链路上每一段都"敢降级"——LLM 失败、TTS 失败、评估失败，都只让状态不更新，不让对话停下来。

### State over prompt

角色的连续性不靠往 prompt 里堆人设文本，而靠真实存在的状态变量。情绪是可读可写的五维向量，关系是可持久化的标量；prompt 只是这些状态在某一时刻的**渲染结果**，不是状态本身。

### Separate talking and thinking

"回答用户"和"消化这轮对话"是两个不同时间尺度的任务，因此由两个独立回路承担：Conversation Brain 同步、亚秒级；Reflection Brain 异步、可合并、可丢弃。

### Local first

语音、推理、合成默认全部跑在你机器上，对话内容不离开本机。同时支持将 LLM 服务指向外部 OpenAI-compatible endpoint。可换模型、可换声音、可换人格。

---

## 4. Agent Architecture

Soul-TTY 的核心是五条分层结构：

```
                          User Voice (PCM)
                                  │
               ┌──────────────────┴──────────────────┐
               ▼                                     ▼
    ┌─────────────────┐               ┌─────────────────────┐
    │ Realtime Path    │               │  Voice Perception  │
    │ Streaming         │               │  SenseVoiceSmall    │
    │ Paraformer       │               │  (异步独立旁路)      │
    └────────┬────────┘               └──────────┬──────────┘
             │                                    │
             │  text                               │  weak evidence
             │                                     │  (voice obs)
             ▼                                     ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  Conversation Brain                        │  实时主对话：永远优先
    │                      (Realtime)                         │  任何旁路计算不阻塞它
    └─────────────────────────┬────────────────────────────┘
                              │
                              │  complete turn
                              │  + voice evidence
                              ▼
    ┌──────────────────────────────────────────────────────────┐
    │                   Reflection Brain                        │  异步旁路：单 worker · 合并多轮 · 限频
    │                      (Async)                             │  文本 + 声音双信号联合评估
    └─────────────────────────┬────────────────────────────┘
                              │
                              │ 串行写入多个状态层
                              ▼
                       ┌──────┼──────┐
                       │      │      │
                       ▼      ▼      ▼
                 ┌────────┐┌────────┐┌────────┐
                 │  Bond  ││Memory  ││Emotion │
                 │  长期   ││ 三层分离 ││ 短期五维 │
                 └────┬───┘└────┬───┘└────┬───┘
                      │          │          │
                      └──────────┼──────────┘
                                 ▼
                       ┌─────────────────────┐
                       │   Expression Layer   │  状态 → 可感知表现
                       └─────────────────────┘
```

### 五层职责

| 层 | 职责 | 时延要求 |
|---|---|---|
| **Perception** | 实时 ASR（文本）+ 异步 Voice（语气/声学事件）；两条独立，Voice 作为弱证据 | 不阻塞主对话 |
| **Conversation Brain** | 用户真正"对话"的回路；同步、永远在主路径 | 亚秒级 |
| **Reflection Brain** | 关系/情绪联合评估 + 记忆抽取；文本 + 声音双信号融合，异步、可合并 | 不阻塞对话 |
| **State Layer** | Bond（跨启动持久化）+ Emotion（单次会话）+ Memory（跨会话持久化） | 无实时要求 |
| **Expression Layer** | 状态 → 可感知表现；[User Context] 常驻段 + Emotion Context + TTS 语气 | 安全时机 |

### 为什么 Reflection 不在主路径上

主对话只向一个有界内存队列投递 `(user_text, agent_text)`，随后立刻返回。后台单 worker 在用户空闲窗口 + 限频条件下把多轮合并成一次 LLM 调用。关系评估、记忆抽取、情绪更新三者串行执行，**共享 idle window / worker，独立 minimum interval**（关系评估默认 60 秒，记忆抽取默认 120 秒）。投递失败、LLM 失败、低 confidence、队列满，任何一种情况都只让状态保持不变——对话本身永远不受影响。

### 为什么 Expression 是独立一层

Expression 是 *状态 → 表现* 的纯映射层，不持有任何状态：

- 输入：`{mood, intensity, expression}`
- 输出：TTS 语气指令、prompt 段落、未来的 avatar 参数

这样换 TTS 后端只影响 Expression 层，不影响 Emotion 本身；未来把 Avatar 接进来也是同一条链路的最后一节。

### 为什么 Emotion 与 Memory 是两件事

- **Emotion** 回答 *"我现在感觉怎么样？"* — 短期内部状态，单次会话内演化，随空闲衰减回归 baseline，默认不跨启动。
- **Memory** 回答 *"之前发生过什么？"* — 长期经验沉淀，跨会话累积。

两者时间尺度不同、存储位置不同、注入 LLM 的时机也不同：Emotion 实时热更新进 prompt，Memory 按阈值选择性注入（画像/偏好常驻、经历按需召回）。**Bond 不能代替 Memory，Memory 也不能直接改写 Bond 或 Emotion**。三者构成完整的状态三角：Bond 衡量关系深度，Emotion 反映当下心情，Memory 沉淀过往经验。

---

## 5. Emotion System

### Concept

Soul-TTY 不使用简单情绪标签（"开心" / "难过"），而是维护一组**连续状态**，让"心情"成为一个有过程、有惯性、会自己恢复的东西。你说了让她高兴的话，她不会立刻从 0 跳到 1；你离开一会儿再回来，她的情绪已经自然平复了一些。

### Model

| 维度 | 含义 | 极端高 | 极端低 |
|---|---|---|---|
| **happiness** | 愉悦度 | 兴高采烈 | 心情低落 |
| **calmness** | 平静度 | 心如止水 | 焦躁不安 |
| **curiosity** | 好奇度 | 主动探索 | 无聊放空 |
| **stress** | 压力值 | 紧绷应激 | 完全放松 |
| **energy** | 活力值 | 兴奋充沛 | 疲倦低迷 |

五个维度构成一个向量，收敛为 `(mood, intensity, expression)` 三元组后交给 Expression Layer。

### Evolution

```
Reflection Brain 评估（旁路）
       │  每轮输出 emotion_delta（5 维）
       ▼
apply_delta  ──  平滑：单次评估不会造成情绪突变
       │         截断：异常大的 delta 被限幅，防止模型抖动污染状态
       ▼
EmotionVector（session state）
       │
       │  idle decay：长时间无交互 → 缓慢回归 baseline
       ▼
resolver → (mood, intensity, expression)
```

### Integration

| 集成点 | 状态 | 说明 |
|---|---|---|
| system prompt 的 `[Emotion Context]` 段 | ✅ | 情绪状态实时渲染进主对话 prompt |
| TTS 语气指令 | ✅ | Expression 映射为合成语气，声音跟着心情变 |
| Avatar 表情与口型 | 🚧 | 链路已预留，等设计资源 |

<details>
<summary><b>Technical details</b></summary>

- 实现位于 `src/soul_tty/emotion/`：`service.py`（单例）、`state.py`（向量与持久化）、`analyzer.py`（LLM delta → 向量）、`resolver.py`（收敛 mood / expression）、`prompt_builder.py`（向量 → prompt 文本）、`updater.py`（按强度阈值节流更新）。
- 平滑采用 EMA，速率由 `EMOTION_EMA_RATE` 控制（默认 `0.2`）。
- 单轮 delta 由 `EMOTION_DELTA_CAP` 限幅（默认 `0.3`）。
- 空闲衰减每 `EMOTION_DECAY_INTERVAL_S`（默认 300 秒）按 `EMOTION_DECAY_RATE`（默认 `0.05`）回归 baseline。
- 情绪默认只存在于内存，`EMOTION_PERSIST=1` 才写盘。

</details>

---

## 6. Bond System

### Concept

Bond（羁绊）是 Soul-TTY 对"长期陪伴关系"的建模。它不是一个游戏化的进度条，也不解锁任何功能，而是一个连续梯度：随着真实的关心、共同的玩笑和逐渐建立的信任缓慢上升，并且**越往后越难涨**。

### Model

单一标量 `bond ∈ [0, 1]`，跨启动持久化，映射为六个阶段：

| bond | 阶段 |
|---|---|
| 0.00 – 0.09 | stranger |
| 0.10 – 0.29 | acquaintance |
| 0.30 – 0.49 | familiar |
| 0.50 – 0.69 | companion |
| 0.70 – 0.89 | close |
| 0.90 – 1.00 | bonded |

欢迎区显示当前阶段，按 `Tab` 可以看到精确分值。

### Evolution

增长采用边际递减：

```
bond ← bond + delta × (1 - bond)
```

同样的一次互动，在 `bond = 0.1` 时带来的增长远大于 `bond = 0.9` 时。评估由 Reflection Brain 在后台完成：主流程只投递问答，后台 worker 合并多轮后调用一次 LLM，输出 `relationship_delta.bond`、`emotion_delta`、`expression` 三路分离的结果。低 confidence 的评估会被直接丢弃。

### Bond 与 Emotion 的边界

| | Bond | Emotion |
|---|---|---|
| 维度 | 单一标量 | 5 维向量 |
| 时间尺度 | 跨启动持久化 | 单次会话 |
| 变化方式 | 边际递减增长 | 平滑演化 + 空闲衰减 |
| 触发 | 确认发生"关心 / 共同玩笑 / 信任"等事件 | 任意情绪 delta |
| 显示 | 阶段标签 + `Tab` 精确分值 | Dashboard 五维条形 + mood 文案 |

<details>
<summary><b>Technical details</b></summary>

- 实现位于 `src/soul_tty/reflection/relationship.py`。
- 单 worker + 有界队列（`RELATIONSHIP_QUEUE_SIZE`，默认 4），队列满即丢弃，不反压主链路。
- 回答结束后等待 `RELATIONSHIP_IDLE_DELAY_S`（默认 3 秒）确认用户空闲；两次评估最小间隔 `RELATIONSHIP_MIN_INTERVAL_S`（默认 60 秒），窗口内多轮合并为一次调用。
- 单次 delta 上限 `RELATIONSHIP_MAX_DELTA`（默认 `0.03`），置信度低于 `RELATIONSHIP_MIN_CONFIDENCE`（默认 `0.65`）不生效。
- 最近关系事件按 FIFO 保留 `RELATIONSHIP_MAX_RECENT_EVENTS` 条（默认 20），避免无限增长。

</details>

---

## 7. Voice Perception

### Concept

> **ASR tells Soul-TTY what you said. Voice perception tells it how you said it. Reflection decides what that means.**

SenseVoiceSmall 是独立于 ASR 的异步旁路：**不进入主对话关键路径，不等待其结果即可开始回复。**它分析的是"这句话听起来像什么"，不是"用户实际是什么"。

### Boundary

```
SenseVoice emotion=sad
≠
Serena 认定"用户在撒谎"
```

Voice Observation 是外部传感器信号，Reflection 才是认知层。两者之间有明确的语义鸿沟——信号 ≠ 结论。

### What voice observation provides

| 字段 | 来源 | 含义 |
|---|---|---|
| `emotion` | SenseVoice 分类 | 这句话的声学情绪：happy / sad / angry / neutral / surprise / fear / disgust |
| `event` | SenseVoice 分类 | 声学事件：speech / laughter / crying / cough / applause |
| `language` | SenseVoice 判断 | 语种：zh / en / ja / ko / yue |

### How Reflection uses it

Reflection 的 `evaluate_relationship()` 收到的是渲染后的 voice context：

```
[voice] turn=2, emotion=sad, event=speech, language=zh
```

解读规则：
- 优先综合文本与声音，不单独依据标签下结论
- 声音与文本冲突时视为**反差信号**，而非"用户在掩饰"
- laughter / crying 等声学事件权重高于普通 emotion 分类
- 声音可影响 emotion_delta 和 expression，**不得直接改写 bond**
- 不得把一次性的声音情绪写入长期记忆描述

### 产品语义

| Voice 影响 | Voice 不影响 |
|---|---|
| `emotion_delta` | Bond 数值 |
| `expression` | 长期 Memory 写入 |
| 跨模态冲突判断 | 用户心理状态的直接判定 |

### 生命周期

Voice Observation 有两条独立的生命周期：

- **Reflection（内部关联）**：由 `VOICE_STATE_RESULT_TTL_S` 控制（默认 120 秒），确保 Reflection 来得及关联
- **Tab 展示（产品感知）**：由 `VOICE_STATE_UI_TTL_S` 控制（默认 45 秒），超时后 Tab 感知行自动消失

超过后均不可见——这不是 bug，而是设计语义：**过了就是过了，Serena 不能反复翻旧账**。

## 8. Full-Duplex Voice Mode

> **半双工是默认；全双工是显式实验开关。开启 = partial 流 + 后台 answer + 取消语义 + backchannel。**

### 三种语音模式

| 模式 | 触发 | 行为 |
|---|---|---|
| `half_duplex` | 默认 | Mic 在 agent 说话期间暂停；用户说一句、agent 答一句 |
| `barge_in` | `BARGE_IN_ENABLED=1`（旧名） | agent 说话期间持续听；离线 ASR 整句识别后决定是否打断 |
| `full_duplex` | `DUPLEX_ENABLED=1` | partial 流式显示；agent 后台跑；用户 partial 命中即取消；backchannel 短词不打断 |

`run_microphone` 是统一入口：先 `_detect_voice_mode()` 选模式 → `_voice_mode_warning()` 一次性提示 → 转到对应 `_run_*_mic()`。

### 关键组件

- **`Mic.add_frame_listener`** — 麦克风采集帧广播给 `DuplexListener` 与 VoiceState 旁路（同一份 PCM）
- **`DuplexListener`** — 把帧送进 `VadGatedSherpaStream`，产出 `SPEECH_START / PARTIAL / FINAL / SPEECH_END` 事件流
- **`FloorManager`** — 谁拥有麦克风的状态机（`IDLE / USER_SPEAKING / AGENT_SPEAKING / INTERRUPTED`），回声判定走 `interaction.echo.is_probable_echo`
- **`PlaybackTranscript`** — agent 实际播放过的文本，用于回声过滤（避免误把"嗯,你说得对"当插话）
- **Backchannel** — `BACKCHANNEL_ENABLED=1` 时，"嗯/好的/是" 等 ≤3 字肯定词不打断，只进 `pending_backchannel`，供下一轮参考
- **`MacOSVoiceIO`** — macOS 13+ 后端，通过 Swift helper 启用 `AVAudioEngine.setVoiceProcessingEnabled`，硬件级 AEC；其他平台走 `PortAudioIO` 需佩戴耳机

### 已知边界

- TTS 播放仍走 `sd.RawOutputStream`，未接入 `MacOSVoiceIO`：外放环境下仍有少量自激，AEC 不完整
- DuplexListener 内部 queue 容量 `64`：极端慢 ASR 时老 partial 会被丢
- Backchannel 白名单写死在 `interaction/floor.BACKCHANNEL_WORDS`，当前不可配置

## 9. Memory System

### Concept

Memory 回答 *"之前发生过什么？"* — 跨会话累积用户画像、交流偏好与共同经历，让 Agent 能在新一轮对话中"记得"你。它不是简单地把历史对话原文倒进 prompt，而是**抽取 → 存储 → 按需注入**：

- 画像 profile 和偏好 preference 常驻 system prompt，作为 [User Context] 段落；
- 经历 experience 只在用户提起过去相关话题时按需检索召回，作为临时 [Relevant Memories] 段注入。

### Three Types

| 类型 | 作用域 | 含义 | 示例 |
|---|---|---|---|
| **`profile`** 用户画像 | global | 关于用户的客观事实 | "用户是一名 AI 应用开发者，正在做终端语音助手项目" |
| **`preference`** 交流偏好 | global | 用户喜欢怎样的交流方式 | "用户喜欢简洁直接的回复，不喜欢太长的解释" |
| **`experience`** 共同经历 | persona | 用户与某个 Agent 的共同经历 | "用户和 Serena 聊过她的项目，Serena 给了架构建议" |

**作用域隔离：** profile 和 preference 是 global 的——换人格依然有效。experience 绑定 persona——换人格不会继承上一人格的共同经历。

### Extraction

记忆抽取由 Reflection Worker 在后台异步完成，与关系评估、情绪更新串行执行：

```
Conversation Brain 投递 (user_text, agent_text)
         │
         ▼
Reflection Worker 空闲窗口 + 限频（默认 120 秒）
         │
         ▼ 多轮合并为一次 LLM 调用
extract_memories(model, known_facts, user_text, agent_text)
         │
         ▼ 返回 {memories: [{type, content, importance}]}
service.remember_many(...)
         │
         ▼ 去重、门槛过滤、写入 SQLite
```

- **已知信息回灌：** 抽取时把已有记忆作为 `known_facts` 拼入 prompt，引导 LLM"只提取新的事实"，避免重复。
- **静默降级：** LLM 调用失败或响应无效时保留待抽取 buffer，下次重试。Memory 的待处理轮次与 Relationship queue 独立——queue 溢出只影响关系评估，不影响记忆抽取。
- **去重：** 新记忆与同类已有记忆的 bigram 重叠超过 0.8 则跳过。

### Recall

当用户说起"还记得之前那个……"时，Memory 系统按需检索相关经历：

```
用户："还记得我上次说的那个 AI 项目吗？"
         │
         ▼ 命中召回关键词（"还记得""之前""上次"等）
is_recall_query → True
         │
         ▼ 按 persona 检索 experience 表
search(query, memories)
         │
         ▼ bigram 相关性 > 0.2 门槛
         ▼ 综合分排序取 top 3
render_recall → [Relevant Memories] 段落
         │
         ▼ 作为临时 system message 注入本轮对话
```

**为什么不一直注入所有记忆？** 上下文窗口有限。常驻段只保留最重要的画像和偏好（默认最多 12 条），经历仅在用户主动提起时检索。检索不到时也不注入——宁可安静，也不错误唤起记忆。

**检索算法：** V1 使用字符 bigram 重叠（零依赖），不引入 embedding 模型。V2 可替换为向量余弦而调用方不变。

### Resident Context

画像和偏好常驻在 system prompt 的 [User Context] 段落：

```
[User Context]
关于用户：
- 用户是一名 AI 应用开发者，正在做终端语音助手项目

交流偏好：
- 用户喜欢简洁直接的回复
- 用户偏好中文交流
```

### CLI Management

`soul-tty memory` 子命令提供运行时查看和管理：

```bash
# 列出全部记忆（按类型分组）
uv run soul-tty memory

# 按类型过滤
uv run soul-tty memory list --type profile

# 查看单条详情
uv run soul-tty memory show 1

# 删除单条
uv run soul-tty memory forget 1

# 清空全部（需二次确认）
uv run soul-tty memory clear
```

### Design Principles

- **Memory 不是控制权：** 它只向 Prompt 提供信息，不修改 Bond、Emotion 或任何其他状态。
- **宁可不说，也不要说错：** 检索不到记忆时直接返回空——错误唤起比不唤起更糟糕。
- **旁路写入：** 抽取永远是异步的，队列满、LLM 失败都不影响主对话。
- **降级透明：** SQLite 打不开、文件损坏时 `available = False`，所有方法静默返回空值，行为与 `MEMORY_ENABLED=0` 一致。

<details>
<summary><b>Technical details</b></summary>

- 实现位于 `src/soul_tty/memory/`：`models.py`（数据模型）、`store.py`（SQLite 存储）、`service.py`（业务装配）、`extractor.py`（LLM 抽取）、`retriever.py`（bigram 检索）、`prompt.py`（文本渲染）、`cli.py`（管理子命令）。
- 存储使用 SQLite WAL 模式，每次操作新开连接后关闭，彻底避免线程安全。
- 抽取间隔 `MEMORY_MIN_INTERVAL_S` 默认 120 秒，窗口内多轮合并为一次调用。
- 单条记忆 importance 下限 `MEMORY_MIN_IMPORTANCE` 默认 0.7（0~1 标度）。
- 去重阈值 `MEMORY_DEDUPE_THRESHOLD` 默认 0.8（bigram Jaccard 变体）。
- 常驻上限 `MEMORY_MAX_RESIDENT` 默认 12 条。
- 召回 top-k `MEMORY_RECALL_TOP_K` 默认 3 条。
- 召回相关性门槛 `MEMORY_RECALL_MIN_RELEVANCE` 默认 0.2。
- 时间衰减半衰期 `MEMORY_RECENCY_HALFLIFE_DAYS` 默认 180 天。

</details>

---

## 10. Outfit & Modes

换装不只是换一张图（运行截图见文首）。每个套装标注一个 `mode`，这个 mode 决定情绪系统如何解释"现在是哪种陪伴状态"：

| 套装 | mode | 含义 | 适合 |
|---|---|---|---|
| **默认装** | `companion` | 标准陪伴形态，情绪中性、活力在线 | 日常闲聊、说点心里话 |
| **深夜装** | `late_night` | 情绪基线压低、节奏放缓 | 临睡前、放空时 |
| **工作装** | `focused` | 警觉度上升、好奇度拉高、不跑题 | 写代码、debug、长期任务 |

启动时通过 `--outfit` 或 `SOUL_TTY_OUTFIT` 指定，运行中按 `0` 在配置顺序之间循环切换。

换装后立即给出一句本地短句，再由后台 LLM 结合当前时段、羁绊阶段和本次会话情绪生成动态台词。连续换装不会阻塞界面，也不会污染正式对话历史。

---

## 11. Technical Implementation

```
麦克风 (16kHz mono PCM)
       │
       ├──► Streaming Paraformer (sherpa-onnx)
       │          │
       │          ▼  text
       │    Conversation Brain
       │          │
       │          ▼  流式回复
       │         TTS → 终端口型
       │
       └──► SenseVoiceSmall (异步，不阻塞主路径)
                  │
                  ▼  VoiceObservation
              Reflection Brain
                  │
                  ▼  emotion_delta / expression
              Emotion / Expression
```

详细链路：

```
麦克风 (16kHz mono PCM)
       │
       ▼
┌─ 音频采集 ──────────────────────────────────────┐
│  sounddevice · WebRTC VAD（30ms 帧门控）         │
│  空闲时只跑 VAD，触发后补回 300ms pre-roll        │
└─────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────┐
       ▼                                              ▼
┌─ Streaming Paraformer ─┐              ┌─ SenseVoiceSmall ─────────┐
│  sherpa-onnx          │              │  进程内离线推理（~228MB）  │
│  流式 ASR，partial 显示 │              │  emotion / event / lang   │
└──────────┬────────────┘              └────────────┬─────────────┘
           │  text                                   │ voice_obs
           ▼                                         ▼
    ┌────────────────────────────┐    ┌─────────────────────┐
    │     Conversation Brain      │    │  Reflection Brain   │
    │  流式 LLM → TTS → 口型    │    │  联合评估          │
    └────────────────────────────┘    │  Bond/Memory/Emotion│
                                      └─────────────────────┘
```

**为什么是这套技术栈：** 全本地、Apple Silicon 友好（MLX）、延迟可压到亚秒级、整条链路没有云依赖。辅助请求（欢迎语、换装台词、空闲短句）可以通过 `AUX_LLM_URL` 指向独立的小模型服务，与主对话彻底隔离算力竞争。

---

## 12. Project Structure

```text
src/soul_tty/
├── cli.py                # 命令行入口与启动信息
├── config.py             # 环境变量配置
├── conversation.py       # 对话流程、句切分、Markdown 净化
├── presence.py           # 启动节奏、低频特殊开场
├── emotion/              # 五维情绪体系
│   ├── service.py        #   EmotionService 单例
│   ├── state.py          #   EmotionVector / 持久化
│   ├── analyzer.py       #   LLM delta → 五维向量
│   ├── resolver.py       #   收敛合法 Mood / Expression
│   ├── prompt_builder.py #   向量 → 注入 LLM 的 context 文本
│   └── updater.py        #   节流更新（强度阈值）
├── memory/               # 长期记忆系统
│   ├── models.py         #   Memory 数据模型（三类记忆、两种作用域）
│   ├── store.py          #   SQLite 存储（WAL、每操作新连接）
│   ├── service.py        #   MemoryService 业务装配
│   ├── extractor.py      #   LLM 记忆抽取器
│   ├── retriever.py      #   bigram 相关性检索
│   ├── prompt.py         #   记忆 → 文本渲染（常驻 / 召回）
│   └── cli.py            #   管理子命令 list/show/forget/clear
├── reflection/           # 异步旁路推理
│   ├── worker.py         #   ReflectionWorker 调度（队列/空闲门控/限频）
│   └── relationship.py   #   Bond 评估与持久化
├── audio/                # 录音 + sherpa-onnx ASR + MLX TTS
├── clients/llm.py        # OpenAI 兼容 LLM 客户端
├── personas/             # YAML 人格加载、校验、运行时应用
└── ui/
    ├── terminal.py       # Rich Dashboard、对话视口、状态栏
    └── avatar.py         # 原生图片协议 / Chafa / 像素画回退

personas/                 # 用户可直接编辑的人格 YAML
assets/avatars/           # 768×768 原创像素风角色图
tests/                    # pytest 单元测试
docs/                     # 设计稿、规划文档
```

---

## 13. Installation

### 前置依赖

Soul-TTY 是一个 Python 进程，但完整跑起来还需要几个外部服务：

| 组件 | 用途 | 部署方式 |
|---|---|---|
| **Python ≥ 3.10** | 主程序 | 系统 Python / pyenv / uv 自带 |
| **uv** | 依赖管理 | `pip install uv` 或 `brew install uv` |
| **llama.cpp / llama-server** | 主对话 + 状态评估 | `llama-server -m your-model.gguf --port 8180 --host 127.0.0.1` |
| **MLX Qwen3-TTS** | 流式中文 TTS | 任意 MLX-Audio 兼容服务（默认端口 `50501`） |
| **chafa** | 终端像素图渲染 | `brew install chafa`（Kitty / Ghostty / iTerm2 使用原生协议时可跳过） |
| **sherpa-onnx** | 流式 ASR | 已作为 pip 依赖打进 soul-tty，无需单独部署 |
| **macOS 麦克风权限** | 语音输入 | 首次运行自动弹出授权；系统设置 → 隐私与安全性 → 麦克风 |

> **推荐运行环境：** macOS（Apple Silicon）+ Ghostty / Kitty / iTerm2。
> 原生图片协议能让头像以接近原画的清晰度显示在终端里；其他终端自动回退为真彩 Unicode 像素画。

### 准备模型

`llama-server` 启动任意支持中文的指令模型即可（例如 Qwen3、Llama-3 中文微调等量化版），暴露 OpenAI 兼容的 `/v1/chat/completions` 与 `/v1/models`。

TTS 端使用 `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` 或兼容版本。

### 可选组件：语音感知

语音感知可感知用户语气/情绪/声学事件，结果作为弱证据供旁路反思系统消费。默认关闭，需要手动下载模型（约 228MB）：

```bash
# 模型下载（首次运行前执行一次即可）
cd sherpa-asr/models
# 方法一：HuggingFace
git lfs install
git clone https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17

# 方法二：ModelScope（国内）
git clone https://www.modelscope.cn/nickyac/SenseVoice.git
# 将 model_quant.onnx 和 tokens.json 放入同一目录

# 验证模型文件
ls sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/
# 应包含：model.int8.onnx（或 model_quant.onnx）+ tokens.txt（或 tokens.json）
```

下载完成后，设置环境变量并启动：

```bash
export VOICE_STATE_ENABLED=1
export SENSEVOICE_MODEL_DIR=/path/to/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
uv run soul-tty
```

### 安装并启动

```bash
git clone https://github.com/bug1024/soul-tty.git
cd soul-tty
uv sync                    # 自动创建 .venv 并安装依赖
cp .env.example .env       # 按需填写；也可直接用环境变量覆盖
uv run soul-tty
```

---

## 14. Configuration

所有配置项都通过环境变量覆盖，完整默认值见 [`src/soul_tty/config.py`](./src/soul_tty/config.py)。

| 类别 | 变量 | 默认 | 说明 |
|---|---|---|---|
| 人格 | `SOUL_TTY_PERSONA` | `serena` | 人格 id 或 YAML 路径 |
| 人格 | `SOUL_TTY_PERSONA_DIR` | 空 | 额外人格搜索目录；项目 `personas/` 始终生效 |
| 人格 | `SOUL_TTY_OUTFIT` | persona `default_outfit` | 启动套装；运行中按 `0` 循环 |
| 人格 | `AGENT_NAME` | persona 名 | 覆盖显示名并注入 LLM |
| 主对话 | `LLM_URL` | `http://127.0.0.1:8180` | 主对话 LLM 地址 |
| 主对话 | `LLM_MODEL` | 自动取 `/v1/models` 第一个 | 主对话模型 id |
| 主对话 | `LLM_MAX_TOKENS` | `256` | 单轮回答硬上限 |
| 主对话 | `LLM_TEMPERATURE` | `0.7` | 采样温度 |
| 主对话 | `LLM_REPEAT_PENALTY` | `1.1` | 重复惩罚 |
| 主对话 | `MAX_HISTORY` | `10` | 保留最近 N 轮上下文 |
| 辅助 | `AUX_LLM_URL` | 回退到 `LLM_URL` | 辅助 LLM 地址（欢迎 / 换装 / 空闲短句） |
| 辅助 | `AUX_LLM_MODEL` | 回退到 `LLM_MODEL` | 辅助 LLM 模型 id |
| ASR | `SHERPA_MODEL_DIR` | `../sherpa-asr/models/...` | Streaming Paraformer int8 模型目录 |
| ASR | `SHERPA_ENDPOINT_SILENCE_S` | `0.60` | 已识别出内容后的尾部静音提交阈值 |
| ASR | `SHERPA_VAD_GATE_ENABLED` | `1` | 空闲时只跑 WebRTC VAD，不持续调用 Paraformer |
| ASR | `SHERPA_VAD_PRE_ROLL_MS` | `300` | 检测到人声时补回的句首音频长度 |
| 切句 | `VAD_AGGRESSIVENESS` | `2` | 0–3，越大越严格 |
| 切句 | `SILENCE_MS` | `700` | 连续静音判定一句话结束 |
| 切句 | `MAX_UTTERANCE_S` | `15` | 单句最长秒数，超出强制切段 |
| 打断 | `DUPLEX_ENABLED` | `0` | 真双工总开关（partial 流 + 打断 + 后台 answer 线程 + cancel） |
| 打断 | `DUPLEX_ECHO_SIMILARITY` | `0.72` | duplex 路径回声判定阈值 |
| 打断 | `BARGE_IN_ENABLED` | (alias) | 旧名，等价 `DUPLEX_ENABLED` |
| 打断 | `BARGE_IN_ECHO_SIMILARITY` | (alias) | 旧路径回声阈值 |
| 打断 | `BACKCHANNEL_ENABLED` | `1` | agent 说话时识别「嗯/好的」等短肯定词，不打断只记录 |
| 音频 I/O | `AUDIO_IO_BACKEND` | `portaudio` | `portaudio` / `macos_voice`（后者需 Swift helper + macOS 13+） |
| 音频 I/O | `TTS_PLAYBACK_GAIN` | `1.0` | TTS 输出增益（>0；1.0 = 原样） |
| TTS | `TTS_BACKEND` | `mlx` | `mlx` / `macos` |
| TTS | `MLX_TTS_URL` | `http://127.0.0.1:50501` | MLX-Audio 服务地址 |
| TTS | `MLX_TTS_VOICE` | `Serena` | Qwen3-TTS 内置音色 |
| TTS | `MLX_TTS_INSTRUCT` | `""` | 可选语气指令，如 `用温柔、亲切的语气说` |
| TTS | `TTS_WHOLE_ANSWER` | `1` | `1`=完整回答后播报；`0`=按句流水线播报（首音更快） |
| 反射 | `REFLECTION_ENABLED` | `1` | 旁路总开关；关闭后 Bond/Emotion/Memory 异步处理全部停用 |
| 反射 | `BOND_ENABLED` | `1` | 关系评估子开关；关闭后评估结果不落库，Memory/Emotion 不受影响 |
| Bond | `RELATIONSHIP_ENABLED` | 同 `REFLECTION_ENABLED` | 旧名兼容 |
| Bond | `RELATIONSHIP_LLM_URL` | 同 `LLM_URL` | 评估服务；可指向独立小模型 |
| Bond | `RELATIONSHIP_IDLE_DELAY_S` | `3` | 回答结束后等用户空闲多久再评估 |
| Bond | `RELATIONSHIP_MIN_INTERVAL_S` | `60` | 两次评估最小间隔（窗口内多轮合并） |
| Memory | `MEMORY_ENABLED` | `1` | 关闭后完全不抽取、不检索、不注入 |
| Memory | `MEMORY_DB_PATH` | `~/.local/state/soul-tty/memory.db` | SQLite 存储路径 |
| Memory | `MEMORY_LLM_URL` | 同 `AUX_LLM_URL` | 抽取服务；可指向独立小模型 |
| Memory | `MEMORY_LLM_MODEL` | 同 `AUX_LLM_MODEL` | 抽取模型 id |
| Memory | `MEMORY_LLM_TIMEOUT` | `8` | 抽取请求超时（秒） |
| Memory | `MEMORY_LLM_MAX_TOKENS` | `256` | 抽取响应最大 token 数 |
| Memory | `MEMORY_MIN_INTERVAL_S` | `120` | 两次抽取最小间隔（窗口内多轮合并） |
| Memory | `MEMORY_BUFFER_TURNS` | `20` | 抽取缓冲区最大轮数 |
| Memory | `MEMORY_MIN_TEXT_CHARS` | `20` | 用户文本不足此字符数时不触发抽取 |
| Memory | `MEMORY_MIN_IMPORTANCE` | `0.7` | 记忆重要度下限（0~1，低于此丢弃） |
| Memory | `MEMORY_DEDUPE_THRESHOLD` | `0.8` | 去重 bigram 重叠阈值 |
| Memory | `MEMORY_MAX_RESIDENT` | `12` | 常驻 system prompt 的 global 记忆上限 |
| Memory | `MEMORY_RECALL_TOP_K` | `3` | 每次召回返回最多条数 |
| Memory | `MEMORY_RECALL_MIN_RELEVANCE` | `0.2` | 召回相关性门槛 |
| Memory | `MEMORY_RECENCY_HALFLIFE_DAYS` | `180` | 记忆时间衰减半衰期（天） |
| Emotion | `EMOTION_ENABLED` | `1` | 关闭则情绪系统不启动 |
| Voice | `VOICE_STATE_ENABLED` | `0` | 开启语音感知；默认关闭（ONNX 模型需单独下载） |
| Voice | `SENSEVOICE_MODEL_DIR` | `../sherpa-asr/models/...` | SenseVoice 模型目录，含 `model.int8.onnx` + `tokens.txt` |
| Voice | `VOICE_STATE_RESULT_TTL_S` | `120` | 感知结果缓存有效期（秒），超时后 Reflection 不可见 |
| Voice | `VOICE_STATE_UI_TTL_S` | `45` | Dashboard 感知行展示 TTL，超出后自动消失；应短于 RESULT_TTL |
| Voice | `VOICE_STATE_MIN_UTTERANCE_MS` | `800` | 最短有效语音长度，过短不提交分析 |
| Emotion | `EMOTION_EMA_RATE` | `0.2` | 平滑率 |
| Emotion | `EMOTION_DECAY_INTERVAL_S` | `300` | 空闲衰减间隔（秒） |
| Emotion | `EMOTION_DECAY_RATE` | `0.05` | 每轮衰减幅度 |
| Emotion | `EMOTION_PERSIST` | `0` | 是否把情绪写盘 |
| Dashboard | `DASHBOARD_DETAILS` | `0` | 启动时是否展开详情（运行中按 `Tab` 切换） |
| Dashboard | `DASHBOARD_MAX_MESSAGES` | `300` | 可滚动消息上限 |
| 状态 | `SOUL_TTY_STATE_DIR` | `~/.local/state/soul-tty` | 持久化目录（Bond、Emotion、Memory） |

### 人格自定义

默认人格是 `serena`（紫色未来感中文女声）。新人格只需在 `personas/` 或 `SOUL_TTY_PERSONA_DIR` 下添加一个 YAML 文件：

```bash
uv run soul-tty --persona serena --name 小夜      # 临时改名（也注入 LLM）
uv run soul-tty --persona /path/to/my-persona.yaml  # 直接加载外部 YAML
SOUL_TTY_PERSONA=my_persona uv run soul-tty         # 用环境变量选人格
```

人格 YAML 可定义：

- 名字、tagline、开场白、告别语
- LLM 系统提示词与生成参数上限
- TTS 音色与语气指令
- 终端主色 / 副色 / 强调色
- 头像渲染器（`auto` / `pixels` / `symbols` / `off`）
- 多套 `outfits`，每套关联一组 `idle / speaking_closed / speaking_half` 图与一个 `mode`

---

## 15. Development

不需要麦克风也能验证 LLM ↔ TTS 链路：

```bash
uv run soul-tty --text "你好"                   # 跳过 ASR，直接发给 LLM（含 TTS 播放）
uv run soul-tty --file /path/to/16k_mono.wav   # 用本地音频跑 ASR → LLM → TTS 全链路
```

查看可用资源：

```bash
uv run soul-tty personas    # 列出可用人格
uv run soul-tty outfits     # 列出当前人格的套装
```

运行测试：

```bash
uv run pytest
```

测试覆盖音频链路、对话切句、情绪系统、Bond 旁路、Memory 系统、人格加载、启动节奏、Avatar 渲染与终端 UI。

---

## 16. Interaction Reference

| 操作 | 行为 |
|---|---|
| 直接说话 | 显示 `◉ 正在聆听` 后开口即可；识别结果实时滚动，0.6s 静音后提交 |
| `0` | 循环切换当前人格的头像套装 |
| `Tab` | 展开 / 收起 Dashboard 详情：精确羁绊值、五维情绪值、互动次数 |
| `Ctrl+C` | 退出（自动写回持久化状态） |
| `soul-tty memory` | 列出全部记忆（按类型分组） |
| `soul-tty memory list --type profile` | 按类型过滤列出记忆 |
| `soul-tty memory show <id>` | 查看单条记忆详情 |
| `soul-tty memory forget <id>` | 删除单条记忆 |
| `soul-tty memory clear` | 清空全部记忆（需二次确认） |

---

## 17. Roadmap

### Memory Layer ✅

V1 已完成。三层分离记忆（画像 / 偏好 / 经历）、异步抽取、常驻段 + 按需召回、CLI 管理均已实现。

- ✅ 画像 profile 与偏好 preference 常驻 system prompt
- ✅ 经历 experience 按用户召回词按需检索注入
- ✅ 异步抽取由 Reflection Worker 旁路完成
- ✅ 去重与 importance 门槛过滤 | 管理与治理 CLI
- 🚧 V2 embedding 检索（替换当前 bigram 方案）
- 🚧 手动编辑与修正记忆

### Expression Layer

- Emotion-aware TTS：情绪 → 语气的实时热更新
- Avatar expression：表情与口型由情绪状态驱动

### Interaction Layer

- Full duplex voice：完整的采集 / 识别 / 取消 / 恢复状态机
- Real-time interruption：外放场景接入 AEC，区分用户插话与扬声器回声

### 明确不做

- 权限式亲密解锁 —— Bond 是连续梯度，不是游戏进度条
- 复杂 Avatar 状态机 —— 先让声音和语言体现状态

---

## 18. License & Thanks

**License：** 待定（开源筹备中）。

**致谢：**

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 嵌入式流式 ASR
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — 通用 LLM 推理
- [Qwen3-TTS](https://huggingface.co/Qwen) + [MLX-Audio](https://github.com/Blaizzy/mlx-audio) — Apple Silicon 流式语音合成
- [Rich](https://github.com/Textualize/rich) — 终端表现层
- [Chafa](https://github.com/hpjansson/chafa) — 终端像素图渲染
