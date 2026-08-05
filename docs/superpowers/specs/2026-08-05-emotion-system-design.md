# Soul-TTY 实时情绪系统设计文档

**日期：** 2026-08-05
**版本：** V1
**作者：** Soul-TTY 团队

---

## 1. 目标

实现一个**短周期、动态变化、有惯性的 AI 情绪状态系统**，让角色具备：

- 每次启动有不同初始状态；
- 对话内容持续影响情绪；
- 情绪不会瞬间切换，而是逐步变化；
- 情绪影响 LLM 回复风格、语音、表情（接口预留）、换装（可选）。

---

## 2. 情绪模型

采用**多维情绪值模型**，不直接维护单一离散情绪。所有维度归一化到 `[0, 1]`。

### 2.1 情绪维度

| 维度           | 范围  | 说明         |
| -------------- | ----- | ------------ |
| Happiness 快乐 | 0~1   | 积极、愉悦程度 |
| Calmness 平静  | 0~1   | 稳定、放松程度 |
| Curiosity 好奇 | 0~1   | 探索欲、兴趣程度 |
| Stress 压力    | 0~1   | 紧张、负荷程度 |
| Energy 能量    | 0~1   | 活跃程度     |

### 2.2 状态示例

```json
{
  "happiness": 0.65,
  "calmness": 0.75,
  "curiosity": 0.70,
  "stress": 0.20,
  "energy": 0.80
}
```

---

## 3. 当前情绪状态（Mood）

**Mood 不直接存储**，由五维情绪值实时计算。

### 3.1 Mood 分类与优先级

优先级从高到低（异常状态优先判定）：

```
Numb → Tired → Sad → Excited → Happy → Curious → Calm
```

### 3.2 Resolver 阈值表

| Mood    | 规则                              |
| ------- | --------------------------------- |
| Numb    | stress ≥ 0.75 AND energy ≤ 0.25   |
| Tired   | energy ≤ 0.35                     |
| Sad     | happiness ≤ 0.35 AND stress ≥ 0.45 |
| Excited | happiness ≥ 0.75 AND energy ≥ 0.75 |
| Happy   | happiness ≥ 0.65 AND energy ≥ 0.4  |
| Curious | curiosity ≥ 0.7                   |
| Calm    | 默认（不满足以上任意条件）           |

### 3.3 Intensity 计算

每个 mood 都附一个 0~1 的强度值，体现该 mood 的"程度"：

| Mood    | Intensity 计算           |
| ------- | ------------------------ |
| Numb    | `(stress + (1-energy)) / 2` |
| Tired   | `1 - energy`             |
| Sad     | `(1-happiness + stress) / 2` |
| Excited | `(happiness + energy) / 2` |
| Happy   | `happiness`              |
| Curious | `curiosity`              |
| Calm    | `calmness`               |

---

## 4. 初始值生成

**不完全随机**——人格基础值 + 随机扰动。

### 4.1 人格基础值

每个 persona 在 YAML 里提供 `personality.mood_baseline`：

```yaml
personality:
  system_prompt: |-
    ...
  mood_baseline:
    happiness: 0.65
    calmness: 0.75
    curiosity: 0.70
    stress: 0.20
    energy: 0.75
```

缺省时使用默认基础值：

```json
{
  "happiness": 0.65,
  "calmness": 0.75,
  "curiosity": 0.70,
  "stress": 0.20,
  "energy": 0.75
}
```

### 4.2 启动扰动

每次启动时给每个维度叠加 ±10% 的随机扰动，并夹到 [0, 1]。

示例：

第一次启动：

```json
{
  "happiness": 0.72,
  "calmness": 0.68,
  "curiosity": 0.75,
  "stress": 0.25,
  "energy": 0.70
}
```

第二次启动：

```json
{
  "happiness": 0.60,
  "calmness": 0.82,
  "curiosity": 0.65,
  "stress": 0.15,
  "energy": 0.80
}
```

---

## 5. 情绪更新机制

### 5.1 完整流程

```
用户输入
   ↓
Interaction Analyzer（旁路 LLM）
   ↓
emotion_delta + relationship_delta
   ↓
Mood Engine
   ↓
更新五维情绪值（EMA 平滑）
   ↓
Mood Resolver
   ↓
当前 mood + intensity
   ↓
Prompt Builder
   ↓
Emotion Context 段落
   ↓
Chat.update_system_prompt（热更新）
   ↓
下一轮 LLM 回复生效
```

### 5.2 Emotion Delta

Interaction Analyzer 每次输出五维情绪的增量（每维度 ∈ [-0.3, +0.3]）：

```json
{
  "happiness": +0.10,
  "stress": -0.05,
  "energy": +0.05
}
```

情绪评估**复用现有的 relationship 旁路评估 pipeline**，不在主对话里塞。

---

## 6. 情绪变化算法

### 6.1 EMA 平滑

每轮收到 emotion_delta 后，对每一维执行：

```
new = old × (1 - rate) + (old + delta) × rate
```

推荐 `rate = 0.2`。

效果：

- 单次对话影响有限；
- 连续同向输入会累积；
- 不会瞬间人格变化。

### 6.2 衰减（Decay）

仅在 idle 时执行，每 5 分钟一次：

```
value += (baseline - value) × 0.05
```

其中 `baseline` 来自当前 persona 的 `mood_baseline`。

### 6.3 规则优先级

- **Active conversation**：只执行对话更新，不执行 decay；
- **Idle（≥ 5 分钟无输入）**：只执行 decay，不接收对话更新。

---

## 7. 情绪存储

V1 **不接入 Memory**，本地 JSON 保存即可。

### 7.1 文件位置

```
~/.local/state/soul-tty/emotion/{persona_id}.json
```

### 7.2 结构

```json
{
  "session_id": "uuid",
  "session_count": 7,
  "emotion": {
    "happiness": 0.65,
    "calmness": 0.75,
    "curiosity": 0.70,
    "stress": 0.20,
    "energy": 0.80
  },
  "updated_at": "2026-08-05 22:00"
}
```

### 7.3 规则

- Session 期间持续保存（每次更新后落盘）；
- 新启动时**情绪值重置**，但 `session_count` 从 JSON 读取并 +1；
- V1 不跨 session 持久化情绪值，V2 可扩展。

---

## 8. System Prompt 注入

### 8.1 结构

System Prompt 现在分三段：

```
[Persona]
你是 Serena，...

[Conversation Mode]
你处于专注模式：...

[Emotion Context]
当前状态：你处于轻松愉悦状态。

行为倾向：
- 回复语气更加积极
- 可以适当表达开心
- 保持自然，不过度兴奋
```

### 8.2 注入规则

- **不暴露原始数值**（"happiness=0.73" 之类）；
- 只输出**行为描述**（"语气更加积极"）；
- Emotion Context 段落由 Prompt Builder 根据当前 mood + intensity 生成；
- 段落追加到 system_prompt 末尾；
- 任何情绪变化都通过 `Chat.update_system_prompt()` 热更新到活跃 Chat 实例。

### 8.3 Prompt Builder 模板

每个 mood 对应一段固定模板，intensity 控制修饰词强度（>=0.8 加"明显"，>=0.5 加"中等"，<0.5 不加）。

---

## 9. 情绪影响范围

### 9.1 LLM Prompt（V1 主战场）

影响：

- 回复语气；
- 回复长度倾向；
- 主动程度；
- 幽默程度；
- 是否表达情绪。

### 9.2 TTS（V1）

通过 `MLX_TTS_INSTRUCT` 注入情绪描述：

| Mood    | TTS Instruct                            |
| ------- | --------------------------------------- |
| Happy   | "用开心上扬的语气说"                      |
| Excited | "用兴奋激动的语气说"                      |
| Sad     | "用低沉平缓的语气说"                      |
| Tired   | "用轻柔缓慢的语气说"                      |
| Calm    | (使用音色默认)                              |
| Curious | "用好奇询问的语气说"                      |
| Caring  | "用温柔关切的语气说"                      |
| Numb    | "用平淡低能量的语气说"                    |

### 9.3 Avatar（V1 预留接口）

不在 outfit schema 扩展，新增独立 `avatar_expression` 概念：

```json
{
  "face": "smile",
  "eye": "open",
  "motion": "slight_nod"
}
```

V1 提供默认 mapping 表，但**不实际接入 renderer**——renderer 后续单独实现。

默认 mapping 表：

| Mood    | face     | eye   | motion       |
| ------- | -------- | ----- | ------------ |
| Happy   | smile    | open  | slight_nod   |
| Excited | bright   | open  | bounce       |
| Sad     | droop    | half  | none         |
| Tired   | flat     | half  | none         |
| Calm    | neutral  | open  | none         |
| Curious | neutral  | wide  | tilt_head    |
| Caring  | soft     | open  | slight_lean  |
| Numb    | flat     | half  | none         |

### 9.4 Outfit（V1 不接入）

V1 不让情绪直接触发换装。换装仍由用户 `0` 键手动控制。

---

## 10. 模块设计

新增模块：

```
src/soul_tty/emotion/
├── __init__.py
├── state.py             # 五维情绪值数据结构 + 持久化
├── analyzer.py          # 从 LLM 输出提取 emotion_delta
├── updater.py           # EMA 平滑 + decay 算法
├── resolver.py          # 五维值 → mood + intensity
├── prompt_builder.py    # mood → Emotion Context 段落
├── expression.py        # mood → avatar_expression mapping
└── service.py           # EmotionService 顶层协调
```

修改现有模块：

```
src/soul_tty/
├── cli.py               # 启动 EmotionService，注入到 RelationshipService
├── relationship.py      # 升级为 InteractionAnalyzer，输出 emotion_delta
├── clients/llm.py       # evaluate_relationship 输出加 emotion_delta 字段
├── personas/loader.py   # 加载 mood_baseline
├── personas/models.py   # Personality 增加 mood_baseline 字段
├── ui/terminal.py       # 暂不显示情绪数值（V1 UI 不变）
├── config.py            # 新增情绪相关配置
└── conversation.py      # 注入 Emotion Context 到 system_prompt
```

---

## 11. 详细数据流

### 11.1 启动

```
cli.main()
   ↓
load_persona() → persona.mood_baseline
   ↓
EmotionService(baseline=persona.mood_baseline)
   ↓
随机扰动 → 初始五维值
   ↓
加载 session_count
   ↓
MoodResolver → 初始 mood + intensity
   ↓
PromptBuilder → 初始 Emotion Context
   ↓
apply_persona() 拼装完整 system_prompt
   ↓
Chat(model) 创建
```

### 11.2 每轮对话

```
用户语音 → ASR → text
   ↓
Chat.ask_stream(text)  # 主对话，注入完整 system_prompt
   ↓
LLM 回复流式输出
   ↓
relationship.record_turn(user, agent)
   ↓
InteractionAnalyzer.evaluate()  # 旁路
   ↓
输出 emotion_delta + relationship_delta
   ↓
EmotionService.apply_delta(emotion_delta)
   ↓
EMA 平滑更新五维值
   ↓
持久化到 JSON
   ↓
MoodResolver → 新 mood + intensity
   ↓
PromptBuilder → 新 Emotion Context
   ↓
Chat.update_system_prompt(new_system_prompt)
   ↓
下一轮对话生效
```

### 11.3 Idle 衰减

```
后台线程，每 5 分钟一次：
   ↓
检查 last_activity
   ↓
如果 idle ≥ 5 分钟：
   ↓
对每个维度执行 decay
   ↓
持久化
   ↓
刷新 mood / system_prompt
```

---

## 12. Interaction Analyzer 输出协议

升级 `evaluate_relationship` 输出 schema：

```json
{
  "event": "用户分享了项目上线",
  "delta": 1,
  "mood": "happy",
  "inner_voice": "替你高兴呢。",
  "confidence": 0.85,
  "emotion_delta": {
    "happiness": 0.15,
    "stress": -0.05,
    "energy": 0.05
  }
}
```

`emotion_delta` 是新增字段，可选。`apply_evaluation` 处理时：

1. 原有 relationship 逻辑不变；
2. 新增情绪 delta 处理：调用 `EmotionService.apply_delta()`。

V1 默认行为：情绪分析由同一个 LLM 输出，避免额外调用。

---

## 13. 配置项

`config.py` 新增：

```python
EMOTION_ENABLED = ...               # 是否启用（默认开）
EMOTION_EMA_RATE = float("0.2")     # EMA 平滑率
EMOTION_DELTA_CAP = float("0.3")    # 单轮 delta 上限
EMOTION_DECAY_INTERVAL_S = 300      # 衰减周期
EMOTION_DECAY_RATE = float("0.05")  # 衰减速率
EMOTION_IDLE_THRESHOLD_S = 300      # 进入 idle 阈值
EMOTION_PERSIST = False             # 是否跨 session 持久化（V1 固定 False）
```

---

## 14. 测试策略

`tests/test_emotion.py` 覆盖：

1. EMA 平滑算法正确性；
2. Mood Resolver 阈值边界（含互斥条件）；
3. Intensity 计算；
4. Decay 算法；
5. JSON 持久化往返；
6. 启动扰动范围；
7. Prompt Builder 输出包含正确行为描述、不包含原始数值。

`tests/test_relationship.py` 更新：

- Interaction Analyzer 输出包含 `emotion_delta` 字段（mock 评估器）。

---

## 15. 范围之外（V2 候选）

- 跨 session 情绪持久化；
- 情绪对换装的自动触发；
- Avatar renderer 实际接入 expression 参数；
- 多 Persona 情绪差异学习；
- 情绪→面部表情的实时插值动画；
- 情绪值的可解释性可视化（雷达图）。

---

## 16. 设计决策表

| 问题             | 决策                                         |
| ---------------- | -------------------------------------------- |
| 情绪评估来源       | 复用 relationship pipeline，加 `emotion_delta` |
| 分析模块名称       | Interaction Analyzer                        |
| Resolver 方式    | 优先级 + 阈值                                |
| 默认 mood        | 不满足任意条件时按情绪最低的 mood 输出        |
| 衰减触发          | 仅 idle 时执行                              |
| Prompt 注入      | system_prompt 独立 Emotion Context 段落     |
| Prompt 是否暴露数值 | 否，只暴露行为描述                          |
| Avatar           | 预留 expression 接口和 mapping 表，不实际渲染 |
| Persona 基础值    | YAML 配置，缺省用全局默认                   |
| 持久化            | 仅本地 JSON，情绪不跨 session              |
| session_count    | 跨 session 同步递增                          |