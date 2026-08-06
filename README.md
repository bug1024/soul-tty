# Soul-TTY · 终端之魂

> **Soul-TTY is a local AI companion and agent architecture experiment — real-time voice in your terminal, backed by a persistent personality, an emotion state, and a relationship that evolves.**
>
> **Soul-TTY 是一个运行在终端中的本地 AI 伙伴，也是一个探索状态化 Agent 架构的开源项目：实时语音、人格、情绪状态与关系成长，全部跑在你自己的机器上。**

麦克风 → 流式识别 → 流式对话 → 流式语音，整条链路没有云依赖。
`uv run soul-tty` 之后，你对着终端说话，她就醒了。

<p align="center">
  <img src="assets/screenshots/serena-0.png" alt="Soul-TTY 运行截图：启动后的对话界面" width="900">
</p>

**换装即换人格。** 每套装映射一种行为语气，按 `0` 循环切换：

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

它不是"另一个 ChatGPT 客户端"。区别在于：普通 CLI 助手每一轮都是无状态的文本生成，上下文窗口一关就什么都不剩；Soul-TTY 在对话之外维护一组**持续演化的内部状态**——情绪是五维连续向量，关系是跨启动持久化的标量，两者共同决定她下一句话怎么说、用什么语气说。

因此这个项目同时是两件事：

- 一个**可以每天打开的终端语音伙伴**；
- 一个**状态化本地 Agent 的架构实验**：实时对话与状态演化分离、状态与表达分离、全部本地运行。

---

## 2. Features

| 能力 | 状态 | 说明 |
|---|---|---|
| **实时语音对话** | ✅ | 麦克风 → sherpa-onnx 流式 ASR → llama.cpp 流式 LLM → MLX Qwen3-TTS 流式语音 |
| **五维情绪体系** | ✅ | happiness / calmness / curiosity / stress / energy 连续向量，平滑演化 + 空闲自然衰减 |
| **Bond System（羁绊）** | ✅ | 跨启动持久化的关系标量，边际递减增长；后台旁路评估，不阻塞对话 |
| **三套装 × 三种模式** | ✅ | companion（默认）/ late_night（深夜）/ focused（工作），每套装映射一种行为语气 |
| **本地人格系统** | ✅ | YAML 描述人格、台词、形象、TTS 语气；不锁死，可热加载 |
| **动态口型** | ✅ | 闭嘴 / 半开双缓存图，由实际播放 PCM 音量驱动切换 |
| **动态台词** | ✅ | 启动欢迎、换装语、长时间安静后的陪伴短句均由 LLM 实时生成，超时回退本地短句 |
| **会话记忆** | 🚧 | 当前只有单次会话的短期上下文；跨会话记忆将作为独立的 Memory Layer 实现 |
| **全双工语音打断** | 🚧 | 实验开关 `BARGE_IN_ENABLED=1`；完整 AEC 状态机尚未稳定 |

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

语音、推理、合成全部跑在你机器上，对话内容不离开本机。可换模型、可换声音、可换人格。

---

## 4. Agent Architecture

```
                    User
                     │
                     ▼
        ┌────────────────────────┐
        │   Conversation Brain   │  实时主对话：流式识别 → 流式回复 → 流式语音
        │      (Realtime)        │  永远优先；任何状态计算不阻塞它
        └────────────┬───────────┘
                     │
                     │ 每轮完整回答
                     ▼
        ┌────────────────────────┐
        │    Reflection Brain    │  异步旁路：单 worker · 合并多轮 · 限频
        │       (Async)          │
        └──────┬─────────┬───────┘
               │         │
               ▼         ▼
     ┌──────────────┐  ┌──────────────────┐
     │ Bond System  │  │  Emotion System  │   State Layer
     │  长期 · 标量  │  │  短期 · 五维向量   │
     │  跨启动持久化 │  │  会话内演化 + 衰减 │
     └──────┬───────┘  └────────┬─────────┘
            │                   │
            └─────────┬─────────┘
                      ▼
        ┌────────────────────────┐
        │   Expression Layer     │  表达层：把内部状态翻译成"怎么说话"
        │  ───────────────────── │  Implemented:
        │                        │  · system prompt 的 Emotion Context
        │                        │  · TTS 语气指令
        │                        │  Planned:
        │                        │  · Avatar 表情与口型
        └────────────────────────┘
```

### 四层职责

| 层 | 职责 | 时延要求 |
|---|---|---|
| **Conversation Brain** | 用户真正"对话"的回路；同步、永远在主路径 | 亚秒级 |
| **Reflection Brain** | 状态演化；异步、可合并、可丢弃 | 不阻塞对话 |
| **State Layer** | Bond（跨启动持久化）+ Emotion（单次会话） | 无实时要求 |
| **Expression Layer** | 状态 → 可感知表现；只在听 / 播 / 写三个安全时机刷新 | 安全时机 |

### 为什么 Reflection 不在主路径上

主对话只向一个有界内存队列投递 `(user_text, agent_text)`，随后立刻返回。后台单 worker 在用户空闲窗口 + 限频条件下把多轮合并成一次 LLM 调用。投递失败、LLM 失败、低 confidence、队列满，任何一种情况都只让状态保持不变——对话本身永远不受影响。

### 为什么 Expression 是独立一层

Expression 是 *状态 → 表现* 的纯映射层，不持有任何状态：

- 输入：`{mood, intensity, expression}`
- 输出：TTS 语气指令、prompt 段落、未来的 avatar 参数

这样换 TTS 后端只影响 Expression 层，不影响 Emotion 本身；未来把 Avatar 接进来也是同一条链路的最后一节。

### 为什么 Emotion 与 Memory 是两件事

- **Emotion** 回答 *"我现在感觉怎么样？"* — 短期内部状态，单次会话内演化，随空闲衰减回归 baseline，默认不跨启动。
- **Memory** 回答 *"之前发生过什么？"* — 长期经验沉淀，跨会话累积。

两者时间尺度不同、存储位置不同、注入 LLM 的时机也不同：Emotion 实时热更新进 prompt，Memory 按上下文预算选择性注入。同理，**Bond 不能代替 Memory，Memory 也不能直接改写 Bond 或 Emotion**。

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

- 实现位于 `src/soul_tty/relationship.py`。
- 单 worker + 有界队列（`RELATIONSHIP_QUEUE_SIZE`，默认 4），队列满即丢弃，不反压主链路。
- 回答结束后等待 `RELATIONSHIP_IDLE_DELAY_S`（默认 3 秒）确认用户空闲；两次评估最小间隔 `RELATIONSHIP_MIN_INTERVAL_S`（默认 60 秒），窗口内多轮合并为一次调用。
- 单次 delta 上限 `RELATIONSHIP_MAX_DELTA`（默认 `0.03`），置信度低于 `RELATIONSHIP_MIN_CONFIDENCE`（默认 `0.65`）不生效。
- 最近关系事件按 FIFO 保留 `RELATIONSHIP_MAX_RECENT_EVENTS` 条（默认 20），避免无限增长。

</details>

---

## 7. Outfit & Modes

换装不只是换一张图（运行截图见文首）。每个套装标注一个 `mode`，这个 mode 决定情绪系统如何解释"现在是哪种陪伴状态"：

| 套装 | mode | 含义 | 适合 |
|---|---|---|---|
| **默认装** | `companion` | 标准陪伴形态，情绪中性、活力在线 | 日常闲聊、说点心里话 |
| **深夜装** | `late_night` | 情绪基线压低、节奏放缓 | 临睡前、放空时 |
| **工作装** | `focused` | 警觉度上升、好奇度拉高、不跑题 | 写代码、debug、长期任务 |

启动时通过 `--outfit` 或 `SOUL_TTY_OUTFIT` 指定，运行中按 `0` 在配置顺序之间循环切换。

换装后立即给出一句本地短句，再由后台 LLM 结合当前时段、羁绊阶段和本次会话情绪生成动态台词。连续换装不会阻塞界面，也不会污染正式对话历史。

---

## 8. Technical Implementation

```
麦克风 (16kHz mono PCM)
       │
       ▼
┌─ 音频采集 ──────────────────────────────────────┐
│  sounddevice · WebRTC VAD（30ms 帧门控）         │
│  空闲时只跑 VAD，触发后补回 300ms pre-roll        │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─ ASR ───────────────────────────────────────────┐
│  sherpa-onnx Streaming Paraformer int8（进程内）  │
│  partial 增量显示 → 0.6s 尾部静音提交 final       │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─ Conversation Brain ────────────────────────────┐
│  llama.cpp 流式 chat（OpenAI 兼容 HTTP）          │
│  流式 token → 句切分 · Markdown 净化 · 重复截断    │
│  system prompt 注入 [Emotion Context]            │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─ TTS ───────────────────────────────────────────┐
│  MLX Qwen3-TTS 1.7B CustomVoice（Apple Silicon）  │
│  流式 PCM 块（≈0.32s）· 音量驱动终端口型           │
│  备用后端：macOS `say`                            │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─ UI ────────────────────────────────────────────┐
│  Rich Dashboard                                  │
│  Kitty / Ghostty / iTerm2 原生图片协议            │
│  Chafa 真彩 Unicode 像素画回退                    │
└─────────────────────────────────────────────────┘
       │
       │  (异步旁路，不阻塞以上任何一段)
       ▼
┌─ Reflection Brain ──────────────────────────────┐
│  RelationshipService（bond · 单 worker · 合并评估）│
│  EmotionService（五维向量 · 平滑 · 空闲衰减）      │
└─────────────────────────────────────────────────┘
```

**为什么是这套技术栈：** 全本地、Apple Silicon 友好（MLX）、延迟可压到亚秒级、整条链路没有云依赖。辅助请求（欢迎语、换装台词、空闲短句）可以通过 `AUX_LLM_URL` 指向独立的小模型服务，与主对话彻底隔离算力竞争。

---

## 9. Project Structure

```text
src/soul_tty/
├── cli.py                # 命令行入口与启动信息
├── config.py             # 环境变量配置
├── conversation.py       # 对话流程、句切分、Markdown 净化
├── relationship.py       # Bond 非阻塞旁路 + 状态持久化
├── presence.py           # 启动节奏、低频特殊开场
├── emotion/              # 五维情绪体系
│   ├── service.py        #   EmotionService 单例
│   ├── state.py          #   EmotionVector / 持久化
│   ├── analyzer.py       #   LLM delta → 五维向量
│   ├── resolver.py       #   收敛合法 Mood / Expression
│   ├── prompt_builder.py #   向量 → 注入 LLM 的 context 文本
│   └── updater.py        #   节流更新（强度阈值）
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

## 10. Installation

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

### 安装并启动

```bash
git clone https://github.com/bug1024/soul-tty.git
cd soul-tty
uv sync                    # 自动创建 .venv 并安装依赖
cp .env.example .env       # 按需填写；也可直接用环境变量覆盖
uv run soul-tty
```

---

## 11. Configuration

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
| 打断 | `BARGE_IN_ENABLED` | `0` | 实验性全双工打断（建议配合耳机） |
| 打断 | `BARGE_IN_ECHO_SIMILARITY` | `0.72` | 回声判定阈值 |
| TTS | `TTS_BACKEND` | `mlx` | `mlx` / `macos` |
| TTS | `MLX_TTS_URL` | `http://127.0.0.1:50501` | MLX-Audio 服务地址 |
| TTS | `MLX_TTS_VOICE` | `Serena` | Qwen3-TTS 内置音色 |
| TTS | `MLX_TTS_INSTRUCT` | `""` | 可选语气指令，如 `用温柔、亲切的语气说` |
| TTS | `TTS_WHOLE_ANSWER` | `1` | `1`=完整回答后播报；`0`=按句流水线播报（首音更快） |
| Bond | `RELATIONSHIP_ENABLED` | `1` | 关闭后完全不评估 |
| Bond | `RELATIONSHIP_LLM_URL` | 同 `LLM_URL` | 评估服务；可指向独立小模型 |
| Bond | `RELATIONSHIP_IDLE_DELAY_S` | `3` | 回答结束后等用户空闲多久再评估 |
| Bond | `RELATIONSHIP_MIN_INTERVAL_S` | `60` | 两次评估最小间隔（窗口内多轮合并） |
| Emotion | `EMOTION_ENABLED` | `1` | 关闭则情绪系统不启动 |
| Emotion | `EMOTION_EMA_RATE` | `0.2` | 平滑率 |
| Emotion | `EMOTION_DECAY_INTERVAL_S` | `300` | 空闲衰减间隔（秒） |
| Emotion | `EMOTION_DECAY_RATE` | `0.05` | 每轮衰减幅度 |
| Emotion | `EMOTION_PERSIST` | `0` | 是否把情绪写盘 |
| Dashboard | `DASHBOARD_DETAILS` | `0` | 启动时是否展开详情（运行中按 `Tab` 切换） |
| Dashboard | `DASHBOARD_MAX_MESSAGES` | `300` | 可滚动消息上限 |
| 状态 | `SOUL_TTY_STATE_DIR` | `~/.local/state/soul-tty` | 持久化目录（Bond、Emotion） |

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

## 12. Development

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

测试覆盖音频链路、对话切句、情绪系统、Bond 旁路、人格加载、启动节奏、Avatar 渲染与终端 UI。

---

## 13. Interaction Reference

| 操作 | 行为 |
|---|---|
| 直接说话 | 显示 `◉ 正在聆听` 后开口即可；识别结果实时滚动，0.6s 静音后提交 |
| `0` | 循环切换当前人格的头像套装 |
| `Tab` | 展开 / 收起 Dashboard 详情：精确羁绊值、五维情绪值、互动次数 |
| `Ctrl+C` | 退出（自动写回持久化状态） |

---

## 14. Roadmap

### Memory Layer

- 跨会话记忆：短期上下文 / 跨会话摘要 / 长期用户事实三层分离
- 经验检索：按上下文预算选择性注入，而非无限增长
- 记忆治理：查看、修正、删除与一键清空

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

## 15. License & Thanks

**License：** 待定（开源筹备中）。

**致谢：**

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 嵌入式流式 ASR
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — 通用 LLM 推理
- [Qwen3-TTS](https://huggingface.co/Qwen) + [MLX-Audio](https://github.com/Blaizzy/mlx-audio) — Apple Silicon 流式语音合成
- [Rich](https://github.com/Textualize/rich) — 终端表现层
- [Chafa](https://github.com/hpjansson/chafa) — 终端像素图渲染
