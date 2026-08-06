# Soul TTY · 终端之魂

> **Soul-TTY is a local AI companion with real-time voice interaction, persistent personality, emotion state, and relationship evolution.**
>
> **Soul-TTY 是一个运行在终端中的本地 AI 伙伴，具备实时语音、人格、情绪状态和关系成长能力。**

> 麦克风 → 流式识别 → 流式对话 → 流式语音，所有链路跑在你自己的机器上。

---

## 写在前面：什么是"终端之魂"

> *Soul TTY，Terminal Soul。*

这个词把两个意向并置：

- **Soul** — 一个有情绪、有温度、会成长的角色。她记得今天是你的夜晚，记得你已经问过几次同一个问题，记得你说"工作装"的时候其实想专注。
- **TTY** — 没有窗口、没有按钮、没有花哨动画。**一个终端就够了。**

我们想要的不是"另一个 ChatGPT 客户端"，而是一种"在命令行里也能被陪伴"的体验：
**`uv run soul-tty` 之后，你对着终端说话，她就醒了。**

围绕这个目标，项目把所有功能都收束在同一件事上 ——

> **让"在终端里陪着你说话"这件事，做到值得每天打开。**

---

## 当前能做什么

| 模块 | 状态 | 说明 |
|---|---|---|
| **实时语音对话** | ✅ | 麦克风 → sherpa-onnx 流式 ASR → llama.cpp 流式 LLM → MLX Qwen3-TTS 流式语音 |
| **动态口型** | ✅ | 闭嘴 / 半开双缓存图，由实际播放 PCM 体积驱动切换；音量未跨阈值不重绘 |
| **动态台词** | ✅ | 启动欢迎、换装语、长时间安静后的陪伴短句均由 LLM 实时生成；超时回退本地短句，不阻塞界面 |
| **五维情绪体系** | ✅ | happiness / calmness / curiosity / stress / energy 五维向量，EMA 平滑 + 空闲自然衰减 |
| **Bond System（羁绊）** | ✅ | 后台非阻塞旁路评估 bond（边际递减）；欢迎区显示阶段，按 `Tab` 看精确分值 |
| **三套装 + 三种模式** | ✅ | companion（默认）/ late_night（深夜）/ focused（工作），每套装映射一种行为语气 |
| **本地人格系统** | ✅ | YAML 描述人格、台词、形象、TTS 语气；不锁死，可热加载 |
| **会话记忆** | 🚧 | 设计中 — 见 [TODO.md](./TODO.md)。当前只有短期上下文，跨会话记忆将在新层实现 |
| **全双工语音打断** | 🚧 | 实验开关 `BARGE_IN_ENABLED=1`；完整 AEC 状态机尚未稳定 |

---

## Emotion System

Soul-TTY 不使用简单情绪标签，而是维护一组**连续状态**，让"心情"是一个有过程、有惯性、会自己恢复的过程。

**五维向量：**

| 维度 | 含义 | 极端高 | 极端低 |
|---|---|---|---|
| **happiness** | 愉悦度 | 兴高采烈 | 心情低落 |
| **calmness** | 平静度 | 心如止水 | 焦躁不安 |
| **curiosity** | 好奇度 | 主动探索 | 无聊放空 |
| **stress** | 压力值 | 紧绷应激 | 完全放松 |
| **energy** | 活力值 | 兴奋充沛 | 疲倦低迷 |

**状态变化路径：**

```
LLM 评估（旁路）
       │
       │  每轮 emotion_delta（5 维 -0.3~+0.3）
       ▼
EmotionService.apply_delta
       │
       │  EMA 平滑（默认 0.2，避免单次突变）
       │  delta_cap 截断（防止 LLM 抖动污染）
       ▼
EmotionVector（内存 / 持久化）
       │
       │  idle_decay：长时间无交互 → 缓慢回归 baseline
       │  （每 5 分钟按 0.05 衰减）
       ▼
resolver → (mood, intensity, expression)
       │
       ▼
Expression Layer
   ├─ system prompt [Emotion Context] 段
   ├─ TTS 语气指令（规划中）
   └─ Avatar 表情（未来）
```

**当前实现状态：**

- ✅ 五维向量 + EMA + idle decay（已在 `src/soul_tty/emotion/`）
- ✅ 收敛到 `mood` + `intensity` + `expression`（resolver）
- ✅ 注入到 system prompt 的 `[Emotion Context]` 段
- 🚧 驱动 TTS 语气指令（Expression Mapper → MLX_TTS_INSTRUCT）
- 🚧 驱动 Avatar 表情与口型（后置）

---

## Bond System

"Bond"（羁绊）是 Soul-TTY 对"长期陪伴关系"的命名。**注意**：在 README 内部统一称 Bond，不混用 relationship / intimacy / 关系。

**与 Emotion 的边界：**

| | Bond | Emotion |
|---|---|---|
| 维度 | 单一标量 `bond ∈ [0, 1]` | 5 维向量 |
| 时间尺度 | 跨启动持久化 | 当次会话内存（可持久化但默认不写盘） |
| 增长方式 | 边际递减 `bond + delta * (1 - bond)` | EMA 平滑 + 衰减 |
| 触发 | LLM 评估确认有"关心/共同玩笑/信任"等事件 | 任意情绪 delta |
| 显示 | 按 Tab 看精确分值 + 阶段标签 | Dashboard 五维条形 + mood 文案 |

**评估节奏：**

- 每轮完整回答结束后，主流程只向有界内存队列投递问答
- 后台单 worker 在空闲窗口 + 限频条件下合并多轮 → 调用 LLM
- LLM 输出 `relationship_delta.bond` + `emotion_delta` + `expression`（三路分离）
- 投递失败 / LLM 失败 / 低 confidence 都不会改变状态

**阶段映射：**

| bond | 阶段 |
|---|---|
| 0.00 – 0.09 | stranger |
| 0.10 – 0.29 | acquaintance |
| 0.30 – 0.49 | familiar |
| 0.50 – 0.69 | companion |
| 0.70 – 0.89 | close |
| 0.90 – 1.00 | bonded |

---

## 三种模式：换装即换人格

每个套装都标注一个 `mode`，这个 mode 决定情绪系统怎么解释"现在是哪种陪伴状态"：

| 套装 | mode | 含义 | 适合 |
|---|---|---|---|
| **默认装** | `companion` | 标准陪伴形态，情绪中性、活力在线 | 日常闲聊、说点心里话 |
| **深夜装** | `late_night` | 情绪基线压低、节奏放缓 | 临睡前、放空时 |
| **工作装** | `focused` | 警觉度上升、好奇度拉高、不跑题 | 写代码、debug、长期任务 |

启动时通过 `--outfit` 或 `SOUL_TTY_OUTFIT` 选；运行中按 `0` 在配置顺序之间循环切换。
换装后立即给出一句本地短句，再由后台 LLM 根据当前时段、羁绊阶段、本次会话情绪生成动态台词；
连续换装不会阻塞界面，也不会污染正式对话历史。

---

## 技术栈

```
麦克风 (16kHz mono PCM)
       │
       ▼
┌─ 音频采集 ──────────────────────────────────────┐
│  sounddevice    WebRTC VAD（30ms 帧门控）        │
│  + 300ms pre-roll + WebRTC VAD 触发 → sherpa     │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─ ASR ───────────────────────────────────────────┐
│  sherpa-onnx  Streaming Paraformer int8（进程内） │
│  partial 增量显示 → 0.6s endpoint 提交 final    │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─ 主对话 ────────────────────────────────────────┐
│  llama.cpp 流式 chat (HTTP)                     │
│  /v1/chat/completions · 流式 token → 句切分      │
│  · Markdown 净化 · 重复截断 · prompt 注入情绪    │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─ TTS ───────────────────────────────────────────┐
│  MLX Qwen3-TTS 1.7B CustomVoice (Apple Silicon) │
│  流式 PCM 块（≈0.32s）· 音量 → 终端口型           │
│  备用后端：macOS `say`                            │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─ 表现层 ────────────────────────────────────────┐
│  Rich Dashboard（Kitty/Ghostty/iTerm 原生图片协议）│
│  Chafa 真彩 Unicode 像素画回退                    │
└─────────────────────────────────────────────────┘
       │
       │  (异步旁路)
       ▼
┌─ Bond + Emotion 后台 ───────────────────────────┐
│  RelationshipService（bond · 单 worker · 合并评估）│
│  EmotionService（五维向量 · EMA · 空闲衰减）       │
└─────────────────────────────────────────────────┘
```

**为什么是这套技术栈：** 全本地、Apple Silicon 友好（MLX）、延迟可压到亚秒级、整条链路没有云依赖。

---

## Agent 架构

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
     ┌─────────────┐  ┌────────────────┐
     │  Bond System│  │ Emotion System │  五维向量：happiness / calmness /
     │  (羁绊)     │  │  (情绪)         │  curiosity / stress / energy
     │ bond 持久化  │  │ EMA 平滑 + 衰减  │
     │ event 累积  │  │ mood + expression 收敛 │
     └──────┬──────┘  └────────┬───────┘
            │                  │
            └────────┬─────────┘
                     ▼
        ┌────────────────────────┐
        │   Expression Layer     │  表达层：把内部状态翻译成"怎么说话"
        │  ───────────────────── │  → system prompt 的 Emotion Context
        │                        │  → TTS 的语气指令（后续）
        │                        │  → Avatar 的口型 / 表情（未来）
        └────────────────────────┘
```

**四层职责：**

| 层 | 职责 | 时延要求 |
|---|---|---|
| **Conversation Brain** | 用户真正"对话"的回路；同步、永远在主路径 | 亚秒级 |
| **Reflection Brain** | 状态演化；异步、可合并、可丢弃 | 不阻塞对话 |
| **State**（Bond / Emotion） | 跨会话记忆 + 当次情绪快照 | 持久化 |
| **Expression** | 状态 → 可感知表现；只在听 / 播 / 写三个安全时机刷新 | 安全时机 |

---

## 核心设计原则

- **主对话链路优先** — 任何状态计算、持久化、旁路评估都不能阻塞回复；旁路失败也不影响对话继续。
- **状态异步演化** — Bond / Emotion 在后台 worker 中评估；冷却窗口内的多轮会合并为一次 LLM 调用。
- **本地优先** — 语音、推理、合成全部跑在你机器上；对话内容不离开本机；可换模型、可换声音、可换人格。
- **表现与状态分离** — Persona / Emotion / Expression 各自独立演进；Bond 不能代替记忆，记忆也不能直接改 Bond。

---

## 项目结构

```text
src/soul_tty/
├── cli.py                # 命令行入口与启动信息
├── config.py             # 环境变量配置
├── conversation.py       # 对话流程、句切分、Markdown 净化
├── relationship.py       # 非阻塞亲密成长旁路 + 状态持久化
├── presence.py           # 启动节奏、低频特殊开场
├── emotion/              # 五维情绪体系（service / state / resolver / mapping）
│   ├── service.py        #   EmotionService 单例
│   ├── state.py          #   EmotionVector / 持久化
│   ├── analyzer.py       #   LLM delta → 五维向量
│   ├── resolver.py       #   收敛合法 Mood / Expression
│   ├── prompt_builder.py #   向量 → 注入 LLM 的 context 文本
│   └── updater.py        #   节流更新（强度阈值）
├── audio/                # 录音 + sherpa-onnx ASR + MLX TTS
├── clients/llm.py        # llama.cpp / Hermes 客户端
├── personas/             # YAML 人格加载、校验、运行时应用
└── ui/
    ├── terminal.py       # Rich Dashboard、对话视口、状态栏
    └── avatar.py         # Chafa / 原生图片协议 / 像素画回退

personas/                 # 用户可直接编辑的人格 YAML
assets/avatars/           # 768×768 原创像素风角色图
docs/                     # 设计稿、规划文档
TODO.md                   # 路线图（会话记忆、稳定语音打断）
```

---

## Roadmap

**已交付 ✅**

- 实时语音对话（ASR + LLM + TTS 全链路流式）
- 五维情绪体系（EMA 平滑 + idle decay）
- Bond System（边际递减 + 三路状态分离）
- 三种模式 × 三套装（companion / late_night / focused）
- 动态口型 + 动态台词 + 启动节奏
- 本地人格系统（YAML 可热加载）

**下一阶段 🚧**

1. **Emotion → Prompt / TTS 消费链**（C）
   - C1 Prompt：[Emotion Context] 段已落地，下一步让 emotion delta 实时触发 prompt 热更新
   - C2 TTS：Emotion Mapper → MLX_TTS_INSTRUCT（V1 走文本指令，不让 TTS 知道 emotion enum）
   - C3 Avatar：Avatar 表情与口型（后置）

2. **会话记忆（Memory Layer）**
   - 短期上下文 / 跨会话摘要 / 长期用户事实 三层分离
   - 见 [TODO.md](./TODO.md) 设计稿

3. **稳定全双工打断**
   - 从 `BARGE_IN_ENABLED=1` 实验开关做成完整状态机 + AEC 接入

**暂不做 ❌**

- 更多目录拆分（当前结构已经够清晰）
- 复杂 Avatar 状态机（先让声音和语言体现状态）
- 权限式亲密解锁（Bond 是连续梯度，不是游戏进度条）
- 拆 ReflectionService / EmotionService / ExpressionService 命名重构（代码稳定，重命名收益有限）

---

## 安装运行

### 0. 前置依赖

Soul TTY 是一段 **Python 进程**，但完整跑起来还需要几个外部服务：

| 组件 | 用途 | 推荐部署 |
|---|---|---|
| **Python ≥ 3.10** | 主程序 | 系统 Python / pyenv / uv 自带 |
| **uv** | 依赖管理 | `pip install uv` 或 `brew install uv` |
| **chafa** | 终端像素图渲染（Kitty/Ghostty/iTerm 上使用原生协议时可跳过） | `brew install chafa` |
| **sherpa-onnx** | 流式 ASR（已作为 `pip` 依赖打进 soul-tty，无需单独部署） | — |
| **llama.cpp / llama-server** | 主对话 + 羁绊评估 | `llama-server -m your-model.gguf --port 8180 --host 127.0.0.1` |
| **MLX Qwen3-TTS** | 流式中文 TTS | 任意 MLX-Audio 兼容服务（端口 `50501`） |
| **macOS 麦克风权限** | 第一次运行时会自动弹出授权 | 系统设置 → 隐私与安全性 → 麦克风 |

> **推荐运行环境：** macOS（M 系）+ Ghostty / Kitty / iTerm2。
> 原生图片协议能让头像以接近原画的清晰度显示在终端里；其他终端自动回退为真彩 Unicode 像素画。

### 1. 准备模型

`llama-server` 启动任意支持中文的指令模型即可（例如 Qwen3、Llama-3 中文微调等量化版），并暴露兼容 OpenAI 的 `/v1/chat/completions` 与 `/v1/models`。

`MLX TTS` 端使用 `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` 或兼容版本。

### 2. 安装并启动

```bash
git clone https://github.com/bug1024/soul-tty.git
cd soul-tty
uv sync                    # 自动创建 .venv 并安装依赖
```

复制示例配置并按需填写：

```bash
cp .env.example .env       # 如有提供；当前可直接用环境变量覆盖
```

启动：

```bash
uv run soul-tty
```

### 3. 调试 / 旁路运行

不需要麦克风也能验证 LLM ↔ TTS 链路：

```bash
uv run soul-tty --text "你好"                  # 跳过 ASR，直接发给 LLM
uv run soul-tty --file /path/to/16k_mono.wav  # 用本地音频跑 ASR→LLM→TTS 全链路
```

---

## 交互速查

| 按键 | 行为 |
|---|---|
| 说话 | 显示 `◉ 正在聆听` 后直接开口；sherpa partial 实时滚动，0.6s 静音后提交 |
| `0` | 循环切换当前人格的头像套装 |
| `Tab` | 展开 / 收起 Dashboard 详情：精确羁绊值、五维情绪值、互动次数 |
| `Ctrl+C` | 退出（自动写回持久化状态） |

---

## 人格与扩展

默认人格是 `serena`（紫色未来感中文女声）。其它人格通过 YAML 在
`personas/` 目录或 `SOUL_TTY_PERSONA_DIR` 自定义目录下添加即可：

```bash
uv run soul-tty personas                   # 列出可用人格
uv run soul-tty --persona serena --name 小夜   # 临时改名（也注入 LLM）
SOUL_TTY_PERSONA=my_persona uv run soul-tty     # 用环境变量选人格
```

人格文件位于 `personas/*.yaml`，可定义：

- 名字、tagline、开场白、告别语
- LLM 系统提示词与生成参数上限
- TTS 音色、语气指令
- 终端主色 / 副色 / 强调色
- 头像渲染器（`auto` / `pixels` / `symbols` / `off`）
- 多套 `outfits`，每套关联一组 `idle / speaking_closed / speaking_half` 图与 `mode`

也可以直接加载外部 YAML：

```bash
uv run soul-tty --persona /path/to/my-persona.yaml
SOUL_TTY_PERSONA_DIR=/path/to/dir uv run soul-tty
```

---

## 配置参考

所有配置都通过环境变量覆盖（见 [`src/soul_tty/config.py`](./src/soul_tty/config.py)）。

| 类别 | 变量 | 默认 | 说明 |
|---|---|---|---|
| 人格 | `SOUL_TTY_PERSONA` | `serena` | 人格 id 或 YAML 路径 |
| 人格 | `SOUL_TTY_OUTFIT` | persona `default_outfit` | 启动时套装；运行时按 `0` 循环 |
| 人格 | `AGENT_NAME` | persona 名 | 覆盖显示名并注入 LLM |
| 主对话 | `LLM_URL` | `http://127.0.0.1:8180` | 主对话 LLM 地址（默认；Chat 流式回答走这里） |
| 主对话 | `LLM_MODEL` | 自动取 `/v1/models` 第一个 | 主对话模型 id |
| 主对话 | `LLM_MAX_TOKENS` | `256` | 单轮回答硬上限 |
| 主对话 | `LLM_TEMPERATURE` | `0.7` | 采样温度 |
| 主对话 | `LLM_REPEAT_PENALTY` | `1.1` | llama.cpp 重复惩罚 |
| 辅助 | `AUX_LLM_URL` | `LLM_URL`（回退） | 辅助 LLM 地址（欢迎/换装/idle） |
| 辅助 | `AUX_LLM_MODEL` | `LLM_MODEL`（回退） | 辅助 LLM 模型 id |
| ASR | `SHERPA_MODEL_DIR` | `../sherpa-asr/models/...` | Streaming Paraformer int8 模型目录 |
| ASR | `SHERPA_ENDPOINT_SILENCE_S` | `0.60` | 已识别出内容后的尾部静音提交阈值 |
| ASR | `SHERPA_VAD_GATE_ENABLED` | `1` | 空闲时只跑 WebRTC VAD，不持续调用 Paraformer |
| ASR | `SHERPA_VAD_PRE_ROLL_MS` | `300` | 检测到人声时补回的句首音频长度 |
| 切句 | `VAD_AGGRESSIVENESS` | `2` | 0-3，越大越严格 |
| 切句 | `SILENCE_MS` | `700` | 连续静音判定一句话结束 |
| 切句 | `MAX_UTTERANCE_S` | `15` | 单句最长秒数，超出强制切段 |
| 打断 | `BARGE_IN_ENABLED` | `0` | 实验性全双工打断（建议耳机） |
| 打断 | `BARGE_IN_ECHO_SIMILARITY` | `0.72` | 回声判定阈值 |
| TTS | `TTS_BACKEND` | `mlx` | `mlx` / `macos` |
| TTS | `MLX_TTS_URL` | `http://127.0.0.1:50501` | MLX-Audio 服务地址 |
| TTS | `MLX_TTS_VOICE` | `Serena` | Qwen3-TTS 内置音色 |
| TTS | `MLX_TTS_INSTRUCT` | `""` | 可选语气指令，如 `用温柔、亲切的语气说` |
| TTS | `TTS_WHOLE_ANSWER` | `1` | `1`=完整回答后播报；`0`=按句流水线播报（首音更快） |
| 羁绊 | `RELATIONSHIP_ENABLED` | `1` | 关闭后完全不评估 |
| 羁绊 | `RELATIONSHIP_LLM_URL` | `LLM_URL` | 评估服务；可指向独立小模型 |
| 羁绊 | `RELATIONSHIP_LLM_MODEL` | `LLM_MODEL` | 评估模型 id |
| 羁绊 | `RELATIONSHIP_IDLE_DELAY_S` | `3` | 回答结束后等用户空闲多久再评估 |
| 羁绊 | `RELATIONSHIP_MIN_INTERVAL_S` | `60` | 两次评估的最小间隔（窗口内多轮合并） |
| 情绪 | `EMOTION_ENABLED` | `1` | 关闭则情绪系统不启动 |
| 情绪 | `EMOTION_EMA_RATE` | `0.2` | EMA 平滑率 |
| 情绪 | `EMOTION_DECAY_INTERVAL_S` | `300` | 空闲衰减间隔 |
| 情绪 | `EMOTION_DECAY_RATE` | `0.05` | 每轮衰减幅度 |
| Dashboard | `DASHBOARD_DETAILS` | `0` | 启动时是否展开详情（运行中按 `Tab` 切换） |
| Dashboard | `DASHBOARD_MAX_MESSAGES` | `300` | 可滚动消息上限 |
| 状态 | `SOUL_TTY_STATE_DIR` | `~/.local/state/soul-tty` | 持久化目录（Bond、Emotion） |

---

## 设计哲学

- **全本地优先** — 语音、推理、合成都在你机器上完成；没有任何对话内容离开本机。
- **链路上每一段都"敢降级"** — LLM 失败、TTS 失败、羁绊评估失败，永远不会让对话停下来。
- **表现层只服务一个判断** — 这个角色在你身边，是不是更像一个活物。
- **状态分层，互不污染** — 短期上下文 / 跨会话摘要 / 长期记忆 / Bond / Emotion 各自独立存储，Bond 不能代替记忆，记忆也不能直接改 Bond。
- **实现克制** — 所有"P0 漂亮但能拖两周"的功能（稳定 AEC、真跨会话记忆）都明确放进 `TODO.md`，不悄悄做一半。

---

## 路线图

- **会话记忆** — 设计已起稿，见 [`TODO.md`](./TODO.md)。
  短期上下文、跨会话摘要、稳定用户事实三层分离，按上下文预算选择性注入 LLM。
- **稳定全双工打断** — 当前是实验开关；目标是从采集到识别到取消到恢复的完整状态机 + AEC 接入。
- **更多人格与社区素材** — 鼓励通过 YAML + `assets/avatars/` 贡献新角色。

---

## 许可

待定（开源筹备中）。

---

## 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 嵌入式流式 ASR
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — 通用 LLM 推理
- [Qwen3-TTS](https://huggingface.co/Qwen) + [MLX-Audio](https://github.com/Blaizzy/mlx-audio) — Apple Silicon 流式语音合成
- [Rich](https://github.com/Textualize/rich) — 终端表现层
- [Chafa](https://github.com/hpjansson/chafa) — 终端像素图渲染