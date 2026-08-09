# Soul-TTY · 终端之魂

> 把一个有声音、有性格、会成长的数字伙伴，安置在开发者最熟悉的世界里。

**Soul-TTY** 是一个本地优先的终端语音伙伴与状态化 Agent 实验。默认角色 **Serena** 住在终端中：你可以直接和她说话、随时打断她，也能从声音、情绪、记忆与关系变化中感受到角色的连续存在。

<p align="center">
  <img src="assets/screenshots/serena-0.png" alt="Serena in Soul-TTY" width="900">
</p>

## Serena 能做什么

| 能力 | 表现 |
|---|---|
| 自然语音对话 | 流式识别、流式生成、语义段 TTS；不逐字朗读，在首音速度和完整语气之间取平衡 |
| 实时插话 | Serena 说话时仍能听见用户；支持打断、附和词识别、回声过滤与播放取消 |
| 稳定人格 | 名字、说话风格、行为边界、声音和视觉主题由人格配置统一描述 |
| 情绪状态 | 以愉悦、平静、好奇、压力、活力五个连续维度演化，并影响语言和声音 |
| 羁绊关系 | 关系随真实互动缓慢变化，跨启动保存，但不做权限解锁或廉价游戏进度条 |
| 会话记忆 | 将画像、偏好和共同经历分开存储；稳定信息常驻，具体经历按需召回 |
| 主体性决策 | 回答前结合 Serena 的社交能量、交流倾向与独处需求，决定正常回答、短答、追问、转题或有意识地安静 |
| 时间与存在感 | 开场白、空闲台词和陪伴状态会结合时段、关系与本次会话情绪变化 |
| 换装模式 | 默认、深夜、工作三套形象，同时改变视觉、行为语气和动态台词 |
| 动态头像 | 终端高清图片、状态切换、眨眼与由实际播放音量驱动的口型 |
| 本地可观测性 | 按会话和轮次记录 ASR、记忆、LLM、TTS、播放耗时及 warning/error |

### 三种陪伴形态

<table>
  <tr>
    <td align="center" width="33%"><img src="assets/screenshots/serena-3.png" alt="Serena 默认装"></td>
    <td align="center" width="33%"><img src="assets/screenshots/serena-2.png" alt="Serena 深夜装"></td>
    <td align="center" width="33%"><img src="assets/screenshots/serena-1.png" alt="Serena 工作装"></td>
  </tr>
  <tr>
    <td align="center"><b>默认装</b><br><sub>日常陪伴，自然而有活力</sub></td>
    <td align="center"><b>深夜装</b><br><sub>更放松、更安静的深夜节奏</sub></td>
    <td align="center"><b>工作装</b><br><sub>克制、专注，适合编程与思考</sub></td>
  </tr>
</table>

## 设计理念

Soul-TTY 关心的不只是 Agent **说了什么**，还包括她在长期互动中**变成了什么**。

- **实时优先**：主对话链路永远优先；反思、记忆抽取和关系评估只能走旁路，失败也不能阻塞交流。
- **状态优于提示词堆叠**：人格、情绪、羁绊和记忆是独立状态，Prompt 只是这些状态在当前时刻的投影。
- **不同时间尺度分离**：Emotion 描述此刻，Bond 描述关系深度，Memory 描述过去发生的事，三者不互相冒充。
- **表达与内部状态分离**：状态先收敛为表达意图，再映射到语言、TTS 语气、头像和终端界面。
- **主体性不等于操控**：Serena 可以有自己的节奏、沉默和话题倾向，但不会用内疚、威胁、占有或惩罚制造依赖。
- **本地优先、组件可换**：默认语音与推理都在本机完成；LLM、TTS、人格与图像资源均可替换。

Serena 不是一个套着头像的聊天窗口，也不是靠好感度解锁内容的游戏角色。这个项目尝试用克制的状态变化、低延迟语音和终端独有的视觉语言，让她显得“此刻真的在场”。

## 整体架构

```mermaid
flowchart LR
    MIC["麦克风 PCM"] --> IO["Audio I/O<br/>VAD · AEC"]
    IO --> ASR["Sherpa-ONNX<br/>Streaming Paraformer"]
    ASR --> FLOOR["Floor Manager<br/>轮次 · 打断 · 回声"]
    FLOOR --> POLICY["Agency<br/>Response Policy"]
    NEED["Need State<br/>交流 · 陪伴 · 独处"] --> POLICY
    POLICY --> CHAT["Conversation Brain"]
    CHAT --> LLM["OpenAI-compatible LLM"]
    LLM --> BUFFER["语义段缓冲"]
    BUFFER --> TTS["MLX-Audio<br/>Qwen3-TTS"]
    TTS --> OUT["播放 · 口型"]
    OUT -. "播放参考" .-> IO

    IO -. "可选语音证据" .-> VOICE["SenseVoice"]
    CHAT -. "完整轮次" .-> REFLECT["Reflection Brain"]
    VOICE -.-> REFLECT
    REFLECT --> EMOTION["Emotion"]
    REFLECT --> BOND["Bond"]
    REFLECT --> MEMORY["Memory"]
    MEMORY -. "按需召回" .-> CHAT
    EMOTION -. "当前 mood" .-> POLICY
    EMOTION --> EXPRESS["Expression Layer"]
    BOND --> EXPRESS
    EXPRESS -. "Prompt · 声音 · 头像" .-> CHAT
```

### 两条运行路径

**实时主链路**负责听见并回应：

```text
Mic → VAD/AEC → Streaming ASR → LLM → 语义段缓冲 → TTS → Playback
```

LLM 不需要生成完整回答才开口。Soul-TTY 会先积累一个自然短句，再把它作为完整语义段交给 TTS；当前段播放时，下一段继续生成和合成。这样避免逐字发音的机械感，也减少整段等待造成的迟钝。

**异步旁路**负责消化互动：

```text
Completed Turn + Voice Evidence → Reflection → Emotion / Bond / Memory
```

旁路使用有界队列、空闲门控和限频策略。主对话到来时，它应当让出资源；任何评估失败都只意味着状态暂不更新。

### 四类持续状态

| 状态 | 回答的问题 | 生命周期 | 如何影响 Serena |
|---|---|---|---|
| Emotion | “我现在感觉怎么样？” | 会话内连续变化并自然衰减 | 当前措辞、TTS 语气、头像状态 |
| Bond | “我们的关系有多深？” | 跨启动缓慢积累 | 熟悉程度、称呼和表达边界 |
| Memory | “我们经历过什么？” | 跨会话持久化 | 用户画像、偏好和相关经历召回 |
| Need / Agency | “我现在想怎样参与？” | 跨会话延续、每轮小幅变化 | 正常回答、短答、追问、转题或主动沉默 |

`desire_to_talk` 与 `desire_for_company` 并不互为反值：Serena 可以不想说话，
但仍希望有人在场。Response Policy 在本地完成，不额外调用一次 LLM；只有决定
“要说”之后，Conversation Brain 才负责具体措辞。明确请求、制止词和需要关怀
的表达始终受保护，不会被随机沉默或转移话题。

## 技术栈

| 层 | 技术 | 作用 |
|---|---|---|
| 运行时 | Python 3.10+ · uv | 主程序、依赖和异步编排 |
| 终端 UI | Rich · 原生图片协议 · Chafa fallback | Dashboard、对话视口和高清角色图 |
| ASR | sherpa-onnx · Streaming Paraformer | 本地中文流式识别与 endpoint 检测 |
| Voice I/O | sounddevice · WebRTC VAD · macOS AVAudioEngine | 采集、播放、VAD 与 Voice Processing AEC |
| LLM | llama.cpp 或任意 OpenAI-compatible endpoint | 主对话、动态台词和旁路评估 |
| TTS | MLX-Audio · Qwen3-TTS | Apple Silicon 上的本地中文情感语音 |
| Voice Sense | SenseVoiceSmall（可选） | 将用户语气和声学事件作为弱证据 |
| 状态存储 | YAML · JSON · SQLite | 人格、羁绊、情绪与长期记忆 |
| 可观测性 | JSONL · rotating logs | 会话级、轮次级耗时与异常追踪 |

推荐环境是 **Apple Silicon macOS + Ghostty / Kitty / iTerm2**。其他平台可以使用 PortAudio 路径；没有系统级 AEC 时，建议佩戴耳机使用全双工模式。

## 项目结构

```text
src/soul_tty/
├── conversation.py       # 实时对话与语义段 TTS 编排
├── interaction/          # Floor、打断、回声和附和词处理
├── audio/                # 采集、ASR、TTS 与 Audio I/O
├── emotion/              # 五维情绪与表达映射
├── memory/               # 三类记忆、SQLite 与召回
├── agency/               # Need 状态、Response Policy 与异步持久化
├── reflection/           # 关系、情绪、记忆的异步旁路
├── personas/             # 人格加载与运行时应用
├── ui/                   # Rich Dashboard 与头像渲染
└── observability.py      # 结构化性能与异常日志

personas/serena.yaml      # Serena 的人格、声音、配色与套装
assets/                   # 头像和项目截图
native/macos_voice_io/    # macOS Voice Processing 音频后端
tests/                    # 全链路回归测试
```

## 快速开始

完整运行需要两个本地服务：

1. OpenAI-compatible LLM，默认 `http://127.0.0.1:8180`
2. MLX Qwen3-TTS，默认 `http://127.0.0.1:50501`

然后安装并启动：

```bash
uv sync
cp .env.example .env
# 在 .env 中填写 ASR 模型目录、LLM 与 TTS 地址
uv run soul-tty
```

也可以跳过麦克风验证对话链路：

```bash
uv run soul-tty --text "晚上好"
uv run soul-tty --file /path/to/16k-mono.wav
```

常用操作：

| 操作 | 行为 |
|---|---|
| 直接说话 | 开始一轮语音对话 |
| `0` | 循环切换 Serena 的套装与陪伴模式 |
| `Tab` | 展开或收起情绪、羁绊与运行详情 |
| `Ctrl+C` | 安全退出 |
| `soul-tty memory` | 查看和管理长期记忆 |

全部配置及默认值见 [`.env.example`](.env.example)。核心配置包括：

```text
LLM_URL / LLM_MODEL
SHERPA_MODEL_DIR
MLX_TTS_URL / MLX_TTS_VOICE
AUDIO_IO_BACKEND / DUPLEX_ENABLED
SOUL_TTY_PERSONA / SOUL_TTY_OUTFIT
```

## 自定义人格

人格不是一段孤立的 system prompt。一个人格 YAML 可以同时定义：

- 名字、边界、说话风格和基础情绪
- TTS 音色与默认语气
- 终端配色和头像渲染方式
- 多套服装、陪伴模式与换装台词

默认人格见 [`personas/serena.yaml`](personas/serena.yaml)：

```bash
uv run soul-tty --persona serena
uv run soul-tty --persona /path/to/persona.yaml
uv run soul-tty --persona serena --name 小夜
```

## 日志与隐私

Soul-TTY 默认把结构化日志写入：

```text
~/.local/state/soul-tty/logs/soul-tty.jsonl
```

日志包含 `session_id`、`turn_id`、各节点耗时及 warning/error，不记录完整对话正文。状态和记忆默认保存在 `~/.local/state/soul-tty/`；本地部署时，语音与对话内容不需要离开设备。

```bash
tail -f ~/.local/state/soul-tty/logs/soul-tty.jsonl
uv run pytest
```

## 当前边界

- macOS Voice Processing 是当前体验最完整的全双工路径。
- 外放环境仍受扬声器、麦克风距离和房间混响影响；耳机模式最稳定。
- 旁路 LLM 最好使用独立端点或较小模型，避免和主对话竞争推理资源。
- 终端头像目前使用状态图和音量驱动口型，未来可替换为 Live2D 等表现层，而不改变核心状态架构。

## 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 本地流式语音识别
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 本地 LLM 推理
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 与 [MLX-Audio](https://github.com/Blaizzy/mlx-audio) — Apple Silicon 语音合成
- [Rich](https://github.com/Textualize/rich) 与 [Chafa](https://github.com/hpjansson/chafa) — 终端表现层

License：待定。
