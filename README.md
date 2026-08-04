# soul-tty

**Soul TTY，终端之魂。** 本地语音对话伙伴：麦克风实时输入 → sherpa-onnx 流式识别 → llama.cpp 流式对话 → MLX Qwen3-TTS 流式语音。

## 依赖服务

由 `scripts/svc` 管理:

```bash
svc start llama
svc start mlx-tts
```

- sherpa-onnx:直接嵌入 soul-tty 进程，无需单独启动服务
- llama:8180,router 模式,按需自动加载模型
- mlx-tts:50501,Qwen3-TTS 1.7B CustomVoice 内置 Serena 中文女声(MLX/Apple Silicon,默认)
- chafa:终端角色图渲染；支持原生图片协议时显示像素图，否则回退为 Truecolor Unicode 像素画

## 使用

```bash
svc start soul-tty     # 前台启动(推荐,任何目录可用)
```

等价于 `cd apps/soul-tty && uv run soul-tty`(首次自动建环境)。

交互:显示 `◉ 正在聆听` 后直接说话。按 `0` 循环切换头像套装，按 `Tab` 展开或收起精确羁绊值与完整技术详情，`Ctrl+C` 退出。空闲时仅由轻量 WebRTC VAD 检查 30ms PCM 帧，连续检测到人声后携带 300ms pre-roll 唤醒 sherpa-onnx；识别期间显示 partial，检测到约 0.6s 尾部静音后提交 final。默认**半双工**:播报期间麦克风停止采集(避免外放回声),播完自动恢复聆听。

调试模式(不需要麦克风):

```bash
uv run soul-tty --file /path/to/audio.wav   # 测 ASR->LLM->TTS 全链路(需 16kHz 16bit 单声道 WAV)
uv run soul-tty --text "你好"               # 跳过 ASR 直测 LLM+TTS
```

## 人格与名字

默认人格是 `serena`，项目同时内置了更简洁的 `assistant`：

```bash
uv run soul-tty personas                       # 列出可用人格
uv run soul-tty --persona assistant            # 切换人格
uv run soul-tty --persona serena --name 小夜  # 临时改名
SOUL_TTY_PERSONA=assistant svc start soul-tty
AGENT_NAME=小夜 svc start soul-tty
```

人格文件位于 `personas/*.yaml`，可以自定义名字、开场白、告别语、LLM 系统提示词、TTS 语气与终端主题色。也可直接加载外部文件：

```bash
uv run soul-tty --persona /path/to/my-persona.yaml
SOUL_TTY_PERSONA_DIR=/path/to/personas uv run soul-tty --persona my-agent
```

配置优先级：命令行名字 > 环境变量 > persona YAML > 程序默认值。`--name` 不仅修改终端标题，也会把新名字注入 LLM 人格。

### 手动换装

Serena 内置默认装、深夜装和工作装。可以在启动时指定，也可以进入 Dashboard 后按 `0` 按配置顺序循环切换。每次只保留当前套装的渲染结果，不预载另外两套素材：

```bash
uv run soul-tty outfits                 # 查看当前人格的可用套装
uv run soul-tty --outfit late-night     # 深夜装
uv run soul-tty --outfit work           # 工作装
SOUL_TTY_OUTFIT=work svc start soul-tty # 通过 svc 启动工作装
```

未传 `--outfit` 或 `SOUL_TTY_OUTFIT` 时使用人格 YAML 中的 `default_outfit`。自定义人格仍兼容原来的单套 `avatar` 写法，也可以使用 `avatar.outfits` 增加多套资源。

换装后会立即显示当前套装配置的本地短句，随后由后台 LLM 根据套装气质、当前时段、羁绊阶段和本次会话情绪生成动态台词；生成失败、超时或快速连续切换都不会阻塞界面，也不会把换装台词写入正式对话历史。

### 羁绊成长

欢迎区默认只显示当前人格的羁绊阶段，例如 `♡ 羁绊  亲近`，避免把陪伴关系表现成游戏进度条；按 `Tab` 后才显示精确分值。每轮完整回答结束后，主流程只向一个有界内存队列投递问答；后台单线程再调用 LLM 判断关心、信任、共同玩笑、冒犯等关系事件，并生成受限分值变化、情绪和一句第一人称的当下感受。机制化或第三人称的画外音会被丢弃。

欢迎语会结合当前时段、羁绊阶段和启动间隔：短时间再次启动时会自然地表达“欢迎回来”，约 5% 的启动使用一次克制的特殊开场。启动表现只在 `~/.local/state/soul-tty/presence/` 保存时间与次数，不包含任何对话内容；`--text` 和 `--file` 调试运行不会计入启动节奏。

这条链路是完全可降级的旁路：投递不等待 LLM，队列满时淘汰旧事件，请求失败或低置信度结果不会改变状态。画外音也不会打断思考或播报，只会在安全的聆听状态刷新。羁绊按人格保存到 `~/.local/state/soul-tty/relationships/`，与正式对话历史相互独立；本次会话的情绪和画外音不会写入磁盘，退出后恢复为平静状态。

长时间没有语音输入时，欢迎区会在原有位置切换为“安静陪伴”，并依据本次会话情绪选择一句行为化提示；用户再次开口后立即恢复正常聆听状态。这些变化使用定时事件而非持续动画，不增加空闲刷新负担。

默认在主回答结束并持续空闲 3 秒后复用当前 LLM 服务，并限制为每分钟最多一次；冷却窗口内的多轮会合并评估。若要隔离模型算力竞争，可把 `RELATIONSHIP_LLM_URL` 和 `RELATIONSHIP_LLM_MODEL` 指向单独常驻的小模型服务。不要把同一个 `--models-max 1` 路由器切换到另一模型，否则反复换模会增加延迟和功耗。

### 角色图像

Serena 自带三套 768×768 原创像素风角色图，位于 `assets/avatars/serena/`。默认装保留完整的待机、聆听、思考和说话表情；深夜装与工作装各使用一张稳定主图和一张说话帧，避免口型切换时构图漂移。交互模式使用固定全屏 Dashboard，在同一角色卡中原位切换状态，并在下方保留最近的对话。

```yaml
appearance:
  avatar:
    renderer: auto       # auto / pixels / symbols / off
    width: 26            # 12-48 个终端字符宽度
    default_outfit: default
    outfits:
      default:
        label: 默认装
        idle: ../assets/avatars/serena/idle.png
        speaking_closed: ../assets/avatars/serena/speaking_closed.png
        speaking_half: ../assets/avatars/serena/speaking_half.png
      work:
        label: 工作装
        description: 克制的衬衫与轻机能背心，专注、利落，适合编程
        switch_greetings:
          - 工作模式，今天从哪里开始？
          - 好了，现在专心陪你把事情做好。
        idle: ../assets/avatars/serena/work/idle.png
        speaking_closed: ../assets/avatars/serena/work/idle.png
        speaking_half: ../assets/avatars/serena/work/speaking.png
```

固定 Dashboard 会在 Kitty、Ghostty、iTerm2 或 WezTerm 中使用 Chafa 原生图片协议，把高清头像覆盖到卡片的固定预留区域，并在状态变化时原位替换；其他终端回退为真彩字符像素画。程序不会主动发送能力探测查询。`NO_COLOR=1`、`SOUL_TTY_AVATAR=0` 或 renderer 为 `off` 时会恢复人格图标。

Ghostty/Kitty 启动 Dashboard 时会一次性缓存闭嘴、半开两张干净的普通图片；TTS 说话期间使用实际播放 PCM 的平滑音量切换固定宽度 placement，音量未跨阈值时不会重复刷新，也不会在播放期间反复传输整张图片。其他终端稳定回退为闭嘴完整帧。可用 `AVATAR_LIP_SYNC_ENABLED=0` 关闭说话动画。Dashboard 会按终端剩余高度固定对话视口，默认持续跟随最新消息；滚轮查看历史时保持当前位置，回到底部后恢复自动跟随，并限制可视历史数量以避免长时间运行后持续增长。

## 项目结构

```text
src/soul_tty/
├── cli.py             # 命令行入口与启动信息
├── config.py          # 环境变量配置
├── conversation.py    # 对话流程与打断策略
├── relationship.py    # 非阻塞亲密成长旁路与状态持久化
├── presence.py        # 时间、启动节奏与低频特殊开场
├── audio/             # 录音、sherpa-onnx ASR、TTS 播放
├── clients/           # llama.cpp 客户端
├── personas/          # 人格加载、校验与运行时应用
└── ui/                # Rich 终端开场、头像渲染与对话状态

personas/                 # 用户可直接编辑的 YAML 人格
assets/avatars/           # 人格角色图资产
TODO.md                   # 会话记忆、稳定语音打断等后续核心能力
```

## TTS 后端

| `TTS_BACKEND` | 特点 |
|---|---|
| `mlx`(默认) | Qwen3-TTS 1.7B 内置 Serena 中文女声，支持语气指令，Apple GPU 流式生成 |
| `macos` | 系统 `say` 命令,首音延迟极低,但音色机械僵硬 |

```bash
TTS_BACKEND=macos svc start soul-tty       # 回退到系统音色
MLX_TTS_INSTRUCT='用温柔、亲切的语气说' svc start soul-tty
```

默认内置音色为 `Serena`。`MLX_TTS_INSTRUCT` 留空时使用自然语气，也可设置为“用特别愤怒的语气说”或“用撒娇、亲昵的语气说”。切换 `Vivian` 等其他内置音色时，需同时修改 `scripts/registry.yaml` 的服务启动参数并重启 `mlx-tts`。

## ASR 后端

默认使用进程内 sherpa-onnx Streaming Paraformer int8。模型启动时加载一次，后续音频持续进入同一个在线识别流，不经过 WAV 封装和 HTTP 上传。

```bash
svc start soul-tty                         # sherpa 流式 ASR（默认）
```

本机 `/Users/bug1024/Documents/sounds/test.wav` 实测：4.9 秒音频纯推理约 0.22 秒，RTF 约 0.044；真实逐帧回放在 0.6 秒 endpoint 下完整识别为“你好我在做一个语音测试”。

### 本机延迟实测

同一段测试文本连续5轮，M5 Pro，首个非空 PCM 字节为“首音”：

| 后端 | 中位首音 | 中位完整响应 | 音频时长 | RTF |
|---|---:|---:|---:|---:|
| MLX Qwen3-TTS 1.7B CustomVoice 8-bit / Serena（当前） | 0.113s | 1.086s | 3.840s | 0.283 |

MLX 服务启动时完成模型与参考音色预热；健康检查通过后的请求不承担冷启动。

## 插话打断(实验)

默认关闭。开启后回答期间仍持续聆听,确认是真人插话(非扬声器回声)即打断当前回答并响应新一句:

```bash
BARGE_IN_ENABLED=1 svc start soul-tty
```

外放环境没有 AEC 时可能误判,**建议只在戴耳机时开启**;回声相似度阈值可用 `BARGE_IN_ECHO_SIMILARITY`(默认 0.72)微调。

## 配置(环境变量)

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SOUL_TTY_PERSONA` | `serena` | 启动时使用的人格 id 或 YAML 路径 |
| `SOUL_TTY_PERSONA_DIR` | `personas/` | 额外的人格目录，同 id 会覆盖内置人格 |
| `SOUL_TTY_OUTFIT` | persona 默认套装 | 启动时选择当前人格的头像套装；Serena 支持 `default` / `late-night` / `work` |
| `AGENT_NAME` | 人格名 | 覆盖显示名称并注入 LLM 人格 |
| `SOUL_TTY_ANIMATIONS` | `1` | 设为 `0` 关闭一次性开场动画 |
| `SOUL_TTY_DASHBOARD` | `1` | 设为 `0` 使用传统滚动输出；固定头像与动态表情仅在 Dashboard 中启用 |
| `SOUL_TTY_AVATAR` | `1` | 设为 `0` 关闭角色图并使用人格图标 |
| `SOUL_TTY_AVATAR_RENDERER` | persona 配置 | 临时覆盖 `auto` / `pixels` / `symbols` / `off` |
| `AVATAR_LIP_SYNC_ENABLED` | `1` | 使用实际播放的 TTS PCM 音量驱动缓存口型 |
| `DASHBOARD_MAX_MESSAGES` | `300` | Dashboard 最多保留的可滚动消息数量 |
| `DASHBOARD_DETAILS` | `0` | 默认收起精确羁绊值和完整技术栈；运行中可按 `Tab` 切换 |
| `SHERPA_MODEL_DIR` | `../sherpa-asr/models/...` | Streaming Paraformer int8 模型目录 |
| `SHERPA_NUM_THREADS` | `1` | ONNX Runtime CPU 推理线程数 |
| `SHERPA_ENDPOINT_SILENCE_S` | `0.60` | 已产生文本后的尾部静音提交阈值；过低会把自然停顿误切成多轮 |
| `SHERPA_PARTIAL_ENABLED` | `1` | 在交互终端原地显示增量识别文本 |
| `SHERPA_VAD_GATE_ENABLED` | `1` | 空闲静音只运行 WebRTC VAD，不持续调用 Paraformer |
| `SHERPA_VAD_PRE_ROLL_MS` | `300` | 检测到人声时补回的句首音频长度 |
| `SHERPA_VAD_TRIGGER_MS` | `120` | 连续人声达到该时长后唤醒 Sherpa；pre-roll 会补回句首 |
| `LLM_URL` | `http://127.0.0.1:8180` | llama-server 地址 |
| `LLM_MODEL` | `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` | 模型 id,设为空则自动取 `/v1/models` 第一个 |
| `LLM_MAX_TOKENS` | `256` | 单轮回答硬上限，防止模型无限生成 |
| `LLM_TEMPERATURE` | `0.7` | LLM 采样温度 |
| `LLM_TOP_P` | `0.9` | LLM nucleus sampling 上限 |
| `LLM_REPEAT_PENALTY` | `1.1` | llama.cpp 重复惩罚 |
| `LLM_REPEAT_LAST_N` | `128` | 重复惩罚回看 token 数 |
| `LLM_GREETING_ENABLED` | `1` | 后台生成欢迎语和换装动态台词；失败时使用 persona 本地短句，不阻塞界面 |
| `LLM_GREETING_TIMEOUT` | `5` | 动态欢迎语请求超时（秒） |
| `PRESENCE_REPEAT_LAUNCH_WINDOW_S` | `600` | 在此秒数内再次启动时使用“欢迎回来”语气 |
| `PRESENCE_SPECIAL_GREETING_RATE` | `0.05` | 每次正常启动采用低频特殊开场的概率 |
| `IDLE_EMOTION_ENABLED` | `1` | Dashboard 长时间未识别到语音时，主动切换一条情绪短句 |
| `IDLE_EMOTION_AFTER_S` | `60` | 进入聆听后首次触发安静陪伴的等待时间（秒） |
| `IDLE_EMOTION_INTERVAL_S` | `120` | 持续安静时后续情绪短句的最小间隔（秒） |
| `LLM_IDLE_EMOTION_ENABLED` | `1` | 后台使用 LLM 润色安静陪伴短句；失败时保留本地短句 |
| `LLM_IDLE_EMOTION_MIN_INTERVAL_S` | `600` | 两次空闲情绪 LLM 请求的最小间隔；期间轮换本地短句 |
| `RELATIONSHIP_ENABLED` | `1` | 启用非阻塞亲密成长旁路；设为 `0` 完全关闭 |
| `RELATIONSHIP_LLM_URL` | `LLM_URL` | 关系评估服务；可指向独立小模型以隔离算力竞争 |
| `RELATIONSHIP_LLM_MODEL` | `LLM_MODEL` | 关系评估使用的模型 id |
| `RELATIONSHIP_LLM_TIMEOUT` | `5` | 单次旁路评估超时（秒）；失败不影响对话 |
| `RELATIONSHIP_QUEUE_SIZE` | `4` | 待评估完整问答的有界队列长度 |
| `RELATIONSHIP_IDLE_DELAY_S` | `3` | 回答结束后等待用户空闲多久再评估；说话会重新计时 |
| `RELATIONSHIP_MIN_INTERVAL_S` | `60` | 两次关系 LLM 评估的最小间隔；窗口内多轮自动合并 |
| `RELATIONSHIP_LLM_MAX_TOKENS` | `96` | 关系 JSON 评估的输出 token 上限 |
| `RELATIONSHIP_INITIAL_SCORE` | `10` | 新人格的初始羁绊值 |
| `RELATIONSHIP_MAX_DELTA` | `2` | LLM 单轮允许增减的绝对上限 |
| `RELATIONSHIP_MIN_CONFIDENCE` | `0.65` | 低于该置信度时忽略本轮评估 |
| `SOUL_TTY_STATE_DIR` | `~/.local/state/soul-tty` | 羁绊等本地运行状态目录 |
| `VAD_AGGRESSIVENESS` | `2` | VAD 严格度 0-3,误触发多就调大 |
| `SILENCE_MS` | `700` | 插话 VAD 的连续静音阈值 |
| `MAX_UTTERANCE_S` | `15` | 单句最长秒数,超出强制切段 |
| `SYSTEM_PROMPT` | 当前 persona | 覆盖人格的系统提示词 |
| `MAX_HISTORY` | `10` | 保留最近 N 轮对话 |
| `TTS_ENABLED` | `1` | 设为 `0` 关闭语音播报 |
| `TTS_BACKEND` | `mlx` | `mlx` / `macos` |
| `MLX_TTS_URL` | `http://127.0.0.1:50501` | MLX-Audio 常驻服务 |
| `MLX_TTS_MODEL` | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | MLX TTS 模型 |
| `MLX_TTS_VOICE` | `Serena` | Qwen3-TTS 内置音色，不可留空 |
| `MLX_TTS_INSTRUCT` | 空 | 可选语气/风格指令 |
| `MLX_TTS_STREAMING_INTERVAL` | `0.32` | 流式音频块间隔(秒) |
| `MLX_TTS_MAX_TOKENS` | `256` | 单句音频 token 硬上限，防止随机采样无限生成静音 |
| `MLX_TTS_REPETITION_PENALTY` | `1.05` | CustomVoice 官方重复惩罚参数；不建议改为 `1.0` |
| `MLX_TTS_TRAILING_SILENCE_S` | `1.5` | 连续长尾静音达到该值时取消异常生成；静音不会送入播放器 |
| `MLX_TTS_MAX_AUDIO_S` | `12` | 单句非静音音频绝对上限；实际还会按文本长度自动缩短 |
| `MACOS_VOICE` | `Tingting` | macOS 系统音色名 |
| `MACOS_SPEECH_RATE` | `205` | 系统语音语速(词/分) |
| `TTS_WHOLE_ANSWER` | `1` | `1`=等完整回答后播报；`0`=LLM 生成期间按句流水线播报 |
| `BARGE_IN_ENABLED` | `0` | 插话打断开关(实验,建议耳机) |
| `BARGE_IN_ECHO_SIMILARITY` | `0.72` | 回声判定相似度阈值 |

## macOS 麦克风权限

首次运行麦克风模式时,系统会弹出权限请求;若无反应,到 系统设置 → 隐私与安全性 → 麦克风 中给所用终端(Terminal/iTerm 等)授权。

## 说明

- sherpa partial 仅用于界面反馈，只有 endpoint final 会进入 LLM，避免模型回复尚未说完的半句话
- LLM 单轮最多生成 256 token；流式输出期间若检测到相同句子或短语循环，会立即取消请求并显示自动截断提示
- 0.45s endpoint 在真实测试音频上会误切；默认 0.60s 是当前测试中不误切的最低值
- MLX 客户端逐句请求 TTS，并在同一轮播报内复用 HTTP 连接；服务端每约 0.32s 返回一块 PCM，客户端边收边播，单句异常不会阻塞整轮对话
- CustomVoice 使用模型官方采样参数，并对音频 token 数和纯静音长尾设置双重上限，确保播报后恢复监听
- 发送 TTS 前会移除 Markdown 加粗、斜体、链接和列表标记，并跳过纯符号片段，避免 `**` 等格式字符被合成为循环怪声
- 长句末尾若只有一个很短的逗号分句，会单独合成该句尾，避免 Qwen3-TTS 偶发把最后几个字说得过快或近似吞掉
- 引号会在朗读前移除，`嗯——` 等拉长拟声词会规范化为普通 `嗯`；短拟声词的异常音频最多保留约 2 秒
- 嘈杂环境 VAD 可能误触发:调大 `VAD_AGGRESSIVENESS`
- 打断后被截断的回答只把"已说出的部分"记入对话历史,保持上下文与人听到的一致
