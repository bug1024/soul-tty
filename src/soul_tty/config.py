"""配置:默认值均可通过环境变量覆盖。"""

import os
from pathlib import Path

# 依赖服务地址(对应 scripts/registry.yaml)
SHERPA_MODEL_DIR = os.environ.get(
    "SHERPA_MODEL_DIR",
    str(
        Path(__file__).resolve().parents[3]
        / "sherpa-asr/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
    ),
)
SHERPA_NUM_THREADS = int(os.environ.get("SHERPA_NUM_THREADS", "1"))
# 已识别出内容后，尾部静音达到该时长即提交给 LLM。
SHERPA_ENDPOINT_SILENCE_S = float(
    os.environ.get("SHERPA_ENDPOINT_SILENCE_S", "0.60")
)
SHERPA_PARTIAL_ENABLED = os.environ.get("SHERPA_PARTIAL_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
# 空闲时先用极轻量 WebRTC VAD 门控，避免把无限静音持续送进 Paraformer。
# 触发时连同 pre-roll 一起送入，防止吞掉句首；触发后保留静音给 Sherpa endpoint。
SHERPA_VAD_GATE_ENABLED = os.environ.get("SHERPA_VAD_GATE_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
SHERPA_VAD_PRE_ROLL_MS = int(os.environ.get("SHERPA_VAD_PRE_ROLL_MS", "300"))
SHERPA_VAD_TRIGGER_MS = int(os.environ.get("SHERPA_VAD_TRIGGER_MS", "120"))
# ---------------------------------------------------------------------------
# LLM 端点：`LLM_URL` 默认服务于主对话；`AUX_LLM_URL` 独立服务于辅助请求。
# 两个端点都按 OpenAI Chat Completions 协议工作，不附带任何专属 header。
# 用户可以把它们指向同一个服务，也可以把辅助请求单独跑在小模型上，
# 让"主对话的算力"和"辅助请求的算力"互不抢资源。
# ---------------------------------------------------------------------------

# 主对话 LLM：用于 Chat 的流式回答。多轮上下文、热更新 system prompt、
# 流式 token 切句、重复截断等都发生在这条链路上。
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8180").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "")  # 空 = 自动取 /v1/models 第一个

# 辅助 LLM：每次启动只发一两个一次性请求，且都不进入正式对话历史：
#   1. 启动欢迎语（短句）
#   2. 换装动态台词（短句）
#   3. 长时间空闲时的陪伴短句（短句）
# 留空时回退到主 LLM；典型用法是指向一个常驻小模型以彻底隔离算力竞争。
AUX_LLM_URL_RAW = os.environ.get("AUX_LLM_URL", "").rstrip("/")
AUX_LLM_MODEL_RAW = os.environ.get("AUX_LLM_MODEL", "")


def _resolve_aux_url() -> str:
    """辅助 URL 留空时回退到主 LLM；延迟求值方便测试与运行时切换。"""
    return AUX_LLM_URL_RAW or LLM_URL


def _resolve_aux_model() -> str:
    """辅助模型留空时回退到主 LLM；延迟求值方便测试与运行时切换。"""
    return AUX_LLM_MODEL_RAW or LLM_MODEL

# 音频采集
SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad 要求 10/20/30ms

# VAD 切句参数
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))  # 0-3,越大越严格
SILENCE_MS = int(os.environ.get("SILENCE_MS", "700"))     # 连续静音判定一句话结束
MAX_UTTERANCE_S = float(os.environ.get("MAX_UTTERANCE_S", "15"))  # 强制切段
MIN_UTTERANCE_MS = int(os.environ.get("MIN_UTTERANCE_MS", "300"))  # 过短丢弃(防误触发)

# 插话打断：回答期间仍持续切句，ASR 确认不是播放回声后取消当前回答。
# 外放环境没有 AEC 时可能误触发，使用耳机效果最稳定。
BARGE_IN_ENABLED = os.environ.get("BARGE_IN_ENABLED", "0") not in ("0", "false", "False")
BARGE_IN_ECHO_SIMILARITY = float(os.environ.get("BARGE_IN_ECHO_SIMILARITY", "0.72"))

# LLM
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "你是一个语音对话助手。请用简短、口语化的中文回答，每次回答不超过三句话，"
    "直接说要说的话，不要输出角色名、旁白或动作描写。"
    "不要使用 Markdown，不要重复同一句话，不要用破折号或连续句点拉长语气。",
)
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "10"))  # 保留最近 N 轮
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "256"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_TOP_P = float(os.environ.get("LLM_TOP_P", "0.9"))
LLM_REPEAT_PENALTY = float(os.environ.get("LLM_REPEAT_PENALTY", "1.1"))
LLM_REPEAT_LAST_N = int(os.environ.get("LLM_REPEAT_LAST_N", "128"))
LLM_GREETING_ENABLED = os.environ.get("LLM_GREETING_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
LLM_GREETING_TIMEOUT = float(os.environ.get("LLM_GREETING_TIMEOUT", "5"))
# 启动节奏只用于表现层，不包含任何对话记忆。
PRESENCE_REPEAT_LAUNCH_WINDOW_S = float(
    os.environ.get("PRESENCE_REPEAT_LAUNCH_WINDOW_S", "600")
)
PRESENCE_SPECIAL_GREETING_RATE = float(
    os.environ.get("PRESENCE_SPECIAL_GREETING_RATE", "0.05")
)
# Dashboard 长时间没有收到语音识别文本时，轮换一句克制的情绪短句。
IDLE_EMOTION_ENABLED = os.environ.get("IDLE_EMOTION_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
IDLE_EMOTION_AFTER_S = float(os.environ.get("IDLE_EMOTION_AFTER_S", "60"))
IDLE_EMOTION_INTERVAL_S = float(
    os.environ.get("IDLE_EMOTION_INTERVAL_S", "120")
)
LLM_IDLE_EMOTION_ENABLED = os.environ.get(
    "LLM_IDLE_EMOTION_ENABLED", "1"
) not in ("0", "false", "False")
LLM_IDLE_EMOTION_MIN_INTERVAL_S = float(
    os.environ.get("LLM_IDLE_EMOTION_MIN_INTERVAL_S", "600")
)

# 反思旁路总开关。关闭后整个 ReflectionWorker 不启动，Bond/Emotion/Memory 的
# 异步处理全部停用。
# 兼容旧版环境变量名 RELATIONSHIP_ENABLED——新代码优先读 REFLECTION_ENABLED。
_REFLECTION_RAW = os.environ.get("REFLECTION_ENABLED") or os.environ.get(
    "RELATIONSHIP_ENABLED", "1"
)
REFLECTION_ENABLED = _REFLECTION_RAW not in ("0", "false", "False")
RELATIONSHIP_ENABLED = REFLECTION_ENABLED  # 旧名兼容

# Bond 子开关。关闭后 ReflectionWorker 仍运行（Memory/Emotion 不受影响），
# 但关系评估结果不落库、不更新 Dashboard。
BOND_ENABLED = os.environ.get("BOND_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
RELATIONSHIP_LLM_URL = os.environ.get("RELATIONSHIP_LLM_URL", LLM_URL)
RELATIONSHIP_LLM_MODEL = os.environ.get("RELATIONSHIP_LLM_MODEL", LLM_MODEL)
RELATIONSHIP_LLM_TIMEOUT = float(
    os.environ.get("RELATIONSHIP_LLM_TIMEOUT", "5")
)
RELATIONSHIP_QUEUE_SIZE = int(os.environ.get("RELATIONSHIP_QUEUE_SIZE", "4"))
RELATIONSHIP_IDLE_DELAY_S = float(
    os.environ.get("RELATIONSHIP_IDLE_DELAY_S", "3")
)
RELATIONSHIP_MIN_INTERVAL_S = float(
    os.environ.get("RELATIONSHIP_MIN_INTERVAL_S", "60")
)
RELATIONSHIP_INITIAL_BOND = float(
    os.environ.get("RELATIONSHIP_INITIAL_BOND", "0.05")
)
RELATIONSHIP_MAX_DELTA = float(
    os.environ.get("RELATIONSHIP_MAX_DELTA", "0.03")
)
RELATIONSHIP_MIN_CONFIDENCE = float(
    os.environ.get("RELATIONSHIP_MIN_CONFIDENCE", "0.65")
)
RELATIONSHIP_LLM_MAX_TOKENS = int(
    os.environ.get("RELATIONSHIP_LLM_MAX_TOKENS", "96")
)
# 最近关系事件保留条数（FIFO）；超过则丢最旧的，避免无限增长。
RELATIONSHIP_MAX_RECENT_EVENTS = int(
    os.environ.get("RELATIONSHIP_MAX_RECENT_EVENTS", "20")
)

# 实时情绪系统（multi-dim emotion with EMA smoothing + idle decay）
EMOTION_ENABLED = os.environ.get("EMOTION_ENABLED", "1") not in (
    "0", "false", "False",
)
EMOTION_EMA_RATE = float(os.environ.get("EMOTION_EMA_RATE", "0.2"))
EMOTION_DELTA_CAP = float(os.environ.get("EMOTION_DELTA_CAP", "0.3"))
EMOTION_DECAY_INTERVAL_S = float(
    os.environ.get("EMOTION_DECAY_INTERVAL_S", "300")
)
EMOTION_DECAY_RATE = float(os.environ.get("EMOTION_DECAY_RATE", "0.05"))
EMOTION_IDLE_THRESHOLD_S = float(
    os.environ.get("EMOTION_IDLE_THRESHOLD_S", "300")
)
EMOTION_PERSIST = os.environ.get("EMOTION_PERSIST", "0") not in (
    "0", "false", "False",
)
EMOTION_PROMPT_UPDATE_INTENSITY = float(
    os.environ.get("EMOTION_PROMPT_UPDATE_INTENSITY", "0.1")
)

SOUL_TTY_STATE_DIR = Path(
    os.environ.get(
        "SOUL_TTY_STATE_DIR",
        str(Path.home() / ".local" / "state" / "soul-tty"),
    )
).expanduser()

# ---------------------------------------------------------------------------
# 长期记忆。写入走 Reflection 旁路，读取分两条：
#   - 用户画像/偏好（global）常驻 system prompt，只在记忆变化时重建
#   - 共同经历（persona）按需检索，作为临时 message 注入，不进 prompt 也不进历史
# 任何环节失败都必须降级为 no-op，主对话表现与 MEMORY_ENABLED=0 一致。
# ---------------------------------------------------------------------------
MEMORY_ENABLED = os.environ.get("MEMORY_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
MEMORY_DB_PATH = Path(
    os.environ.get("MEMORY_DB_PATH", str(SOUL_TTY_STATE_DIR / "memory.db"))
).expanduser()
# 留空回落辅助 LLM；典型用法是指向常驻小模型，与主对话彻底隔离算力。
MEMORY_LLM_URL = os.environ.get("MEMORY_LLM_URL", "").rstrip("/")
MEMORY_LLM_MODEL = os.environ.get("MEMORY_LLM_MODEL", "")
MEMORY_LLM_TIMEOUT = float(os.environ.get("MEMORY_LLM_TIMEOUT", "8"))
MEMORY_LLM_MAX_TOKENS = int(os.environ.get("MEMORY_LLM_MAX_TOKENS", "256"))
# 比关系评估的 60s 稀疏一档：记忆不需要跟得那么紧。
MEMORY_MIN_INTERVAL_S = float(os.environ.get("MEMORY_MIN_INTERVAL_S", "120"))
# 待抽取轮次的环形缓冲；独立于关系评估队列，后者溢出丢轮但记忆不能丢。
MEMORY_BUFFER_TURNS = int(os.environ.get("MEMORY_BUFFER_TURNS", "20"))
# 纯「嗯」「好的」不值得一次推理。
MEMORY_MIN_TEXT_CHARS = int(os.environ.get("MEMORY_MIN_TEXT_CHARS", "20"))
MEMORY_MIN_IMPORTANCE = float(os.environ.get("MEMORY_MIN_IMPORTANCE", "0.7"))
# 落库前的兜底去重：与同类已有记忆的 bigram 重叠超过此值即跳过。
MEMORY_DEDUPE_THRESHOLD = float(os.environ.get("MEMORY_DEDUPE_THRESHOLD", "0.8"))
# 常驻 system prompt 的 global 记忆条数上限。
MEMORY_MAX_RESIDENT = int(os.environ.get("MEMORY_MAX_RESIDENT", "12"))
MEMORY_RECALL_TOP_K = int(os.environ.get("MEMORY_RECALL_TOP_K", "3"))
# 相关性硬门槛。必须是门槛而不是加权项：importance 有 0.7 下限，
# 若卡加权总分，零重叠的新记忆也能穿过任何合理阈值。
MEMORY_RECALL_MIN_RELEVANCE = float(
    os.environ.get("MEMORY_RECALL_MIN_RELEVANCE", "0.2")
)
# 经历的时间衰减半衰参数（天）。
MEMORY_RECENCY_HALFLIFE_DAYS = float(
    os.environ.get("MEMORY_RECENCY_HALFLIFE_DAYS", "180")
)

# 固定 Dashboard 只保留有限滚动历史；长期记忆由未来的会话记忆层负责。
DASHBOARD_MAX_MESSAGES = int(os.environ.get("DASHBOARD_MAX_MESSAGES", "300"))
# 欢迎区默认以角色信息为主；诊断模式展开精确羁绊值与完整技术栈。
DASHBOARD_DETAILS = os.environ.get("DASHBOARD_DETAILS", "0") not in (
    "0",
    "false",
    "False",
)

# TTS
TTS_ENABLED = os.environ.get("TTS_ENABLED", "1") not in ("0", "false", "False")
TTS_BACKEND = os.environ.get("TTS_BACKEND", "mlx")  # mlx(默认)/macos
MLX_TTS_URL = os.environ.get("MLX_TTS_URL", "http://127.0.0.1:50501")
MLX_TTS_MODEL = os.environ.get(
    "MLX_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
)
MLX_TTS_VOICE = os.environ.get("MLX_TTS_VOICE", "Serena")
# 留空表示使用音色默认的自然语气；也可传“用特别愤怒的语气说”等指令。
MLX_TTS_INSTRUCT = os.environ.get("MLX_TTS_INSTRUCT", "")
MLX_TTS_STREAMING_INTERVAL = float(
    os.environ.get("MLX_TTS_STREAMING_INTERVAL", "0.32")
)
# CustomVoice 官方生成参数。repetition_penalty=1.0 在部分中文句子上会退化为
# 数十秒纯静音，必须显式使用模型 generation_config 的 1.05。
MLX_TTS_MAX_TOKENS = int(os.environ.get("MLX_TTS_MAX_TOKENS", "256"))
MLX_TTS_TEMPERATURE = float(os.environ.get("MLX_TTS_TEMPERATURE", "0.9"))
MLX_TTS_TOP_P = float(os.environ.get("MLX_TTS_TOP_P", "1.0"))
MLX_TTS_TOP_K = int(os.environ.get("MLX_TTS_TOP_K", "50"))
MLX_TTS_REPETITION_PENALTY = float(
    os.environ.get("MLX_TTS_REPETITION_PENALTY", "1.05")
)
# 即使模型未输出 EOS，连续静音达到该时长也主动结束当前句，避免麦克风
# 长时间无法恢复。静音先缓存在客户端，不会实际播放出来。
MLX_TTS_TRAILING_SILENCE_S = float(
    os.environ.get("MLX_TTS_TRAILING_SILENCE_S", "1.5")
)
MLX_TTS_SILENCE_RMS = float(os.environ.get("MLX_TTS_SILENCE_RMS", "0.002"))
# 非静音退化（例如纯 Markdown 符号被合成为循环怪声）的播放硬上限。
# 实际上限还会按当前句字符数缩短。
MLX_TTS_MAX_AUDIO_S = float(os.environ.get("MLX_TTS_MAX_AUDIO_S", "12"))
MLX_TTS_AUDIO_S_PER_CHAR = float(
    os.environ.get("MLX_TTS_AUDIO_S_PER_CHAR", "0.25")
)
MLX_TTS_MIN_AUDIO_S = float(os.environ.get("MLX_TTS_MIN_AUDIO_S", "2"))
MLX_TTS_AUDIO_PADDING_S = float(
    os.environ.get("MLX_TTS_AUDIO_PADDING_S", "1.5")
)
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
# 使用实际播放 PCM 的平滑音量驱动缓存口型；不会运行固定帧率动画。
AVATAR_LIP_SYNC_ENABLED = os.environ.get("AVATAR_LIP_SYNC_ENABLED", "1") not in (
    "0", "false", "False"
)
# 默认等 LLM 完整回答后开始播报；MLX 客户端仍会逐句请求，以隔离单句的
# 随机采样退化。设为 0 则在 LLM 生成期间就按句进入合成队列，首音更快。
TTS_WHOLE_ANSWER = os.environ.get("TTS_WHOLE_ANSWER", "1") not in ("0", "false", "False")
MACOS_VOICE = os.environ.get("MACOS_VOICE", "Tingting")
MACOS_SPEECH_RATE = int(os.environ.get("MACOS_SPEECH_RATE", "205"))

REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))
