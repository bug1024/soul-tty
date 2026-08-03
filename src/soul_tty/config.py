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
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8180")
LLM_MODEL = os.environ.get(
    "LLM_MODEL", "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
)  # 空 = 自动取 /v1/models 第一个

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
    "不要使用 Markdown，不要重复同一句话，不要用破折号拉长拟声词。",
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
# 使用实际播放 PCM 的平滑音量驱动三段口型。
AVATAR_LIP_SYNC_ENABLED = os.environ.get("AVATAR_LIP_SYNC_ENABLED", "1") not in (
    "0", "false", "False"
)
# 默认等 LLM 完整回答后开始播报；MLX 客户端仍会逐句请求，以隔离单句的
# 随机采样退化。设为 0 则在 LLM 生成期间就按句进入合成队列，首音更快。
TTS_WHOLE_ANSWER = os.environ.get("TTS_WHOLE_ANSWER", "1") not in ("0", "false", "False")
MACOS_VOICE = os.environ.get("MACOS_VOICE", "Tingting")
MACOS_SPEECH_RATE = int(os.environ.get("MACOS_SPEECH_RATE", "205"))

REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))
