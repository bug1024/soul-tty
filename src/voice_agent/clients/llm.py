"""LLM 客户端:多轮对话历史 + OpenAI 兼容流式输出。"""

import json
import re
import threading
from collections.abc import Iterator

import httpx

from .. import config

_SENTENCE_END = re.compile(r"[。！？!?\n]")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def _normalized_sentence(text: str) -> str:
    return _NON_WORD.sub("", text.lower())


def _tail_is_repeating(text: str) -> bool:
    """捕获没有标点的短语循环，连续三次即判定为模型退化。"""
    compact = _normalized_sentence(text)[-240:]
    for size in range(8, min(80, len(compact) // 3) + 1):
        unit = compact[-size:]
        if compact.endswith(unit * 3):
            return True
    return False


def pick_model() -> str:
    """从 llama router 取第一个可用模型 id。"""
    if config.LLM_MODEL:
        return config.LLM_MODEL
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{config.LLM_URL}/v1/models")
        resp.raise_for_status()
    models = resp.json().get("data", [])
    if not models:
        raise RuntimeError(f"{config.LLM_URL} 没有可用模型,请检查 llama 服务")
    return models[0]["id"]


class Chat:
    def __init__(self, model: str):
        self.model = model
        self.messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        self.last_stop_reason: str | None = None

    def ask_stream(
        self, text: str, cancel: threading.Event | None = None
    ) -> Iterator[str]:
        """发送一轮用户输入,流式产出回答 token,并把本轮记入历史。"""
        self.last_stop_reason = None
        self.messages.append({"role": "user", "content": text})
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "max_tokens": config.LLM_MAX_TOKENS,
            "repeat_penalty": config.LLM_REPEAT_PENALTY,
            "repeat_last_n": config.LLM_REPEAT_LAST_N,
            # 关闭思考链:语音对话要即问即答,thinking 会拖慢首 token
            "chat_template_kwargs": {"enable_thinking": False},
        }
        parts: list[str] = []
        pending = ""
        recent_sentences: list[str] = []
        stop_generation = False
        with httpx.Client(timeout=config.REQUEST_TIMEOUT) as client:
            with client.stream(
                "POST", f"{config.LLM_URL}/v1/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if cancel is not None and cancel.is_set():
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"]
                    token = delta.get("content") or ""
                    if not token:
                        continue
                    pending += token
                    if _tail_is_repeating("".join(parts) + pending):
                        self.last_stop_reason = "repetition"
                        pending = ""
                        break
                    while match := _SENTENCE_END.search(pending):
                        sentence = pending[: match.end()]
                        pending = pending[match.end() :]
                        normalized = _normalized_sentence(sentence)
                        if (
                            len(normalized) >= 4
                            and normalized in recent_sentences[-8:]
                        ):
                            self.last_stop_reason = "repetition"
                            pending = ""
                            stop_generation = True
                            break
                        parts.append(sentence)
                        if normalized:
                            recent_sentences.append(normalized)
                        yield sentence
                    if stop_generation:
                        break
        if pending.strip() and not (cancel is not None and cancel.is_set()):
            parts.append(pending)
            yield pending
        answer = "".join(parts).strip()
        if answer:
            # 被插话时保留已经说出的部分，下一轮上下文才与人真正听到的一致。
            self.messages.append({"role": "assistant", "content": answer})
        else:
            self.messages.pop()  # 在首 token 前被取消，本轮不进历史
        # 裁剪历史:system + 最近 MAX_HISTORY 轮(每轮 2 条)
        if len(self.messages) > 1 + config.MAX_HISTORY * 2:
            self.messages = [self.messages[0]] + self.messages[-config.MAX_HISTORY * 2:]
