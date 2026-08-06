"""Memory 检索：召回门控 + 相关性打分 + 排序。

V1 不做 embedding，也不新增任何依赖：相关性用字符 bigram 重叠。
中文没有分词器可用，bigram 是零依赖下唯一能用的粒度——「AI 项目」
靠「项目」这个 bigram 对上「Soul-TTY 项目」。

对外只有 `search(query, memories, ...)`。打分实现全部是模块私有，
V2 换成向量余弦时调用方一行不用改。
"""

from __future__ import annotations

import math
import re
from datetime import datetime

from .. import config
from .models import Memory

# 明确指向过去的召回信号。刻意不含「那个」和「你知道」：
# 前者是中文最高频的口语填充词且 ASR 高频产出，后者是口头禅，
# 两者都没有时间指向。假阳性的代价是错误唤起记忆，比不检索更糟。
MEMORY_RECALL_HINTS: tuple[str, ...] = (
    "还记得",
    "记得",
    "之前",
    "以前",
    "上次",
    "曾经",
    "聊过",
    "说过",
    "提过",
    "有没有印象",
)

# 打分前从 query 里剥掉的口语填充成分。
# 不剥的话，「你还记得我的 AI 项目吗」里真正有信息量的只有「AI 项目」，
# 其余全是分母——相关性会被稀释到门槛之下，短而具体的问句反而检索不到。
# 这是 V1 的粗糙启发式；接入 embedding 后整块删除。
_QUERY_NOISE: tuple[str, ...] = (
    *MEMORY_RECALL_HINTS,
    "我们",
    "什么",
    "那件",
    "那个",
    "这个",
    "一下",
    "后来",
    "怎么样",
    "怎么",
    "你",
    "我",
    "的",
    "了",
    "吗",
    "呢",
    "吧",
    "啊",
    "是",
    "在",
    "有",
    "过",
    "那",
    "这",
    "个",
    "件",
    "事",
)

_NON_WORD = re.compile(r"[^0-9a-z一-鿿]+")


def _normalize(text: str) -> str:
    return _NON_WORD.sub("", (text or "").lower())


def bigrams(text: str) -> set[str]:
    """字符二元组。单字返回其本身，空文本返回空集合。"""
    normalized = _normalize(text)
    if not normalized:
        return set()
    if len(normalized) == 1:
        return {normalized}
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def _denoise(query: str) -> str:
    text = query or ""
    for noise in _QUERY_NOISE:
        text = text.replace(noise, " ")
    return text


def is_recall_query(text: str) -> bool:
    """用户是否在主动提起过去。不命中就整条检索链路都不执行。"""
    return any(hint in (text or "") for hint in MEMORY_RECALL_HINTS)


def relevance(query: str, content: str) -> float:
    """query 中有多少信息量落在这条记忆上，0~1。"""
    query_grams = bigrams(_denoise(query))
    if not query_grams:
        return 0.0
    content_grams = bigrams(content)
    if not content_grams:
        return 0.0
    return len(query_grams & content_grams) / len(query_grams)


def recency(
    created_at: str,
    *,
    now: datetime | None = None,
    halflife_days: float | None = None,
) -> float:
    """指数时间衰减，0~1。时间戳解析不了时返回 0（不参与加分）。"""
    try:
        created = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return 0.0
    reference = now or datetime.now().astimezone()
    if created.tzinfo is None:
        created = created.astimezone()
    if reference.tzinfo is None:
        reference = reference.astimezone()
    days = max(0.0, (reference - created).total_seconds() / 86400.0)
    halflife = halflife_days or config.MEMORY_RECENCY_HALFLIFE_DAYS
    if halflife <= 0:
        return 1.0
    return math.exp(-days / halflife)


def search(
    query: str,
    memories: list[Memory],
    *,
    limit: int | None = None,
    min_relevance: float | None = None,
    halflife_days: float | None = None,
    now: datetime | None = None,
) -> list[Memory]:
    """按相关性门槛筛选，再按综合分排序取前 N 条。

    相关性是**门槛**而不是加权项：importance 有 MEMORY_MIN_IMPORTANCE
    的下限（0.7），若把三项加权和拿来卡阈值，一条零重叠的新记忆能拿到
    0.4×0 + 0.4×0.7 + 0.2×1.0 = 0.48，穿过任何合理阈值。importance 和
    recency 只有在「已经相关」的前提下才有资格参与排序。

    没有任何记忆越过门槛时返回空列表——宁可不说，也不要错误唤起记忆。
    """
    threshold = (
        config.MEMORY_RECALL_MIN_RELEVANCE
        if min_relevance is None
        else min_relevance
    )
    top_k = config.MEMORY_RECALL_TOP_K if limit is None else limit

    scored: list[tuple[float, int, Memory]] = []
    for memory in memories:
        score = relevance(query, memory.content)
        if score < threshold:
            continue
        total = (
            0.4 * score
            + 0.4 * float(memory.importance)
            + 0.2 * recency(
                memory.created_at, now=now, halflife_days=halflife_days
            )
        )
        scored.append((total, memory.id, memory))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [memory for _, _, memory in scored[:top_k]]
