"""Memory 检索：门控、相关性、排序。"""

import unittest
from datetime import datetime, timedelta

from soul_tty.memory.models import (
    SCOPE_PERSONA,
    SOURCE_REFLECTION,
    TYPE_EXPERIENCE,
    Memory,
)
from soul_tty.memory.retriever import (
    bigrams,
    is_recall_query,
    recency,
    relevance,
    search,
)

NOW = datetime(2026, 8, 6, 12, 0, 0).astimezone()


def make(content, *, importance=0.8, days_ago=0, memory_id=1):
    stamp = (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")
    return Memory(
        id=memory_id,
        scope=SCOPE_PERSONA,
        persona_id="serena",
        type=TYPE_EXPERIENCE,
        content=content,
        importance=importance,
        source=SOURCE_REFLECTION,
        created_at=stamp,
        updated_at=stamp,
    )


class RecallGateTests(unittest.TestCase):
    def test_explicit_recall_phrases_trigger(self):
        for text in (
            "你还记得我的项目吗",
            "我之前跟你说过的那件事",
            "以前我们聊过这个",
            "上次那个方案后来怎么样了",
            "你对这个有没有印象",
            "我曾经提过一次",
        ):
            self.assertTrue(is_recall_query(text), text)

    def test_plain_conversation_does_not_trigger(self):
        for text in (
            "今天天气怎么样",
            "帮我写个函数",
            "我有点累",
            "现在几点了",
        ):
            self.assertFalse(is_recall_query(text), text)

    def test_asr_filler_words_do_not_trigger(self):
        """「那个」「你知道」是口语填充词，ASR 高频产出，不能当召回信号。"""
        self.assertFalse(is_recall_query("那个……嗯……我想问一下"))
        self.assertFalse(is_recall_query("你知道吗，今天挺顺利的"))

    def test_empty_text_does_not_trigger(self):
        self.assertFalse(is_recall_query(""))
        self.assertFalse(is_recall_query("   "))


class BigramTests(unittest.TestCase):
    def test_chinese_bigrams(self):
        self.assertEqual(bigrams("项目"), {"项目"})
        self.assertEqual(bigrams("做项目"), {"做项", "项目"})

    def test_punctuation_and_case_are_normalized(self):
        self.assertEqual(bigrams("Soul-TTY!"), bigrams("soultty"))

    def test_single_character_yields_itself(self):
        self.assertEqual(bigrams("我"), {"我"})

    def test_empty_yields_empty(self):
        self.assertEqual(bigrams(""), set())
        self.assertEqual(bigrams("！？。"), set())


class RelevanceTests(unittest.TestCase):
    def test_identical_text_scores_one(self):
        self.assertEqual(relevance("Soul-TTY 项目", "Soul-TTY 项目"), 1.0)

    def test_unrelated_text_scores_zero(self):
        self.assertEqual(relevance("今天天气不错", "用户是医药研发工程师"), 0.0)

    def test_hint_words_are_stripped_before_scoring(self):
        """门控词和口语填充词不能稀释分母。"""
        query = "你还记得我的 AI 项目吗"
        content = "用户在做 Soul-TTY 项目"
        # 未剥离时「项目」只占 1/9，会被 0.2 门槛砍掉
        self.assertGreater(relevance(query, content), 0.2)

    def test_partial_overlap_between_zero_and_one(self):
        score = relevance("Soul-TTY 项目", "用户完成了 Soul-TTY 的情绪系统")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_query_scores_zero(self):
        self.assertEqual(relevance("", "任意内容"), 0.0)
        self.assertEqual(relevance("记得吗", "任意内容"), 0.0)


class RecencyTests(unittest.TestCase):
    def test_today_is_one(self):
        self.assertAlmostEqual(
            recency(NOW.isoformat(timespec="seconds"), now=NOW), 1.0, places=3
        )

    def test_decays_over_time(self):
        stamp = (NOW - timedelta(days=180)).isoformat(timespec="seconds")
        self.assertAlmostEqual(recency(stamp, now=NOW), 0.5, places=3)

    def test_older_is_smaller(self):
        recent = (NOW - timedelta(days=30)).isoformat(timespec="seconds")
        old = (NOW - timedelta(days=300)).isoformat(timespec="seconds")
        self.assertGreater(recency(recent, now=NOW), recency(old, now=NOW))

    def test_unparseable_timestamp_scores_zero(self):
        self.assertEqual(recency("not a date", now=NOW), 0.0)


class SearchTests(unittest.TestCase):
    def test_returns_relevant_memory(self):
        memories = [make("用户在做 Soul-TTY 项目", memory_id=1)]
        found = search("你还记得我的 Soul-TTY 项目吗", memories, now=NOW)
        self.assertEqual([m.id for m in found], [1])

    def test_irrelevant_query_returns_nothing(self):
        """Memory 最大的风险不是忘记，而是乱想起来。"""
        memories = [
            make("用户完成了 Emotion 系统设计", memory_id=1),
            make("用户和 Serena 一起调试了 TTS", memory_id=2),
        ]
        self.assertEqual(search("你还记得上次吃的火锅吗", memories, now=NOW), [])

    def test_high_importance_recent_memory_still_needs_relevance(self):
        """importance 有 0.7 下限，若不是硬门槛，零重叠的新记忆会穿过任何加权阈值。"""
        memories = [make("用户完成了量子计算课程", importance=0.95, days_ago=0)]
        self.assertEqual(search("你还记得我养的那只猫吗", memories, now=NOW), [])

    def test_more_relevant_ranks_higher_than_more_important(self):
        memories = [
            make("用户提到过 Soul-TTY 项目", importance=0.7, memory_id=1),
            make("用户完成了别的事情", importance=0.99, memory_id=2),
        ]
        found = search("Soul-TTY 项目", memories, now=NOW)
        self.assertEqual(found[0].id, 1)

    def test_importance_breaks_tie_between_equally_relevant(self):
        memories = [
            make("Soul-TTY 项目", importance=0.7, memory_id=1),
            make("Soul-TTY 项目", importance=0.95, memory_id=2),
        ]
        found = search("Soul-TTY 项目", memories, now=NOW)
        self.assertEqual(found[0].id, 2)

    def test_recency_breaks_tie_between_equal_relevance_and_importance(self):
        memories = [
            make("Soul-TTY 项目", importance=0.8, days_ago=300, memory_id=1),
            make("Soul-TTY 项目", importance=0.8, days_ago=1, memory_id=2),
        ]
        found = search("Soul-TTY 项目", memories, now=NOW)
        self.assertEqual(found[0].id, 2)

    def test_respects_limit(self):
        memories = [
            make("Soul-TTY 项目一", memory_id=1),
            make("Soul-TTY 项目二", memory_id=2),
            make("Soul-TTY 项目三", memory_id=3),
        ]
        self.assertEqual(len(search("Soul-TTY 项目", memories, limit=2, now=NOW)), 2)

    def test_empty_corpus_returns_empty(self):
        self.assertEqual(search("Soul-TTY 项目", [], now=NOW), [])

    def test_min_relevance_is_configurable(self):
        memories = [make("用户完成了 Soul-TTY 的情绪系统", memory_id=1)]
        # query 只被部分覆盖：Soul-TTY 命中，「语音识别」没有
        query = "Soul-TTY 的语音识别"
        loose = search(query, memories, min_relevance=0.0, now=NOW)
        strict = search(query, memories, min_relevance=0.99, now=NOW)
        self.assertEqual(len(loose), 1)
        self.assertEqual(strict, [])

    def test_query_fully_covered_by_content_scores_one(self):
        memories = [make("用户完成了 Soul-TTY 的情绪系统", memory_id=1)]
        self.assertEqual(relevance("Soul-TTY", memories[0].content), 1.0)


if __name__ == "__main__":
    unittest.main()
