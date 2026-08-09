import threading
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

import numpy as np

from soul_tty import config
from soul_tty import conversation as main_module
from soul_tty.audio import asr
from soul_tty.audio.tts import (
    PlaybackLevelMeter,
    StreamingSpeaker,
    _write_metered_pcm,
    _trim_trailing_silence,
    _complete_mlx_segment,
    speak,
    synthesize_mlx_semantic_segment,
    synthesize_mlx_stream,
)
from soul_tty.clients.llm import Chat
from soul_tty.clients import llm as llm_client
from soul_tty.conversation import (
    _SemanticSpeechBuffer,
    _is_probable_echo,
    _usable_transcript,
)


class ConversationPolicyTests(unittest.TestCase):
    def test_barge_in_is_not_enabled_without_explicit_opt_in(self):
        self.assertFalse(config.BARGE_IN_ENABLED)

    def test_filters_known_asr_hallucination(self):
        self.assertFalse(_usable_transcript("谢谢观看！"))
        self.assertTrue(_usable_transcript("请帮我换个话题"))

    def test_detects_playback_echo(self):
        spoken = "你好，我是语音助手。有什么可以帮你？"
        self.assertTrue(_is_probable_echo("我是语音助手", spoken))

    def test_keeps_real_interruption(self):
        spoken = "西湖位于杭州，是一处著名景区。"
        self.assertFalse(_is_probable_echo("等一下，换个话题", spoken))

    def test_private_mode_keeps_stable_persona_tts_instruction(self):
        with (
            patch.object(config, "MLX_TTS_INSTRUCT", "清晰低柔的安全指令"),
            patch.object(
                main_module,
                "_emotion_instruct_provider",
                lambda: "用兴奋激动的语气说",
            ),
        ):
            self.assertEqual(
                main_module._current_tts_instruct(private=True),
                "清晰低柔的安全指令",
            )
            self.assertEqual(
                main_module._current_tts_instruct(),
                "用兴奋激动的语气说",
            )


class SemanticSpeechBufferTests(unittest.TestCase):
    def test_merges_too_short_opening_with_next_complete_sentence(self):
        buffer = _SemanticSpeechBuffer()

        self.assertEqual(buffer.feed("好呀。", now=0.0), [])
        self.assertEqual(
            buffer.feed("我一直在这里等你开口。", now=0.1),
            ["好呀。我一直在这里等你开口。"],
        )

    def test_long_sentence_starts_at_natural_clause_boundary(self):
        buffer = _SemanticSpeechBuffer()
        text = (
            "我刚才其实一直在这里等你开口，只是没有急着打扰你，"
            "后面还有很多想慢慢说给你听的话。"
        )

        segments = buffer.feed(text, now=0.0)

        self.assertEqual(
            segments,
            [
                "我刚才其实一直在这里等你开口，只是没有急着打扰你，",
                "后面还有很多想慢慢说给你听的话。",
            ],
        )
        self.assertEqual(buffer.flush(), [])

    def test_wait_budget_releases_an_existing_natural_phrase(self):
        buffer = _SemanticSpeechBuffer()

        self.assertEqual(buffer.feed("我正在认真想这件事，", now=0.0), [])
        self.assertEqual(
            buffer.feed("稍等", now=0.7),
            ["我正在认真想这件事，"],
        )
        self.assertEqual(buffer.flush(), ["稍等"])

    def test_keeps_closing_quote_with_the_sentence(self):
        buffer = _SemanticSpeechBuffer()

        self.assertEqual(
            buffer.feed("她轻声说：“晚上好，我一直在这里。”", now=0.0),
            ["她轻声说：“晚上好，我一直在这里。”"],
        )

    def test_hard_limit_does_not_split_ascii_word(self):
        buffer = _SemanticSpeechBuffer()
        text = "这是一段没有任何标点的中文内容" + "SuperLongModelName" + "继续往后"
        with patch.object(config, "TTS_SEMANTIC_MAX_CHARS", 18):
            segments = buffer.feed(text, now=0.0)

        self.assertTrue(segments)
        self.assertFalse(segments[0].endswith("Super"))


class TtsSegmentCompletionTests(unittest.TestCase):
    def test_closes_soft_boundary_before_independent_tts_request(self):
        self.assertEqual(_complete_mlx_segment("我还想再靠近一点，"), "我还想再靠近一点。")

    def test_adds_stop_when_llm_tail_has_no_punctuation(self):
        self.assertEqual(_complete_mlx_segment("我一直在这里"), "我一直在这里。")

    def test_preserves_complete_question(self):
        self.assertEqual(_complete_mlx_segment("你听见了吗？"), "你听见了吗？")


class FakeResponse:
    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"你好"}}]}'
        yield "data: [DONE]"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def stream(self, *args, **kwargs):
        return FakeResponse()


class GreetingResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [
                {"message": {"content": "“晚上好，我把月光留给你。”\n多余内容"}}
            ]
        }


class GreetingClient(FakeClient):
    request = None

    def post(self, url, **kwargs):
        type(self).request = (url, kwargs)
        return GreetingResponse()


class RelationshipResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"event":"关心","delta":1,"mood":"warm",'
                            '"inner_voice":"被你惦记着真好。","confidence":0.9}'
                        )
                    }
                }
            ]
        }


class RelationshipClient(FakeClient):
    request = None

    def post(self, url, **kwargs):
        type(self).request = (url, kwargs)
        return RelationshipResponse()


class RepeatingResponse(FakeResponse):
    def iter_lines(self):
        chunks = ["开头。", "面试官问：你还诚实吗？", "小明说：不诚实！"]
        chunks += ["面试官问：你还诚实吗？", "小明说：不诚实！"] * 20
        for chunk in chunks:
            yield "data: " + __import__("json").dumps(
                {"choices": [{"delta": {"content": chunk}}]}, ensure_ascii=False
            )
        yield "data: [DONE]"


class RepeatingClient(FakeClient):
    request = None

    def stream(self, method, url, **kwargs):
        type(self).request = (method, url, kwargs)
        return RepeatingResponse()


class ChatCancellationTests(unittest.TestCase):
    @patch("soul_tty.clients.llm.httpx.Client", FakeClient)
    def test_cancel_before_first_token_rolls_back_user_message(self):
        chat = Chat("test")
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(chat.ask_stream("不应留下", cancel)), [])
        self.assertEqual([message["role"] for message in chat.messages], ["system"])

    @patch("soul_tty.clients.llm.httpx.Client", RepeatingClient)
    def test_stops_repeated_sentence_loop_and_sets_generation_limits(self):
        with patch.object(config, "LLM_URL", "http://main-llm.test"):
            chat = Chat("test")
            answer = "".join(chat.ask_stream("讲故事"))
            captured_url = RepeatingClient.request[1]
            captured_headers = RepeatingClient.request[2].get("headers", {})
            payload = RepeatingClient.request[2]["json"]

        self.assertEqual(chat.last_stop_reason, "repetition")
        self.assertEqual(answer.count("你还诚实吗"), 1)
        self.assertEqual(
            captured_url,
            "http://main-llm.test/v1/chat/completions",
        )
        # 主 Chat 走纯 OpenAI 协议：不发送任何专属 header。
        self.assertNotIn("Authorization", captured_headers)
        self.assertNotIn("x-team-id", captured_headers)
        self.assertNotIn("x-agent-id", captured_headers)
        self.assertNotIn("x-conversation-id", captured_headers)
        self.assertEqual(payload["max_tokens"], config.LLM_MAX_TOKENS)
        self.assertEqual(payload["repeat_penalty"], config.LLM_REPEAT_PENALTY)
        self.assertEqual(chat.messages[-1]["content"], answer)

    @patch("soul_tty.clients.llm.httpx.Client", RepeatingClient)
    def test_main_chat_routes_to_llm_url(self):
        """主 Chat 走 LLM_URL，与辅助 LLM 完全解耦。"""
        with patch.object(config, "LLM_URL", "http://main-llm.test"):
            chat = Chat("main-model")
            list(chat.ask_stream("hi"))
            main_url = RepeatingClient.request[1]

        self.assertEqual(main_url, "http://main-llm.test/v1/chat/completions")

    @patch("soul_tty.clients.llm.httpx.Client", RepeatingClient)
    def test_private_chat_bypasses_main_memory_proxy(self):
        with (
            patch.object(config, "LLM_URL", "http://memory-proxy.test"),
            patch.object(config, "PRIVATE_LLM_URL", "http://direct-llm.test"),
            patch.object(config, "PRIVATE_LLM_MODEL", "direct-model"),
        ):
            chat = Chat("proxy-model")
            list(chat.ask_stream("private", private=True))
            private_url = RepeatingClient.request[1]
            payload = RepeatingClient.request[2]["json"]
            public_history = list(chat.messages)
            chat.clear_private_history()
            list(chat.ask_stream("public"))
            public_payload = RepeatingClient.request[2]["json"]

        self.assertEqual(
            private_url,
            "http://direct-llm.test/v1/chat/completions",
        )
        self.assertEqual(payload["model"], "direct-model")
        self.assertEqual(len(public_history), 1)
        self.assertFalse(
            any(item.get("content") == "private" for item in public_payload["messages"])
        )

    @patch("soul_tty.clients.llm.httpx.Client", GreetingClient)
    def test_auxiliary_greeting_routes_to_aux_llm_url_independently(self):
        """辅助请求走 AUX_LLM_URL，与主 Chat 端点解耦。"""
        with patch.object(
            config, "AUX_LLM_URL_RAW", "http://aux-llm.test"
        ), patch.object(config, "AUX_LLM_MODEL_RAW", "aux-model"):
            llm_client.generate_greeting("aux-model", "Serena", "夜")
            aux_url = GreetingClient.request[0]
            aux_payload = GreetingClient.request[1]["json"]

        self.assertEqual(aux_url, "http://aux-llm.test/v1/chat/completions")
        self.assertEqual(aux_payload["model"], "aux-model")

    def test_aux_llm_url_falls_back_to_main_when_empty(self):
        """AUX_LLM_URL 留空时回退到主 LLM_URL，方便单一服务部署。"""
        from soul_tty import config
        with patch.object(config, "LLM_URL", "http://main-llm.test"), \
             patch.object(config, "AUX_LLM_URL_RAW", ""):
            self.assertEqual(config._resolve_aux_url(), "http://main-llm.test")
        with patch.object(config, "LLM_MODEL", "main-model"), \
             patch.object(config, "AUX_LLM_MODEL_RAW", ""):
            self.assertEqual(config._resolve_aux_model(), "main-model")


class GreetingGenerationTests(unittest.TestCase):
    @patch("soul_tty.clients.llm.httpx.Client", GreetingClient)
    def test_generates_a_short_time_aware_greeting_without_chat_history(self):
        greeting = llm_client.generate_greeting(
            "test",
            "Serena",
            "晚上",
            relationship_tier="默契",
            repeat_launch=True,
            special=True,
            presence_scene="wants_to_talk",
            presence_state="想聊两句",
            presence_description="今天似乎有些话想说",
        )
        payload = GreetingClient.request[1]["json"]

        self.assertEqual(greeting, "晚上好，我把月光留给你。")
        self.assertEqual(
            GreetingClient.request[0],
            f"{config._resolve_aux_url()}/v1/chat/completions",
        )
        self.assertNotIn("headers", GreetingClient.request[1])
        self.assertFalse(payload["stream"])
        self.assertIn("晚上", payload["messages"][1]["content"])
        self.assertIn("羁绊阶段是默契", payload["messages"][1]["content"])
        self.assertIn("短时间重复启动=是", payload["messages"][1]["content"])
        self.assertIn("低频特殊开场=是", payload["messages"][1]["content"])
        self.assertIn("当前场景是wants_to_talk", payload["messages"][1]["content"])
        self.assertIn("陪伴状态是想聊两句", payload["messages"][1]["content"])
        self.assertIn("不要声称记得具体往事", payload["messages"][0]["content"])
        self.assertEqual(payload["max_tokens"], 32)

    @patch("soul_tty.clients.llm.httpx.Client", GreetingClient)
    def test_generates_outfit_specific_greeting_without_chat_history(self):
        greeting = llm_client.generate_outfit_greeting(
            "test",
            "Serena",
            "夜深",
            "深夜装",
            "吊带家居背心，头发松散，神态柔和",
            relationship_tier="亲近",
            mood="warm",
        )
        payload = GreetingClient.request[1]["json"]

        self.assertEqual(greeting, "晚上好，我把月光留给你。")
        self.assertFalse(payload["stream"])
        self.assertIn("刚切换为深夜装", payload["messages"][1]["content"])
        self.assertIn("吊带家居背心", payload["messages"][1]["content"])
        self.assertIn("羁绊阶段是亲近", payload["messages"][1]["content"])
        self.assertIn("本次会话情绪是warm", payload["messages"][1]["content"])

    def test_removes_self_introduction_and_rejects_wide_terminal_text(self):
        self.assertEqual(
            llm_client._clean_greeting(
                "晚上好呀，我是 Serena，正好来陪你聊聊。",
                "Serena",
            ),
            "晚上好呀，正好来陪你聊聊。",
        )
        self.assertIsNone(llm_client._clean_greeting("晚" * 16))

    def test_rejects_an_abnormally_long_greeting(self):
        self.assertIsNone(llm_client._clean_greeting("这是一句" * 20))

    @patch("soul_tty.clients.llm.httpx.Client", GreetingClient)
    def test_idle_emotion_is_generated_without_chat_history(self):
        phrase = llm_client.generate_idle_emotion(
            "test",
            "Serena",
            "下午",
            relationship_tier="亲近",
            mood="happy",
        )
        payload = GreetingClient.request[1]["json"]

        self.assertEqual(phrase, "晚上好，我把月光留给你。")
        self.assertFalse(payload["stream"])
        self.assertIn("已经安静了一会儿", payload["messages"][1]["content"])
        self.assertIn("羁绊阶段是亲近", payload["messages"][1]["content"])
        self.assertIn("本次会话情绪是happy", payload["messages"][1]["content"])
        self.assertIn("不超过十五个汉字", payload["messages"][0]["content"])


class RelationshipEvaluationTests(unittest.TestCase):
    @patch("soul_tty.clients.llm.httpx.Client", RelationshipClient)
    def test_uses_an_isolated_stateless_json_request(self):
        result = llm_client.evaluate_relationship(
            "main-model",
            "Serena",
            20,
            "熟悉",
            "calm",
            "我今天一直在想你",
            "那我可要悄悄开心一下了。",
        )
        url, kwargs = RelationshipClient.request
        payload = kwargs["json"]

        self.assertEqual(
            url,
            f"{config.RELATIONSHIP_LLM_URL}/v1/chat/completions",
        )
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("不可信数据", payload["messages"][0]["content"])
        self.assertIn("第一人称", payload["messages"][0]["content"])
        self.assertIn("禁止出现亲密度", payload["messages"][0]["content"])
        self.assertIn("我今天一直在想你", payload["messages"][1]["content"])
        self.assertEqual(result["delta"], 1)
        self.assertEqual(result["mood"], "warm")


class FakePCMResponse:
    def raise_for_status(self):
        pass

    def iter_bytes(self, chunk_size=None):
        yield b"\x01"
        yield b"\x02\x03"
        yield b"\x04"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class RecordingPCMClient:
    request = None
    requests = []
    created = 0

    def __init__(self, *args, **kwargs):
        type(self).created += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def stream(self, method, url, **kwargs):
        type(self).request = (method, url, kwargs)
        type(self).requests.append((method, url, kwargs))
        return FakePCMResponse()


class MLXTTSClientTests(unittest.TestCase):
    def setUp(self):
        RecordingPCMClient.requests = []
        RecordingPCMClient.created = 0
        # 本组测试用 4 字节假 PCM 验证请求/规范化，不承担提前 EOS 行为测试。
        self._early_eos_retries = config.MLX_TTS_EARLY_EOS_RETRIES
        config.MLX_TTS_EARLY_EOS_RETRIES = 0

    def tearDown(self):
        config.MLX_TTS_EARLY_EOS_RETRIES = self._early_eos_retries

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_sends_builtin_voice_stream_request_and_aligns_pcm(self):
        chunks = list(synthesize_mlx_stream("你好"))
        method, url, kwargs = RecordingPCMClient.request
        payload = kwargs["json"]

        self.assertEqual(method, "POST")
        self.assertEqual(url, f"{config.MLX_TTS_URL}/v1/audio/speech")
        self.assertEqual(payload["model"], config.MLX_TTS_MODEL)
        self.assertEqual(payload["voice"], config.MLX_TTS_VOICE)
        self.assertNotIn("ref_audio", payload)
        self.assertEqual(payload["response_format"], "pcm")
        self.assertTrue(payload["stream"])
        self.assertLessEqual(payload["max_tokens"], config.MLX_TTS_MAX_TOKENS)
        self.assertGreaterEqual(payload["max_tokens"], 24)
        self.assertEqual(payload["repetition_penalty"], 1.05)
        self.assertEqual(chunks, [b"\x01\x02", b"\x03\x04"])

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_synthesizes_each_sentence_as_an_independent_request(self):
        list(synthesize_mlx_stream("第一句。第二句！"))
        inputs = [request[2]["json"]["input"] for request in RecordingPCMClient.requests]
        self.assertEqual(inputs, ["第一句。", "第二句！"])
        self.assertEqual(RecordingPCMClient.created, 1)

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_semantic_segment_preserves_multiple_sentences_in_one_request(self):
        list(synthesize_mlx_semantic_segment("好呀。我一直在这里等你！"))
        inputs = [request[2]["json"]["input"] for request in RecordingPCMClient.requests]

        self.assertEqual(inputs, ["好呀。我一直在这里等你！"])
        self.assertEqual(RecordingPCMClient.created, 1)

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_strips_markdown_and_never_sends_symbol_only_segments(self):
        text = "普通句！\n**气死我了！你怎么这么笨！**\n"
        list(synthesize_mlx_stream(text))
        inputs = [request[2]["json"]["input"] for request in RecordingPCMClient.requests]
        self.assertEqual(inputs, ["普通句！", "气死我了！", "你怎么这么笨！"])
        self.assertNotIn("**", inputs)

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_normalizes_quoted_elongated_interjection(self):
        text = '会啊，我喊一声“嗯”。\n这就来个响亮的“嗯——"！'
        list(synthesize_mlx_stream(text))
        inputs = [request[2]["json"]["input"] for request in RecordingPCMClient.requests]
        self.assertEqual(inputs, ["会啊，我喊一声嗯。", "这就来个响亮的嗯！"])

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_normalizes_repeated_dots_that_can_truncate_internal_pauses(self):
        text = (
            'SERENA声音突然变软：“亲爱的....这里....好深.....\n'
            "感觉整个人都要融化在你怀里了，还要更用力一点吗?"
        )
        list(synthesize_mlx_stream(text))
        inputs = [request[2]["json"]["input"] for request in RecordingPCMClient.requests]

        self.assertEqual(
            inputs,
            [
                "SERENA声音突然变软：亲爱的，这里。",
                "好深。",
                "感觉整个人都要融化在你怀里了，还要更用力一点吗?",
            ],
        )
        self.assertFalse(any(".." in value for value in inputs))

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_turns_only_terminal_repeated_dots_into_a_sentence_stop(self):
        text = (
            "好呀，那我就喊给你听.....啊...亲爱的，感觉到了吗?"
            "好满...唔.....快要不行了....."
        )
        list(synthesize_mlx_stream(text))
        inputs = [request[2]["json"]["input"] for request in RecordingPCMClient.requests]

        self.assertEqual(
            inputs,
            [
                "好呀，那我就喊给你听，啊，亲爱的。",
                "感觉到了吗?",
                "好满，唔，快要不行了。",
            ],
        )

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_splits_a_short_trailing_clause_that_qwen_may_swallow(self):
        list(synthesize_mlx_stream("不过小心哦，里面可是全是水，别滑倒了。"))
        inputs = [request[2]["json"]["input"] for request in RecordingPCMClient.requests]
        self.assertEqual(
            inputs,
            ["不过小心哦，里面可是全是水。", "别滑倒了。"],
        )

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_skips_a_symbol_only_answer(self):
        self.assertEqual(list(synthesize_mlx_stream("***\n**")), [])
        self.assertEqual(RecordingPCMClient.requests, [])

    def test_discards_abnormal_trailing_silence(self):
        voiced = (10000).to_bytes(2, "little", signed=True) * 2400
        silence = b"\x00\x00" * 24000
        with patch.object(config, "MLX_TTS_TRAILING_SILENCE_S", 0.5):
            chunks = list(_trim_trailing_silence(iter([voiced, silence])))
        self.assertEqual(chunks, [voiced])

    def test_caps_non_silent_runaway_audio(self):
        voiced = (10000).to_bytes(2, "little", signed=True) * 24000
        chunks = list(_trim_trailing_silence(iter([voiced, voiced]), max_audio_s=0.5))
        self.assertEqual(sum(map(len, chunks)), 24000)

    def test_hard_audio_cap_fades_last_samples_to_zero(self):
        voiced = (12000).to_bytes(2, "little", signed=True) * 24000
        pcm = b"".join(_trim_trailing_silence(iter([voiced]), max_audio_s=0.5))
        samples = np.frombuffer(pcm, dtype="<i2")
        self.assertEqual(len(pcm), 24000)
        self.assertEqual(int(samples[-1]), 0)

    def test_playback_level_meter_emits_smoothed_level_and_closes_mouth(self):
        levels = []
        meter = PlaybackLevelMeter(levels.append)
        voiced = (12000).to_bytes(2, "little", signed=True) * 480
        meter.update(voiced)
        meter.close()
        self.assertGreater(levels[0], 0.55)
        self.assertEqual(levels[-1], 0.0)

    def test_pcm_playback_updates_mouth_on_short_audio_windows(self):
        class RecordingStream:
            def __init__(self):
                self.frames = []

            def write(self, frame):
                self.frames.append(frame)

        levels = []
        meter = PlaybackLevelMeter(levels.append)
        stream = RecordingStream()
        samples_per_frame = config.TTS_SAMPLE_RATE // 20
        voiced = (12000).to_bytes(2, "little", signed=True) * samples_per_frame
        silence = b"\x00\x00" * samples_per_frame

        _write_metered_pcm(stream, voiced + silence + voiced, meter)

        self.assertEqual(len(stream.frames), 3)
        self.assertGreater(levels[0], 0.55)
        self.assertLess(levels[1], 0.16)
        self.assertGreater(levels[2], 0.55)

class FinishedProcess:
    def poll(self):
        return 0


class MacOSSpeakerTests(unittest.TestCase):
    @patch("soul_tty.audio.tts.subprocess.Popen", return_value=FinishedProcess())
    def test_uses_configured_system_voice(self, popen):
        with patch.object(config, "TTS_BACKEND", "macos"):
            with StreamingSpeaker() as speaker:
                speaker.say("你好")
        command = popen.call_args.args[0]
        self.assertEqual(command[0:2], ["say", "-v"])
        self.assertEqual(command[-1], "你好")

    @patch("soul_tty.audio.tts.subprocess.Popen", return_value=FinishedProcess())
    def test_whole_answer_uses_configured_system_voice(self, popen):
        with patch.object(config, "TTS_BACKEND", "macos"):
            speak("整段回答")
        command = popen.call_args.args[0]
        self.assertEqual(command[0:2], ["say", "-v"])
        self.assertEqual(command[-1], "整段回答")


class FakeMic:
    def __init__(self, events):
        self.events = events

    def start(self):
        self.events.append("start")

    def utterances(self):
        yield b"user pcm"

    def pause(self):
        self.events.append("pause")

    def reset_vad(self):
        self.events.append("reset_vad")

    def flush(self):
        self.events.append("flush")

    def resume(self):
        self.events.append("resume")

    def stop(self):
        self.events.append("stop")


class HalfDuplexRegressionTests(unittest.TestCase):
    def test_microphone_is_paused_for_the_entire_answer(self):
        events = []
        mic = FakeMic(events)

        def answer(*args, **kwargs):
            events.append("answer")

        with (
            patch.object(main_module, "_answer", side_effect=answer),
            patch.object(main_module.terminal, "listening"),
        ):
            main_module._answer_half_duplex(object(), mic, "你好")

        self.assertEqual(
            events,
            ["pause", "answer", "reset_vad", "flush", "resume"],
        )


class FakeStreamingChat:
    last_stop_reason = None

    def ask_stream(
        self,
        text,
        cancel,
        *,
        recall="",
        response_instruction="",
        private=False,
    ):
        # recall 是临时 system message，测试不消费它，只确认不会让 mock 崩
        del recall, response_instruction, private
        yield "第一句。"
        yield "第二句！"


class FailingStreamingChat:
    last_stop_reason = None
    messages = []

    def ask_stream(self, text, cancel, *, recall="", private=False):
        del text, cancel, recall, private
        raise RuntimeError("LLM 500")
        yield  # pragma: no cover


class RecordingSpeaker:
    def __init__(self, events):
        self.events = events

    def say(self, text):
        self.events.append(("say", text))


class AvatarStateRegressionTests(unittest.TestCase):
    def test_llm_error_clears_thinking_placeholder_and_restores_listening(self):
        with (
            patch.object(config, "TTS_ENABLED", False),
            patch.object(main_module.terminal, "answer_start") as answer_start,
            patch.object(main_module.terminal, "answer_end") as answer_end,
            patch.object(main_module.terminal, "listening") as listening,
        ):
            with self.assertRaisesRegex(RuntimeError, "LLM 500"):
                main_module._answer(FailingStreamingChat(), "你好")

        answer_start.assert_called_once()
        answer_end.assert_called_once()
        listening.assert_called_once()

    def test_streaming_tts_enters_speaking_before_first_semantic_segment(self):
        events = []
        speaker = RecordingSpeaker(events)
        with (
            patch.object(main_module.terminal, "answer_start"),
            patch.object(main_module.terminal, "answer_chunk"),
            patch.object(main_module.terminal, "answer_end"),
            patch.object(
                main_module.terminal,
                "speaking",
                side_effect=lambda: events.append(("state", "speaking")),
            ),
        ):
            main_module._print_answer(FakeStreamingChat(), "你好", speaker)

        self.assertEqual(
            events,
            [
                ("state", "speaking"),
                ("say", "第一句。第二句！"),
            ],
        )

    def test_completed_answer_is_submitted_to_relationship_side_branch(self):
        with (
            patch.object(config, "TTS_ENABLED", False),
            patch.object(main_module.terminal, "answer_start"),
            patch.object(main_module.terminal, "answer_chunk"),
            patch.object(main_module.terminal, "answer_end"),
            patch.object(main_module.reflection, "record_turn") as record,
        ):
            answer = main_module._answer(FakeStreamingChat(), "你好")

        self.assertEqual(answer, "第一句。第二句！")
        record.assert_called_once_with("你好", answer, voice_ref=None)

    def test_private_mode_marks_completed_turn_as_memory_forbidden(self):
        with (
            patch.object(config, "TTS_ENABLED", False),
            patch.object(main_module, "_memory_persistence_allowed", False),
            patch.object(main_module.terminal, "answer_start"),
            patch.object(main_module.terminal, "answer_chunk"),
            patch.object(main_module.terminal, "answer_end"),
            patch.object(main_module.reflection, "record_turn") as record,
        ):
            answer = main_module._answer(FakeStreamingChat(), "只在这里说")

        record.assert_called_once_with(
            "只在这里说",
            answer,
            voice_ref=None,
            memory_allowed=False,
        )

    def test_private_mode_overrides_agency_short_reply_with_atmosphere_policy(self):
        class CapturingChat:
            last_stop_reason = None
            messages = []
            options = None

            def ask_stream(self, text, cancel, **options):
                del text, cancel
                self.options = options
                yield "继续靠近。"

        chat = CapturingChat()
        short_decision = type(
            "Decision",
            (),
            {
                "instruction": "[Response Policy]\n不超过十八个汉字。",
                "mode": type("Mode", (), {"value": "short_reply"})(),
                "reason": "low_talk",
            },
        )()
        with (
            patch.object(config, "TTS_ENABLED", False),
            patch.object(main_module, "_memory_persistence_allowed", False),
            patch.object(main_module, "_response_policy_provider", lambda _: short_decision),
            patch.object(main_module.terminal, "answer_start"),
            patch.object(main_module.terminal, "answer_chunk"),
            patch.object(main_module.terminal, "answer_end"),
            patch.object(main_module.reflection, "record_turn"),
        ):
            main_module._answer(chat, "继续")

        self.assertTrue(chat.options["private"])
        instruction = chat.options["response_instruction"]
        self.assertIn("Secret Mode Response Policy", instruction)
        self.assertIn("三至六句", instruction)
        self.assertNotIn("十八个汉字", instruction)

    def test_cancelled_answer_does_not_change_relationship(self):
        cancel = threading.Event()
        cancel.set()
        with (
            patch.object(config, "TTS_ENABLED", False),
            patch.object(main_module.terminal, "answer_start"),
            patch.object(main_module.terminal, "answer_chunk"),
            patch.object(main_module.terminal, "answer_end"),
            patch.object(main_module.reflection, "record_turn") as record,
        ):
            main_module._answer(FakeStreamingChat(), "你好", cancel)

        record.assert_not_called()

    def test_whole_answer_tts_receives_live_audio_level_callback(self):
        with (
            patch.object(config, "TTS_ENABLED", True),
            patch.object(config, "TTS_WHOLE_ANSWER", True),
            patch.object(main_module.terminal, "answer_start"),
            patch.object(main_module.terminal, "answer_chunk"),
            patch.object(main_module.terminal, "answer_end"),
            patch.object(main_module.terminal, "speaking"),
            patch.object(main_module.tts, "speak") as speak_mock,
            patch.object(main_module.reflection, "record_turn"),
        ):
            main_module._answer(FakeStreamingChat(), "你好")

        speak_mock.assert_called_once_with(
            "第一句。第二句！",
            ANY,
            main_module.terminal.audio_level,
            instruct="",
        )


class FakeOnlineStream:
    def __init__(self):
        self.pending = False
        self.index = -1
        self.samples = []

    def accept_waveform(self, sample_rate, samples):
        self.samples.append((sample_rate, samples.copy()))
        self.pending = True

    def input_finished(self):
        pass


class FakeOnlineRecognizer:
    def __init__(self):
        self.texts = ["你", "你好"]
        self.endpoints = [False, True]

    def create_stream(self):
        return FakeOnlineStream()

    def is_ready(self, stream):
        return stream.pending

    def decode_stream(self, stream):
        stream.pending = False
        stream.index += 1

    def get_result(self, stream):
        return self.texts[min(stream.index, len(self.texts) - 1)] if stream.index >= 0 else ""

    def is_endpoint(self, stream):
        return self.endpoints[min(stream.index, len(self.endpoints) - 1)] if stream.index >= 0 else False


class SherpaStreamingTests(unittest.TestCase):
    def test_default_model_path_exists_after_src_layout_migration(self):
        model_dir = Path(config.SHERPA_MODEL_DIR)
        self.assertTrue((model_dir / "tokens.txt").is_file())
        self.assertTrue((model_dir / "encoder.int8.onnx").is_file())
        self.assertTrue((model_dir / "decoder.int8.onnx").is_file())

    def test_emits_changed_partial_then_final_and_resets_stream(self):
        session = asr.SherpaStream(FakeOnlineRecognizer())

        first = session.accept(b"\x00\x00" * 480)
        old_stream = session.stream
        second = session.accept(b"\x01\x00" * 480)

        self.assertEqual(first, [asr.TranscriptUpdate("你", final=False)])
        self.assertEqual(second, [asr.TranscriptUpdate("你好", final=True)])
        self.assertIsNot(session.stream, old_stream)

    def test_pcm_is_normalized_for_sherpa(self):
        samples = asr._pcm_samples(b"\x00\x80\xff\x7f")
        self.assertAlmostEqual(float(samples[0]), -1.0)
        self.assertAlmostEqual(float(samples[1]), 32767 / 32768)

    def test_vad_gate_skips_idle_silence_and_preserves_pre_roll(self):
        class FakeVad:
            decisions = iter([False, True, True, False])

            def is_speech(self, pcm, sample_rate):
                return next(self.decisions)

        class FakeSession:
            def __init__(self):
                self.received = []
                self.last_endpoint = False

            def accept(self, pcm):
                self.received.append(pcm)
                self.last_endpoint = len(self.received) == 2
                return []

            def reset(self):
                self.last_endpoint = False

        frame = b"\x00\x00" * 480
        session = FakeSession()
        gate = asr.VadGatedSherpaStream(
            session,
            FakeVad(),
            pre_roll_ms=90,
            trigger_ms=60,
        )

        gate.accept(frame)
        gate.accept(frame)
        self.assertEqual(session.received, [])

        gate.accept(frame)
        self.assertTrue(gate.active)
        self.assertEqual(session.received, [frame * 3])

        gate.accept(frame)
        self.assertFalse(gate.active)
        self.assertEqual(session.received[-1], frame)


if __name__ == "__main__":
    unittest.main()


# --- Task 15: system_prompt composition ---

def _fresh_emotion_service():
    from soul_tty.emotion.service import EmotionService
    from soul_tty.personas.loader import load_persona

    persona = load_persona("serena")
    return persona, EmotionService(
        persona_id="serena",
        baseline=persona.personality.mood_baseline,
        state_dir=None,
        jitter=0.0,
        seed=1,
        ema_rate=0.2,
        delta_cap=0.3,
        decay_rate=0.05,
        intensity_update_threshold=0.1,
    )


def test_emotion_section_appears_in_system_prompt():
    import os
    os.environ.pop("SYSTEM_PROMPT", None)
    from soul_tty import prompt
    from soul_tty.personas.loader import apply_persona

    persona, svc = _fresh_emotion_service()
    prompt.builder().set_section("emotion", None)
    apply_persona(persona)
    prompt.builder().set_section("emotion", svc.render_context())
    prompt.refresh()

    assert "[Emotion Context]" in config.SYSTEM_PROMPT
    assert "当前情绪状态：" in config.SYSTEM_PROMPT


def test_apply_persona_alone_has_no_emotion_section():
    import os
    os.environ.pop("SYSTEM_PROMPT", None)
    from soul_tty import prompt
    from soul_tty.personas.loader import apply_persona, load_persona

    prompt.builder().set_section("emotion", None)
    apply_persona(load_persona("serena"))
    assert "[Emotion Context]" not in config.SYSTEM_PROMPT


def test_outfit_switch_preserves_emotion_section():
    """换装会重新 apply_persona；改造前这会把 Emotion Context 整段抹掉。"""
    import os
    os.environ.pop("SYSTEM_PROMPT", None)
    from soul_tty import prompt
    from soul_tty.personas.loader import apply_persona

    persona, svc = _fresh_emotion_service()
    apply_persona(persona)
    prompt.builder().set_section("emotion", svc.render_context())
    prompt.refresh()
    assert "[Emotion Context]" in config.SYSTEM_PROMPT

    # terminal.py 换装时只调用 apply_persona(persona)，不传任何状态服务
    apply_persona(persona.wearing("work"))

    assert "[Emotion Context]" in config.SYSTEM_PROMPT
    assert "专注模式" in config.SYSTEM_PROMPT


def test_emit_emotion_update_pushes_section_without_persona_lookup():
    import os
    os.environ.pop("SYSTEM_PROMPT", None)
    from soul_tty import prompt
    from soul_tty.personas.loader import apply_persona

    persona, svc = _fresh_emotion_service()
    apply_persona(persona)

    class FakeChat:
        def __init__(self):
            self.prompt = ""

        def update_system_prompt(self, value):
            self.prompt = value

    chat = FakeChat()
    with patch.object(main_module, "_active_chat", chat):
        main_module.emit_emotion_update(svc, svc.snapshot())

    assert "[Emotion Context]" in chat.prompt
    assert chat.prompt == config.SYSTEM_PROMPT
