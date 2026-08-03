import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from soul_tty import config
from soul_tty import conversation as main_module
from soul_tty.audio import asr
from soul_tty.audio.tts import (
    PlaybackLevelMeter,
    StreamingSpeaker,
    _trim_trailing_silence,
    speak,
    synthesize_mlx_stream,
)
from soul_tty.clients.llm import Chat
from soul_tty.clients import llm as llm_client
from soul_tty.conversation import _is_probable_echo, _usable_transcript


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
        chat = Chat("test")
        answer = "".join(chat.ask_stream("讲故事"))
        payload = RepeatingClient.request[2]["json"]

        self.assertEqual(chat.last_stop_reason, "repetition")
        self.assertEqual(answer.count("你还诚实吗"), 1)
        self.assertEqual(payload["max_tokens"], config.LLM_MAX_TOKENS)
        self.assertEqual(payload["repeat_penalty"], config.LLM_REPEAT_PENALTY)
        self.assertEqual(chat.messages[-1]["content"], answer)


class GreetingGenerationTests(unittest.TestCase):
    @patch("soul_tty.clients.llm.httpx.Client", GreetingClient)
    def test_generates_a_short_time_aware_greeting_without_chat_history(self):
        greeting = llm_client.generate_greeting("test", "Serena", "晚上")
        payload = GreetingClient.request[1]["json"]

        self.assertEqual(greeting, "晚上好，我把月光留给你。")
        self.assertFalse(payload["stream"])
        self.assertIn("晚上", payload["messages"][1]["content"])
        self.assertEqual(payload["max_tokens"], 32)

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
        phrase = llm_client.generate_idle_emotion("test", "Serena", "下午")
        payload = GreetingClient.request[1]["json"]

        self.assertEqual(phrase, "晚上好，我把月光留给你。")
        self.assertFalse(payload["stream"])
        self.assertIn("已经安静了一会儿", payload["messages"][1]["content"])
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

    def __init__(self, *args, **kwargs):
        pass

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
        self.assertEqual(payload["max_tokens"], config.MLX_TTS_MAX_TOKENS)
        self.assertEqual(payload["repetition_penalty"], 1.05)
        self.assertEqual(chunks, [b"\x01\x02", b"\x03\x04"])

    @patch("soul_tty.audio.tts.httpx.Client", RecordingPCMClient)
    def test_synthesizes_each_sentence_as_an_independent_request(self):
        list(synthesize_mlx_stream("第一句。第二句！"))
        inputs = [request[2]["json"]["input"] for request in RecordingPCMClient.requests]
        self.assertEqual(inputs, ["第一句。", "第二句！"])

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

    def test_playback_level_meter_emits_smoothed_level_and_closes_mouth(self):
        levels = []
        meter = PlaybackLevelMeter(levels.append)
        voiced = (12000).to_bytes(2, "little", signed=True) * 480
        meter.update(voiced)
        meter.close()
        self.assertGreater(levels[0], 0.55)
        self.assertEqual(levels[-1], 0.0)

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

    def ask_stream(self, text, cancel):
        yield "第一句。"
        yield "第二句！"


class RecordingSpeaker:
    def __init__(self, events):
        self.events = events

    def say(self, text):
        self.events.append(("say", text))


class AvatarStateRegressionTests(unittest.TestCase):
    def test_streaming_tts_enters_speaking_before_first_sentence(self):
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
                ("say", "第一句。"),
                ("say", "第二句！"),
            ],
        )

    def test_completed_answer_is_submitted_to_relationship_side_branch(self):
        with (
            patch.object(config, "TTS_ENABLED", False),
            patch.object(main_module.terminal, "answer_start"),
            patch.object(main_module.terminal, "answer_chunk"),
            patch.object(main_module.terminal, "answer_end"),
            patch.object(main_module.relationship, "record_turn") as record,
        ):
            answer = main_module._answer(FakeStreamingChat(), "你好")

        self.assertEqual(answer, "第一句。第二句！")
        record.assert_called_once_with("你好", answer)

    def test_cancelled_answer_does_not_change_relationship(self):
        cancel = threading.Event()
        cancel.set()
        with (
            patch.object(config, "TTS_ENABLED", False),
            patch.object(main_module.terminal, "answer_start"),
            patch.object(main_module.terminal, "answer_chunk"),
            patch.object(main_module.terminal, "answer_end"),
            patch.object(main_module.relationship, "record_turn") as record,
        ):
            main_module._answer(FakeStreamingChat(), "你好", cancel)

        record.assert_not_called()


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


if __name__ == "__main__":
    unittest.main()
