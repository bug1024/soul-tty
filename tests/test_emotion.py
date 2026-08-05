"""实时情绪系统测试。"""

import pytest

from src.soul_tty.emotion.state import EmotionVector


def test_emotion_vector_clamp_low():
    v = EmotionVector(
        happiness=-0.5, calmness=0.5, curiosity=0.5, stress=0.5, energy=0.5
    )
    assert v.happiness == 0.0


def test_emotion_vector_clamp_high():
    v = EmotionVector(
        happiness=1.5, calmness=0.5, curiosity=0.5, stress=0.5, energy=0.5
    )
    assert v.happiness == 1.0


def test_emotion_vector_dict_round_trip():
    v = EmotionVector(
        happiness=0.6, calmness=0.7, curiosity=0.8, stress=0.2, energy=0.5
    )
    d = v.to_dict()
    assert d == {
        "happiness": 0.6,
        "calmness": 0.7,
        "curiosity": 0.8,
        "stress": 0.2,
        "energy": 0.5,
    }
    assert EmotionVector.from_dict(d) == v


def test_emotion_vector_immutable():
    v = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.5, stress=0.5, energy=0.5
    )
    with pytest.raises(Exception):
        v.happiness = 0.9  # frozen dataclass


def test_default_baseline():
    from src.soul_tty.emotion.state import DEFAULT_BASELINE

    assert DEFAULT_BASELINE == EmotionVector(
        happiness=0.65, calmness=0.75, curiosity=0.70, stress=0.20, energy=0.75
    )

# --- Task 3: Mood Resolver ---

from src.soul_tty.emotion.resolver import resolve_mood, MOODS


def test_resolve_calm_default():
    mood, intensity = resolve_mood(EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.5, stress=0.5, energy=0.5
    ))
    assert mood == "calm"


def test_resolve_numb_priority():
    v = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.5, stress=0.8, energy=0.2
    )
    assert resolve_mood(v)[0] == "numb"


def test_resolve_curious_priority_over_happy():
    """Curious 必须在 Happy 之前判定。"""
    # energy<0.75 让 Excited 不触发；happiness>=0.65 让 Happy 可触发
    v = EmotionVector(
        happiness=0.8, calmness=0.5, curiosity=0.9, stress=0.1, energy=0.5
    )
    assert resolve_mood(v)[0] == "curious"


def test_resolve_excited_needs_both():
    v = EmotionVector(
        happiness=0.8, calmness=0.5, curiosity=0.3, stress=0.1, energy=0.8
    )
    assert resolve_mood(v)[0] == "excited"


def test_resolve_sad_needs_both():
    v = EmotionVector(
        happiness=0.2, calmness=0.5, curiosity=0.3, stress=0.6, energy=0.5
    )
    assert resolve_mood(v)[0] == "sad"


def test_resolve_tired_energy_only():
    v = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.3, stress=0.1, energy=0.3
    )
    assert resolve_mood(v)[0] == "tired"


def test_resolve_happy_minimum():
    v = EmotionVector(
        happiness=0.7, calmness=0.5, curiosity=0.3, stress=0.1, energy=0.5
    )
    assert resolve_mood(v)[0] == "happy"


def test_resolve_intensity_calm():
    v = EmotionVector(
        happiness=0.5, calmness=0.8, curiosity=0.3, stress=0.1, energy=0.5
    )
    mood, intensity = resolve_mood(v)
    assert mood == "calm"
    assert intensity == pytest.approx(0.8)


def test_resolve_intensity_excited():
    v = EmotionVector(
        happiness=0.8, calmness=0.5, curiosity=0.3, stress=0.1, energy=0.9
    )
    mood, intensity = resolve_mood(v)
    assert mood == "excited"
    assert intensity == pytest.approx(0.85)


def test_moods_contains_all():
    assert MOODS == ("numb", "tired", "sad", "excited", "curious", "happy", "calm")


# --- Task 4: Expression Resolver ---

from src.soul_tty.emotion.expression import resolve_expression, EXPRESSIONS


def test_default_expression_neutral():
    v = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.5, stress=0.3, energy=0.6
    )
    assert resolve_expression(v, "") == "neutral"


def test_caring_when_user_negative_event():
    v = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.3, stress=0.5, energy=0.5
    )
    assert resolve_expression(v, "caring") == "caring"


def test_caring_persists_with_user_signal():
    v = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.3, stress=0.6, energy=0.5
    )
    assert resolve_expression(v, "caring") == "caring"


def test_invalid_hint_falls_back_to_neutral():
    assert resolve_expression(EmotionVector(0.5,0.5,0.5,0.3,0.6), "WEIRD") == "neutral"


def test_expressions_list():
    assert "neutral" in EXPRESSIONS
    assert "caring" in EXPRESSIONS


# --- Task 5: EMA Updater + Decay ---

from src.soul_tty.emotion.updater import apply_delta, apply_decay, perturb_baseline


def test_apply_delta_target_is_old_plus_delta_clamped():
    old = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.5, stress=0.5, energy=0.5
    )
    delta = {
        "happiness": 0.3,
        "stress": -0.5,  # -> clamp
        "energy": 0.05,
    }
    new = apply_delta(old, delta, rate=0.2)
    # happiness: target=0.8; new=0.5+(0.8-0.5)*0.2=0.56
    assert new.happiness == pytest.approx(0.56)
    # stress: delta_capped to -0.3; target=clamp(0.5-0.3)=0.2; new=0.5+(0.2-0.5)*0.2=0.44
    assert new.stress == pytest.approx(0.44)
    # energy: target=0.55; new=0.5+(0.05)*0.2=0.51
    assert new.energy == pytest.approx(0.51)
    # untouched
    assert new.calmness == pytest.approx(0.5)
    assert new.curiosity == pytest.approx(0.5)


def test_apply_delta_ignores_unknown_dims():
    old = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.5, stress=0.5, energy=0.5
    )
    new = apply_delta(old, {"happiness": 0.1, "bogus": 0.5}, rate=0.2)
    assert new.happiness == pytest.approx(0.52)
    # calmness untouched
    assert new.calmness == pytest.approx(0.5)


def test_apply_decay_returns_to_baseline():
    baseline = EmotionVector(
        happiness=0.6, calmness=0.6, curiosity=0.6, stress=0.6, energy=0.6
    )
    current = EmotionVector(
        happiness=0.9, calmness=0.2, curiosity=0.3, stress=0.9, energy=0.9
    )
    new = apply_decay(current, baseline, rate=0.05)
    # happiness: 0.9 + (0.6-0.9)*0.05 = 0.9 - 0.015 = 0.885
    assert new.happiness == pytest.approx(0.885)
    # calmness: 0.2 + (0.6-0.2)*0.05 = 0.22
    assert new.calmness == pytest.approx(0.22)


def test_perturb_baseline_within_bounds():
    base = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.5, stress=0.5, energy=0.5
    )
    perturbed = perturb_baseline(base, jitter=0.1, seed=42)
    for dim in ("happiness", "calmness", "curiosity", "stress", "energy"):
        v = getattr(perturbed, dim)
        assert 0.0 <= v <= 1.0
        assert abs(v - 0.5) <= 0.1


def test_perturb_baseline_zero_jitter_is_identity():
    base = EmotionVector(
        happiness=0.5, calmness=0.5, curiosity=0.5, stress=0.5, energy=0.5
    )
    perturbed = perturb_baseline(base, jitter=0.0, seed=1)
    assert perturbed == base


# --- Task 6: Emotion Delta Analyzer ---

from src.soul_tty.emotion.analyzer import parse_emotion_delta


def test_parse_emotion_delta_valid():
    raw = {"happiness": 0.1, "stress": -0.05, "energy": 0.05}
    assert parse_emotion_delta(raw, delta_cap=0.3) == raw


def test_parse_emotion_delta_caps_large_values():
    raw = {"happiness": 1.0, "stress": -2.0}
    parsed = parse_emotion_delta(raw, delta_cap=0.3)
    assert parsed["happiness"] == 0.3
    assert parsed["stress"] == -0.3


def test_parse_emotion_delta_ignores_unknown_dims():
    raw = {"happiness": 0.1, "weird": 0.5}
    parsed = parse_emotion_delta(raw, delta_cap=0.3)
    assert "weird" not in parsed
    assert parsed["happiness"] == 0.1


def test_parse_emotion_delta_empty_returns_empty_dict():
    assert parse_emotion_delta({}, delta_cap=0.3) == {}


def test_parse_emotion_delta_invalid_type_returns_empty():
    assert parse_emotion_delta({"happiness": "abc"}, delta_cap=0.3) == {}
    assert parse_emotion_delta(None, delta_cap=0.3) == {}


# --- Task 7: Prompt Builder ---

from src.soul_tty.emotion.prompt_builder import build_emotion_context


def test_build_emotion_context_calm_high_intensity():
    text = build_emotion_context("calm", 0.85)
    assert "明显" in text


def test_build_emotion_context_happy_low_intensity():
    text = build_emotion_context("happy", 0.6)
    assert "开心" in text or "积极" in text


def test_build_emotion_context_with_caring_expression():
    text = build_emotion_context("calm", 0.5, expression="caring")
    assert "关心" in text or "温柔" in text or "陪伴" in text or "关切" in text


def test_build_emotion_context_never_includes_numeric_values():
    text = build_emotion_context("happy", 0.85)
    assert "happiness=" not in text
    assert "0.85" not in text


def test_build_emotion_context_unknown_mood_returns_minimal():
    text = build_emotion_context("unknown_mood", 0.5)
    assert isinstance(text, str)
    assert len(text) > 0


def test_build_emotion_context_low_intensity_no_intensity_word():
    text = build_emotion_context("calm", 0.3)
    assert "明显" not in text
    assert "中等" not in text


# --- Task 8: TTS Mapping ---

from src.soul_tty.emotion.tts_mapping import build_tts_instruct


def test_calm_returns_empty_instruct():
    assert build_tts_instruct("calm", 0.5, expression="neutral") == ""


def test_happy_returns_instruct():
    text = build_tts_instruct("happy", 0.8, expression="neutral")
    assert "开心" in text or "上扬" in text


def test_caring_expression_overrides_mood_instruct():
    text = build_tts_instruct("calm", 0.5, expression="caring")
    assert "关心" in text or "温柔" in text


def test_unknown_mood_returns_empty():
    assert build_tts_instruct("weird_mood", 0.5) == ""


# --- Task 9: Avatar Mapping ---

from src.soul_tty.emotion.avatar_mapping import build_avatar_expression


def test_happy_avatar_expression():
    expr = build_avatar_expression("happy", 0.8, expression="neutral")
    assert expr["face"] == "smile"
    assert expr["eye"] == "open"


def test_caring_expression_overrides_motion():
    expr = build_avatar_expression("calm", 0.5, expression="caring")
    assert expr["motion"] == "slight_lean"


def test_unknown_mood_falls_back_to_neutral():
    expr = build_avatar_expression("weird_mood", 0.5)
    assert expr["face"] == "neutral"
    assert expr["eye"] == "open"
    assert expr["motion"] == "none"
