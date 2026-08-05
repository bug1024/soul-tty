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
    assert MOODS == {"numb", "tired", "sad", "excited", "curious", "happy", "calm"}
