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