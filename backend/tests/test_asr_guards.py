"""ASR hallucination guards — keep real quiet speech, drop fluent inventions."""

from __future__ import annotations

from models.asr import _hallucination_reason


def _result(no_speech: float, avg_logprob: float, text: str = "hello") -> dict:
    return {
        "segments": [
            {"no_speech_prob": no_speech, "avg_logprob": avg_logprob, "text": text},
        ]
    }


def test_keeps_uncertain_decode_despite_high_no_speech_prob():
    # Measured on a real 30s upload Silero missed: no_speech=0.94, logprob=-1.66.
    reason = _hallucination_reason(
        _result(0.94, -1.66),
        "I'm not sure if you can hear me.",
        duration=30.0,
    )
    assert reason is None


def test_drops_confident_hallucination_on_silence():
    reason = _hallucination_reason(
        _result(0.94, -0.3),
        "Thank you for watching.",
        duration=2.0,
    )
    assert reason is not None
    assert "no_speech_prob" in reason


def test_drops_moderate_garble_not_extreme_uncertainty():
    reason = _hallucination_reason(
        _result(0.75, -1.2),
        "static noise decode",
        duration=2.0,
    )
    assert reason is not None


def test_keeps_clean_speech():
    reason = _hallucination_reason(
        _result(0.2, -0.4),
        "Good morning everyone.",
        duration=3.0,
    )
    assert reason is None


def test_drops_impossible_word_rate():
    reason = _hallucination_reason(
        _result(0.1, -0.2),
        "one two three four five six seven eight nine ten eleven twelve",
        duration=1.0,
    )
    assert reason is not None
    assert "words/sec" in reason
