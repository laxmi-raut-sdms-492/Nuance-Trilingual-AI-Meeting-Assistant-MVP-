"""
Mixed-language (code-switch) detection.

A segment with no pause in it gets ONE language and ONE engine, so a speaker
who switches mid-breath has half their words transcribed by the wrong model.
Nothing splits a segment on language yet — this is the detector that says the
line is suspect instead of presenting it as a confident transcription, and the
gate a future language-change splitter would branch on.

The signal is free: language_ranking() already computes the full ranking and
only the winner was ever read.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from config import SAMPLE_RATE
from models.asr import mixed_language_signal, transcribe


def _tone(seconds: float) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


# ------------------------------------------------------------ the raw signal


def test_confident_detection_is_not_mixed():
    margin, mixed = mixed_language_signal([("en", 0.97), ("mr", 0.02), ("hi", 0.01)])
    assert margin == 0.95
    assert mixed is False


def test_english_against_devanagari_scoring_evenly_is_mixed():
    """The case this exists for: both languages really were spoken."""
    margin, mixed = mixed_language_signal([("en", 0.45), ("mr", 0.40), ("hi", 0.15)])
    assert margin == 0.05
    assert mixed is True


def test_close_hindi_marathi_call_is_not_flagged():
    """
    hi and mr are closely related and the detector confuses them constantly on
    short segments — asr.py already carries a hi<->mr retry for exactly that.
    Flagging these would bury the real signal in noise.
    """
    margin, mixed = mixed_language_signal([("hi", 0.44), ("mr", 0.42), ("en", 0.14)])
    assert margin == 0.02
    assert mixed is False


def test_devanagari_against_english_is_flagged_in_either_order():
    _, mixed = mixed_language_signal([("mr", 0.46), ("en", 0.44), ("hi", 0.10)])
    assert mixed is True


def test_single_candidate_is_never_mixed():
    """language_ranking returns one entry when detection produced nothing."""
    margin, mixed = mixed_language_signal([("en", 0.0)])
    assert margin == 1.0
    assert mixed is False


def test_script_boundary_requirement_can_be_disabled():
    with patch("models.asr.ASR_MIXED_REQUIRES_SCRIPT_BOUNDARY", True):
        assert mixed_language_signal([("hi", 0.44), ("mr", 0.42)])[1] is False
    with patch("models.asr.ASR_MIXED_REQUIRES_SCRIPT_BOUNDARY", False):
        assert mixed_language_signal([("hi", 0.44), ("mr", 0.42)])[1] is True


# ------------------------------------------------------ plumbed into results


def test_transcribe_reports_the_flag_on_a_mixed_segment():
    audio = _tone(4.0)
    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.46), ("mr", 0.44)]),
        patch("models.asr._decode", return_value="The deadline is Friday manje udya"),
    ):
        result = transcribe(audio, hint_language=None)

    assert result["language_mixed_suspected"] is True
    assert result["language_margin"] == 0.02
    # The text is still produced — this flags the line, it does not drop it.
    assert result["text"]


def test_transcribe_reports_clean_segment_as_not_mixed():
    audio = _tone(4.0)
    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.98), ("mr", 0.01)]),
        patch("models.asr._decode", return_value="Good morning everyone."),
    ):
        result = transcribe(audio, hint_language=None)

    assert result["language_mixed_suspected"] is False
    assert result["language_margin"] == 0.97


def test_flag_is_independent_of_the_weak_detection_fallback():
    """
    language_fallback answers "was the winner confident enough to trust".
    language_mixed_suspected answers "were there two winners". A segment can be
    either, both, or neither, so one must not be inferred from the other.
    """
    audio = _tone(4.0)
    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.36), ("mr", 0.34)]),
        patch("models.asr._decode", return_value="mixed line"),
    ):
        result = transcribe(audio, hint_language="mr")

    assert result["language_fallback"] is True      # 0.36 < LANGUAGE_DETECT_MIN_PROB
    assert result["language_mixed_suspected"] is True
