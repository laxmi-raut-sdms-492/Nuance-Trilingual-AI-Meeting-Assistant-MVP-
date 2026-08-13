"""
Indic Conformer recovery must not eat genuine English.

The recovery step exists for a real failure: Whisper hears Marathi speech,
labels it English, and emits a romanized transliteration ("mala udya bhetu
ya"). Re-decoding that with Indic Conformer gets the real Devanagari back.

The danger is that its acceptance test — "did the Indic decode come back as
Devanagari?" — cannot fail. Indic Conformer answers in Devanagari whatever it
is given, and Devanagari always passes the script check for an Indic language.
So without a guard on the way IN, every English segment in the meeting is
replaced by a Devanagari transliteration of itself, fluently and silently.

These tests fake both decoders. They are about which branch runs, not about
what the models say, so they need no weights and no network.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from config import SAMPLE_RATE
from models.asr import transcribe


def _tone(seconds: float = 2.0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_real_english_is_not_replaced_by_a_devanagari_decode():
    """
    The regression. Plain English, confidently detected. Indic Conformer would
    happily return Devanagari for it, and that Devanagari would pass every
    check downstream — so the only thing that can save the line is refusing to
    ask in the first place.
    """
    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.97), ("hi", 0.02)]),
        patch("models.asr._decode_whisper", return_value="the deadline is Friday"),
        patch("models.asr._decode_indic", return_value="द डेडलाइन इज फ्रायडे") as indic,
    ):
        result = transcribe(_tone(), hint_language=None)

    assert result["text"] == "the deadline is Friday"
    assert result["language"] == "en"
    indic.assert_not_called()


def test_english_is_kept_even_when_the_meeting_is_mostly_marathi():
    """
    The dominant-language hint must not be enough on its own. In a mostly
    Marathi meeting every English sentence would otherwise be converted.
    """
    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.95), ("mr", 0.03)]),
        patch("models.asr._decode_whisper", return_value="I will send the report"),
        patch("models.asr._decode_indic", return_value="आय विल सेंड द रिपोर्ट") as indic,
    ):
        result = transcribe(_tone(), hint_language="mr")

    assert result["text"] == "I will send the report"
    assert result["language"] == "en"
    indic.assert_not_called()


def test_romanized_marathi_mislabelled_english_is_still_recovered():
    """
    The case the feature was built for must keep working. Marathi function
    words in Latin script ("mala", "ahe") are the lexical evidence that this
    "English" is not English.
    """
    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.88), ("mr", 0.09)]),
        patch("models.asr._decode_whisper", return_value="mala udya bhetu ya ahe"),
        patch("models.asr._decode_indic", return_value="मला उद्या भेटूया आहे"),
    ):
        result = transcribe(_tone(), hint_language=None)

    assert result["language"] in ("mr", "hi")
    assert result["text"] == "मला उद्या भेटूया आहे"


def test_empty_english_decode_does_not_trigger_recovery():
    """No text means no lexical evidence, so there is nothing to act on."""
    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.9), ("hi", 0.05)]),
        patch("models.asr._decode_whisper", return_value=""),
        patch("models.asr._decode_indic", return_value="काहीतरी") as indic,
    ):
        result = transcribe(_tone(), hint_language=None)

    assert result["text"] == ""
    indic.assert_not_called()


def test_devanagari_languages_are_left_alone():
    """Recovery is an en-only path; a Hindi segment must not re-enter it."""
    with (
        patch("models.asr.language_ranking", return_value=[("hi", 0.93), ("mr", 0.04)]),
        patch("models.asr._decode_indic", return_value="मैं कल आऊंगा"),
        patch("models.asr._decode_whisper", return_value="should not be used"),
    ):
        result = transcribe(_tone(), hint_language=None)

    assert result["language"] in ("hi", "mr")
    assert result["text"] == "मैं कल आऊंगा"
