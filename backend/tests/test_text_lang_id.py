"""Text-based Hindi/Marathi/English disambiguation (bug fix regression tests)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from config import SAMPLE_RATE
from models.asr import transcribe
from models.text_lang_id import classify_text_language


def _tone(seconds: float = 2.0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Me Saloni ahe", "mr"),
        ("Majha nav Saloni ahe", "mr"),
        ("मी शिक्षक आहे", "mr"),
        ("मी तिचर आहे", "mr"),
        ("Mera naam Saloni hai", "hi"),
        ("Main Hindi bolti hoon", "hi"),
        ("मेरा नाम सलोनी है", "hi"),
        ("My name is Saloni", "en"),
    ],
)
def test_classify_text_language(text, expected):
    lang, _confidence = classify_text_language(text)
    assert lang == expected


def test_classify_mixed_sentence_favors_marathi_markers():
    lang, confidence = classify_text_language("Hello, mi Saloni ahe.")
    assert lang == "mr"
    assert confidence > 0


def test_classify_unknown_devanagari_text_returns_none():
    # Devanagari with no recognizable hi/mr marker words (e.g. a lone name).
    lang, confidence = classify_text_language("सलोनी")
    assert lang is None
    assert confidence == 0.0


def test_marathi_audio_mislabeled_as_hindi_gets_corrected():
    """
    Reproduces the reported bug: Whisper's acoustic detector guesses 'hi'
    confidently, IndicConformer decodes with Hindi weights and produces a
    fluent Devanagari string that nonetheless contains unambiguous Marathi
    markers ('मी', 'आहे'). The text-based check should override the label
    and re-decode with 'mr'.
    """
    audio = _tone()

    # The Hindi and Marathi IndicConformer heads share an encoder and mostly
    # agree on which Devanagari words were spoken — what differs is fluency
    # on grammar/orthography, not that they invent a totally different
    # sentence. So the initial ("wrong") hi decode still comes back with the
    # actual Marathi words, which is exactly what makes the old script-only
    # check blind to this failure mode.
    def fake_decode_indic(_audio, _language, _duration):
        return "मी शिक्षक आहे"

    with (
        patch("models.asr.language_ranking", return_value=[("hi", 0.92)]),
        patch("models.asr._decode_indic", side_effect=fake_decode_indic),
        patch("models.asr.INDIC_CONFORMER_ENABLED", True),
    ):
        result = transcribe(audio, hint_language=None)

    assert result["language"] == "mr"
    assert result["text"] == "मी शिक्षक आहे"
    # language_detected still reflects the raw acoustic guess for debugging.
    assert result["language_detected"] == "hi"


def test_romanized_marathi_not_forced_to_english():
    """
    'Me Saloni ahe' has no Devanagari at all, so the old script-only check
    can't catch it. If Whisper's acoustic guess lands on English (a common
    failure mode for romanized Indic speech) and the decode is romanized
    Marathi, the lexical check should relabel it 'mr' rather than leaving it
    tagged as English.
    """
    audio = _tone()

    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.55)]),
        patch("models.asr._decode_whisper", return_value="Me Saloni ahe"),
    ):
        result = transcribe(audio, hint_language=None)

    assert result["language"] == "mr"
    assert result["text"] == "Me Saloni ahe"


def test_confident_correct_hindi_is_not_overridden():
    audio = _tone()

    with (
        patch("models.asr.language_ranking", return_value=[("hi", 0.92)]),
        patch("models.asr._decode_indic", return_value="मेरा नाम सलोनी है"),
    ):
        result = transcribe(audio, hint_language=None)

    assert result["language"] == "hi"
    assert result["text"] == "मेरा नाम सलोनी है"


def test_plain_english_is_not_overridden():
    audio = _tone()

    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.95)]),
        patch("models.asr._decode_whisper", return_value="My name is Saloni"),
    ):
        result = transcribe(audio, hint_language=None)

    assert result["language"] == "en"
    assert result["text"] == "My name is Saloni"
