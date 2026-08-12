"""Hybrid ASR routing — English via Whisper, Hindi/Marathi via Indic Conformer."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from config import SAMPLE_RATE
from models.asr import _decode, _uses_indic_conformer, transcribe


def _tone(seconds: float) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_uses_indic_for_hindi_and_marathi():
    assert _uses_indic_conformer("hi") is True
    assert _uses_indic_conformer("mr") is True
    assert _uses_indic_conformer("en") is False


def test_decode_routes_english_to_whisper():
    audio = _tone(1.5)
    with (
        patch("models.asr._decode_whisper", return_value="hello team") as whisper,
        patch("models.asr._decode_indic") as indic,
    ):
        text = _decode(audio, "en", 1.5)
    assert text == "hello team"
    whisper.assert_called_once()
    indic.assert_not_called()


def test_decode_routes_hindi_to_indic():
    audio = _tone(1.5)
    with (
        patch("models.asr._decode_whisper") as whisper,
        patch("models.asr._decode_indic", return_value="नमस्ते") as indic,
    ):
        text = _decode(audio, "hi", 1.5)
    assert text == "नमस्ते"
    indic.assert_called_once()
    whisper.assert_not_called()


def test_transcribe_hindi_calls_indic_decode():
    audio = _tone(2.0)
    with (
        patch("models.asr.language_ranking", return_value=[("hi", 0.92)]),
        patch("models.asr._decode_indic", return_value="ठीक आहे") as indic,
        patch("models.asr._decode_whisper") as whisper,
    ):
        result = transcribe(audio, hint_language=None)
    assert result["language"] == "hi"
    assert result["text"] == "ठीक आहे"
    indic.assert_called_once()
    whisper.assert_not_called()


def test_transcribe_english_calls_whisper_decode():
    audio = _tone(2.0)
    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.95)]),
        patch("models.asr._decode_whisper", return_value="good morning") as whisper,
        patch("models.asr._decode_indic") as indic,
    ):
        result = transcribe(audio, hint_language=None)
    assert result["language"] == "en"
    assert result["text"] == "good morning"
    whisper.assert_called_once()
    indic.assert_not_called()
