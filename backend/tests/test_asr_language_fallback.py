"""ASR trilingual fallback — weak detection should not default to English."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from config import SAMPLE_RATE
from models.asr import transcribe


def _tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_weak_hindi_detection_kept_without_hint():
    audio = _tone(2.0)
    with (
        patch("models.asr.language_ranking", return_value=[("hi", 0.30)]),
        patch("models.asr._decode", return_value="नमस्ते सर्वांना"),
    ):
        result = transcribe(audio, hint_language=None)
    assert result["language"] == "hi"
    assert result["text"] == "नमस्ते सर्वांना"


def test_very_weak_detection_defaults_to_english():
    audio = _tone(2.0)
    with (
        patch("models.asr.language_ranking", return_value=[("mr", 0.10)]),
        patch("models.asr._decode", return_value="hello"),
    ):
        result = transcribe(audio, hint_language=None)
    assert result["language"] == "en"


def test_weak_detection_uses_meeting_hint():
    audio = _tone(2.0)
    with (
        patch("models.asr.language_ranking", return_value=[("en", 0.32)]),
        patch("models.asr._decode", return_value="ठीक आहे"),
    ):
        result = transcribe(audio, hint_language="mr")
    assert result["language"] == "mr"
