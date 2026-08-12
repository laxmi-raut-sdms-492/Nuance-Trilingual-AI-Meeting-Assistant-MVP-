"""VAD fallback segmentation when Silero finds no speech boundaries."""

from __future__ import annotations

import numpy as np

from config import MAX_SEGMENT_SECONDS, MIN_SPEECH_SECONDS, SAMPLE_RATE, SILENCE_RMS_THRESHOLD
from models.vad import fallback_segments


def _tone(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_fallback_returns_empty_for_silence():
    audio = np.zeros(int(MIN_SPEECH_SECONDS * SAMPLE_RATE), dtype=np.float32)
    assert fallback_segments(audio) == []


def test_fallback_returns_empty_when_too_short():
    audio = _tone(MIN_SPEECH_SECONDS - 0.2)
    assert fallback_segments(audio) == []


def test_fallback_whole_file_for_short_non_silent_upload():
    audio = _tone(30.0)
    segments = fallback_segments(audio)
    assert len(segments) == 1
    assert segments[0]["start"] == 0.0
    assert abs(segments[0]["end"] - 30.0) < 0.01
    np.testing.assert_array_equal(segments[0]["audio"], audio)


def test_fallback_splits_long_non_silent_upload():
    duration = 65.0  # above VAD_FALLBACK_WHOLE_FILE_MAX_SECONDS
    audio = _tone(duration)
    segments = fallback_segments(audio)
    assert len(segments) == 9  # ceil(65 / 8)
    assert segments[0]["start"] == 0.0
    assert abs(segments[-1]["end"] - duration) < 0.01


def test_fallback_skips_quiet_windows_in_long_file():
    loud = _tone(MAX_SEGMENT_SECONDS, amplitude=0.2)
    quiet = np.zeros(int(MAX_SEGMENT_SECONDS * SAMPLE_RATE), dtype=np.float32)
    # 10 x 8s blocks = 80s, above the whole-file cap; every other window is silent.
    blocks = []
    for i in range(10):
        blocks.append(loud if i % 2 == 0 else quiet)
    audio = np.concatenate(blocks)
    segments = fallback_segments(audio)
    assert len(segments) == 5
    assert segments[0]["start"] == 0.0
    assert segments[1]["start"] == 2 * MAX_SEGMENT_SECONDS
