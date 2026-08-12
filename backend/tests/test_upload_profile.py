"""Per-upload dynamic audio profile."""

from __future__ import annotations

import numpy as np

from audio_utils import analyze_upload_audio, prepare_upload_audio
from config import SAMPLE_RATE, SILENCE_RMS_THRESHOLD


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.2) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_flat_tone_profile_lowers_vad_and_sets_warning():
    profile = analyze_upload_audio(_tone(30.0))
    assert profile.is_flat_tone
    assert profile.vad_threshold <= 0.15
    assert profile.warning is not None
    assert profile.k_max >= 4


def test_long_meeting_gets_higher_k_max_and_longer_silence():
    loud = np.random.randn(int(200 * SAMPLE_RATE)).astype(np.float32) * 0.05
    profile = analyze_upload_audio(loud)
    assert profile.k_max >= 10
    assert profile.min_silence_ms >= 300


def test_prepare_upload_audio_notches_flat_tone():
    audio = _tone(10.0)
    cleaned, profile = prepare_upload_audio(audio)
    assert profile.is_flat_tone
    assert cleaned.shape == audio.shape
    assert not np.array_equal(cleaned, audio)


def test_normal_audio_keeps_default_silence_floor():
    loud = np.random.randn(int(1.5 * SAMPLE_RATE)).astype(np.float32) * 0.05
    quiet = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([loud, quiet, loud, quiet, loud])
    profile = analyze_upload_audio(audio)
    assert not profile.is_flat_tone
    assert profile.silence_rms_threshold >= SILENCE_RMS_THRESHOLD
