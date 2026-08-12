"""Upload audio preprocessing — flat-tone / hum detection."""

from __future__ import annotations

import numpy as np

from audio_utils import is_flat_tone, per_second_rms_std, preprocess_upload_audio, remove_steady_hum
from config import SAMPLE_RATE


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.2) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_flat_tone_detection():
    assert is_flat_tone(_tone(30.0))
    speech = _tone(1.0, freq=220.0, amp=0.05)
    quiet = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    mixed = np.concatenate([speech, quiet, speech, quiet])
    assert not is_flat_tone(mixed)


def test_preprocess_applies_notch_to_flat_tone():
    audio = _tone(10.0)
    cleaned, warning = preprocess_upload_audio(audio)
    assert warning is not None
    assert per_second_rms_std(cleaned) > per_second_rms_std(audio)


def test_preprocess_leaves_normal_audio_unchanged():
    loud = np.random.randn(int(1.5 * SAMPLE_RATE)).astype(np.float32) * 0.05
    quiet = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([loud, quiet, loud, quiet, loud])
    cleaned, warning = preprocess_upload_audio(audio)
    assert warning is None
    np.testing.assert_array_equal(cleaned, audio)


def test_remove_steady_hum_reduces_target_frequency():
    audio = _tone(2.0, freq=220.0)
    cleaned = remove_steady_hum(audio)
    assert np.sqrt(np.mean(cleaned**2)) < np.sqrt(np.mean(audio**2))
