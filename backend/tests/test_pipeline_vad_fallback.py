"""Pipeline upload path — VAD fallback when Silero emits nothing."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from audio_utils import UploadAudioProfile
from config import MIN_SILENCE_MS, SAMPLE_RATE, SILENCE_RMS_THRESHOLD, VAD_BATCH_FALLBACK_THRESHOLD
from pipeline import MeetingSession
from tests.test_pipeline_labels import _FakeIdentifier


def _tone(seconds: float) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _neutral_profile() -> UploadAudioProfile:
    return UploadAudioProfile(
        silence_rms_threshold=SILENCE_RMS_THRESHOLD,
        vad_threshold=0.5,
        min_silence_ms=MIN_SILENCE_MS,
        batch_vad_threshold=VAD_BATCH_FALLBACK_THRESHOLD,
        batch_vad_min_speech_ratio=0.10,
        k_max=12,
        hum_freq_hz=None,
        is_flat_tone=False,
        warning=None,
    )


@pytest.fixture
def session():
    return MeetingSession("MTG-fallback", _FakeIdentifier())


def test_process_audio_uses_fallback_when_vad_emits_nothing(session):
    audio = _tone(5.0)
    fake_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with (
        patch("pipeline.prepare_upload_audio", return_value=(audio, _neutral_profile())),
        patch("pipeline.SpeechSegmenter", return_value=session.segmenter),
        patch.object(session.segmenter, "process", return_value=[]),
        patch.object(session.segmenter, "flush", return_value=[]),
        patch("pipeline.get_embedding", return_value=fake_embedding),
        patch.object(session.stt_adapter, "transcribe_with_context", return_value={
            "text": "hello from fallback",
            "language": "en",
            "language_name": "English",
            "language_detected": "en",
            "language_prob": 0.9,
            "language_fallback": False,
        }),
        patch.object(session.stt_adapter, "transcribe", return_value={
            "text": "hello from fallback",
            "language": "en",
            "language_name": "English",
            "language_detected": "en",
            "language_prob": 0.9,
            "language_fallback": False,
        }),
        patch("pipeline.split_on_speaker_change", return_value=[(0, len(audio))]),
        patch("models.transcript_cleanup.cleanup_turns", side_effect=lambda turns, **kw: [
            {**t, "cleaned_text": t.get("raw_text") or t.get("text"), "text": t.get("raw_text") or t.get("text")}
            for t in turns
        ]),
    ):
        session.process_audio(audio)

    assert len(session.transcript) == 1
    assert session.transcript[0]["text"] == "hello from fallback"


def test_process_audio_skips_fallback_when_vad_already_produced_lines(session):
    audio = _tone(5.0)
    vad_seg = {"start": 0.0, "end": 5.0, "audio": audio}
    fake_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with (
        patch("pipeline.prepare_upload_audio", return_value=(audio, _neutral_profile())),
        patch("pipeline.SpeechSegmenter", return_value=session.segmenter),
        patch.object(session.segmenter, "process", return_value=[vad_seg]),
        patch.object(session.segmenter, "flush", return_value=[]),
        patch("pipeline.get_embedding", return_value=fake_embedding),
        patch.object(session.stt_adapter, "transcribe_with_context", return_value={
            "text": "normal vad path",
            "language": "en",
            "language_name": "English",
            "language_detected": "en",
            "language_prob": 0.9,
            "language_fallback": False,
        }),
        patch.object(session.stt_adapter, "transcribe", return_value={
            "text": "normal vad path",
            "language": "en",
            "language_name": "English",
            "language_detected": "en",
            "language_prob": 0.9,
            "language_fallback": False,
        }),
        patch("pipeline.split_on_speaker_change", return_value=[(0, len(audio))]),
        patch("models.transcript_cleanup.cleanup_turns", side_effect=lambda turns, **kw: [
            {**t, "cleaned_text": t.get("raw_text") or t.get("text"), "text": t.get("raw_text") or t.get("text")}
            for t in turns
        ]),
    ):
        session.process_audio(audio)

    assert len(session.transcript) == 1
    assert session.transcript[0]["text"] == "normal vad path"
