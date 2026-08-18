"""
Dedicated test suite for Overlap Source Separation & Attribution module.
"""

import numpy as np
import pytest

from config import OVERLAP_MIN_CONFIDENCE, OVERLAP_SEPARATION_ENABLED
from models.diarizer import SessionDiarizer
from models.identifier import SpeakerIdentifier
from models.overlap_attribution import resolve_overlap_attribution
from stt.base import STTAdapter


class MockSTTAdapter(STTAdapter):
    @property
    def adapter_type(self) -> str:
        return "local"

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock"

    def transcribe(self, audio: np.ndarray, language=None, hint_language=None) -> dict:
        return {
            "text": "Hello world test",
            "language": "en",
            "language_name": "English",
            "language_prob": 1.0,
            "language_detected": "en",
            "language_fallback": False,
        }

    def transcribe_with_context(self, full_audio, start_sec, end_sec, hint_language=None, padding_sec=None) -> dict:
        return self.transcribe(full_audio, hint_language=hint_language)


def test_overlap_disabled_by_default(monkeypatch):
    """Confirm OVERLAP_SEPARATION_ENABLED is False by default."""
    import config
    assert config.OVERLAP_SEPARATION_ENABLED is False, "OVERLAP_SEPARATION_ENABLED must be False by default"


def test_normal_non_overlap_speech_bypass():
    """Confirm non-overlap speech bypasses overlap resolution."""
    diarizer = SessionDiarizer()
    identifier = SpeakerIdentifier()
    stt = MockSTTAdapter()
    dummy_audio = np.zeros(16000, dtype=np.float32)

    # When OVERLAP_SEPARATION_ENABLED is False, resolve_overlap_attribution returns None
    res = resolve_overlap_attribution(
        dummy_audio,
        start=0.0,
        end=1.0,
        candidate_labels=["Speaker_00", "Speaker_01"],
        diarizer=diarizer,
        identifier=identifier,
        stt_adapter=stt,
        dominant_language="en",
    )
    assert res is None, "Disabled overlap separation must return None"


def test_low_confidence_fallback(monkeypatch):
    """Confirm low confidence separation falls back to safe behavior."""
    import config
    monkeypatch.setattr(config, "OVERLAP_SEPARATION_ENABLED", True)
    monkeypatch.setattr(config, "OVERLAP_MIN_CONFIDENCE", 0.99)  # Ultra high floor forces fallback

    # Mock source separation returning 2 dummy streams
    monkeypatch.setattr(
        "models.overlap_attribution.separate_audio_streams",
        lambda audio: (np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.float32)),
    )

    diarizer = SessionDiarizer()
    diarizer._new_cluster(np.random.randn(192).astype(np.float32))
    diarizer._new_cluster(np.random.randn(192).astype(np.float32))
    identifier = SpeakerIdentifier()
    stt = MockSTTAdapter()

    dummy_audio = np.random.randn(16000).astype(np.float32)

    res = resolve_overlap_attribution(
        dummy_audio,
        start=0.0,
        end=1.0,
        candidate_labels=["Speaker_00", "Speaker_01"],
        diarizer=diarizer,
        identifier=identifier,
        stt_adapter=stt,
        dominant_language="en",
    )
    assert res is None, "Low confidence must return None to trigger safe fallback"


def test_existing_diarization_behavior_intact():
    """Verify existing SessionDiarizer clustering behavior remains intact."""
    diarizer = SessionDiarizer()
    emb1 = np.ones(192, dtype=np.float32) / np.sqrt(192)
    label1 = diarizer.add_segment(0.0, 2.0, emb1)
    assert label1 == "Speaker_00"

    emb2 = np.ones(192, dtype=np.float32) / np.sqrt(192)
    label2 = diarizer.add_segment(2.0, 4.0, emb2)
    assert label2 == "Speaker_00"

    emb_new = -np.ones(192, dtype=np.float32) / np.sqrt(192)
    label3 = diarizer.add_segment(4.0, 6.0, emb_new)
    assert label3 != "Speaker_00"
