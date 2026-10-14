"""A Cloud STT failure stays on Sarvam and never calls the local model."""

from __future__ import annotations

import numpy as np


class _FailingSpeech:
    def transcribe(self, **kwargs):
        raise RuntimeError("quota exceeded")


class _FailingClient:
    speech_to_text = _FailingSpeech()


def _adapter(monkeypatch):
    monkeypatch.setattr("sarvamai.SarvamAI", lambda **kwargs: _FailingClient())
    from stt.sarvam_adapter import SarvamSTTAdapter

    return SarvamSTTAdapter(api_key="test-key")


def test_cloud_failure_stays_on_sarvam(monkeypatch):
    adapter = _adapter(monkeypatch)
    audio = np.zeros(1600, dtype=np.float32)

    first = adapter.transcribe(audio)
    second = adapter.transcribe(audio)

    assert first["text"] == "" and first["provider"] == "sarvam"
    assert second["text"] == "" and second["provider"] == "sarvam"


def test_cloud_failure_does_not_construct_the_local_model(monkeypatch):
    adapter = _adapter(monkeypatch)

    def _forbidden(*args, **kwargs):
        raise AssertionError("Cloud STT must not construct the local model")

    monkeypatch.setattr("stt.local_adapter.LocalSTTAdapter", _forbidden)
    result = adapter.transcribe(np.zeros(1600, dtype=np.float32))

    assert result["provider"] == "sarvam"
    assert result["model"] == "saaras:v3"


def test_local_runs_only_when_the_upload_asks_for_it():
    from api import _stt_choice

    assert _stt_choice("local", "local", "") == ("local", None)
    assert _stt_choice("", "local", "") == ("local", None)
    assert _stt_choice("cloud", "cloud", "") == ("cloud", "sarvam")
    assert _stt_choice("cloud", "", "") == ("cloud", "sarvam")
    assert _stt_choice("", "", "") == ("cloud", "sarvam")
    # A cloud choice is never rewritten to the local model.
    assert _stt_choice("cloud", "local", "") == ("cloud", "sarvam")
