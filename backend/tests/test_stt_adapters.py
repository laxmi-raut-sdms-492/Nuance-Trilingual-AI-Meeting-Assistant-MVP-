"""Unit tests for STT Adapter Architecture, Factory, and Providers."""

from __future__ import annotations

import numpy as np
import pytest

from stt.base import STTProviderError
from stt.factory import get_stt_adapter
from stt.local.local_adapter import LocalSTTAdapter
from stt.cloud.cloud_adapter import CloudSTTAdapter
from stt.cloud.sarvam_provider import SarvamProvider, LANGUAGE_MAPPING
from stt.cloud.google_chirp_provider import GoogleChirpProvider


@pytest.fixture
def dummy_audio():
    return np.zeros(16000, dtype=np.float32)


def test_stt_factory_default():
    adapter = get_stt_adapter()
    assert isinstance(adapter, LocalSTTAdapter)
    assert adapter.adapter_type == "local"


def test_stt_factory_local():
    adapter = get_stt_adapter("local")
    assert isinstance(adapter, LocalSTTAdapter)
    assert adapter.adapter_type == "local"
    assert adapter.provider_name == "whisper+indic_conformer"


def test_stt_factory_cloud_sarvam():
    adapter = get_stt_adapter("cloud", "sarvam")
    assert isinstance(adapter, CloudSTTAdapter)
    assert adapter.adapter_type == "cloud"
    assert adapter.provider_name == "sarvam"
    assert adapter.model_name == "saaras:v3"


def test_stt_factory_cloud_google():
    adapter = get_stt_adapter("cloud", "google")
    assert isinstance(adapter, CloudSTTAdapter)
    assert adapter.adapter_type == "cloud"
    assert adapter.provider_name == "google"
    assert adapter.model_name == "chirp_3"


def test_local_adapter_transcribe_structure(dummy_audio):
    adapter = LocalSTTAdapter()
    res = adapter.transcribe(dummy_audio, language="en")
    assert "text" in res
    assert "language" in res
    assert res["adapter"] == "local"
    assert res["provider"] == "whisper+indic_conformer"


def test_sarvam_language_mappings():
    assert LANGUAGE_MAPPING["en"] == "en-IN"
    assert LANGUAGE_MAPPING["en-IN"] == "en-IN"
    assert LANGUAGE_MAPPING["hi"] == "hi-IN"
    assert LANGUAGE_MAPPING["hi-IN"] == "hi-IN"
    assert LANGUAGE_MAPPING["mr"] == "mr-IN"
    assert LANGUAGE_MAPPING["mr-IN"] == "mr-IN"


def test_sarvam_provider_missing_key(dummy_audio):
    # STTProviderError, not ValueError: a missing key is a provider failure
    # like auth or timeout, and api.py/main.py catch that one type to turn any
    # of them into a clear message on the meeting.
    provider = SarvamProvider(api_key="")
    with pytest.raises(STTProviderError, match="SARVAM_API_KEY is not configured"):
        provider.transcribe(dummy_audio)


def test_google_chirp_provider_missing_project(dummy_audio):
    provider = GoogleChirpProvider(project_id="")
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT is not configured"):
        provider.transcribe(dummy_audio)
