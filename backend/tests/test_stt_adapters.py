"""
Adapter contract tests.

Verifies LocalSTTAdapter and SarvamSTTAdapter both satisfy the shared
STTAdapter contract (backend/stt/base.py) and that pipeline.py's expected
dict shape is never silently changed by either implementation.

No real Sarvam API call is made anywhere in this file — the SDK client's
speech_to_text.transcribe is monkeypatched.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stt.base import STTAdapter, STTProviderError
from stt.local_adapter import LocalSTTAdapter

EXPECTED_KEYS = {
    "text",
    "language",
    "language_name",
    "language_detected",
    "language_prob",
    "language_fallback",
    "language_margin",
    "language_mixed_suspected",
}


def _silence(seconds: float = 1.5, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def _tone(seconds: float = 1.5, sample_rate: int = 16000, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------- local ----


def test_local_adapter_is_stt_adapter():
    assert isinstance(LocalSTTAdapter(), STTAdapter)


def test_local_adapter_transcribe_shape(monkeypatch):
    import models.asr as local_asr

    monkeypatch.setattr(
        local_asr,
        "transcribe",
        lambda audio, hint_language=None: {
            "text": "hello",
            "language": "en",
            "language_name": "English",
            "language_detected": "en",
            "language_prob": 0.9,
            "language_fallback": False,
            "language_margin": 0.8,
            "language_mixed_suspected": False,
        },
    )
    result = LocalSTTAdapter().transcribe(_tone())
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["text"] == "hello"


def test_local_adapter_transcribe_with_context_shape(monkeypatch):
    import models.asr as local_asr

    monkeypatch.setattr(
        local_asr,
        "transcribe_with_context",
        lambda full_audio, start_sec, end_sec, hint_language=None, padding_sec=None: {
            "text": "hi",
            "language": "hi",
            "language_name": "Hindi",
            "language_detected": "hi",
            "language_prob": 0.7,
            "language_fallback": False,
            "language_margin": 0.5,
            "language_mixed_suspected": False,
        },
    )
    result = LocalSTTAdapter().transcribe_with_context(_tone(4.0), 0.0, 2.0)
    assert set(result.keys()) == EXPECTED_KEYS


def test_local_adapter_delegates_hint_language(monkeypatch):
    import models.asr as local_asr

    seen = {}

    def fake_transcribe(audio, hint_language=None):
        seen["hint"] = hint_language
        return {
            "text": "", "language": "en", "language_name": "English",
            "language_detected": "en", "language_prob": 0.1,
            "language_fallback": True, "language_margin": 1.0,
            "language_mixed_suspected": False,
        }

    monkeypatch.setattr(local_asr, "transcribe", fake_transcribe)
    LocalSTTAdapter().transcribe(_silence(), hint_language="mr")
    assert seen["hint"] == "mr"


# --------------------------------------------------------------- sarvam ----


class _FakeSTTResponse:
    def __init__(self, transcript, language_code, language_probability):
        self.transcript = transcript
        self.language_code = language_code
        self.language_probability = language_probability


def _install_fake_sarvamai(monkeypatch, transcribe_fn=None, raise_exc=None):
    """Installs a fake `sarvamai` package into sys.modules so
    SarvamSTTAdapter's lazy `from sarvamai import SarvamAI` picks it up
    without touching the network or requiring the real SDK's internals.
    """

    class _FakeApiError(Exception):
        pass

    class _FakeSTT:
        def transcribe(self, **kwargs):
            if raise_exc is not None:
                raise raise_exc
            return transcribe_fn(**kwargs)

    class _FakeClient:
        def __init__(self, *, api_subscription_key=None, timeout=None):
            self.api_subscription_key = api_subscription_key
            self.speech_to_text = _FakeSTT()

    fake_sarvamai = SimpleNamespace(SarvamAI=_FakeClient)
    fake_core = SimpleNamespace(api_error=SimpleNamespace(ApiError=_FakeApiError))

    monkeypatch.setitem(sys.modules, "sarvamai", fake_sarvamai)
    monkeypatch.setitem(sys.modules, "sarvamai.core", fake_core)
    monkeypatch.setitem(sys.modules, "sarvamai.core.api_error", fake_core.api_error)
    return _FakeApiError


def _adapter(monkeypatch, transcribe_fn=None, raise_exc=None, api_key="test-key"):
    from stt.sarvam_adapter import SarvamSTTAdapter

    _install_fake_sarvamai(monkeypatch, transcribe_fn=transcribe_fn, raise_exc=raise_exc)
    return SarvamSTTAdapter(api_key=api_key)


def test_sarvam_adapter_is_stt_adapter(monkeypatch):
    adapter = _adapter(monkeypatch, transcribe_fn=lambda **kw: _FakeSTTResponse("x", "en-IN", 0.9))
    assert isinstance(adapter, STTAdapter)


def test_sarvam_missing_api_key_raises_at_construction(monkeypatch):
    from stt.sarvam_adapter import SarvamSTTAdapter
    import stt.sarvam_adapter as sarvam_mod

    monkeypatch.setattr(sarvam_mod, "SARVAM_API_KEY", "")
    with pytest.raises(STTProviderError):
        SarvamSTTAdapter(api_key="")


@pytest.mark.parametrize(
    "sarvam_lang,expected",
    [("en-IN", "en"), ("hi-IN", "hi"), ("mr-IN", "mr")],
)
def test_sarvam_transcribe_shape_and_language_mapping(monkeypatch, sarvam_lang, expected):
    adapter = _adapter(
        monkeypatch,
        transcribe_fn=lambda **kw: _FakeSTTResponse("hello world", sarvam_lang, 0.87),
    )
    result = adapter.transcribe(_tone())
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["language"] == expected
    assert result["language_detected"] == expected
    assert result["text"] == "hello world"
    assert result["language_prob"] == pytest.approx(0.87)
    assert result["language_fallback"] is False
    assert result["language_margin"] == 1.0
    assert result["language_mixed_suspected"] is False


def test_sarvam_unsupported_language_falls_back_to_hint(monkeypatch):
    adapter = _adapter(
        monkeypatch,
        transcribe_fn=lambda **kw: _FakeSTTResponse("bonjour", "fr-FR", 0.5),
    )
    result = adapter.transcribe(_tone(), hint_language="hi")
    assert result["language"] == "hi"
    assert result["language_fallback"] is True


def test_sarvam_unsupported_language_no_hint_uses_default(monkeypatch):
    adapter = _adapter(
        monkeypatch,
        transcribe_fn=lambda **kw: _FakeSTTResponse("bonjour", "fr-FR", 0.5),
    )
    result = adapter.transcribe(_tone())
    assert result["language"] == "en"  # config.DEFAULT_LANGUAGE
    assert result["language_fallback"] is True


def test_sarvam_empty_audio_returns_empty_text_no_call(monkeypatch):
    called = {"n": 0}

    def fake(**kw):
        called["n"] += 1
        return _FakeSTTResponse("should not happen", "en-IN", 0.9)

    adapter = _adapter(monkeypatch, transcribe_fn=fake)
    result = adapter.transcribe(np.array([], dtype=np.float32))
    assert result["text"] == ""
    assert called["n"] == 0


def test_sarvam_empty_transcript_response(monkeypatch):
    adapter = _adapter(
        monkeypatch,
        transcribe_fn=lambda **kw: _FakeSTTResponse("", "en-IN", 0.3),
    )
    result = adapter.transcribe(_tone())
    assert result["text"] == ""


def test_sarvam_malformed_response_missing_fields(monkeypatch):
    class _Bare:
        pass

    adapter = _adapter(monkeypatch, transcribe_fn=lambda **kw: _Bare())
    result = adapter.transcribe(_tone())
    # No transcript/language_code/language_probability attrs at all -> safe
    # defaults, not a crash.
    assert result["text"] == ""
    assert result["language"] in {"en", "hi", "mr"}


def test_sarvam_api_error_wrapped(monkeypatch):
    FakeApiError = _install_fake_sarvamai(monkeypatch, raise_exc=None)
    from stt.sarvam_adapter import SarvamSTTAdapter

    adapter = SarvamSTTAdapter(api_key="test-key")

    def boom(**kw):
        raise FakeApiError("401 unauthorized")

    adapter._client.speech_to_text.transcribe = boom

    with pytest.raises(STTProviderError):
        adapter.transcribe(_tone())


def test_sarvam_network_timeout_wrapped(monkeypatch):
    adapter = _adapter(monkeypatch, raise_exc=TimeoutError("timed out"))
    with pytest.raises(STTProviderError):
        adapter.transcribe(_tone())


def test_sarvam_transcribe_with_context_slices_audio(monkeypatch):
    seen = {}

    def fake(**kw):
        seen["called"] = True
        return _FakeSTTResponse("segment text", "mr-IN", 0.8)

    adapter = _adapter(monkeypatch, transcribe_fn=fake)
    full_audio = _tone(10.0)
    result = adapter.transcribe_with_context(full_audio, 2.0, 4.0)
    assert seen["called"]
    assert result["language"] == "mr"
