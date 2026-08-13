"""
STT adapter resolver tests. No real Sarvam API calls — cloud-resolution
tests only check that the resolver hands back the right class / raises the
right error; SarvamSTTAdapter's own network behaviour is covered in
test_stt_adapters.py with a fully faked SDK.
"""

from __future__ import annotations

import pytest

from stt.base import STTProviderError
from stt.local_adapter import LocalSTTAdapter
from stt.resolver import resolve_stt_adapter


def test_local_mode_returns_local_adapter():
    adapter = resolve_stt_adapter("local")
    assert isinstance(adapter, LocalSTTAdapter)


def test_missing_mode_defaults_to_local():
    assert isinstance(resolve_stt_adapter(None), LocalSTTAdapter)
    assert isinstance(resolve_stt_adapter(""), LocalSTTAdapter)


def test_unknown_mode_raises():
    with pytest.raises(STTProviderError):
        resolve_stt_adapter("quantum")


def test_cloud_unknown_provider_raises():
    with pytest.raises(STTProviderError):
        resolve_stt_adapter("cloud", "not-a-real-provider")


def test_cloud_sarvam_missing_api_key_raises(monkeypatch):
    import stt.sarvam_adapter as sarvam_mod

    monkeypatch.setattr(sarvam_mod, "SARVAM_API_KEY", "")
    with pytest.raises(STTProviderError):
        resolve_stt_adapter("cloud", "sarvam")


def test_cloud_sarvam_resolves_with_api_key(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    import stt.sarvam_adapter as sarvam_mod
    from stt.sarvam_adapter import SarvamSTTAdapter

    monkeypatch.setattr(sarvam_mod, "SARVAM_API_KEY", "test-key")

    # Avoid importing the real sarvamai internals for this resolution check.
    class _Fake:
        def __init__(self, *a, **kw):
            pass

    mock_module = sys.modules.get("sarvamai", MagicMock())
    mock_module.SarvamAI = _Fake
    monkeypatch.setitem(sys.modules, "sarvamai", mock_module)

    adapter = resolve_stt_adapter("cloud", "sarvam")
    assert isinstance(adapter, SarvamSTTAdapter)


def test_cloud_missing_provider_falls_back_to_configured_default(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    import stt.resolver as resolver_mod
    import stt.sarvam_adapter as sarvam_mod

    monkeypatch.setattr(resolver_mod, "CLOUD_STT_DEFAULT_PROVIDER", "sarvam")
    monkeypatch.setattr(sarvam_mod, "SARVAM_API_KEY", "test-key")

    class _Fake:
        def __init__(self, *a, **kw):
            pass

    mock_module = sys.modules.get("sarvamai", MagicMock())
    mock_module.SarvamAI = _Fake
    monkeypatch.setitem(sys.modules, "sarvamai", mock_module)

    from stt.sarvam_adapter import SarvamSTTAdapter

    adapter = resolve_stt_adapter("cloud", None)
    assert isinstance(adapter, SarvamSTTAdapter)


def test_case_and_whitespace_insensitive():
    assert isinstance(resolve_stt_adapter("  LOCAL  "), LocalSTTAdapter)
