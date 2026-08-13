"""
Every STT adapter must actually satisfy stt.base.STTAdapter.

This exists because of a real outage-shaped bug. The adapter layer landed twice
under two naming schemes, adapter_type / provider_name / model_name became
abstract on the base class, and one of the two LocalSTTAdapter copies never
grew them. Nothing failed at import — the class was fine to *define*. It failed
at `LocalSTTAdapter()`, inside the resolver, on the default processing mode,
which meant every meeting died on construction.

None of the existing tests caught it because they each imported whichever copy
their half of the codebase used. These checks are class-level and need no model
weights, no API key, and no network, so they run everywhere and fail loudly the
moment an adapter drifts from the interface again.
"""

from __future__ import annotations

import inspect

import pytest

from stt.base import STTAdapter


def _adapter_classes() -> list[tuple[str, type]]:
    """Every concrete adapter the app can actually construct."""
    from stt.cloud.cloud_adapter import CloudSTTAdapter
    from stt.local.local_adapter import LocalSTTAdapter
    from stt.sarvam_adapter import SarvamSTTAdapter

    return [
        ("stt.local.local_adapter.LocalSTTAdapter", LocalSTTAdapter),
        ("stt.cloud.cloud_adapter.CloudSTTAdapter", CloudSTTAdapter),
        ("stt.sarvam_adapter.SarvamSTTAdapter", SarvamSTTAdapter),
    ]


@pytest.mark.parametrize("name,cls", _adapter_classes(), ids=lambda v: v if isinstance(v, str) else "")
def test_adapter_has_no_unimplemented_abstract_members(name, cls):
    """
    The exact check that was missing. An abstract member left unimplemented
    makes the class uninstantiable, and Python only says so at construction.
    """
    missing = sorted(getattr(cls, "__abstractmethods__", frozenset()))
    assert not missing, f"{name} cannot be instantiated — missing {missing}"


@pytest.mark.parametrize("name,cls", _adapter_classes(), ids=lambda v: v if isinstance(v, str) else "")
def test_adapter_is_registered_as_an_stt_adapter(name, cls):
    assert issubclass(cls, STTAdapter), f"{name} does not implement the STT interface"


@pytest.mark.parametrize("name,cls", _adapter_classes(), ids=lambda v: v if isinstance(v, str) else "")
def test_adapter_accepts_the_declared_call_signature(name, cls):
    """
    pipeline.py calls these by keyword. A adapter that renamed or dropped a
    parameter would only fail once a meeting was already running.
    """
    transcribe = inspect.signature(cls.transcribe).parameters
    assert "hint_language" in transcribe, f"{name}.transcribe lost hint_language"

    with_context = inspect.signature(cls.transcribe_with_context).parameters
    for expected in ("full_audio", "start_sec", "end_sec", "hint_language", "padding_sec"):
        assert expected in with_context, f"{name}.transcribe_with_context lost {expected}"


def test_both_local_adapter_import_paths_are_the_same_class():
    """
    stt/local_adapter.py and stt/local/local_adapter.py are both imported by
    live code — the resolver uses one, the factory the other. Two separate
    implementations behind those two paths is what caused the original break,
    so they must resolve to one object, not merely to similar ones.
    """
    from stt.local.local_adapter import LocalSTTAdapter as FromPackage
    from stt.local_adapter import LocalSTTAdapter as FromModule

    assert FromModule is FromPackage


def test_the_default_processing_mode_can_be_constructed():
    """
    Local is the default. If this raises, no meeting can be processed at all —
    which is precisely how the original bug presented.
    """
    from stt.local.local_adapter import LocalSTTAdapter
    from stt.resolver import resolve_stt_adapter

    adapter = resolve_stt_adapter("local")

    assert isinstance(adapter, LocalSTTAdapter)
    assert adapter.adapter_type == "local"
    assert adapter.provider_name
    assert adapter.model_name
