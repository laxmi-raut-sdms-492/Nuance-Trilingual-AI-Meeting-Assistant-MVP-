"""
STT adapter resolver.

The one place in the codebase that maps a meeting's (processing_mode,
stt_provider) to a concrete STTAdapter instance. Everything else —
pipeline.py, MeetingSession, api.py, main.py — talks to the result through
the generic stt.base.STTAdapter interface and never imports a concrete
adapter class directly.

Adding a new cloud provider later means:
  1. writing stt/<provider>_adapter.py implementing STTAdapter
  2. adding one entry to _CLOUD_PROVIDERS below
  3. adding its credential(s) to config.py
No change to pipeline.py, MeetingSession, or this function's callers.
"""

from __future__ import annotations

from config import CLOUD_STT_DEFAULT_PROVIDER, DEFAULT_PROCESSING_MODE
from stt.base import STTAdapter, STTProviderError
from stt.local_adapter import LocalSTTAdapter

PROCESSING_MODES = ("local", "cloud")

# provider name -> zero-arg (or default-arg) factory. Registered lazily
# (lambdas, not instances) so importing this module never requires a cloud
# SDK to be installed unless that provider is actually resolved.
_CLOUD_PROVIDERS = {
    "sarvam": lambda: _sarvam_adapter(),
}


def _sarvam_adapter() -> STTAdapter:
    from stt.sarvam_adapter import SarvamSTTAdapter

    return SarvamSTTAdapter()


def resolve_stt_adapter(
    processing_mode: str | None,
    stt_provider: str | None = None,
) -> STTAdapter:
    """
    processing_mode: "local" | "cloud" | None/"" (missing/null -> local,
        matching the DB default and pre-Phase-2 behaviour).
    stt_provider: only consulted when processing_mode == "cloud". None/""
        falls back to config.CLOUD_STT_DEFAULT_PROVIDER ("sarvam" today).

    Raises STTProviderError for cloud + unknown/unregistered provider, or
    for cloud + a provider whose credentials are missing (e.g. Sarvam
    without SARVAM_API_KEY) — callers should treat this the same as any
    other meeting-setup failure (mark the meeting Failed with the message),
    not silently fall back to local, since that would processing the
    meeting under settings the user didn't choose.
    """
    mode = (processing_mode or DEFAULT_PROCESSING_MODE or "local").strip().lower()

    if mode not in PROCESSING_MODES:
        raise STTProviderError(
            f"Unknown processing_mode {mode!r}. Expected one of {PROCESSING_MODES}."
        )

    if mode == "local":
        return LocalSTTAdapter()

    # mode == "cloud"
    provider = (stt_provider or CLOUD_STT_DEFAULT_PROVIDER or "").strip().lower()
    if not provider:
        raise STTProviderError(
            "processing_mode is 'cloud' but no stt_provider was given and "
            "CLOUD_STT_DEFAULT_PROVIDER is not configured."
        )

    factory = _CLOUD_PROVIDERS.get(provider)
    if factory is None:
        raise STTProviderError(
            f"Unknown cloud STT provider {provider!r}. "
            f"Registered providers: {sorted(_CLOUD_PROVIDERS)}."
        )

    return factory()
