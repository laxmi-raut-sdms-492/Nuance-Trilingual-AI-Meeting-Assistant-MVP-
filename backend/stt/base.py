"""
Generic STT adapter interface.

This is the seam between MeetingSession (backend/pipeline.py) and whatever
actually turns audio into text — the local Whisper/Indic-Conformer stack
(models/asr.py) or a cloud provider (e.g. Sarvam).

pipeline.py must depend ONLY on this interface. No provider-specific type,
SDK object, or exception may appear in pipeline.py or MeetingSession — see
backend/stt/resolver.py for how a concrete adapter gets chosen per meeting.

Contract (must match models/asr.py exactly — pipeline.py does not know or
care which adapter produced it):

    {
        "text": str,
        "language": str,                 # one of config.ALLOWED_LANGUAGES
        "language_name": str,
        "language_detected": str,
        "language_prob": float,
        "language_fallback": bool,
        "language_margin": float,
        "language_mixed_suspected": bool,
    }

`text` is "" when the segment is judged non-speech / undecodable. Adapters
must never raise for "no speech" or "empty transcript" — that is a normal,
valid result. Adapters MAY raise STTProviderError (or a subclass) for
transport/auth/provider failures; MeetingSession/pipeline.py treats that the
same way it already treats any other per-segment exception (counted as a
failed segment, meeting continues).
"""

from __future__ import annotations

import abc

import numpy as np


class STTProviderError(Exception):
    """Base class for adapter-level failures (auth, network, timeout, bad
    response, unsupported language, ...). Provider-specific exceptions must
    be caught INSIDE the adapter and re-raised as this (or a subclass of
    this) — no SDK-specific exception type may leak past the adapter
    boundary, so pipeline.py never needs to know which provider is in use.
    """


class STTAdapter(abc.ABC):
    """One instance = one meeting's STT strategy. Stateless with respect to
    audio (no per-call caching beyond what an individual provider needs),
    so a single instance is safely reused across every segment of a
    meeting.
    """

    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, hint_language: str | None = None) -> dict:
        """audio: 1-D float32 numpy array, 16kHz mono — one single-speaker
        segment with no surrounding context. Used on the live/streaming path.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transcribe_with_context(
        self,
        full_audio: np.ndarray,
        start_sec: float,
        end_sec: float,
        hint_language: str | None = None,
        padding_sec: float | None = None,
    ) -> dict:
        """Transcribe the [start_sec, end_sec) window of full_audio, optionally
        using padding_sec of surrounding context for a better decode. Used on
        the upload path, where the whole file is already in memory.
        """
        raise NotImplementedError
