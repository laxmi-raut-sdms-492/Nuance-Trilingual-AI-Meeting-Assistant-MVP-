"""
LocalSTTAdapter — wraps the existing local ASR pipeline (models/asr.py)
behind the generic STTAdapter interface.

This introduces a seam and nothing else. It does not change models/asr.py,
does not change the decode strategy, does not change the returned dict
shape. Local processing_mode meetings must behave byte-for-byte the same as
before this adapter layer existed.
"""

from __future__ import annotations

import numpy as np

from models import asr as _local_asr
from stt.base import STTAdapter


class LocalSTTAdapter(STTAdapter):
    """Delegates every call straight through to models/asr.py. No new logic,
    no new state — this class exists purely so pipeline.py can hold a
    reference to "the STT adapter for this meeting" without caring whether
    it is local or cloud.
    """

    def transcribe(self, audio: np.ndarray, hint_language: str | None = None) -> dict:
        return _local_asr.transcribe(audio, hint_language=hint_language)

    def transcribe_with_context(
        self,
        full_audio: np.ndarray,
        start_sec: float,
        end_sec: float,
        hint_language: str | None = None,
        padding_sec: float | None = None,
    ) -> dict:
        return _local_asr.transcribe_with_context(
            full_audio,
            start_sec,
            end_sec,
            hint_language=hint_language,
            padding_sec=padding_sec,
        )
