"""Local STT Adapter wrapping Whisper Medium (English) & AI4Bharat IndicConformer (Hindi/Marathi)."""

from __future__ import annotations

import logging
import numpy as np

from models.asr import transcribe as local_transcribe, transcribe_with_context as local_transcribe_with_context
from stt.base import STTAdapter

logger = logging.getLogger("local_stt_adapter")


class LocalSTTAdapter(STTAdapter):
    """
    Adapter 1: Local STT.
    
    Routes:
        English (en / en-IN) -> Whisper Medium
        Hindi (hi / hi-IN)   -> AI4Bharat IndicConformer-600M
        Marathi (mr / mr-IN) -> AI4Bharat IndicConformer-600M
    """

    @property
    def adapter_type(self) -> str:
        return "local"

    @property
    def provider_name(self) -> str:
        return "whisper+indic_conformer"

    @property
    def model_name(self) -> str:
        return "whisper-medium / indic-conformer-600m"

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        hint_language: str | None = None,
    ) -> dict:
        target_hint = language or hint_language
        result = local_transcribe(audio, hint_language=target_hint)
        result["adapter"] = self.adapter_type
        result["provider"] = self.provider_name
        result["model"] = self.model_name
        return result

    def transcribe_with_context(
        self,
        full_audio: np.ndarray,
        start_sec: float,
        end_sec: float,
        hint_language: str | None = None,
    ) -> dict:
        result = local_transcribe_with_context(
            full_audio, start_sec, end_sec, hint_language=hint_language
        )
        result["adapter"] = self.adapter_type
        result["provider"] = self.provider_name
        result["model"] = self.model_name
        return result
