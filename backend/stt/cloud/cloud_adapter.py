"""Cloud STT Adapter wrapping active cloud provider (Sarvam or Google Chirp)."""

from __future__ import annotations

import logging
import numpy as np

from config import CLOUD_STT_PROVIDER
from stt.base import STTAdapter
from stt.cloud.google_chirp_provider import GoogleChirpProvider
from stt.cloud.sarvam_provider import SarvamProvider

logger = logging.getLogger("cloud_stt_adapter")


class CloudSTTAdapter(STTAdapter):
    """
    Adapter 2: Cloud STT.
    
    Wraps provider abstraction (SarvamProvider or GoogleChirpProvider).
    """

    def __init__(self, provider_choice: str | None = None):
        choice = (provider_choice or CLOUD_STT_PROVIDER or "sarvam").lower()
        if choice == "sarvam":
            self.provider = SarvamProvider()
        elif choice == "google":
            self.provider = GoogleChirpProvider()
        else:
            logger.warning(f"Unknown CLOUD_STT_PROVIDER '{choice}', defaulting to Sarvam")
            self.provider = SarvamProvider()

    @property
    def adapter_type(self) -> str:
        return "cloud"

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        hint_language: str | None = None,
    ) -> dict:
        result = self.provider.transcribe(audio, language=language, hint_language=hint_language)
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
        padding_sec: float | None = None,
    ) -> dict:
        # Accepted for interface parity and deliberately unused: a cloud
        # provider transcribes whatever clip it is posted, so there is no
        # windowed decode of a larger buffer for extra context to inform.
        # Local honours it; taking it here keeps every caller identical.
        # For cloud adapters, extract segment audio slice directly
        sr = 16000
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        audio = full_audio[start_idx:end_idx]
        return self.transcribe(audio, hint_language=hint_language)
