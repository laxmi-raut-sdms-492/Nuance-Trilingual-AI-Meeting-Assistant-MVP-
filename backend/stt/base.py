"""Abstract Base Class for Speech-to-Text Adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class STTAdapter(ABC):
    """Common interface for all Speech-to-Text (STT) adapters."""

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """Return adapter type string ('local' or 'cloud')."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (e.g. 'whisper+indic_conformer', 'sarvam', 'google')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return active model identifier."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        hint_language: str | None = None,
    ) -> dict:
        """
        Transcribe audio numpy array.

        Returns dict matching existing ASR structure:
            {
                "text": str,
                "language": str,
                "language_name": str,
                "language_detected": str,
                "language_prob": float,
                "language_fallback": bool,
                "adapter": str,
                "provider": str,
                "model": str,
            }
        """
        pass

    @abstractmethod
    def transcribe_with_context(
        self,
        full_audio: np.ndarray,
        start_sec: float,
        end_sec: float,
        hint_language: str | None = None,
    ) -> dict:
        """Transcribe a subsegment with surrounding audio context when available."""
        pass
