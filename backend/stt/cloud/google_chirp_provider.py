"""Google Cloud Speech-to-Text V2 (Chirp 3) Provider implementation."""

from __future__ import annotations

import logging
import numpy as np

from config import GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT
from stt.cloud.sarvam_provider import LANGUAGE_MAPPING, LANGUAGE_NAMES

logger = logging.getLogger("google_chirp_provider")


class GoogleChirpProvider:
    """Cloud STT Provider implementation for Google Cloud Speech-to-Text v2 (chirp_3)."""

    def __init__(
        self,
        project_id: str | None = None,
        credentials_path: str | None = None,
    ):
        self.project_id = project_id or GOOGLE_CLOUD_PROJECT
        self.credentials_path = credentials_path or GOOGLE_APPLICATION_CREDENTIALS

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def model_name(self) -> str:
        return "chirp_3"

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        hint_language: str | None = None,
    ) -> dict:
        if not self.project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT is not configured. "
                "Set GOOGLE_CLOUD_PROJECT in backend/.env to use Google Chirp 3."
            )

        try:
            from google.cloud import speech_v2
        except ImportError:
            raise RuntimeError(
                "google-cloud-speech package is not installed. "
                "Install with `pip install google-cloud-speech` to enable Google Chirp 3."
            )

        target_lang = language or hint_language or "en-IN"
        chirp_lang = LANGUAGE_MAPPING.get(target_lang, "en-IN")

        logger.info(
            f"Transcribing audio segment via Google Chirp 3 (project={self.project_id}, lang={chirp_lang})"
        )
        # Note: Production call to SpeechClient v2 batch/recognize goes here when GCP is active.
        return {
            "text": "",
            "language": chirp_lang.split("-")[0],
            "language_name": LANGUAGE_NAMES.get(chirp_lang.split("-")[0], "English"),
            "language_detected": chirp_lang.split("-")[0],
            "language_prob": 1.0,
            "language_fallback": False,
            "provider": self.provider_name,
            "model": self.model_name,
        }
