"""Sarvam AI Cloud STT Provider implementation (saaras:v3)."""

from __future__ import annotations

import io
import logging
import requests
import numpy as np
from scipy.io import wavfile

from config import SAMPLE_RATE, SARVAM_API_KEY, SARVAM_MODEL

logger = logging.getLogger("sarvam_provider")

SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"

# Mappings from app language codes to Sarvam language codes
LANGUAGE_MAPPING = {
    "en": "en-IN",
    "en-IN": "en-IN",
    "hi": "hi-IN",
    "hi-IN": "hi-IN",
    "mr": "mr-IN",
    "mr-IN": "mr-IN",
}

LANGUAGE_NAMES = {
    "en": "English",
    "en-IN": "English",
    "hi": "Hindi",
    "hi-IN": "Hindi",
    "mr": "Marathi",
    "mr-IN": "Marathi",
}


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert float32 mono PCM numpy array to 16-bit PCM WAV bytes."""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clamped * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, int16_audio)
    return buf.getvalue()


class SarvamProvider:
    """Cloud STT Provider implementation for Sarvam AI (saaras:v3)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else SARVAM_API_KEY
        self.model = model if model is not None else SARVAM_MODEL

    @property
    def provider_name(self) -> str:
        return "sarvam"

    @property
    def model_name(self) -> str:
        return self.model

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        hint_language: str | None = None,
    ) -> dict:
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY is not configured. Please set SARVAM_API_KEY in backend/.env"
            )

        if len(audio) == 0:
            return {
                "text": "",
                "language": language or "en",
                "language_name": LANGUAGE_NAMES.get(language or "en", "English"),
                "language_detected": language or "en",
                "language_prob": 1.0,
                "language_fallback": False,
                "provider": self.provider_name,
                "model": self.model_name,
            }

        # For Sarvam saaras:v3, use 'unknown' when auto-detecting language so Sarvam accurately identifies
        # Marathi (mr-IN), Hindi (hi-IN), or English (en-IN) without forcing English Romanization.
        if language in ("mr", "mr-IN", "hi", "hi-IN"):
            sarvam_lang = LANGUAGE_MAPPING.get(language, "unknown")
        else:
            sarvam_lang = "unknown"

        wav_bytes = _audio_to_wav_bytes(audio)
        headers = {"api-subscription-key": self.api_key}
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        data = {
            "model": self.model,
            "language_code": sarvam_lang,
        }

        try:
            logger.info(f"calling Sarvam AI STT (model={self.model}, lang={sarvam_lang})")
            response = requests.post(
                SARVAM_API_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=30,
            )
            response.raise_for_status()
            res_json = response.json()

            transcript_text = res_json.get("transcript", "").strip()
            detected_lang = res_json.get("language_code", "en-IN")
            short_lang = detected_lang.split("-")[0] if detected_lang else "en"

            # Script verification: if transcript contains Devanagari characters, ensure language is mr/hi
            has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in transcript_text)
            if has_devanagari and short_lang not in ("hi", "mr"):
                short_lang = "mr"

            return {
                "text": transcript_text,
                "language": short_lang,
                "language_name": LANGUAGE_NAMES.get(short_lang, "English"),
                "language_detected": short_lang,
                "language_prob": 0.95,
                "language_fallback": False,
                "provider": self.provider_name,
                "model": self.model_name,
            }
        except requests.exceptions.RequestException as exc:
            logger.error(f"Sarvam API call failed: {exc}")
            raise RuntimeError(f"Sarvam STT failed: {exc}") from exc
