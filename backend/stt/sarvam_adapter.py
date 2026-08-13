"""
SarvamSTTAdapter — the current (and, for now, only) CloudSTTAdapter
implementation.

Every Sarvam-specific detail lives in this one file: the SDK client, its
request/response shapes, its exception types, its audio-encoding
requirements. Nothing outside stt/ may import sarvamai directly. A future
provider (Google/Azure/AWS) is added by writing a sibling module that
satisfies stt.base.STTAdapter and registering it in stt/resolver.py — this
file is not touched and pipeline.py is not touched.

Sarvam's REST speech-to-text endpoint (SDK: sarvamai, client.speech_to_text.
transcribe) takes a short audio clip and returns one best transcript plus a
single detected BCP-47 language code and a confidence — it does not return
ranked alternative languages the way the local Whisper-based detector does.
That means two fields in the shared contract cannot be produced the same way
a cloud call:

  - language_margin: no runner-up language to measure a gap against, so this
    is fixed at 1.0 (maximally confident / no ambiguity signal available).
  - language_mixed_suspected: Sarvam's single-language response can't reveal
    code-switching within one segment the way the local ranked-detection
    signal can. Fixed at False.

Both are documented here rather than silently faked so a caller comparing
local vs. cloud output for the same meeting understands why those two
fields behave differently.
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np

from config import (
    ALLOWED_LANGUAGES,
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    SAMPLE_RATE,
    SARVAM_API_KEY,
    SARVAM_STT_MODEL,
    SARVAM_STT_MODE,
    SARVAM_TIMEOUT_SECONDS,
)
from stt.base import STTAdapter, STTProviderError

logger = logging.getLogger("stt.sarvam")

# Our 2-letter codes -> Sarvam's BCP-47 codes, and back. Only the three
# languages this app supports are mapped; Sarvam supports many more, but a
# code outside ALLOWED_LANGUAGES is not something the rest of the pipeline
# (or the UI) knows how to render.
_TO_SARVAM_LANGUAGE = {"en": "en-IN", "hi": "hi-IN", "mr": "mr-IN"}
_FROM_SARVAM_LANGUAGE = {v: k for k, v in _TO_SARVAM_LANGUAGE.items()}


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """1-D float32 [-1, 1] mono -> 16-bit PCM WAV bytes. Sarvam's REST
    endpoint takes a file; wav/pcm_s16le is the least lossy, best-supported
    option in input_audio_codec and needs no extra encoder dependency
    (stdlib `wave` only).
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class SarvamSTTAdapter(STTAdapter):
    """One instance per meeting/process; the underlying SDK client is safe
    to reuse across calls, so it is created once in __init__.
    """

    def __init__(self, api_key: str | None = None):
        api_key = api_key or SARVAM_API_KEY
        if not api_key:
            # Fail at construction, not on the first segment — the resolver
            # calls this eagerly so a meeting created with cloud+no-key fails
            # fast with a clear message instead of silently losing every
            # segment 400 requests in.
            raise STTProviderError(
                "SARVAM_API_KEY is not configured. Set it in the environment "
                "to use cloud (Sarvam) processing, or use processing_mode=local."
            )

        # Imported lazily so importing stt.sarvam_adapter (and therefore
        # stt.resolver) never requires the sarvamai package to be installed
        # unless cloud mode is actually used.
        try:
            from sarvamai import SarvamAI
        except (ImportError, ModuleNotFoundError) as err:
            raise STTProviderError(
                "The 'sarvamai' Python package is not installed in the backend environment. "
                "Run 'pip install sarvamai' or 'pip install -r requirements.txt'."
            ) from err

        self._client = SarvamAI(api_subscription_key=api_key, timeout=SARVAM_TIMEOUT_SECONDS)

    # -- provenance (stt.base.STTAdapter) --
    #
    # Abstract on the base class, so omitting these makes the adapter
    # impossible to instantiate rather than merely under-described. They are
    # also what stamps each transcript line with where its text came from,
    # which is the whole point of being able to switch providers per meeting.

    @property
    def adapter_type(self) -> str:
        return "cloud"

    @property
    def provider_name(self) -> str:
        return "sarvam"

    @property
    def model_name(self) -> str:
        return SARVAM_STT_MODEL

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        hint_language: str | None = None,
    ) -> dict:
        return self._transcribe_clip(audio, hint_language=language or hint_language)

    def transcribe_with_context(
        self,
        full_audio: np.ndarray,
        start_sec: float,
        end_sec: float,
        hint_language: str | None = None,
        padding_sec: float | None = None,
    ) -> dict:
        # Sarvam's REST endpoint has no timestamp-windowed decode of a larger
        # buffer the way local Indic Conformer/Whisper context-padding does;
        # it transcribes whatever clip it is given. So the "context" here is
        # just: slice out the segment and send exactly that.
        s0 = max(0, int(start_sec * SAMPLE_RATE))
        s1 = min(len(full_audio), int(end_sec * SAMPLE_RATE))
        return self._transcribe_clip(full_audio[s0:s1], hint_language=hint_language)

    # -- internal --

    def _stamp(self, result: dict) -> dict:
        """Record which engine produced this text — part of the shared contract."""
        result["adapter"] = self.adapter_type
        result["provider"] = self.provider_name
        result["model"] = self.model_name
        return result

    def _transcribe_clip(self, audio: np.ndarray, hint_language: str | None) -> dict:
        if audio is None or len(audio) == 0:
            return self._stamp(self._empty_result(hint_language))

        wav_bytes = _float32_to_wav_bytes(audio)

        # `unknown` lets Sarvam auto-detect. hint_language is the meeting's
        # running dominant language (see pipeline.py); it is only a hint for
        # local's own weak-detection fallback and is deliberately NOT forced
        # here — locking every segment to the first-detected language is
        # exactly the code-switching failure mode this app exists to avoid.
        language_code = "unknown"

        from sarvamai.core.api_error import ApiError

        try:
            response = self._client.speech_to_text.transcribe(
                file=("segment.wav", wav_bytes, "audio/wav"),
                model=SARVAM_STT_MODEL,
                mode=SARVAM_STT_MODE,
                language_code=language_code,
                with_timestamps=False,
                input_audio_codec="wav",
            )
        except ApiError as exc:
            raise STTProviderError(f"Sarvam API error ({exc})") from exc
        except Exception as exc:  # noqa: BLE001 - network/timeout/SDK internals
            raise STTProviderError(f"Sarvam request failed: {exc}") from exc

        return self._stamp(self._to_contract(response, hint_language))

    def _to_contract(self, response, hint_language: str | None) -> dict:
        text = (getattr(response, "transcript", None) or "").strip()

        raw_lang = getattr(response, "language_code", None)
        detected = _FROM_SARVAM_LANGUAGE.get(raw_lang or "", None)

        prob = getattr(response, "language_probability", None)
        prob = float(prob) if prob is not None else (1.0 if detected else 0.0)

        used_fallback = detected is None
        if detected is None:
            # Sarvam returned a language outside en/hi/mr (or none at all —
            # e.g. silence). Fall back to the meeting's running language,
            # same policy the local adapter uses, else the configured default.
            language = hint_language if hint_language in ALLOWED_LANGUAGES else DEFAULT_LANGUAGE
            detected = language
            if raw_lang:
                logger.info(f"Sarvam returned unsupported language {raw_lang!r} -> {language}")
        else:
            language = detected

        return {
            "text": text,
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language),
            "language_detected": detected,
            "language_prob": prob,
            "language_fallback": used_fallback,
            # See module docstring — Sarvam gives one answer, not a ranking.
            "language_margin": 1.0,
            "language_mixed_suspected": False,
        }

    def _empty_result(self, hint_language: str | None) -> dict:
        language = hint_language if hint_language in ALLOWED_LANGUAGES else DEFAULT_LANGUAGE
        return {
            "text": "",
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language),
            "language_detected": language,
            "language_prob": 0.0,
            "language_fallback": True,
            "language_margin": 1.0,
            "language_mixed_suspected": False,
        }
