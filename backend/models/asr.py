"""
Speech-to-text using OpenAI Whisper, with per-segment language detection
constrained to the three languages this product supports (English, Hindi,
Marathi).

Why detect per segment instead of once per meeting:

  A real trilingual meeting code-switches. One person answers in Marathi, the
  next replies in English, someone quotes a number in Hindi. Detecting once
  for the whole file locks every later segment to whatever language happened
  to be spoken first, and Whisper then transliterates or mistranslates
  everything else into that language's script instead of transcribing it.

Why constrain the candidate set:

  Whisper's `detect_language` scores all ~99 languages it knows. On a 2-4
  second meeting segment — short, noisy, possibly accented — the top guess is
  frequently a language nobody in the room speaks (Urdu and Nepali for Hindi,
  Sanskrit for Marathi are the common ones). Restricting the argmax to
  ALLOWED_LANGUAGES turns a 99-way guess into a 3-way one, which is far more
  reliable at this segment length, and passing the winner explicitly as
  `language=` stops Whisper re-detecting internally.
"""

import logging

import numpy as np

from config import (
    WHISPER_MODEL_SIZE,
    SAMPLE_RATE,
    ALLOWED_LANGUAGES,
    LANGUAGE_NAMES,
    DEFAULT_LANGUAGE,
    LANGUAGE_DETECT_MIN_PROB,
    ASR_MAX_NO_SPEECH_PROB,
    ASR_MIN_AVG_LOGPROB,
    ASR_STANDALONE_NO_SPEECH_PROB,
    ASR_MAX_WORDS_PER_SECOND,
    ASR_DEVANAGARI_LANGUAGES,
    ASR_MIN_DEVANAGARI_RATIO,
)

logger = logging.getLogger("asr")

_model = None
_load_error: Exception | None = None


def _get_model():
    global _model, _load_error

    # A failed load is permanent for this process — the usual cause is the
    # model not fitting in VRAM, which retrying cannot fix. Without this,
    # every segment re-attempts the load and a fatal error takes minutes to
    # surface instead of seconds: `medium` on a 4 GB card spent 376s failing
    # 20 times over before the meeting was finally marked Failed.
    if _load_error is not None:
        raise _load_error

    if _model is None:
        import whisper

        # The English-only variants ("base.en", "small.en", ...) have no
        # language tokens at all, so detect_language raises on them and Hindi
        # or Marathi audio comes back as garbled English. Fail loudly at load
        # time rather than confusingly on the first segment.
        if WHISPER_MODEL_SIZE.endswith(".en"):
            raise ValueError(
                f"WHISPER_MODEL_SIZE is '{WHISPER_MODEL_SIZE}', an English-only model. "
                f"This app transcribes {', '.join(ALLOWED_LANGUAGES)} and needs a "
                f"multilingual model — drop the '.en' suffix."
            )

        logger.info(f"loading Whisper model '{WHISPER_MODEL_SIZE}' (first run downloads it)")
        try:
            _model = whisper.load_model(WHISPER_MODEL_SIZE)
        except Exception as exc:
            _load_error = exc
            logger.error(f"Whisper model '{WHISPER_MODEL_SIZE}' failed to load: {exc}")
            raise
    return _model


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def language_ranking(audio: np.ndarray) -> list[tuple[str, float]]:
    """
    Returns [(language_code, probability), ...] over ALLOWED_LANGUAGES, best
    first. Probabilities are renormalized across only the allowed languages,
    so they answer "which of our three is this" rather than "how confident are
    we against all ninety-nine".

    The full ranking matters, not just the winner: when a Devanagari decode
    comes back in Latin script the runner-up is the natural second attempt.
    """
    import whisper

    model = _get_model()

    clip = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(clip, model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)

    allowed = {code: float(probs.get(code, 0.0)) for code in ALLOWED_LANGUAGES}
    total = sum(allowed.values())
    if total <= 0:
        return [(DEFAULT_LANGUAGE, 0.0)]

    ranked = sorted(
        ((code, round(p / total, 3)) for code, p in allowed.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked


def detect_language(audio: np.ndarray) -> tuple[str, float]:
    """Top-ranked language and its probability. See language_ranking()."""
    return language_ranking(audio)[0]


def transcribe(audio: np.ndarray, hint_language: str | None = None) -> dict:
    """
    audio: 1-D float32 numpy array, 16kHz mono — one single-speaker segment.

    hint_language: the meeting's dominant language so far. Used only as a
    fallback when this segment's own detection is too weak to trust, which
    happens on very short or noisy segments.

    Returns {"text", "language", "language_prob", "language_name",
             "language_detected", "language_fallback"}.
    `text` is "" when the segment is judged to be non-speech.

    `language_prob` is always the detector's confidence in `language_detected`,
    its own top choice. When the fallback fires, `language` differs from
    `language_detected` and `language_fallback` is True — without that flag a
    reader sees something like "en (0.70)" and cannot tell that the 0.70
    belonged to a language which was then thrown away.
    """
    duration = len(audio) / SAMPLE_RATE

    ranked = language_ranking(audio)
    detected, prob = ranked[0]

    language = detected
    used_fallback = prob < LANGUAGE_DETECT_MIN_PROB
    if used_fallback:
        language = hint_language or DEFAULT_LANGUAGE
        logger.info(
            f"language detect near chance ({detected} @ {prob}) -> falling back to {language}"
        )

    text = _decode(audio, language, duration)

    # A Devanagari language that decoded into Latin script means the decode
    # failed, however confident the detector was. Drop it.
    #
    # Retrying with the detector's runner-up language was tried and measured
    # WORSE, so don't reintroduce it. A failed Marathi decode retried as
    # English returned "Khokhla, this is the end of today's episode" — a
    # fluent hallucination that passes every guard below, because English is
    # exactly where Whisper's outro-caption prior lives. Obvious garbage is
    # safer than convincing garbage: a reader spots "comme c.o-pid Jay" as
    # broken immediately and cannot spot the other one at all. The retry also
    # drained Marathi from 28.6% of the meeting to 11.7% by reassigning its
    # segments to whichever language decoded more fluently.
    if text and _is_script_mismatch(text, language):
        logger.info(f"dropping script-mismatched {language} decode: {text!r}")
        text = ""

    return _result(text, language, detected, prob, used_fallback)


def _decode(audio: np.ndarray, language: str, duration: float) -> str:
    """
    One Whisper pass in a fixed language, with the hallucination guards
    applied. Returns "" when the output fails a guard.
    """
    model = _get_model()

    result = model.transcribe(
        audio,
        language=language,
        fp16=False,
        # Each segment here is already an independent, single-speaker piece of
        # audio. Carrying decoder context across them makes Whisper repeat the
        # previous speaker's words into the next segment.
        condition_on_previous_text=False,
    )

    text = result.get("text", "").strip()
    if not text:
        return ""

    reason = _hallucination_reason(result, text, duration)
    if reason:
        logger.info(f"dropping likely hallucination ({reason}): {text!r}")
        return ""

    return text


def _result(text: str, language: str, detected: str, prob: float, used_fallback: bool) -> dict:
    return {
        "text": text,
        "language": language,
        "language_name": language_name(language),
        "language_detected": detected,
        "language_prob": prob,
        "language_fallback": used_fallback,
    }


def _hallucination_reason(result: dict, text: str, duration: float) -> str | None:
    """
    Returns a short reason string when this output looks invented rather than
    transcribed, else None. Three independent tests — see the ASR guard notes
    in config.py for why one signal is not enough.
    """
    # Guard 3 first: it needs no model internals and catches the confident
    # hallucinations that the probability-based guards below miss entirely.
    if duration > 0:
        words_per_second = len(text.split()) / duration
        if words_per_second > ASR_MAX_WORDS_PER_SECOND:
            return f"{words_per_second:.1f} words/sec over {duration:.1f}s"

    segments = result.get("segments") or []
    if not segments:
        return None

    no_speech = max(s.get("no_speech_prob", 0.0) for s in segments)
    avg_logprob = min(s.get("avg_logprob", 0.0) for s in segments)

    if no_speech > ASR_STANDALONE_NO_SPEECH_PROB:
        return f"no_speech_prob={no_speech:.2f}"

    if no_speech > ASR_MAX_NO_SPEECH_PROB and avg_logprob < ASR_MIN_AVG_LOGPROB:
        return f"no_speech_prob={no_speech:.2f}, avg_logprob={avg_logprob:.2f}"

    return None


def _is_script_mismatch(text: str, language: str) -> bool:
    """
    True when a Hindi/Marathi decode produced too little Devanagari to be a
    real transcription. Only letters are counted — digits, punctuation and
    whitespace are script-neutral and would otherwise skew short lines.
    """
    if language not in ASR_DEVANAGARI_LANGUAGES:
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        # No letters at all — "? ?", "...", stray punctuation. Not a
        # transcription in any script, so treat it as a failed decode rather
        # than passing it through as valid text.
        return True

    devanagari = sum(1 for ch in letters if "ऀ" <= ch <= "ॿ")
    return (devanagari / len(letters)) < ASR_MIN_DEVANAGARI_RATIO
