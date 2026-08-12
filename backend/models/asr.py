"""
Hybrid speech-to-text for English, Hindi, and Marathi.

  - Language detection: Whisper (restricted to en / hi / mr per segment)
  - English transcription: Whisper medium (local)
  - Hindi / Marathi transcription: AI4Bharat IndicConformer-600M (local,
    HuggingFace — see models/indic_conformer.py)

Why detect per segment instead of once per meeting:

  A real trilingual meeting code-switches. One person answers in Marathi, the
  next replies in English, someone quotes a number in Hindi. Detecting once
  for the whole file locks every later segment to whatever language happened
  to be spoken first.

Why constrain the candidate set:

  Whisper's `detect_language` scores all ~99 languages it knows. Restricting
  the argmax to ALLOWED_LANGUAGES turns a 99-way guess into a 3-way one.
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
    LANGUAGE_DETECT_WEAK_FLOOR,
    ASR_MAX_NO_SPEECH_PROB,
    ASR_MIN_AVG_LOGPROB,
    ASR_WHISPER_NO_SPEECH_THRESHOLD,
    ASR_WHISPER_LOGPROB_THRESHOLD,
    ASR_UNCERTAIN_LOGPROB_CUTOFF,
    ASR_STANDALONE_NO_SPEECH_PROB,
    ASR_MAX_WORDS_PER_SECOND,
    ASR_DEVANAGARI_LANGUAGES,
    ASR_MIN_DEVANAGARI_RATIO,
    ASR_CONTEXT_PADDING_SEC,
    ASR_BEAM_SIZE,
    INDIC_CONFORMER_ENABLED,
    INDIC_DECODE_LANGUAGES,
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
        if hint_language:
            language = hint_language
        elif prob >= LANGUAGE_DETECT_WEAK_FLOOR:
            # No meeting context yet — a weak 3-way guess beats defaulting to English.
            language = detected
        else:
            language = DEFAULT_LANGUAGE
        logger.info(
            f"language detect near chance ({detected} @ {prob}) -> using {language}"
        )

    text = _decode(audio, language, duration)

    # A Devanagari language that decoded into Latin script means the decode failed.
    # Retry with the other Devanagari language (hi↔mr) before dropping.
    if text and _is_script_mismatch(text, language):
        if language in ASR_DEVANAGARI_LANGUAGES:
            alt = "mr" if language == "hi" else "hi"
            alt_text = _decode(audio, alt, duration)
            if alt_text and not _is_script_mismatch(alt_text, alt):
                logger.info(f"script-mismatch retry {language}->{alt} succeeded")
                text = alt_text
                language = alt
            else:
                logger.info(f"dropping script-mismatched {language} decode: {text!r}")
                text = ""
        else:
            logger.info(f"dropping script-mismatched {language} decode: {text!r}")
            text = ""

    return _result(text, language, detected, prob, used_fallback)


def transcribe_with_context(
    full_audio: np.ndarray,
    start_sec: float,
    end_sec: float,
    hint_language: str | None = None,
    padding_sec: float | None = None,
) -> dict:
    """
    Transcribe a time range. English uses Whisper with padded context; Hindi and
    Marathi use Indic Conformer on the exact segment (no timestamp trimming).
    """
    pad = ASR_CONTEXT_PADDING_SEC if padding_sec is None else padding_sec
    seg_duration = end_sec - start_sec
    total_sec = len(full_audio) / SAMPLE_RATE

    if seg_duration >= 10.0 or total_sec <= 0:
        s0 = max(0, int(start_sec * SAMPLE_RATE))
        s1 = min(len(full_audio), int(end_sec * SAMPLE_RATE))
        return transcribe(full_audio[s0:s1], hint_language=hint_language)

    ranked = language_ranking(
        full_audio[
            int(max(0.0, start_sec - (ASR_CONTEXT_PADDING_SEC if padding_sec is None else padding_sec)) * SAMPLE_RATE) :
            int(min(total_sec, end_sec + (ASR_CONTEXT_PADDING_SEC if padding_sec is None else padding_sec)) * SAMPLE_RATE)
        ]
    )
    detected, prob = ranked[0]
    language = detected
    used_fallback = prob < LANGUAGE_DETECT_MIN_PROB
    if used_fallback:
        if hint_language:
            language = hint_language
        elif prob >= LANGUAGE_DETECT_WEAK_FLOOR:
            language = detected
        else:
            language = DEFAULT_LANGUAGE

    s0 = max(0, int(start_sec * SAMPLE_RATE))
    s1 = min(len(full_audio), int(end_sec * SAMPLE_RATE))
    seg_audio = full_audio[s0:s1]

    if _uses_indic_conformer(language):
        text = _decode_indic(seg_audio, language, seg_duration)
    else:
        ctx_start = max(0.0, start_sec - pad)
        ctx_end = min(total_sec, end_sec + pad)
        clip = full_audio[int(ctx_start * SAMPLE_RATE) : int(ctx_end * SAMPLE_RATE)]
        clip_duration = len(clip) / SAMPLE_RATE
        text = _decode_whisper(
            clip,
            language,
            clip_duration,
            abs_start=start_sec,
            abs_end=end_sec,
            clip_start=ctx_start,
        )

    if text and _is_script_mismatch(text, language):
        if language in ASR_DEVANAGARI_LANGUAGES:
            alt = "mr" if language == "hi" else "hi"
            if _uses_indic_conformer(alt):
                alt_text = _decode_indic(seg_audio, alt, seg_duration)
            else:
                ctx_start = max(0.0, start_sec - pad)
                ctx_end = min(total_sec, end_sec + pad)
                clip = full_audio[int(ctx_start * SAMPLE_RATE) : int(ctx_end * SAMPLE_RATE)]
                clip_duration = len(clip) / SAMPLE_RATE
                alt_text = _decode_whisper(
                    clip, alt, clip_duration,
                    abs_start=start_sec, abs_end=end_sec, clip_start=ctx_start,
                )
            if alt_text and not _is_script_mismatch(alt_text, alt):
                text = alt_text
                language = alt
            else:
                text = ""
        else:
            text = ""

    return _result(text, language, detected, prob, used_fallback)


def _text_for_time_range(
    result: dict,
    abs_start: float,
    abs_end: float,
    clip_start: float,
) -> str:
    """Keep only Whisper segment text overlapping the target time window."""
    parts: list[str] = []
    for seg in result.get("segments") or []:
        seg_start = clip_start + float(seg.get("start", 0))
        seg_end = clip_start + float(seg.get("end", 0))
        if seg_end > abs_start + 0.05 and seg_start < abs_end - 0.05:
            chunk = (seg.get("text") or "").strip()
            if chunk:
                parts.append(chunk)
    if parts:
        return " ".join(parts).strip()
    return (result.get("text") or "").strip()


def _uses_indic_conformer(language: str) -> bool:
    return INDIC_CONFORMER_ENABLED and language in INDIC_DECODE_LANGUAGES


def _decode(
    audio: np.ndarray,
    language: str,
    duration: float,
    *,
    abs_start: float | None = None,
    abs_end: float | None = None,
    clip_start: float = 0.0,
) -> str:
    """Route to Whisper (en) or Indic Conformer (hi/mr)."""
    if _uses_indic_conformer(language):
        return _decode_indic(audio, language, duration)
    return _decode_whisper(
        audio, language, duration,
        abs_start=abs_start, abs_end=abs_end, clip_start=clip_start,
    )


def _decode_indic(audio: np.ndarray, language: str, duration: float) -> str:
    """Indic Conformer pass for Hindi/Marathi."""
    try:
        from models.indic_conformer import transcribe_indic

        text = transcribe_indic(audio, language)
    except Exception as exc:
        logger.warning(
            f"IndicConformer unavailable for {language} ({exc}); falling back to Whisper"
        )
        return _decode_whisper(audio, language, duration)

    if not text:
        return ""

    reason = _simple_hallucination_reason(text, duration)
    if reason:
        logger.info(f"dropping likely hallucination ({reason}): {text!r}")
        return ""

    if _is_script_mismatch(text, language):
        logger.info(f"dropping script-mismatched {language} decode: {text!r}")
        return ""

    return text


def _decode_whisper(
    audio: np.ndarray,
    language: str,
    duration: float,
    *,
    abs_start: float | None = None,
    abs_end: float | None = None,
    clip_start: float = 0.0,
) -> str:
    """
    One Whisper pass in a fixed language, with the hallucination guards
    applied. Returns "" when the output fails a guard.
    """
    model = _get_model()

    decode_kwargs: dict = {
        "language": language,
        "fp16": False,
        "condition_on_previous_text": False,
        "no_speech_threshold": ASR_WHISPER_NO_SPEECH_THRESHOLD,
        "logprob_threshold": ASR_WHISPER_LOGPROB_THRESHOLD,
    }
    if language in ASR_DEVANAGARI_LANGUAGES and ASR_BEAM_SIZE > 1:
        decode_kwargs["beam_size"] = ASR_BEAM_SIZE

    result = model.transcribe(audio, **decode_kwargs)

    if abs_start is not None and abs_end is not None:
        text = _text_for_time_range(result, abs_start, abs_end, clip_start)
        guard_duration = abs_end - abs_start
    else:
        text = result.get("text", "").strip()
        guard_duration = duration
    if not text:
        return ""

    reason = _hallucination_reason(result, text, guard_duration)
    if reason:
        logger.info(f"dropping likely hallucination ({reason}): {text!r}")
        return ""

    return text


def _simple_hallucination_reason(text: str, duration: float) -> str | None:
    """Word-rate guard for engines that don't expose Whisper segment scores."""
    if duration > 0:
        words_per_second = len(text.split()) / duration
        if words_per_second > ASR_MAX_WORDS_PER_SECOND:
            return f"{words_per_second:.1f} words/sec over {duration:.1f}s"
    return None


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

    # Guard 2 — confident decode despite overwhelming non-speech evidence.
    # A low avg_logprob means the model struggled; keep that rather than treat
    # high no_speech_prob alone as silence (measured on quiet real speech at
    # no_speech_prob=0.94, avg_logprob=-1.66).
    if no_speech > ASR_STANDALONE_NO_SPEECH_PROB and avg_logprob > ASR_MIN_AVG_LOGPROB:
        return f"no_speech_prob={no_speech:.2f}, avg_logprob={avg_logprob:.2f}"

    # Guard 1 — moderate non-speech evidence plus a struggling decode, but not
    # when the decode is extremely uncertain (likely quiet real speech).
    if (
        no_speech > ASR_MAX_NO_SPEECH_PROB
        and avg_logprob < ASR_MIN_AVG_LOGPROB
        and avg_logprob > ASR_UNCERTAIN_LOGPROB_CUTOFF
    ):
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
