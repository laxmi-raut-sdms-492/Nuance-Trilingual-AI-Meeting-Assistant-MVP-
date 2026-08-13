"""
Spoken language identification — a cheap per-window language posterior.

Whisper's detect_language is the accurate detector and this does not replace
it. Whisper's encoder always consumes a padded 30s mel, so asking it "which
language is this 2 second window" costs the same as transcribing eight
seconds. Sliding it across a segment to find a code-switch — a dozen forward
passes for one segment — is not affordable.

This is the cheap ear that makes that search possible: SpeechBrain's
VoxLingua107 ECAPA-TDNN, the same architecture already loaded in
models/embedding.py for speaker vectors, run on the window directly.

It is deliberately trusted for one thing only. lcd.py uses it to decide WHERE
the language changed; the languages themselves are re-decided per resulting
piece by asr.language_ranking(). VoxLingua107 is known to confuse closely
related Indic languages, so its Hindi-vs-Marathi call is not reliable — but
its English-vs-Devanagari call, which is all a boundary search needs, is.

Posteriors are restricted and renormalized over ALLOWED_LANGUAGES, exactly as
asr.language_ranking does, so a 107-way guess becomes a 3-way one.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from config import ALLOWED_LANGUAGES, DEFAULT_LANGUAGE, LID_MODEL_SOURCE, SAMPLE_RATE

logger = logging.getLogger("lid")

_classifier = None
_code_to_index: dict[str, int] | None = None
_load_error: Exception | None = None


def _get_classifier():
    """
    Lazy singleton, and a load failure is permanent for the process.

    Same reasoning as asr._get_model: the usual cause is the weights not
    downloading or not fitting, which retrying cannot fix, and retrying once
    per window would turn a clear failure into a very slow one.
    """
    global _classifier, _code_to_index, _load_error

    if _load_error is not None:
        raise _load_error

    if _classifier is None:
        logger.info(f"loading language-ID model '{LID_MODEL_SOURCE}' (first run downloads it)")
        try:
            # Imported inside the try: a missing speechbrain is a load failure
            # like any other and must be recorded as permanent. Python does not
            # cache failed imports, so leaving it outside would re-walk the
            # import path once per segment forever.
            from speechbrain.inference.classifiers import EncoderClassifier

            _classifier = EncoderClassifier.from_hparams(
                source=LID_MODEL_SOURCE,
                savedir="pretrained_models/lang-id-voxlingua107-ecapa",
            )
            _code_to_index = _build_code_index(_classifier)
        except Exception as exc:
            _load_error = exc
            logger.error(f"language-ID model failed to load: {exc}")
            raise

        if not _code_to_index:
            _load_error = RuntimeError(
                f"'{LID_MODEL_SOURCE}' exposes none of {ALLOWED_LANGUAGES} in its "
                f"label set — wrong model for this pipeline."
            )
            logger.error(str(_load_error))
            raise _load_error

        missing = set(ALLOWED_LANGUAGES) - set(_code_to_index)
        if missing:
            # Not fatal: the remaining languages still separate. But a missing
            # language can never win a window, which silently biases every
            # boundary, so it must not pass unremarked.
            logger.warning(
                f"language-ID model has no label for {sorted(missing)} — those "
                f"languages cannot be detected by the boundary search"
            )

    return _classifier


def _build_code_index(classifier) -> dict[str, int]:
    """
    Map 'en'/'hi'/'mr' to their index in the model's output layer.

    VoxLingua107 labels look like 'en: English', so the code is the part
    before the colon. Read once at load rather than per call.
    """
    encoder = getattr(getattr(classifier, "hparams", None), "label_encoder", None)
    ind2lab = getattr(encoder, "ind2lab", None) if encoder is not None else None
    if not ind2lab:
        return {}

    mapping: dict[str, int] = {}
    for index, label in dict(ind2lab).items():
        code = str(label).split(":")[0].strip().lower()
        if code in ALLOWED_LANGUAGES:
            mapping[code] = int(index)
    return mapping


def language_posterior(audio: np.ndarray) -> dict[str, float]:
    """
    audio: 1-D float32, 16kHz mono — typically one short analysis window.

    Returns {code: probability} over ALLOWED_LANGUAGES, renormalized so the
    values answer "which of our three is this" rather than "how sure are we
    against all one hundred and seven". An all-zero or empty input returns a
    flat distribution rather than raising — an uninformative window is a
    normal event in a boundary search, not an error.
    """
    if audio is None or len(audio) == 0:
        return _uniform()

    classifier = _get_classifier()
    assert _code_to_index is not None  # set by _get_classifier on success

    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        out_prob, _score, _index, _text_lab = classifier.classify_batch(tensor)

    # SpeechBrain classifiers emit log-probabilities.
    probs = torch.exp(out_prob).squeeze().cpu().numpy().reshape(-1)

    raw = {
        code: float(probs[index])
        for code, index in _code_to_index.items()
        if index < len(probs)
    }
    total = sum(raw.values())
    if total <= 0:
        return _uniform()
    return {code: value / total for code, value in raw.items()}


def top_language(audio: np.ndarray) -> tuple[str, float]:
    """Most likely of ALLOWED_LANGUAGES for this window, and its probability."""
    posterior = language_posterior(audio)
    if not posterior:
        return DEFAULT_LANGUAGE, 0.0
    code = max(posterior, key=posterior.get)
    return code, posterior[code]


def is_available() -> bool:
    """
    True when the model can be used. Never raises.

    lcd.py calls this to decide whether a boundary search is possible at all;
    when it is not, segments are left unsplit and behaviour is exactly what it
    was before language change detection existed.
    """
    try:
        _get_classifier()
        return True
    except Exception:
        return False


def _uniform() -> dict[str, float]:
    share = 1.0 / len(ALLOWED_LANGUAGES)
    return {code: share for code in ALLOWED_LANGUAGES}


def window_samples(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE)
