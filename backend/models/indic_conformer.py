"""
Local Hindi/Marathi ASR via AI4Bharat IndicConformer-600M.

Model: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual

First run downloads weights from HuggingFace (~600M params). You must accept
the model license on HuggingFace and set HF_TOKEN if the repo requires auth.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from config import (
    INDIC_CONFORMER_DECODE,
    INDIC_CONFORMER_MODEL,
    INDIC_DECODE_LANGUAGES,
    SAMPLE_RATE,
)

logger = logging.getLogger("indic_conformer")

_model = None
_load_error: Exception | None = None


def _get_model():
    global _model, _load_error

    if _load_error is not None:
        raise _load_error

    if _model is None:
        from transformers import AutoModel

        logger.info(
            f"loading IndicConformer '{INDIC_CONFORMER_MODEL}' "
            f"(first run downloads from HuggingFace — accept the license on the model page)"
        )
        try:
            _model = AutoModel.from_pretrained(
                INDIC_CONFORMER_MODEL,
                trust_remote_code=True,
            )
            _model.eval()
        except Exception as exc:
            _load_error = exc
            logger.error(f"IndicConformer failed to load: {exc}")
            raise
    return _model


def transcribe_indic(audio: np.ndarray, language: str) -> str:
    """
    Transcribe one float32 mono segment at SAMPLE_RATE.

    language must be 'hi' or 'mr' — Indic Conformer does not support English.
    """
    if language not in INDIC_DECODE_LANGUAGES:
        raise ValueError(
            f"IndicConformer supports {INDIC_DECODE_LANGUAGES}, got {language!r}"
        )

    if len(audio) == 0:
        return ""

    decode = INDIC_CONFORMER_DECODE
    if decode not in ("ctc", "rnnt"):
        logger.warning(f"unknown INDIC_CONFORMER_DECODE={decode!r}, using ctc")
        decode = "ctc"

    model = _get_model()
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)

    with torch.no_grad():
        out = model(wav, language, decode)

    if isinstance(out, (list, tuple)):
        text = out[0] if out else ""
    else:
        text = out

    return str(text or "").strip()
