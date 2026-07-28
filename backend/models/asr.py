"""Speech-to-text using OpenAI Whisper."""

import numpy as np

from config import WHISPER_MODEL_SIZE

_model = None


def _get_model():
    global _model
    if _model is None:
        import whisper

        _model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _model


def transcribe(audio: np.ndarray) -> str:
    """audio: 1-D float32 numpy array, 16kHz mono. Returns the transcribed text."""
    model = _get_model()
    result = model.transcribe(audio, fp16=False)
    return result.get("text", "").strip()
