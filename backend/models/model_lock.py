"""
One lock around every call into a shared model.

The VAD, ECAPA, Whisper and Indic Conformer models are loaded once and
shared by every meeting. None of them is safe to run from two threads at
once: Whisper installs forward hooks on its decoder per call, Silero keeps
its recurrent state inside the model, and SpeechBrain's encoder is plain
PyTorch with no guard. An upload transcribing in the background while a
live meeting streams in hit all three, with garbled text, shape errors and
VAD state wiped mid-utterance.

Re-entrant, so a model call that uses another model (transcribe -> language
ranking) does not deadlock on itself.
"""

from __future__ import annotations

import functools
import threading

MODEL_LOCK = threading.RLock()


def serialized(func):
    """Run func while holding MODEL_LOCK."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with MODEL_LOCK:
            return func(*args, **kwargs)

    return wrapper
