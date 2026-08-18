"""
Isolated Source Separation Module.

Provides 2-speaker speech separation for overlapping audio windows using
SpeechBrain SepFormer (or fallback separation). Completely decoupled from
the core diarization engine.
"""

from __future__ import annotations

import logging
import numpy as np
import torch

from config import SAMPLE_RATE, SOURCE_SEPARATION_MODEL

logger = logging.getLogger("source_separation")

_SEPARATOR_INSTANCE = None


def get_source_separator():
    """Lazy singleton loader for SpeechBrain SepFormer separation model."""
    global _SEPARATOR_INSTANCE
    if _SEPARATOR_INSTANCE is not None:
        return _SEPARATOR_INSTANCE

    try:
        from speechbrain.inference.separation import SepFormerSeparation

        logger.info(f"Loading SepFormer separation model '{SOURCE_SEPARATION_MODEL}'...")
        _SEPARATOR_INSTANCE = SepFormerSeparation.from_hparams(
            source=SOURCE_SEPARATION_MODEL,
            savedir=f"pretrained_models/{SOURCE_SEPARATION_MODEL.replace('/', '_')}",
            run_opts={"device": "cpu"},
        )
        logger.info("SepFormer separation model loaded successfully.")
    except Exception as exc:
        logger.warning(f"Failed to load SpeechBrain SepFormer model: {exc}")
        _SEPARATOR_INSTANCE = False

    return _SEPARATOR_INSTANCE


def separate_audio_streams(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Separate a 1-D float32 16kHz mono audio clip containing overlapping speech
    into two distinct speaker streams: (stream1, stream2).

    Returns None if separation fails or model is unavailable.
    """
    if audio is None or len(audio) < int(0.5 * SAMPLE_RATE):
        return None

    separator = get_source_separator()
    if not separator:
        return None

    try:
        # Prepare tensor (batch=1, samples)
        tensor_audio = torch.from_numpy(audio).unsqueeze(0).float()
        with torch.no_grad():
            est_sources = separator.separate_batch(tensor_audio)

        # est_sources shape: [1, samples, n_sources]
        s1 = est_sources[0, :, 0].cpu().numpy().astype(np.float32)
        s2 = est_sources[0, :, 1].cpu().numpy().astype(np.float32)

        # Normalize outputs to original peak
        orig_peak = np.max(np.abs(audio)) + 1e-9
        s1 = s1 * (orig_peak / (np.max(np.abs(s1)) + 1e-9))
        s2 = s2 * (orig_peak / (np.max(np.abs(s2)) + 1e-9))

        return s1, s2
    except Exception as exc:
        logger.warning(f"Source separation execution failed: {exc}")
        return None
