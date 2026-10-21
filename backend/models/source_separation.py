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
        import torchaudio.functional as AF

        # The default SepFormer (wsj02mix) is an 8 kHz model. Fed 16 kHz
        # audio as if it were 8 kHz it hears everything an octave down and
        # separates badly, so the clip is resampled to the model's rate and
        # the streams brought back to ours.
        model_sr = int(getattr(getattr(separator, "hparams", None), "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        tensor_audio = torch.from_numpy(audio).unsqueeze(0).float()
        if model_sr != SAMPLE_RATE:
            tensor_audio = AF.resample(tensor_audio, SAMPLE_RATE, model_sr)
        with torch.no_grad():
            est_sources = separator.separate_batch(tensor_audio)

        # est_sources shape: [1, samples, n_sources]
        streams = est_sources[0].T  # [n_sources, samples]
        if model_sr != SAMPLE_RATE:
            streams = AF.resample(streams, model_sr, SAMPLE_RATE)
        streams = streams[:, : len(audio)]
        if streams.shape[1] < len(audio):
            streams = torch.nn.functional.pad(streams, (0, len(audio) - streams.shape[1]))
        s1 = streams[0].cpu().numpy().astype(np.float32)
        s2 = streams[1].cpu().numpy().astype(np.float32)

        # Normalize outputs to original peak
        orig_peak = np.max(np.abs(audio)) + 1e-9
        s1 = s1 * (orig_peak / (np.max(np.abs(s1)) + 1e-9))
        s2 = s2 * (orig_peak / (np.max(np.abs(s2)) + 1e-9))

        return s1, s2
    except Exception as exc:
        logger.warning(f"Source separation execution failed: {exc}")
        return None
