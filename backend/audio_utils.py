"""
Audio format helpers.

The frontend sends raw 16-bit PCM, mono, 16kHz audio directly (no container
format like WebM/MP3) — this sidesteps a common real-time-audio pitfall
where chunked container formats can't be decoded independently.
"""

import numpy as np
import torch

from config import SAMPLE_RATE


def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM bytes into a normalized float32 numpy array in [-1, 1]."""
    int16_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    float32_array = int16_array.astype(np.float32) / 32768.0
    return float32_array


def float32_to_tensor(audio: np.ndarray) -> torch.Tensor:
    """Wrap a 1-D float32 numpy array as a (1, N) torch tensor for SpeechBrain."""
    return torch.from_numpy(audio).unsqueeze(0)


def rms(audio: np.ndarray) -> float:
    """Root-mean-square amplitude — a cheap stand-in for 'is there speech here'."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def wav_bytes_to_float32(wav_path: str) -> np.ndarray:
    """Load a WAV file (e.g. an enrollment sample) and resample to SAMPLE_RATE mono."""
    import torchaudio

    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stereo -> mono
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0).numpy()
