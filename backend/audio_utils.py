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


def load_audio_file(path: str) -> np.ndarray:
    """
    Decode any uploaded recording to 1-D float32, 16kHz mono.

    Uses Whisper's ffmpeg-backed loader rather than torchaudio because the
    frontend can hand us containers torchaudio won't open — browser
    MediaRecorder produces .webm/Opus, and meeting exports are often .mp4 or
    .m4a. ffmpeg handles all of them identically, and it is already a hard
    dependency of Whisper, so this adds nothing new to install.
    """
    import whisper

    return whisper.load_audio(path, sr=SAMPLE_RATE)


def wav_bytes_to_float32(wav_path: str) -> np.ndarray:
    """Load a WAV file (e.g. an enrollment sample) and resample to SAMPLE_RATE mono."""
    import torchaudio

    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stereo -> mono
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0).numpy()
