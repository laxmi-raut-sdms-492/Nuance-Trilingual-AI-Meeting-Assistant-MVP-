"""
Audio format helpers.

The frontend sends raw 16-bit PCM, mono, 16kHz audio directly (no container
format like WebM/MP3) — this sidesteps a common real-time-audio pitfall
where chunked container formats can't be decoded independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import (
    SAMPLE_RATE,
    FLAT_TONE_RMS_STD_MAX,
    HUM_NOTCH_FREQ_HZ,
    MIN_SILENCE_MS,
    SILENCE_RMS_THRESHOLD,
    VAD_BATCH_FALLBACK_THRESHOLD,
)


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
    arr = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(arr))))


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


def per_second_rms_std(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Standard deviation of per-second RMS — near zero means a steady tone/hum."""
    window = sample_rate
    if len(audio) < window * 2:
        return 0.0
    energies = [
        float(np.sqrt(np.mean(np.square(audio[i : i + window]))))
        for i in range(0, len(audio) - window, window)
    ]
    return float(np.std(energies))


def is_flat_tone(audio: np.ndarray) -> bool:
    """True when the waveform has energy but almost no second-to-second variation."""
    return (
        rms(audio) >= SILENCE_RMS_THRESHOLD
        and per_second_rms_std(audio) <= FLAT_TONE_RMS_STD_MAX
    )


def remove_steady_hum(audio: np.ndarray, freq_hz: float = HUM_NOTCH_FREQ_HZ) -> np.ndarray:
    """
    Attenuate a dominant steady tone (often ~220 Hz on bad exports) so VAD and
    Whisper can see the speech underneath.
    """
    from scipy.signal import iirnotch, filtfilt

    b, a = iirnotch(freq_hz, 30, SAMPLE_RATE)
    cleaned = filtfilt(b, a, np.asarray(audio, dtype=np.float64))
    return cleaned.astype(np.float32)


@dataclass(frozen=True)
class UploadAudioProfile:
    """Per-upload thresholds derived from the recording itself — not global config."""

    silence_rms_threshold: float
    vad_threshold: float
    min_silence_ms: int
    batch_vad_threshold: float
    batch_vad_min_speech_ratio: float
    k_max: int
    hum_freq_hz: float | None
    is_flat_tone: bool
    warning: str | None = None


def detect_dominant_tone_hz(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float | None:
    """Peak frequency in the speech band when a steady tone dominates the file."""
    from scipy.fft import rfft, rfftfreq

    chunk = audio[: min(len(audio), sample_rate * 5)]
    if len(chunk) < sample_rate:
        return None
    spectrum = np.abs(rfft(chunk))
    freqs = rfftfreq(len(chunk), 1 / sample_rate)
    mask = (freqs > 50) & (freqs < 2000)
    if not np.any(mask):
        return None
    band = spectrum[mask]
    freqs_band = freqs[mask]
    peak_idx = int(np.argmax(band))
    if band[peak_idx] / (float(np.mean(band)) + 1e-9) < 3.0:
        return None
    return float(freqs_band[peak_idx])


def analyze_upload_audio(audio: np.ndarray) -> UploadAudioProfile:
    """
    Measure each upload and derive VAD/diarization knobs for that file alone.
    """
    duration = len(audio) / SAMPLE_RATE
    energy_std = per_second_rms_std(audio)
    flat = is_flat_tone(audio)
    hum_freq = detect_dominant_tone_hz(audio) if flat else None

    window = SAMPLE_RATE
    energies = [
        float(np.sqrt(np.mean(np.square(audio[i : i + window]))))
        for i in range(0, max(len(audio) - window, 1), window)
    ]
    noise_floor = float(np.percentile(energies, 10)) if energies else SILENCE_RMS_THRESHOLD
    silence_threshold = float(
        np.clip(max(SILENCE_RMS_THRESHOLD, noise_floor * 1.5), SILENCE_RMS_THRESHOLD, 0.05)
    )

    if flat:
        vad_threshold = VAD_BATCH_FALLBACK_THRESHOLD
    elif energy_std < 0.02:
        vad_threshold = 0.25
    else:
        vad_threshold = 0.35

    min_silence = 300 if duration > 120 else MIN_SILENCE_MS
    batch_ratio = 0.05 if duration > 120 else 0.10
    k_max = max(4, min(20, int(duration / 20) + 2))

    warning = None
    if flat:
        freq = hum_freq or HUM_NOTCH_FREQ_HZ
        warning = (
            f"This recording has a steady background tone near {freq:.0f} Hz that can hide "
            "speech. Hum removal was applied automatically; for best results, upload the "
            "original source file."
        )

    return UploadAudioProfile(
        silence_rms_threshold=silence_threshold,
        vad_threshold=vad_threshold,
        min_silence_ms=min_silence,
        batch_vad_threshold=vad_threshold,
        batch_vad_min_speech_ratio=batch_ratio,
        k_max=k_max,
        hum_freq_hz=hum_freq,
        is_flat_tone=flat,
        warning=warning,
    )


def prepare_upload_audio(audio: np.ndarray) -> tuple[np.ndarray, UploadAudioProfile]:
    """Analyze and optionally clean an upload before VAD/ASR run."""
    import logging

    profile = analyze_upload_audio(audio)
    if not profile.is_flat_tone:
        return audio, profile

    logger = logging.getLogger("audio_utils")
    freq = profile.hum_freq_hz or HUM_NOTCH_FREQ_HZ
    logger.warning(
        f"flat-tone upload (per-second RMS std={per_second_rms_std(audio):.4f}) — "
        f"applying {freq:.0f} Hz notch"
    )
    return remove_steady_hum(audio, freq), profile


def preprocess_upload_audio(audio: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Backward-compatible wrapper around prepare_upload_audio()."""
    cleaned, profile = prepare_upload_audio(audio)
    return cleaned, profile.warning


def wav_bytes_to_float32(wav_path: str) -> np.ndarray:
    """Load a WAV file (e.g. an enrollment sample) and resample to SAMPLE_RATE mono."""
    import torchaudio

    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stereo -> mono
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0).numpy()
