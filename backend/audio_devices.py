"""
Audio input device enumeration — Jabra Speak2 75 and other microphones.

The AI pipeline does not run inside the Jabra. This module discovers OS-level
input devices so the frontend can let the user pick one without hardcoding
device_index.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("audio_devices")


def list_input_devices() -> list[dict]:
    """
    Return available audio input devices: [{index, name, channels, default}, ...].
    Uses sounddevice when installed; otherwise returns an empty list gracefully.
    """
    try:
        import sounddevice as sd
    except ImportError:
        logger.warning("sounddevice not installed — cannot enumerate audio devices")
        return []

    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0] if sd.default.device else None
        inputs: list[dict] = []
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) < 1:
                continue
            inputs.append(
                {
                    "index": i,
                    "name": dev.get("name", f"Device {i}"),
                    "channels": dev.get("max_input_channels", 1),
                    "default": i == default_in,
                    "sample_rate": dev.get("default_samplerate"),
                }
            )
        return inputs
    except Exception as exc:
        logger.warning(f"audio device enumeration failed: {exc}")
        return []


def find_device_by_name(substring: str) -> dict | None:
    """Case-insensitive partial match (e.g. 'Jabra Speak2 75')."""
    needle = (substring or "").lower()
    for dev in list_input_devices():
        if needle in dev.get("name", "").lower():
            return dev
    return None
