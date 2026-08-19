"""
Diagnostic Script: Direct HTTP request to Sarvam Cloud STT API
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from stt.sarvam_adapter import _float32_to_wav_bytes


def test_sarvam_direct():
    print(f"Loaded SARVAM_API_KEY from env: {config.SARVAM_API_KEY[:10]}...{config.SARVAM_API_KEY[-6:] if config.SARVAM_API_KEY else 'EMPTY'}")

    from sarvamai import SarvamAI
    from sarvamai.core.api_error import ApiError

    client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)

    # 1 second sine wave audio
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_bytes = _float32_to_wav_bytes(audio)

    print("Sending test audio clip to Sarvam STT API (api.sarvam.ai)...")
    try:
        response = client.speech_to_text.transcribe(
            file=("segment.wav", wav_bytes, "audio/wav"),
            model=config.SARVAM_STT_MODEL if hasattr(config, "SARVAM_STT_MODEL") else "saaras:v3",
            mode="transcribe",
            language_code="unknown",
            with_timestamps=False,
            input_audio_codec="wav",
        )
        print("\n==========================================================================")
        print("SARVAM API SUCCESS (200 OK)!")
        print("==========================================================================")
        print("Response:", response)

    except ApiError as exc:
        print("\n==========================================================================")
        print("SARVAM API ERROR RESPONSE:")
        print("==========================================================================")
        print(f"Status Code: {exc.status_code}")
        print(f"Error Body : {exc.body}")
        print(f"Headers    : {exc.headers}")

    except Exception as exc:
        print(f"\nUNEXPECTED EXCEPTION: {exc}")


if __name__ == "__main__":
    test_sarvam_direct()
