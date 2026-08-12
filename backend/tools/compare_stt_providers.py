"""
Developer CLI Utility to compare Local STT, Sarvam Saaras v3, and Google Chirp 3 side-by-side.

Usage:
    python tools/compare_stt_providers.py path/to/sample.wav --lang en-IN
"""

from __future__ import annotations

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from stt.local.local_adapter import LocalSTTAdapter
from stt.cloud.sarvam_provider import SarvamProvider
from stt.cloud.google_chirp_provider import GoogleChirpProvider


def compare_file(audio_path: str, hint_language: str = "en-IN"):
    print(f"\n=======================================================")
    print(f"   STT PROVIDER COMPARISON TOOL")
    print(f"   Audio File: {audio_path}")
    print(f"   Language Hint: {hint_language}")
    print(f"=======================================================\n")

    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    audio = load_audio_file(audio_path)

    providers = [
        ("1. Local STT (Whisper / IndicConformer)", LocalSTTAdapter()),
        ("2. Cloud STT - Sarvam AI (saaras:v3)", SarvamProvider()),
        ("3. Cloud STT - Google Cloud (chirp_3)", GoogleChirpProvider()),
    ]

    results = []
    for title, provider in providers:
        print(f"Testing {title}...")
        start_t = time.time()
        try:
            res = provider.transcribe(audio, hint_language=hint_language)
            latency = round(time.time() - start_t, 3)
            results.append((title, res, latency, None))
        except Exception as exc:
            latency = round(time.time() - start_t, 3)
            results.append((title, None, latency, str(exc)))

    print("\n---------------- COMPARISON SUMMARY ----------------\n")
    for title, res, latency, error in results:
        print(f"[{title}]")
        print(f"  Latency:      {latency}s")
        if error:
            print(f"  Status:       ERROR ({error})")
        else:
            print(f"  Detected Lang:{res.get('language')} ({res.get('language_name')})")
            print(f"  Transcript:   {res.get('text')}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STT Provider Comparison CLI Tool")
    parser.add_argument("audio_path", help="Path to audio WAV/MP3 file")
    parser.add_argument("--lang", default="en-IN", help="Language code (en-IN, hi-IN, mr-IN)")
    args = parser.parse_args()

    compare_file(args.audio_path, hint_language=args.lang)
