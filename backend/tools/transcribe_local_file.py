"""
Ad-hoc: run the full offline pipeline (STT + diarization) over an arbitrary
audio file that was never uploaded through the app, and print a per-speaker
transcript to stdout. Nothing is written to the meetings DB.

    python3 -m tools.transcribe_local_file /path/to/recording.webm
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402  (boots identifier + model loading, same as the server)
from audio_utils import load_audio_file  # noqa: E402
from pipeline import MeetingSession, format_duration  # noqa: E402
from stt.resolver import resolve_stt_adapter  # noqa: E402


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )

    stt_adapter = resolve_stt_adapter(None)  # local, matches the app default
    audio = load_audio_file(args.audio_path)

    session = MeetingSession("adhoc-local", main.identifier, stt_adapter=stt_adapter)
    session.process_audio(audio)

    wall_clock = len(audio) / 16000
    print(f"\n=== {os.path.basename(args.audio_path)} ({format_duration(wall_clock)}) ===\n")

    if not session.transcript:
        print("(no speech detected)")
        if session.failed_segments:
            print(f"{session.failed_segments} segment(s) failed. Last error: {session.last_error}")
        return

    for entry in session.transcript:
        print(f"[{entry['time']}] {entry['speaker']}: {entry['text']}")

    print("\n--- speaker stats ---")
    for s in session.speaker_stats():
        print(s)


if __name__ == "__main__":
    main_cli()
