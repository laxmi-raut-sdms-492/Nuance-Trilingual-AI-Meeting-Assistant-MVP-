"""
Benchmark pipeline directly against all 12 real meeting audio files.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.identifier import SpeakerIdentifier
from pipeline import MeetingSession
from stt.sarvam_adapter import SarvamSTTAdapter
import models.speaker_turns as speaker_turns_module


def benchmark():
    audio_files = sorted(glob.glob("storage/audio/*/*.webm"))
    print(f"Found {len(audio_files)} real meeting audio files to benchmark.\n")

    sarvam = SarvamSTTAdapter()

    for idx, audio_path in enumerate(audio_files, 1):
        meeting_dir = os.path.basename(os.path.dirname(audio_path))
        file_name = os.path.basename(audio_path)
        print(f"==========================================================================")
        print(f"[{idx}/{len(audio_files)}] BENCHMARKING: {meeting_dir} / {file_name}")
        print(f"==========================================================================")

        try:
            audio = load_audio_file(audio_path)
            dur = len(audio) / 16000.0
            print(f"Audio loaded: {dur:.2f}s ({dur/60:.2f} mins)")

            session = MeetingSession(session_id=f"BENCH_{meeting_dir}", identifier=SpeakerIdentifier(), stt_adapter=sarvam)
            session.process_audio(audio)

            turns = session.transcript
            stats = session.speaker_stats()

            print(f"Result: {len(turns)} turns, {len(stats)} speaker categories.")

            # Check for foreign script leaks
            foreign_leaks = 0
            for t in turns:
                txt = t.get("text", "")
                # Check for Kannada, Malayalam, Gujarati, Odia, Telugu, Tamil, Bengali
                import re
                m = re.search(r"[\u0980-\u09FF\u0A00-\u0AFF\u0B00-\u0BFF\u0C00-\u0CFF\u0D00-\u0D7F]", txt)
                if m:
                    foreign_leaks += 1

            print(f"Foreign Script Garbage Leaks: {foreign_leaks} / {len(turns)} turns")
            print("Speakers Breakdown:")
            for s in stats:
                print(f"  - {s['name']}: {s['seconds']}s ({s['pct']}%)")

            print("\nFirst 3 Turns Sample:")
            for t in turns[:3]:
                print(f"  [{t.get('start_sec'):5.1f}s - {t.get('end_sec'):5.1f}s] {t.get('speaker')}: \"{t.get('text')[:100]}\"")
            print("\n")

        except Exception as e:
            print(f"ERROR processing {audio_path}: {e}\n")


if __name__ == "__main__":
    benchmark()
