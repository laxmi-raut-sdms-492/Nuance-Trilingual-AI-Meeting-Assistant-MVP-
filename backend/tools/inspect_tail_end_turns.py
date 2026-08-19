"""
Diagnostic Script: Inspect subsegments #45 and #46 around 124.5s - 130.0s
"""

import os
import sys
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.identifier import SpeakerIdentifier
from models.speaker_turns import build_speaker_turns
from pipeline import MeetingSession
from stt.sarvam_adapter import SarvamSTTAdapter


def inspect_tail():
    audio_path = "/home/stark/Downloads/audio_testing /audio.mp3"
    if not os.path.exists(audio_path):
        audio_path = "/home/stark/Downloads/audio_testing/audio.mp3"

    audio = load_audio_file(audio_path)
    sarvam = SarvamSTTAdapter()

    session = MeetingSession(session_id="INSPECT_TAIL", identifier=SpeakerIdentifier(), stt_adapter=sarvam)
    session.process_audio(audio)

    raw = session.transcript

    print("==========================================================================")
    print("SUBSEGMENTS AROUND 120s - 135s TIMELINE")
    print("==========================================================================\n")

    tail_segs = [s for s in raw if s.get("start_sec", 0.0) >= 120.0]
    for idx, s in enumerate(tail_segs, 1):
        start = s.get("start_sec", 0.0)
        end = s.get("end_sec", 0.0)
        spk = s.get("speaker")
        lbl = s.get("speaker_label")
        txt = s.get("text")
        print(f"  Tail Subsegment #{idx:02d} [{start:6.2f}s - {end:6.2f}s] {spk:<15} ({lbl}): \"{txt}\"")

    print("\n==========================================================================")
    print("FINAL TURNS AFTER BUILD_SPEAKER_TURNS")
    print("==========================================================================\n")

    turns = build_speaker_turns(raw)
    tail_turns = [t for t in turns if t.get("start_sec", 0.0) >= 120.0]
    for idx, t in enumerate(tail_turns, 1):
        print(f"  Tail Turn #{idx:02d} [{t.get('start_sec'):6.2f}s - {t.get('end_sec'):6.2f}s] {t.get('speaker'):<15} ({t.get('speaker_label')}): \"{t.get('text')}\"")


if __name__ == "__main__":
    inspect_tail()
