"""
Comprehensive Verification Script: Test SCD calibration and check for over-splitting / overlap regressions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.identifier import SpeakerIdentifier
from models.speaker_turns import build_speaker_turns
from pipeline import MeetingSession
from stt.sarvam_adapter import SarvamSTTAdapter


def run_full_verification():
    audio_path = "/home/stark/Downloads/audio_testing /audio.mp3"
    if not os.path.exists(audio_path):
        audio_path = "/home/stark/Downloads/audio_testing/audio.mp3"

    print("Loading audio and running full pipeline with new SCD/VAD settings...")
    audio = load_audio_file(audio_path)
    sarvam = SarvamSTTAdapter()

    session = MeetingSession(session_id="SCD_VERIFICATION", identifier=SpeakerIdentifier(), stt_adapter=sarvam)
    session.process_audio(audio)

    raw_transcript = session.transcript
    final_turns = build_speaker_turns(raw_transcript)

    print("\n==========================================================================")
    print("1. 00:12s - 00:22s TIMELINE BREAKDOWN (FAST ALTERNATION WINDOW)")
    print("==========================================================================\n")

    alternation_window = [s for s in raw_transcript if s.get("start_sec", 0.0) >= 10.0 and s.get("end_sec", 0.0) <= 22.0]
    print(f"Subsegments in 10s - 22s window ({len(alternation_window)} subsegments):")
    for s in alternation_window:
        print(f"  [{s.get('start_sec'):5.2f}s - {s.get('end_sec'):5.2f}s] {s.get('speaker'):<15} ({s.get('speaker_label')}): \"{s.get('text')}\"")

    print("\n==========================================================================")
    print("2. REGRESSION CHECK: LONG SINGLE-SPEAKER STRETCHES (OVER-SPLITTING CHECK)")
    print("==========================================================================\n")

    # Long speaker sections to monitor:
    # Section A: ~20.99s - 32.20s (Yashraj continuous monologue)
    # Section B: ~61.27s - 77.00s (Siddesh continuous question)
    # Section C: ~77.00s - 89.47s (Yashraj explanation)
    monologue_a = [s for s in raw_transcript if s.get("start_sec", 0.0) >= 20.0 and s.get("end_sec", 0.0) <= 33.0]
    monologue_b = [s for s in raw_transcript if s.get("start_sec", 0.0) >= 61.0 and s.get("end_sec", 0.0) <= 77.5]
    monologue_c = [s for s in raw_transcript if s.get("start_sec", 0.0) >= 77.0 and s.get("end_sec", 0.0) <= 90.0]

    print(f"Section A (20s - 33s Yashraj Monologue): {len(monologue_a)} subsegment(s)")
    for s in monologue_a:
        print(f"  [{s.get('start_sec'):5.2f}s - {s.get('end_sec'):5.2f}s] {s.get('speaker')} ({s.get('speaker_label')}): \"{s.get('text')[:60]}...\"")

    print(f"\nSection B (61s - 77s Siddesh Monologue): {len(monologue_b)} subsegment(s)")
    for s in monologue_b:
        print(f"  [{s.get('start_sec'):5.2f}s - {s.get('end_sec'):5.2f}s] {s.get('speaker')} ({s.get('speaker_label')}): \"{s.get('text')[:60]}...\"")

    print(f"\nSection C (77s - 90s Yashraj Monologue): {len(monologue_c)} subsegment(s)")
    for s in monologue_c:
        print(f"  [{s.get('start_sec'):5.2f}s - {s.get('end_sec'):5.2f}s] {s.get('speaker')} ({s.get('speaker_label')}): \"{s.get('text')[:60]}...\"")

    print("\n==========================================================================")
    print("3. OVERLAP DETECTION VERIFICATION (MUST BE UNCHANGED)")
    print("==========================================================================\n")

    overlap_turns = [t for t in final_turns if t.get("is_overlap")]
    print(f"Total Overlap Turns Detected: {len(overlap_turns)}")
    for t in overlap_turns:
        print(f"  [{t.get('time')}] Speaker: '{t.get('speaker')}' | Candidates: {t.get('candidate_speakers')} | Text: \"{t.get('text')}\"")

    print("\n==========================================================================")
    print("4. FULL MEETING SUBSEGMENT BREAKDOWN (BEFORE vs AFTER IMPACT)")
    print("==========================================================================\n")

    print(f"Total Raw Subsegments: {len(raw_transcript)} (Before calibration: 21)")
    print(f"Total Final Turns    : {len(final_turns)} (Before calibration: 14)")
    print("\nFull Turn List:")
    for idx, t in enumerate(final_turns, 1):
        print(f"  Turn #{idx:02d} [{t.get('time')}] {t.get('speaker'):<20} (is_overlap={t.get('is_overlap')}): \"{t.get('text')}\"")


if __name__ == "__main__":
    run_full_verification()
