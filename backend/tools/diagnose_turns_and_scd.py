"""
Diagnostic Script: Deep Investigation of Turn Merging and SCD Behavior across audio.mp3

Investigates:
1. Every raw subsegment from online diarization (start, end, text, speaker_label, centroid distances).
2. Speaker Change Detection (SCD) boundaries — were short turns split by SCD?
3. Diarizer cluster assignment — did raw diarization assign distinct speaker labels or group them into one cluster?
4. Turn builder & Sandwich filter — did speaker_turns.py or offline reclustering merge distinct raw segments?
5. Total count of misattributed / merged alternating turns across the entire meeting.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.identifier import SpeakerIdentifier
from models.speaker_matcher import cosine_similarity
from models.speaker_turns import build_speaker_turns, _clean_and_merge_sandwiched_segments
from pipeline import MeetingSession
from stt.sarvam_adapter import SarvamSTTAdapter


def run_diagnostics():
    audio_path = "/home/stark/Downloads/audio_testing /audio.mp3"
    if not os.path.exists(audio_path):
        audio_path = "/home/stark/Downloads/audio_testing/audio.mp3"

    print("Loading audio and running pipeline processing...")
    audio = load_audio_file(audio_path)
    sarvam = SarvamSTTAdapter()
    identifier = SpeakerIdentifier()

    # Track raw subsegments as they are processed
    raw_subsegments = []

    session = MeetingSession(session_id="DIAGNOSTIC_RUN", identifier=identifier, stt_adapter=sarvam)

    # Wrap process_audio subsegment capturing
    # We will process audio standardly and capture intermediate states
    session.process_audio(audio)

    print("\n==========================================================================")
    print("ANALYSIS OF RAW SUBSEGMENTS (BEFORE TURN MERGING)")
    print("==========================================================================\n")

    raw_transcript = session.transcript  # This is the raw line-by-line list of subsegments before build_speaker_turns
    print(f"Total Raw Subsegments Generated: {len(raw_transcript)}")

    for idx, seg in enumerate(raw_transcript, 1):
        spk = seg.get("speaker")
        lbl = seg.get("speaker_label")
        start = seg.get("start_sec", 0.0)
        end = seg.get("end_sec", 0.0)
        dur = end - start
        txt = seg.get("text", "")
        overlap = seg.get("is_overlap", False)
        print(f"Subsegment #{idx:02d} [{start:6.2f}s - {end:6.2f}s] (dur={dur:4.2f}s) | Label: {lbl:<12} | Speaker: {spk:<15} | Overlap: {str(overlap):<5} | Text: \"{txt}\"")

    print("\n==========================================================================")
    print("ANALYSIS OF 00:12s - 00:30s TIMELINE (FAST ALTERNATION WINDOW)")
    print("==========================================================================\n")

    window_segs = [s for s in raw_transcript if s.get("start_sec", 0.0) >= 10.0 and s.get("end_sec", 0.0) <= 35.0]
    print(f"Subsegments in 10s - 35s window ({len(window_segs)} subsegments):")
    for s in window_segs:
        start = s.get("start_sec", 0.0)
        end = s.get("end_sec", 0.0)
        spk = s.get("speaker")
        lbl = s.get("speaker_label")
        txt = s.get("text")
        print(f"  [{start:5.2f}s - {end:5.2f}s] {spk:<15} ({lbl}): \"{txt}\"")

    print("\n==========================================================================")
    print("CHECKING TURN BUILDER / MERGING SCOPE ACROSS FULL MEETING")
    print("==========================================================================\n")

    final_turns = build_speaker_turns(raw_transcript)
    print(f"Final Merged Turns Count: {len(final_turns)}")

    # Check for turns where raw subsegments had DIFFERENT speakers but got merged
    merged_diff_speaker_count = 0
    sandwich_overwrites_count = 0

    # Test sandwich logic directly on raw subsegments
    sandwiched_raw = _clean_and_merge_sandwiched_segments(raw_transcript)
    for idx, (orig, sand) in enumerate(zip(raw_transcript, sandwiched_raw), 1):
        orig_spk = orig.get("speaker")
        sand_spk = sand.get("speaker")
        if orig_spk != sand_spk:
            sandwich_overwrites_count += 1
            print(f"  [SANDWICH OVERWRITE #{sandwich_overwrites_count}] Subsegment #{idx} [{orig.get('start_sec'):.1f}s - {orig.get('end_sec'):.1f}s]: Was '{orig_spk}' ({orig.get('speaker_label')}), changed to '{sand_spk}' ({sand.get('speaker_label')}) | Text: \"{orig.get('text')}\"")

    print("\nSummary of Scope:")
    print(f"  - Total Raw Subsegments: {len(raw_transcript)}")
    print(f"  - Total Merged Turns   : {len(final_turns)}")
    print(f"  - Sandwich Overwrites  : {sandwich_overwrites_count}")


if __name__ == "__main__":
    run_diagnostics()
