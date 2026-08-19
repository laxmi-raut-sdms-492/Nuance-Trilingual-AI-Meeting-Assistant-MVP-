"""
Diagnostic Script: Deep Dive on Real Meeting MTG-26ac2b9ef3dc
"""

import os
import sys
import numpy as np
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.embedding import get_embedding
from models.identifier import SpeakerIdentifier
from models.scd import split_on_speaker_change
from models.vad import SpeechSegmenter
from pipeline import MeetingSession
from stt.sarvam_adapter import SarvamSTTAdapter


def diagnose():
    audio_path = "storage/audio/MTG-26ac2b9ef3dc/Recording Aug 19, 10:51 AM.webm"
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    print(f"Loading audio from {audio_path}...")
    audio = load_audio_file(audio_path)
    dur = len(audio) / 16000.0
    print(f"Audio loaded successfully! Duration: {dur:.2f}s ({dur/60:.2f} mins)\n")

    sarvam = SarvamSTTAdapter()
    session = MeetingSession(session_id="DIAG_26AC", identifier=SpeakerIdentifier(), stt_adapter=sarvam)

    # Let's inspect step-by-step what process_audio does
    session.process_audio(audio)

    raw = session.raw_segments or session.transcript
    print(f"Total raw ASR subsegments emitted before turn merging: {len(raw)}")
    print("==========================================================================")
    print("RAW SUBSEGMENTS & DIARIZATION ASSIGNMENTS")
    print("==========================================================================\n")

    for idx, seg in enumerate(raw, 1):
        start = seg.get("start_sec", 0.0)
        end = seg.get("end_sec", 0.0)
        seg_dur = end - start
        lbl = seg.get("speaker_label")
        spk = seg.get("speaker")
        txt = seg.get("text", "")
        lang = seg.get("language")

        # Get embedding distance to Speaker_00 centroid
        emb = session._embeddings[idx - 1] if idx - 1 < len(session._embeddings) else None
        dist_str = ""
        if emb is not None and session.diarizer.clusters:
            dists = []
            for c_lbl, c_data in session.diarizer.clusters.items():
                d = float(cosine(emb, c_data["centroid"]))
                dists.append(f"{c_lbl}={d:.3f}")
            dist_str = ", ".join(dists)

        print(f"Subsegment #{idx:02d} [{start:6.2f}s - {end:6.2f}s] (dur={seg_dur:4.2f}s)")
        print(f"   Label: {lbl:<12} | Speaker: {spk:<15} | Lang: {lang}")
        print(f"   Distances to Centroids: {dist_str}")
        print(f"   Text snippet: \"{txt[:90]}...\"")
        print("-" * 75)

    print("\n==========================================================================")
    print("FINAL TURNS AFTER TURN MERGING")
    print("==========================================================================\n")
    for idx, turn in enumerate(session.transcript, 1):
        start = turn.get("start_sec", 0.0)
        end = turn.get("end_sec", 0.0)
        spk = turn.get("speaker")
        lbl = turn.get("speaker_label")
        txt = turn.get("text", "")
        print(f"Turn #{idx:02d} [{start:6.2f}s - {end:6.2f}s] {spk:<15} ({lbl}): \"{txt[:120]}...\"")


if __name__ == "__main__":
    diagnose()
