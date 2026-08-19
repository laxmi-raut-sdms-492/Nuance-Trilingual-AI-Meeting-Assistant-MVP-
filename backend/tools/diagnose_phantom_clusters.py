"""
Diagnostic Script: Trace Phantom Speaker Clusters and Language Hallucination on audio.mp3
"""

import os
import sys
import numpy as np
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.identifier import SpeakerIdentifier
from pipeline import MeetingSession
from stt.sarvam_adapter import SarvamSTTAdapter


def trace_phantom_clusters():
    audio_path = "/home/stark/Downloads/audio_testing /audio.mp3"
    if not os.path.exists(audio_path):
        audio_path = "/home/stark/Downloads/audio_testing/audio.mp3"

    print("Loading audio and running MeetingSession...")
    audio = load_audio_file(audio_path)
    sarvam = SarvamSTTAdapter()

    session = MeetingSession(session_id="TRACE_PHANTOM", identifier=SpeakerIdentifier(), stt_adapter=sarvam)
    session.process_audio(audio)

    raw_transcript = session.transcript

    print("==========================================================================")
    print("DETAILED TRACE OF ALL SUBSEGMENTS & CLUSTER ASSIGNMENTS")
    print("==========================================================================\n")

    for idx, seg in enumerate(raw_transcript, 1):
        start = seg.get("start_sec", 0.0)
        end = seg.get("end_sec", 0.0)
        dur = end - start
        lbl = seg.get("speaker_label")
        spk = seg.get("speaker")
        txt = seg.get("text", "")
        lang = seg.get("language")
        lang_prob = seg.get("language_prob")
        raw_txt = seg.get("raw_text", "")

        # Compute embedding distances to existing clusters if stored
        emb = session._embeddings[idx - 1] if idx - 1 < len(session._embeddings) else None

        dist_str = ""
        if emb is not None and session.diarizer.clusters:
            dists = []
            for c_lbl, c_data in session.diarizer.clusters.items():
                d = float(cosine(emb, c_data["centroid"]))
                dists.append(f"{c_lbl}={d:.3f}")
            dist_str = ", ".join(dists)

        print(f"Subsegment #{idx:02d} [{start:6.2f}s - {end:6.2f}s] (dur={dur:4.2f}s)")
        print(f"   Label: {lbl:<12} | Speaker: {spk:<15} | Lang: {lang} (p={lang_prob})")
        print(f"   Distances to Centroids: {dist_str}")
        print(f"   Text: \"{txt}\"")
        if "ಅದು" in txt or "ಕನ್ನಡ" in txt or lang == "kn":
            print(f"   *** KANNADA HALLUCINATION DETECTED *** Raw text: \"{raw_txt}\"")
        print("-" * 75)


if __name__ == "__main__":
    trace_phantom_clusters()
