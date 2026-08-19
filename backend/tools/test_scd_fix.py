"""
Test SCD fix on MTG-26ac2b9ef3dc
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
import models.scd as scd_module


def custom_find_change_points(distances: list[float], window_starts: list[int]) -> list[int]:
    # Cap threshold at 0.35 max
    if len(distances) < 3:
        threshold = 0.13
    else:
        arr = np.asarray(distances, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median))) + 1e-6
        threshold = min(0.35, max(0.13, median + max(1.5 * mad, 0.05)))

    points = []
    for i in range(len(distances)):
        d = distances[i]
        if d < threshold:
            continue
        # Check if it's a peak or plateau top
        is_left_ok = (i == 0 or d >= distances[i - 1])
        is_right_ok = (i == len(distances) - 1 or d >= distances[i + 1])
        if is_left_ok or is_right_ok:
            # Check if this is the maximum in a 3-window neighborhood
            window_slice = distances[max(0, i - 1) : min(len(distances), i + 2)]
            if d == max(window_slice):
                change_sample = window_starts[i] + scd_module.HOP_SAMPLES
                points.append(change_sample)
    return points


def test_fix():
    audio_path = "storage/audio/MTG-26ac2b9ef3dc/Recording Aug 19, 10:51 AM.webm"
    audio = load_audio_file(audio_path)

    # Monkey patch scd_module._find_change_points
    scd_module._find_change_points = custom_find_change_points

    sarvam = SarvamSTTAdapter()
    session = MeetingSession(session_id="TEST_SCD_FIX", identifier=SpeakerIdentifier(), stt_adapter=sarvam)
    session.process_audio(audio)

    turns = session.transcript
    print(f"Total turns after SCD fix: {len(turns)}\n")
    for idx, t in enumerate(turns, 1):
        print(f"Turn #{idx:02d} [{t.get('start_sec'):6.2f}s - {t.get('end_sec'):6.2f}s] {t.get('speaker'):<15} ({t.get('speaker_label')}):")
        print(f"  \"{t.get('text')[:100]}...\"")
        print("-" * 60)

    # Speaker stats
    stats = session.speaker_stats()
    print("\nSPEAKER STATS:")
    for s in stats:
        print(f"  {s['name']}: {s['seconds']}s ({s['pct']}%)")


if __name__ == "__main__":
    test_fix()
