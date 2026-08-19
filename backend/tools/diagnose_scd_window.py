"""
Diagnostic Script: Investigate VAD & SCD cuts across the full audio.mp3
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.scd import split_on_speaker_change
from models.vad import SpeechSegmenter


def inspect_scd():
    audio_path = "/home/stark/Downloads/audio_testing /audio.mp3"
    if not os.path.exists(audio_path):
        audio_path = "/home/stark/Downloads/audio_testing/audio.mp3"

    audio = load_audio_file(audio_path)
    segmenter = SpeechSegmenter()

    print("1. Running VAD (SpeechSegmenter) on full audio...")
    vad_segments = segmenter.process(audio)
    print(f"VAD returned {len(vad_segments)} raw speech segments:")
    for idx, seg in enumerate(vad_segments, 1):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        print(f"  VAD Segment #{idx:02d}: [{start:6.2f}s - {end:6.2f}s] (dur={end-start:4.2f}s)")

        # Run SCD on this VAD segment
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        seg_audio = audio[start_sample:end_sample]

        scd_pieces = split_on_speaker_change(seg_audio)
        if len(scd_pieces) > 1:
            print(f"     ---> SCD split this segment into {len(scd_pieces)} subsegments:")
            for p_idx, (p_start, p_end) in enumerate(scd_pieces, 1):
                abs_start = start + p_start / 16000
                abs_end = start + p_end / 16000
                print(f"          Piece #{p_idx}: [{abs_start:6.2f}s - {abs_end:6.2f}s] (dur={abs_end-abs_start:4.2f}s)")
        else:
            print("     ---> SCD detected 0 speaker changes (left unbroken).")


if __name__ == "__main__":
    inspect_scd()
