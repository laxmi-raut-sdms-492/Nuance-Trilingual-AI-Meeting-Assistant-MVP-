"""
Diagnostic Script: Test SCD distances and adaptive threshold on MTG-26ac2b9ef3dc
"""

import os
import sys
import numpy as np
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.embedding import get_embedding
from models.scd import WINDOW_SAMPLES, HOP_SAMPLES, _adaptive_change_threshold


def analyze_scd():
    audio_path = "storage/audio/MTG-26ac2b9ef3dc/Recording Aug 19, 10:51 AM.webm"
    audio = load_audio_file(audio_path)

    # Let's inspect Subsegment #03 (15.9s - 38.74s) and Subsegment #04 (38.74s - 63.74s)
    sample_start = int(15.9 * 16000)
    sample_end = int(63.74 * 16000)
    seg_audio = audio[sample_start:sample_end]

    print(f"Segment length: {len(seg_audio)/16000:.2f}s ({len(seg_audio)} samples)")

    # Compute window embeddings
    window_embeddings = []
    window_starts = []
    pos = 0
    while pos + WINDOW_SAMPLES <= len(seg_audio):
        w = seg_audio[pos : pos + WINDOW_SAMPLES]
        window_embeddings.append(get_embedding(w))
        window_starts.append(pos)
        pos += HOP_SAMPLES

    distances = [
        float(cosine(window_embeddings[i], window_embeddings[i + 1]))
        for i in range(len(window_embeddings) - 1)
    ]

    print(f"Window count: {len(window_embeddings)}, Distances count: {len(distances)}")
    print("Distance sequence:")
    for idx, d in enumerate(distances):
        t_sec = (window_starts[idx] + HOP_SAMPLES) / 16000.0 + 15.9
        print(f"  @{t_sec:5.2f}s (hop #{idx:02d}): dist={d:.4f}")

    arr = np.asarray(distances)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    adaptive_orig = max(0.13, median + max(2.5 * mad, 0.08))

    print(f"\nSCD Stats:")
    print(f"  Min dist   : {np.min(arr):.4f}")
    print(f"  Max dist   : {np.max(arr):.4f}")
    print(f"  Mean dist  : {np.mean(arr):.4f}")
    print(f"  Median dist: {median:.4f}")
    print(f"  MAD        : {mad:.4f}")
    print(f"  Adaptive Threshold (current code): {adaptive_orig:.4f}")

    peaks_orig = [d for d in distances if d > adaptive_orig]
    print(f"  Peaks exceeding current threshold ({adaptive_orig:.4f}): {len(peaks_orig)}")

    candid_threshold = max(0.13, min(adaptive_orig, 0.22))
    peaks_new = [d for d in distances if d > candid_threshold]
    print(f"  Peaks exceeding capped threshold ({candid_threshold:.4f}): {len(peaks_new)}")


if __name__ == "__main__":
    analyze_scd()
