"""
Diagnostic Script: Deep Dive on Turn 1 (1.25s - 14.00s) of MTG-26ac2b9ef3dc
"""

import os
import sys
import numpy as np
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.embedding import get_embedding
from models.vad import SpeechSegmenter


def inspect_turn1():
    audio_path = "storage/audio/MTG-26ac2b9ef3dc/Recording Aug 19, 10:51 AM.webm"
    audio = load_audio_file(audio_path)

    # Cut Turn 1 audio (1.25s - 14.00s)
    s_start = int(1.25 * 16000)
    s_end = int(14.00 * 16000)
    turn1_audio = audio[s_start:s_end]

    print(f"Turn 1 Audio Length: {len(turn1_audio)/16000:.2f}s")

    # 1. First test Silero VAD on Turn 1 audio with smaller min_silence_ms (e.g. 100ms or 150ms)
    for silence_ms in [300, 200, 150, 100]:
        segger = SpeechSegmenter(min_silence_ms=silence_ms, threshold=0.25)
        segs = segger.process(turn1_audio) + segger.flush()
        print(f"\nVAD with min_silence_ms={silence_ms}: {len(segs)} segments emitted:")
        for idx, s in enumerate(segs, 1):
            st = s["start"] + 1.25
            en = s["end"] + 1.25
            print(f"  VAD #{idx:02d} [{st:5.2f}s - {en:5.2f}s] (dur={en-st:4.2f}s)")

    # 2. Sliding window ECAPA embeddings across Turn 1 with 0.4s window and 0.15s hop
    window_samples = int(0.4 * 16000)
    hop_samples = int(0.15 * 16000)

    embs = []
    starts = []
    pos = 0
    while pos + window_samples <= len(turn1_audio):
        w = turn1_audio[pos : pos + window_samples]
        embs.append(get_embedding(w))
        starts.append(pos)
        pos += hop_samples

    dists = [
        float(cosine(embs[i], embs[i + 1]))
        for i in range(len(embs) - 1)
    ]

    print(f"\nSliding Window Distance Profile (0.4s window, 0.15s hop, {len(dists)} steps):")
    for idx, d in enumerate(dists):
        t_sec = (starts[idx] + hop_samples) / 16000.0 + 1.25
        bar = "#" * int(d * 40)
        print(f"  @{t_sec:5.2f}s: dist={d:.4f} | {bar}")


if __name__ == "__main__":
    inspect_turn1()
