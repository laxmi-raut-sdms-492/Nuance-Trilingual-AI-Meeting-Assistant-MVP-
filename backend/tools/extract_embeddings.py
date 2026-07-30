"""
Cache one recording's segment embeddings so diarization can be replayed offline.

Diarization tuning needs many runs over the same audio, and re-running the
whole pipeline each time re-pays for Whisper — which has nothing to do with
clustering. VAD, SCD and the ECAPA embeddings are deterministic given the
audio, so they are computed once here and written to an .npz; replay_diarizer.py
then sweeps clustering parameters over the cached vectors in milliseconds.

The segmentation below mirrors MeetingSession._consume / _process_subsegment
exactly (VAD -> SCD split -> length and RMS gate). If that changes, this must.

    python3 -m tools.extract_embeddings MTG-85407e28d5eb
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file, rms  # noqa: E402
from config import (  # noqa: E402
    MIN_SPEECH_SECONDS,
    SAMPLE_RATE,
    SILENCE_RMS_THRESHOLD,
)
from db import repository as repo  # noqa: E402
from models.embedding import get_embedding  # noqa: E402
from models.scd import split_on_speaker_change  # noqa: E402
from models.vad import SpeechSegmenter  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")

MIN_SPEECH_SAMPLES = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage", "embeddings")


def extract(meeting_id: str) -> str:
    path = repo.audio_path(meeting_id)
    if not path:
        raise SystemExit(f"no audio stored for {meeting_id}")

    audio = load_audio_file(path)
    print(f"{meeting_id}: {len(audio) / SAMPLE_RATE:.1f}s of audio from {os.path.basename(path)}")

    segmenter = SpeechSegmenter()
    vad_segments = list(segmenter.process(audio)) + list(segmenter.flush())
    print(f"VAD produced {len(vad_segments)} segments")

    embeddings, starts, ends = [], [], []
    for vad_seg in vad_segments:
        seg_audio = vad_seg["audio"]
        seg_start = vad_seg["start"]
        if len(seg_audio) == 0:
            continue

        for a, b in split_on_speaker_change(seg_audio):
            sub = seg_audio[a:b]
            if len(sub) < MIN_SPEECH_SAMPLES or rms(sub) < SILENCE_RMS_THRESHOLD:
                continue
            embeddings.append(get_embedding(sub))
            starts.append(seg_start + a / SAMPLE_RATE)
            ends.append(seg_start + b / SAMPLE_RATE)

    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.abspath(os.path.join(CACHE_DIR, f"{meeting_id}.npz"))
    np.savez(
        out,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        starts=np.asarray(starts, dtype=np.float32),
        ends=np.asarray(ends, dtype=np.float32),
    )
    print(f"wrote {len(embeddings)} embeddings -> {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 -m tools.extract_embeddings <meeting-id>")
    extract(sys.argv[1])
