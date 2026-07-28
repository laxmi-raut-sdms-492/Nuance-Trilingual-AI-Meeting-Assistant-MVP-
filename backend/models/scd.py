"""
Speaker Change Detection (SCD).

VAD (models/vad.py) tells us WHEN there's speech vs silence. It does NOT
tell us whether the speech inside one continuous (no-pause) stretch is a
single person or several people talking back-to-back with no gap — which
is common in fast-paced conversation. Feeding a multi-speaker stretch into
one embedding call produces a single blended/wrong embedding, and whichever
speaker's voice happens to dominate "wins" the label for the whole segment.

SCD fixes this by sliding a short window across the segment, embedding each
window, and looking for large jumps in embedding similarity between
consecutive windows — a jump means the voice changed. The segment is then
split at those points BEFORE diarization/identification/transcription ever
see it, so each piece handed downstream is acoustically homogeneous
(one speaker only).

This runs only on segments long enough for a transition to plausibly fit
(see SCD_MIN_SEGMENT_SECONDS) — most VAD segments in a real conversation
are already short and single-speaker, so this is a targeted extra pass,
not a rewrite of the whole pipeline.
"""

import logging

import numpy as np
from scipy.spatial.distance import cosine

from config import (
    SAMPLE_RATE,
    SCD_WINDOW_SECONDS,
    SCD_HOP_SECONDS,
    SCD_CHANGE_THRESHOLD,
    SCD_MIN_SEGMENT_SECONDS,
    SCD_MIN_SUBSEGMENT_SECONDS,
)
from models.embedding import get_embedding

logger = logging.getLogger("scd")

WINDOW_SAMPLES = int(SCD_WINDOW_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(SCD_HOP_SECONDS * SAMPLE_RATE)
MIN_SUBSEGMENT_SAMPLES = int(SCD_MIN_SUBSEGMENT_SECONDS * SAMPLE_RATE)


def split_on_speaker_change(audio: np.ndarray) -> list[tuple[int, int]]:
    """
    audio: 1-D float32 PCM, 16kHz mono — one VAD-bounded speech segment.
    Returns a list of (start_sample, end_sample) sub-segments, each
    intended to be single-speaker. If no change is detected (or the
    segment is too short to analyze), returns the whole segment as one.
    """
    n = len(audio)

    if n < SCD_MIN_SEGMENT_SECONDS * SAMPLE_RATE or n < WINDOW_SAMPLES * 2:
        return [(0, n)]

    window_embeddings = []
    window_starts = []
    pos = 0
    while pos + WINDOW_SAMPLES <= n:
        window = audio[pos : pos + WINDOW_SAMPLES]
        window_embeddings.append(get_embedding(window))
        window_starts.append(pos)
        pos += HOP_SAMPLES

    if len(window_embeddings) < 3:
        return [(0, n)]

    # Distance between each pair of consecutive windows — a spike means
    # the voice changed somewhere in that hop.
    distances = [
        cosine(window_embeddings[i], window_embeddings[i + 1])
        for i in range(len(window_embeddings) - 1)
    ]

    change_points = _find_change_points(distances, window_starts)
    change_points = _enforce_min_spacing(change_points, n)

    if not change_points:
        return [(0, n)]

    logger.info(f"speaker change detected at {[round(c / SAMPLE_RATE, 2) for c in change_points]}s within segment")

    boundaries = [0] + change_points + [n]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _find_change_points(distances: list[float], window_starts: list[int]) -> list[int]:
    """Local maxima in the distance sequence that clear the change threshold."""
    points = []
    for i in range(1, len(distances) - 1):
        is_local_max = distances[i] > distances[i - 1] and distances[i] > distances[i + 1]
        if is_local_max and distances[i] > SCD_CHANGE_THRESHOLD:
            # the change happened between window i and window i+1 — split at
            # the boundary between them (start of the next window)
            change_sample = window_starts[i + 1]
            points.append(change_sample)
    return points


def _enforce_min_spacing(change_points: list[int], total_samples: int) -> list[int]:
    """Drop change points that would create a sub-segment shorter than the minimum."""
    if not change_points:
        return []

    kept = []
    last = 0
    for cp in sorted(change_points):
        if cp - last >= MIN_SUBSEGMENT_SAMPLES and total_samples - cp >= MIN_SUBSEGMENT_SAMPLES:
            kept.append(cp)
            last = cp
    return kept
