"""
Overlapping speech detection, run once the whole meeting is known.

The streaming diarizer (models/diarizer.py) flags overlap only when a whole
segment's voice print sits almost exactly halfway between two speakers. Real
overlap rarely looks like that: someone cuts in for a second inside another
person's sentence, or one voice is louder. The segment's print then follows
the main speaker and the overlap is never flagged. The streaming flag is also
taken before the offline recluster renames everyone, so the speaker names it
records can point at clusters that no longer exist.

This pass replaces it for uploads. It slides a short window over every line
and compares each window's voice print with the final speakers' centroids.
A window where two people talk at once sounds like neither of them alone, but
like the blend of both — its print is closer to the mix of the two centroids
than to either one. A line with such a window is marked as overlap between its
own speaker and the one who joined in.

WHEN two people talk comes from pyannote's segmentation model when it is
available: a model trained for exactly this, unlike a voice-print model,
which is built to describe one voice. The voice-print blend test above is the
fallback, set conservatively — measured on two-voice mixes built from real
meetings it catches about 30% of overlaps and wrongly flags about 2% of clean
lines. WHO joined in always comes from voice prints, since the segmentation
model only knows "a second voice", not whose.
"""

from __future__ import annotations

import logging
import re

import numpy as np
from scipy.spatial.distance import cosine

from config import (
    HF_TOKEN,
    OVERLAP_HOP_SECONDS,
    OVERLAP_MIN_REGION_SECONDS,
    OVERLAP_MODEL,
    OVERLAP_MIN_LINE_SECONDS,
    OVERLAP_MIN_SCORE,
    OVERLAP_MIN_WINDOWS,
    OVERLAP_WINDOW_SECONDS,
    SAMPLE_RATE,
)
from models.speaker_matcher import UNKNOWN

logger = logging.getLogger("overlap_detection")

_GENERIC_LABEL = re.compile(r"(?i)^speaker[_\s]?\d+$")


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / (np.linalg.norm(vec) + 1e-9)).astype(np.float32)


def speaker_centroids(transcript: list[dict], embeddings: list) -> dict[str, np.ndarray]:
    """Mean voice print per speaker, from single-speaker lines only."""
    groups: dict[str, list[np.ndarray]] = {}
    for line, emb in zip(transcript, embeddings):
        label = line.get("speaker_label")
        if line.get("is_overlap") or not label or emb is None:
            continue
        vec = np.asarray(emb, dtype=np.float32)
        if np.linalg.norm(vec) < 1e-6:
            continue
        groups.setdefault(label, []).append(vec)
    return {label: _unit(np.mean(vecs, axis=0)) for label, vecs in groups.items()}


def window_overlap(
    embedding: np.ndarray,
    main: str,
    centroids: dict[str, np.ndarray],
) -> tuple[float, str]:
    """
    How much better the blend of `main` and its nearest other speaker fits
    this window than either voice alone (> 0 means it fits better), and who
    that other speaker is.
    """
    distances = {label: float(cosine(embedding, c)) for label, c in centroids.items()}
    others = {label: d for label, d in distances.items() if label != main}
    other = min(others, key=others.get)
    blend = _unit(centroids[main] + centroids[other])
    score = min(distances[main], distances[other]) - float(cosine(embedding, blend))
    return score, other


def _line_windows(audio: np.ndarray, start: float, end: float) -> list[np.ndarray]:
    s0 = max(0, int(start * SAMPLE_RATE))
    s1 = min(len(audio), int(end * SAMPLE_RATE))
    clip = audio[s0:s1]
    width = int(OVERLAP_WINDOW_SECONDS * SAMPLE_RATE)
    hop = int(OVERLAP_HOP_SECONDS * SAMPLE_RATE)
    if len(clip) <= width:
        return [clip]
    starts = list(range(0, len(clip) - width + 1, hop))
    if starts[-1] + width < len(clip):
        starts.append(len(clip) - width)
    return [clip[i : i + width] for i in starts]


def _nearest(embedding, centroids: dict[str, np.ndarray]) -> str:
    return min(centroids, key=lambda label: float(cosine(embedding, centroids[label])))


_pipeline = None
_load_error: Exception | None = None


def overlap_regions(audio: np.ndarray) -> list[tuple[float, float]] | None:
    """
    (start, end) seconds where two or more people talk at once, from
    pyannote's segmentation model. None when the model cannot be loaded (no
    pyannote, no Hugging Face token, terms not accepted); the caller then
    falls back to the voice-print test. A failed load is not retried.
    """
    global _pipeline, _load_error

    if _load_error is not None:
        return None
    if _pipeline is None:
        try:
            from huggingface_hub import hf_hub_download
            from pyannote.audio import Model
            from pyannote.audio.pipelines import OverlappedSpeechDetection

            # Downloaded here rather than by pyannote: pyannote 3.3 passes
            # use_auth_token=, which the installed huggingface_hub removed.
            checkpoint = hf_hub_download(OVERLAP_MODEL, "pytorch_model.bin", token=HF_TOKEN or None)
            pipeline = OverlappedSpeechDetection(segmentation=Model.from_pretrained(checkpoint))
            pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
            _pipeline = pipeline
        except Exception as exc:
            _load_error = exc
            logger.warning(
                f"overlap model '{OVERLAP_MODEL}' unavailable ({exc}); using the voice-print "
                f"fallback. Set HF_TOKEN and accept the model terms on huggingface.co to enable it."
            )
            return None

    import torch

    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
    annotation = _pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    return [(seg.start, seg.end) for seg in annotation.get_timeline().support()]


def _intersection(start: float, end: float, regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [
        (max(start, r0), min(end, r1))
        for r0, r1 in regions
        if min(end, r1) - max(start, r0) > 0
    ]


def _joiner(audio, main, centroids, parts, line_start, line_end, embed) -> str:
    """Who talked over `main` in these parts of the line: nearest other voice."""
    votes: dict[str, float] = {}
    for p0, p1 in parts:
        # A voice print needs about a second of audio; widen short regions
        # around their centre, inside the line.
        mid = (p0 + p1) / 2
        half = max((p1 - p0) / 2, 0.75)
        w0, w1 = max(line_start, mid - half), min(line_end, mid + half)
        clip = audio[int(w0 * SAMPLE_RATE) : int(w1 * SAMPLE_RATE)]
        if len(clip) < int(0.5 * SAMPLE_RATE):
            continue
        _, other = window_overlap(embed(clip.astype(np.float32)), main, centroids)
        votes[other] = votes.get(other, 0.0) + (p1 - p0)
    return max(votes, key=votes.get) if votes else ""


def detect_overlaps(
    transcript: list[dict],
    embeddings: list,
    audio: np.ndarray,
    *,
    embed=None,
    find_regions=None,
) -> tuple[list[dict], dict]:
    """
    Re-decide which lines hold two voices at once.

    transcript and embeddings are aligned one-to-one (one voice print per
    line). Returns a new transcript list and a small info dict. Lines whose
    overlap came from real source separation are left alone.
    """
    if embed is None:
        from models.embedding import get_embedding as embed
    if find_regions is None:
        find_regions = overlap_regions

    info = {"applied": False, "overlaps": 0}
    if not transcript or audio is None or len(embeddings) != len(transcript):
        info["reason"] = "no_data"
        return transcript, info

    centroids = speaker_centroids(transcript, embeddings)
    lines = [dict(line) for line in transcript]

    # Undo the streaming guess first: its labels may predate the recluster.
    for line, emb in zip(lines, embeddings):
        if line.get("is_overlap") and not line.get("is_separated_overlap") and centroids:
            label = _nearest(np.asarray(emb, dtype=np.float32), centroids)
            _make_single(line, label)

    if len(centroids) < 2:
        info["reason"] = "single_speaker"
        info["applied"] = True
        return lines, info

    regions = find_regions(audio)
    info["source"] = "model" if regions is not None else "voice_prints"

    for line in lines:
        if line.get("is_separated_overlap"):
            continue
        main = line.get("speaker_label")
        start = float(line.get("start_sec") or 0)
        end = float(line.get("end_sec") or 0)
        if main not in centroids:
            continue

        if regions is not None:
            parts = _intersection(start, end, regions)
            if sum(p1 - p0 for p0, p1 in parts) < OVERLAP_MIN_REGION_SECONDS:
                continue
            other = _joiner(audio, main, centroids, parts, start, end, embed)
            if other:
                _make_overlap(line, main, other)
                info["overlaps"] += 1
                logger.info(f"[{start:.1f}-{end:.1f}s] overlap: {main} + {other} (model)")
            continue

        if end - start < OVERLAP_MIN_LINE_SECONDS:
            continue

        joined: dict[str, int] = {}
        for window in _line_windows(audio, start, end):
            if len(window) < int(0.5 * SAMPLE_RATE):
                continue
            score, other = window_overlap(embed(window.astype(np.float32)), main, centroids)
            if score > OVERLAP_MIN_SCORE:
                joined[other] = joined.get(other, 0) + 1

        if joined:
            other = max(joined, key=joined.get)
            if joined[other] >= OVERLAP_MIN_WINDOWS:
                _make_overlap(line, main, other)
                info["overlaps"] += 1
                logger.info(f"[{start:.1f}-{end:.1f}s] overlap: {main} + {other} ({joined[other]} window(s))")

    info["applied"] = True
    return lines, info


def _make_single(line: dict, label: str) -> None:
    line["is_overlap"] = False
    line["speaker_label"] = label
    line["speaker"] = label
    line["identified_as"] = UNKNOWN
    line["confidence"] = 0.0
    line["candidate_labels"] = [label]
    line["candidate_speakers"] = [label]


def _make_overlap(line: dict, main: str, other: str) -> None:
    # The line's own speaker first: they were talking, the other joined in.
    combined = f"{main} + {other}"
    line["is_overlap"] = True
    line["speaker_label"] = combined
    line["speaker"] = combined
    line["identified_as"] = combined
    line["confidence"] = 0.5
    line["candidate_labels"] = [main, other]
    line["candidate_speakers"] = [main, other]


def detect_overlaps_from_audio(audio_path: str, transcript: list[dict]) -> list[dict]:
    """
    Same pass for a meeting that is already stored (the identify-speakers
    action re-clusters it, which renames speakers under the overlap lines).
    Returns the transcript unchanged when the audio cannot be read.
    """
    from audio_utils import load_audio_file
    from models.offline_diarizer import _embedding_for_line

    try:
        audio = load_audio_file(audio_path)
    except Exception:
        logger.exception("overlap detection: failed to load audio")
        return transcript

    embeddings = []
    for line in transcript:
        start = float(line.get("start_sec") or 0)
        emb = _embedding_for_line(audio, start, float(line.get("end_sec") or start))
        embeddings.append(emb if emb is not None else np.zeros(192, dtype=np.float32))
    out, info = detect_overlaps(transcript, embeddings, audio)
    logger.info(f"overlap detection on stored meeting: {info.get('overlaps', 0)} line(s)")
    return out
