"""
Isolated Overlap Resolution & Word/Phrase Attribution Engine.

Performs source separation on overlapping speech windows, transcribes each
separated stream, matches stream embeddings to candidate speaker clusters,
and yields individual speaker turns if confidence threshold is met.

Falls back safely to "Speaker A & Speaker B [Overlapping Speech]" if confidence
is low or feature flag is disabled.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.spatial.distance import cosine

from config import OVERLAP_MIN_CONFIDENCE, OVERLAP_SEPARATION_ENABLED
from models.embedding import get_embedding
from models.source_separation import separate_audio_streams
from models.speaker_matcher import UNKNOWN

logger = logging.getLogger("overlap_attribution")


def resolve_overlap_attribution(
    audio: np.ndarray,
    start: float,
    end: float,
    candidate_labels: list[str],
    diarizer,
    identifier,
    stt_adapter,
    dominant_language: str,
) -> list[dict] | None:
    """
    Attempt source separation and speaker attribution on an overlapping audio clip.

    Returns a list of attributed transcript entry dicts if resolution succeeds with
    high confidence. Returns None if feature is disabled, confidence is low, or
    separation fails (triggering safe fallback).
    """
    if not OVERLAP_SEPARATION_ENABLED:
        return None

    if not candidate_labels or len(candidate_labels) < 2:
        return None

    streams = separate_audio_streams(audio)
    if not streams:
        logger.info(f"[{start:.1f}-{end:.1f}s] source separation unavailable or failed -> fallback")
        return None

    s1, s2 = streams

    # Compute ECAPA embeddings for separated streams
    emb1 = get_embedding(s1)
    emb2 = get_embedding(s2)

    # Match stream embeddings to candidate centroids
    best_c1_spk, best_c1_dist = _match_to_candidates(emb1, candidate_labels, diarizer)
    best_c2_spk, best_c2_dist = _match_to_candidates(emb2, candidate_labels, diarizer)

    # If both streams matched the exact same candidate, enforce distinct assignment if 2 candidates exist
    if best_c1_spk == best_c2_spk and len(candidate_labels) >= 2:
        other = candidate_labels[1] if candidate_labels[0] == best_c1_spk else candidate_labels[0]
        if best_c1_dist <= best_c2_dist:
            best_c2_spk = other
        else:
            best_c1_spk = other

    # Transcribe separated streams with explicit dominant language hint to prevent script corruption
    asr1 = stt_adapter.transcribe(s1, language=dominant_language, hint_language=dominant_language)
    asr2 = stt_adapter.transcribe(s2, language=dominant_language, hint_language=dominant_language)

    text1 = (asr1.get("text") or "").strip()
    text2 = (asr2.get("text") or "").strip()

    if not text1 and not text2:
        logger.info(f"[{start:.1f}-{end:.1f}s] both separated streams transcribed empty -> fallback")
        return None

    # Calculate separation confidence metric based on stream distinctness and embedding match
    emb_dist_between_streams = float(cosine(emb1, emb2))
    match_conf = max(0.0, 1.0 - (best_c1_dist + best_c2_dist) / 2.0)
    separation_confidence = min(1.0, (emb_dist_between_streams + match_conf) / 2.0)

    logger.info(
        f"[{start:.1f}-{end:.1f}s] overlap separation result: "
        f"stream1 ({best_c1_spk})='{text1[:30]}...', stream2 ({best_c2_spk})='{text2[:30]}...' "
        f"(confidence={separation_confidence:.3f})"
    )

    if separation_confidence < OVERLAP_MIN_CONFIDENCE:
        logger.info(
            f"[{start:.1f}-{end:.1f}s] overlap confidence {separation_confidence:.3f} < "
            f"{OVERLAP_MIN_CONFIDENCE} -> falling back to safe overlap entry"
        )
        return None

    # Construct attributed transcript entries & attributed_spans list
    attributed_spans = []
    entries = []
    for spk_label, asr_res, emb in [(best_c1_spk, asr1, emb1), (best_c2_spk, asr2, emb2)]:
        txt = (asr_res.get("text") or "").strip()
        if not txt:
            continue

        disp, conf = identifier.identify(emb)
        display_name = disp if disp != UNKNOWN else spk_label

        span = {
            "speaker": display_name,
            "speaker_label": spk_label,
            "text": txt,
        }
        attributed_spans.append(span)

        entries.append({
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "speaker": display_name,
            "speaker_label": spk_label,
            "identified_as": disp,
            "confidence": conf,
            "is_overlap": True,
            "is_separated_overlap": True,
            "separation_confidence": round(separation_confidence, 3),
            "candidate_speakers": [display_name],
            "candidate_labels": [spk_label],
            "language": asr_res.get("language", dominant_language),
            "language_name": asr_res.get("language_name"),
            "language_prob": asr_res.get("language_prob", 1.0),
            "raw_text": txt,
            "text": txt,
            "is_turn": False,
        })

    for entry in entries:
        entry["attributed_spans"] = attributed_spans

    return entries if entries else None


def _match_to_candidates(
    embedding: np.ndarray,
    candidate_labels: list[str],
    diarizer,
) -> tuple[str, float]:
    best_label = candidate_labels[0]
    best_dist = 1.0
    for label in candidate_labels:
        if label in diarizer.clusters:
            centroid = diarizer.get_centroid(label)
            dist = float(cosine(embedding, centroid))
            if dist < best_dist:
                best_dist = dist
                best_label = label
    return best_label, best_dist
