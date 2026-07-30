"""
Match an unknown voice embedding against enrolled speaker profiles.

Pure functions — no DB, no SpeechBrain. SpeakerIdentifier and the pipeline
call into this so cosine matching, thresholding, and ambiguity rejection live
in one place rather than being reimplemented at every call site.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cosine

from config import (
    IDENTIFICATION_AMBIGUITY_MARGIN,
    IDENTIFICATION_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger("speaker_matcher")

UNKNOWN = "Unknown"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [−1, 1]. Both vectors should already be L2-normalized."""
    return float(1.0 - cosine(a, b))


def rank_matches(
    embedding: np.ndarray,
    profiles: dict[str, dict],
) -> list[tuple[str, float]]:
    """
    Score every enrolled profile. Returns [(name, similarity), ...] sorted
    best-first. `profiles` maps name -> {"centroid": np.ndarray, ...}.
    """
    scored: list[tuple[str, float]] = []
    for name, profile in profiles.items():
        scored.append((name, cosine_similarity(embedding, profile["centroid"])))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def match_speaker(
    embedding: np.ndarray,
    profiles: dict[str, dict],
    *,
    threshold: float = IDENTIFICATION_SIMILARITY_THRESHOLD,
    ambiguity_margin: float = IDENTIFICATION_AMBIGUITY_MARGIN,
) -> tuple[str, float]:
    """
    Pick the best enrolled match, or Unknown.

    Rules:
      - no profiles            -> Unknown, 0.0
      - best < threshold       -> Unknown (confidence still reported)
      - top-two within margin  -> Unknown (ambiguous)
      - otherwise              -> best name + similarity
    """
    if not profiles:
        return UNKNOWN, 0.0

    ranked = rank_matches(embedding, profiles)
    best_name, best_score = ranked[0]

    if best_score < threshold:
        logger.info(
            f"match: best '{best_name}' at {best_score:.3f} "
            f"< threshold {threshold} -> {UNKNOWN}"
        )
        return UNKNOWN, round(best_score, 3)

    if len(ranked) > 1:
        second_name, second_score = ranked[1]
        if (best_score - second_score) < ambiguity_margin and second_score >= threshold:
            logger.info(
                f"match: ambiguous '{best_name}' ({best_score:.3f}) vs "
                f"'{second_name}' ({second_score:.3f}) "
                f"(margin {ambiguity_margin}) -> {UNKNOWN}"
            )
            return UNKNOWN, round(best_score, 3)

    logger.info(f"match: '{best_name}' at {best_score:.3f}")
    return best_name, round(best_score, 3)
