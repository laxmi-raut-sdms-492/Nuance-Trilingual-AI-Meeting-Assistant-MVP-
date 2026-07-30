"""
Incremental speaker diarization for a live, ongoing meeting.

v3 redesign — fixes fragmentation and long-meeting drift via:

  1. EMA (not plain running mean) centroid updates — keeps a constant
     adaptation rate instead of freezing after ~15-20 segments.
  2. Hysteresis — the cluster that was just active gets a small distance
     discount, preventing flip-flopping between two close-together voices.
  3. Dynamic threshold — mature, well-established clusters (many segments,
     tight internal variance) get a looser matching threshold, since we
     trust their profile more and real speakers legitimately vary more
     than a fixed threshold assumes (distance from mic, volume, etc).
  4. Periodic cluster merging — if two "different" clusters turn out to be
     close to each other after all, merge them and recover history.

Embeddings arriving here are assumed to already be L2-normalized
(see models/embedding.py) — this makes cosine distance meaningful and
keeps averaging magnitude-invariant.
"""

import logging

import numpy as np
from scipy.spatial.distance import cosine

from config import (
    DIARIZATION_DISTANCE_THRESHOLD,
    DIARIZATION_EMA_ALPHA,
    HYSTERESIS_BONUS,
    THRESHOLD_GROWTH_PER_SEGMENT,
    THRESHOLD_GROWTH_CAP,
    CLUSTER_MERGE_DISTANCE,
    CLUSTER_MERGE_CHECK_EVERY,
)

logger = logging.getLogger("diarizer")


class SessionDiarizer:
    def __init__(self):
        self.clusters: dict[str, dict] = {}  # label -> {"centroid": np.array, "count": int}
        self._next_label_id = 0
        self._last_assigned_label: str | None = None
        self._segments_since_merge_check = 0

    MIN_NEW_CLUSTER_SECONDS = 2.5       

    def add_segment(self, start: float, end: float, embedding: np.ndarray) -> str:
        """Add one segment's embedding and return the speaker label it was assigned to."""
        duration = end - start
        if not self.clusters:
            label = self._new_cluster(embedding)
            logger.info(f"[{start:.1f}-{end:.1f}s] no existing clusters -> new {label}")
            return label

        best_label, best_dist, raw_dists = self._find_best_match(embedding)

        threshold = self._effective_threshold(best_label)
        too_short_for_new = duration < self.MIN_NEW_CLUSTER_SECONDS
        if best_dist < threshold or too_short_for_new:
            self._update_cluster(best_label, embedding)
            reason = "too short for new cluster" if too_short_for_new and best_dist >= threshold else "matched"
            logger.info(
                f"[{start:.1f}-{end:.1f}s] {reason} {best_label} "
                f"(dist={best_dist:.3f}, threshold={threshold:.3f}, "
                f"count={self.clusters[best_label]['count']})"
            )
            self._last_assigned_label = best_label
            self._maybe_merge_clusters()
            return best_label

        new_label = self._new_cluster(embedding)
        logger.info(
            f"[{start:.1f}-{end:.1f}s] no match within threshold "
            f"(closest={best_label} dist={best_dist:.3f}, threshold={threshold:.3f}) "
            f"-> new {new_label}"
        )
        self._last_assigned_label = new_label
        self._maybe_merge_clusters()
        return new_label

    def get_centroid(self, label: str) -> np.ndarray:
        return self.clusters[label]["centroid"]

    def get_confidence(self, label: str) -> float:
        """0-1 score for how established this cluster's profile is (more segments = more confidence)."""
        count = self.clusters[label]["count"]
        return min(count / 20.0, 1.0)

    # -- internals --

    def _find_best_match(self, embedding: np.ndarray) -> tuple[str, float, dict]:
        dists = {}
        for label, data in self.clusters.items():
            dist = cosine(embedding, data["centroid"])
            if label == self._last_assigned_label:
                dist = max(0.0, dist - HYSTERESIS_BONUS)  # sticky: prefer continuing the same speaker
            dists[label] = dist
        best_label = min(dists, key=dists.get)
        return best_label, dists[best_label], dists

    def _effective_threshold(self, label: str) -> float:
        """Mature clusters (many confirmed segments) get a looser threshold."""
        count = self.clusters[label]["count"]
        growth = min(count * THRESHOLD_GROWTH_PER_SEGMENT, THRESHOLD_GROWTH_CAP)
        return DIARIZATION_DISTANCE_THRESHOLD + growth

    def _new_cluster(self, embedding: np.ndarray) -> str:
        label = f"Speaker_{self._next_label_id:02d}"
        self._next_label_id += 1
        self.clusters[label] = {"centroid": embedding.copy(), "count": 1}
        return label

    def _update_cluster(self, label: str, embedding: np.ndarray):
        """EMA update — constant adaptation rate, doesn't freeze in long meetings."""
        data = self.clusters[label]
        alpha = DIARIZATION_EMA_ALPHA
        data["centroid"] = alpha * embedding + (1 - alpha) * data["centroid"]
        data["count"] += 1

    def _maybe_merge_clusters(self):
        """
        Periodically check whether any two clusters have drifted close enough
        together that they're almost certainly the same person (recovers
        from an earlier premature split, e.g. Speaker_01 and Speaker_03
        that are actually one voice).
        """
        self._segments_since_merge_check += 1
        if self._segments_since_merge_check < CLUSTER_MERGE_CHECK_EVERY:
            return
        self._segments_since_merge_check = 0

        labels = list(self.clusters.keys())
        for i, label_a in enumerate(labels):
            for label_b in labels[i + 1 :]:
                if label_a not in self.clusters or label_b not in self.clusters:
                    continue  # one may have been merged away already this pass
                dist = cosine(self.clusters[label_a]["centroid"], self.clusters[label_b]["centroid"])
                if dist < CLUSTER_MERGE_DISTANCE:
                    logger.info(f"merging {label_b} into {label_a} (centroid dist={dist:.3f})")
                    self._merge(label_a, label_b)

    def _merge(self, keep_label: str, absorb_label: str):
        keep = self.clusters[keep_label]
        absorb = self.clusters.pop(absorb_label)
        total = keep["count"] + absorb["count"]
        # weighted average of the two centroids by how much history each has
        keep["centroid"] = (
            keep["centroid"] * keep["count"] + absorb["centroid"] * absorb["count"]
        ) / total
        keep["count"] = total
        if self._last_assigned_label == absorb_label:
            self._last_assigned_label = keep_label
