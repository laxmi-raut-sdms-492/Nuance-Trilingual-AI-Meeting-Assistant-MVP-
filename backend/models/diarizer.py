"""
Incremental speaker diarization for a live, ongoing meeting.

v4 — evidence-based clustering with embedding profiles and candidate deferral:

  1. EMA centroid updates with a rolling embedding profile per cluster.
  2. Match distance = min(cosine to centroid, cosine to recent profile samples).
  3. Hysteresis — sticky to the speaker who just spoke.
  4. Dynamic threshold — mature clusters get looser matching; threshold also
     adapts to this meeting's observed embedding spread.
  5. Candidate deferral — one anomalous embedding does not mint Speaker_01;
     require consistent evidence before creating a new permanent cluster.
  6. Periodic cluster merging — recover from premature splits.

Embeddings arriving here are assumed L2-normalized (see models/embedding.py).
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
    MIN_NEW_CLUSTER_SECONDS,
    SHORT_FORCE_MATCH_MAX_DISTANCE,
    CLUSTER_PROFILE_MAX_SAMPLES,
    NEW_CLUSTER_EVIDENCE_COUNT,
    NEW_CLUSTER_MARGIN,
    SINGLE_CLUSTER_SPLIT_MULTIPLIER,
)

logger = logging.getLogger("diarizer")


class SessionDiarizer:
    def __init__(self):
        self.clusters: dict[str, dict] = {}
        self._next_label_id = 0
        self._last_assigned_label: str | None = None
        self._segments_since_merge_check = 0
        # Pending evidence for a potential new speaker before minting a label.
        self._candidates: list[dict] = []
        # All segment distances seen — used for adaptive thresholds.
        self._meeting_distances: list[float] = []

    def add_segment(self, start: float, end: float, embedding: np.ndarray) -> str:
        duration = end - start
        if not self.clusters:
            label = self._new_cluster(embedding)
            logger.info(f"[{start:.1f}-{end:.1f}s] no existing clusters -> new {label}")
            return label

        best_label, best_dist, raw_dists = self._find_best_match(embedding)
        self._meeting_distances.append(best_dist)

        threshold = self._effective_threshold(best_label)
        force_same = (
            duration < MIN_NEW_CLUSTER_SECONDS
            and best_dist < SHORT_FORCE_MATCH_MAX_DISTANCE
        )

        if best_dist < threshold or force_same:
            self._update_cluster(best_label, embedding)
            reason = (
                "short but close — keep same speaker"
                if force_same and best_dist >= threshold
                else "matched"
            )
            logger.info(
                f"[{start:.1f}-{end:.1f}s] {reason} {best_label} "
                f"(dist={best_dist:.3f}, threshold={threshold:.3f}, "
                f"count={self.clusters[best_label]['count']})"
            )
            self._last_assigned_label = best_label
            self._maybe_merge_clusters()
            return best_label

        # No confident match — defer or create new cluster based on evidence.
        if self._should_defer_new_cluster(best_label, best_dist, threshold, duration):
            self._record_candidate(embedding, best_label, best_dist, start, end)
            # Do NOT update centroid with outlier — that would drift the cluster
            # toward a different voice and prevent ever splitting.
            logger.info(
                f"[{start:.1f}-{end:.1f}s] deferred new cluster — tentatively {best_label} "
                f"(dist={best_dist:.3f}, threshold={threshold:.3f}, "
                f"candidates={len(self._candidates)})"
            )
            self._last_assigned_label = best_label
            return best_label

        if self._promote_candidate_if_ready(embedding, start, end):
            label = self._last_assigned_label
            self._maybe_merge_clusters()
            return label

        new_label = self._new_cluster(embedding)
        logger.info(
            f"[{start:.1f}-{end:.1f}s] new speaker evidence "
            f"(closest={best_label} dist={best_dist:.3f}, threshold={threshold:.3f}) "
            f"-> {new_label}"
        )
        self._last_assigned_label = new_label
        self._candidates.clear()
        self._segments_since_merge_check = CLUSTER_MERGE_CHECK_EVERY
        self._maybe_merge_clusters()
        return new_label

    def get_centroid(self, label: str) -> np.ndarray:
        return self.clusters[label]["centroid"]

    def get_confidence(self, label: str) -> float:
        count = self.clusters[label]["count"]
        return min(count / 20.0, 1.0)

    # -- internals --

    def _distance_to_cluster(self, embedding: np.ndarray, label: str) -> float:
        """Min distance to centroid or any stored profile sample."""
        data = self.clusters[label]
        dists = [cosine(embedding, data["centroid"])]
        for sample in data.get("embeddings", []):
            dists.append(cosine(embedding, sample))
        return float(min(dists))

    def _cluster_internal_spread(self, label: str) -> float:
        """Typical within-cluster distance — same speaker's natural variation."""
        data = self.clusters[label]
        samples = list(data.get("embeddings", [])) + [data["centroid"]]
        if len(samples) < 2:
            return 0.0
        dists = [
            cosine(samples[i], samples[j])
            for i in range(len(samples))
            for j in range(i + 1, len(samples))
        ]
        return float(np.percentile(dists, 75)) if dists else 0.0

    def _meeting_adaptive_bonus(self) -> float:
        """Looser matching when this meeting's voices sit close together."""
        if len(self._meeting_distances) < 5:
            return 0.0
        p75 = float(np.percentile(self._meeting_distances, 75))
        # Meetings where embeddings are naturally tight get extra tolerance.
        if p75 < 0.35:
            return min(0.12, (0.35 - p75) * 0.5)
        return 0.0

    def _find_best_match(self, embedding: np.ndarray) -> tuple[str, float, dict]:
        dists = {}
        for label in self.clusters:
            dist = self._distance_to_cluster(embedding, label)
            if label == self._last_assigned_label:
                dist = max(0.0, dist - HYSTERESIS_BONUS)
            dists[label] = dist
        best_label = min(dists, key=dists.get)
        return best_label, dists[best_label], dists

    def _effective_threshold(self, label: str) -> float:
        count = self.clusters[label]["count"]
        growth = min(count * THRESHOLD_GROWTH_PER_SEGMENT, THRESHOLD_GROWTH_CAP)
        spread = self._cluster_internal_spread(label)
        # Mature clusters with high internal variation need looser thresholds.
        spread_bonus = min(spread * 0.8, 0.10)
        return (
            DIARIZATION_DISTANCE_THRESHOLD
            + growth
            + spread_bonus
            + self._meeting_adaptive_bonus()
        )

    def _should_defer_new_cluster(
        self,
        best_label: str,
        best_dist: float,
        threshold: float,
        duration: float,
    ) -> bool:
        margin = best_dist - threshold
        # Slightly above threshold — likely same speaker with mic/style variation.
        if margin < NEW_CLUSTER_MARGIN:
            return True

        # Single-cluster meeting: require distance well beyond natural spread.
        if len(self.clusters) == 1:
            only = next(iter(self.clusters))
            spread = self._cluster_internal_spread(only)
            adaptive = threshold + spread * SINGLE_CLUSTER_SPLIT_MULTIPLIER
            if best_dist < adaptive:
                return True

        # Short segment with moderate distance — wait for more evidence.
        if duration < MIN_NEW_CLUSTER_SECONDS * 1.5 and margin < NEW_CLUSTER_MARGIN * 2:
            return True

        return False

    def _record_candidate(
        self,
        embedding: np.ndarray,
        assigned_label: str,
        dist: float,
        start: float,
        end: float,
    ) -> None:
        self._candidates.append(
            {
                "embedding": embedding.copy(),
                "assigned": assigned_label,
                "dist": dist,
                "start": start,
                "end": end,
            }
        )
        if len(self._candidates) > 20:
            self._candidates.pop(0)

    def _promote_candidate_if_ready(
        self, embedding: np.ndarray, start: float, end: float
    ) -> bool:
        """
        When enough consecutive outliers cluster together, mint a new speaker.
        Returns True if a new label was created and assigned.
        """
        if len(self._candidates) < NEW_CLUSTER_EVIDENCE_COUNT:
            return False

        recent = self._candidates[-NEW_CLUSTER_EVIDENCE_COUNT:]
        embs = np.stack([c["embedding"] for c in recent] + [embedding])
        # Candidate group must be internally consistent (same novel voice).
        internal = [
            cosine(embs[i], embs[j])
            for i in range(len(embs))
            for j in range(i + 1, len(embs))
        ]
        if internal and float(np.mean(internal)) > 0.25:
            return False

        # And consistently far from the cluster they were tentatively assigned to.
        assigned = recent[0]["assigned"]
        if assigned not in self.clusters:
            return False
        to_assigned = [
            self._distance_to_cluster(e, assigned) for e in embs
        ]
        if float(np.mean(to_assigned)) < self._effective_threshold(assigned):
            return False

        new_label = self._new_cluster(embedding)
        logger.info(
            f"[{start:.1f}-{end:.1f}s] promoted candidate -> {new_label} "
            f"after {NEW_CLUSTER_EVIDENCE_COUNT} consistent outliers"
        )
        self._last_assigned_label = new_label
        self._candidates.clear()
        return True

    def _new_cluster(self, embedding: np.ndarray) -> str:
        label = f"Speaker_{self._next_label_id:02d}"
        self._next_label_id += 1
        self.clusters[label] = {
            "centroid": embedding.copy(),
            "count": 1,
            "embeddings": [embedding.copy()],
        }
        return label

    def _update_cluster(self, label: str, embedding: np.ndarray):
        data = self.clusters[label]
        alpha = DIARIZATION_EMA_ALPHA
        data["centroid"] = alpha * embedding + (1 - alpha) * data["centroid"]
        norm = float(np.linalg.norm(data["centroid"])) + 1e-9
        data["centroid"] = (data["centroid"] / norm).astype(np.float32)
        data["count"] += 1
        samples: list = data.setdefault("embeddings", [])
        samples.append(embedding.copy())
        if len(samples) > CLUSTER_PROFILE_MAX_SAMPLES:
            data["embeddings"] = samples[-CLUSTER_PROFILE_MAX_SAMPLES:]

    def _maybe_merge_clusters(self):
        self._segments_since_merge_check += 1
        if self._segments_since_merge_check < CLUSTER_MERGE_CHECK_EVERY:
            return
        self._segments_since_merge_check = 0

        labels = list(self.clusters.keys())
        merge_dist = self._adaptive_merge_distance()
        for i, label_a in enumerate(labels):
            for label_b in labels[i + 1 :]:
                if label_a not in self.clusters or label_b not in self.clusters:
                    continue
                dist = cosine(
                    self.clusters[label_a]["centroid"],
                    self.clusters[label_b]["centroid"],
                )
                if dist < merge_dist:
                    logger.info(
                        f"merging {label_b} into {label_a} (centroid dist={dist:.3f})"
                    )
                    self._merge(label_a, label_b)

    def _adaptive_merge_distance(self) -> float:
        if len(self.clusters) < 2:
            return CLUSTER_MERGE_DISTANCE
        dists = [
            cosine(self.clusters[a]["centroid"], self.clusters[b]["centroid"])
            for i, a in enumerate(self.clusters)
            for b in list(self.clusters)[i + 1 :]
        ]
        if not dists:
            return CLUSTER_MERGE_DISTANCE
        p30 = float(np.percentile(dists, 30))
        return min(CLUSTER_MERGE_DISTANCE, max(p30, 0.40))

    def _merge(self, keep_label: str, absorb_label: str):
        keep = self.clusters[keep_label]
        absorb = self.clusters.pop(absorb_label)
        total = keep["count"] + absorb["count"]
        keep["centroid"] = (
            keep["centroid"] * keep["count"] + absorb["centroid"] * absorb["count"]
        ) / total
        norm = float(np.linalg.norm(keep["centroid"])) + 1e-9
        keep["centroid"] = (keep["centroid"] / norm).astype(np.float32)
        keep["count"] = total
        keep_samples = keep.setdefault("embeddings", [])
        absorb_samples = absorb.get("embeddings", [])
        keep["embeddings"] = (keep_samples + absorb_samples)[-CLUSTER_PROFILE_MAX_SAMPLES:]
        if self._last_assigned_label == absorb_label:
            self._last_assigned_label = keep_label
