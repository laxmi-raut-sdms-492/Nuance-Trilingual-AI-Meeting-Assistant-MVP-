"""
Offline speaker re-clustering for uploaded meetings.

Streaming diarization is good for live WebSocket, but over-splits on file
uploads (one person → many Speaker_XX). After the full meeting is available,
re-cluster segment embeddings with agglomerative clustering and pick k by
silhouette — no ground-truth speaker count required.

Guard: if the best silhouette is weak, keep the streaming labels (especially
important for true 1-speaker recordings — silhouette cannot score k=1).

TODO(production-upgrade): pyannote/speaker-diarization-3.1 was evaluated and is the
recommended upgrade path once a GPU server deployment is available. It natively handles
unknown speaker counts via neural powerset segmentation and multi-speaker overlap detection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

DEFAULT_K_MAX = 12

try:
    from config import (
        CLUSTER_MERGE_DISTANCE,
        OFFLINE_RECLUSTER_MIN_SILHOUETTE,
        OFFLINE_SILHOUETTE_TIE_EPSILON,
        SINGLE_SPEAKER_P90_DISTANCE,
    )

    SILHOUETTE_MIN_SCORE = float(OFFLINE_RECLUSTER_MIN_SILHOUETTE)
    SILHOUETTE_TIE_EPSILON = float(OFFLINE_SILHOUETTE_TIE_EPSILON)
except Exception:
    CLUSTER_MERGE_DISTANCE = 0.55
    SILHOUETTE_MIN_SCORE = 0.18
    SILHOUETTE_TIE_EPSILON = 0.025
    SINGLE_SPEAKER_P90_DISTANCE = 0.38

# Stricter bar before inventing a split on a meeting the streamer called solo.
SOLO_SPLIT_MIN_SILHOUETTE = 0.28


def likely_single_speaker(embeddings: np.ndarray) -> bool:
    """
    True when all segment embeddings sit within one voice's natural variation.
    Used to collapse false Speaker_00/Speaker_01 splits on solo recordings.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    valid_emb = embeddings[norms > 1e-6]
    n = len(valid_emb)
    if n < 3:
        return True
    dists = [
        float(cosine(valid_emb[i], valid_emb[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    if not dists:
        return True
    p90 = float(np.percentile(dists, 90))
    max_d = float(np.max(dists))
    logger.info(f"single-speaker check: p90 pairwise dist={p90:.3f}, max dist={max_d:.3f}")
    if max_d >= 0.42:
        return False
    return p90 < SINGLE_SPEAKER_P90_DISTANCE


def collapse_to_single_speaker(transcript: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Force all lines to Speaker_00 when embedding spread indicates one voice."""
    updated = []
    for entry in transcript:
        row = dict(entry)
        row["speaker_label"] = "Speaker_00"
        row["speaker"] = row.get("speaker") if not str(row.get("speaker", "")).startswith("Speaker_") else "Speaker_00"
        updated.append(row)
    return updated, {"applied": True, "k": 1, "reason": "single_speaker_collapse"}


def pick_speaker_count(
    embeddings: np.ndarray,
    *,
    k_max: int = DEFAULT_K_MAX,
    min_score: float = SILHOUETTE_MIN_SCORE,
) -> tuple[int, float, np.ndarray] | None:
    """
    Choose k by silhouette over average-linkage cosine agglomerative clusters.

    Returns (k, score, assignment) or None if clustering should not override
    the streaming labels.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms < 1e-6):
        valid_mask = norms > 1e-6
        if np.sum(valid_mask) < 3:
            return None
        embeddings = embeddings[valid_mask]

    n = len(embeddings)
    if n < 3:
        return None

    upper = min(k_max, n - 1)
    if upper < 2:
        return None

    scores: list[tuple[int, float, np.ndarray]] = []
    for k in range(2, upper + 1):
        assignment = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(embeddings)
        if len(set(int(a) for a in assignment)) < 2:
            continue
        try:
            score = float(
                silhouette_score(embeddings, assignment, metric="cosine")
            )
        except Exception:
            continue
        scores.append((k, score, assignment))

    if not scores:
        return None

    peak_score = max(row[1] for row in scores)
    # Prefer the smallest k whose silhouette is nearly as good — stops inventing
    # a 6th/7th speaker when k=5 fits almost as well (common on real meetings).
    close = [row for row in scores if row[1] >= peak_score - SILHOUETTE_TIE_EPSILON]
    # Smallest k among the near-ties, as the comment above describes. Taking
    # the peak score instead makes this line a no-op — `close` is built around
    # the peak — and on real meetings the peak is routinely a k that has split
    # one person in two. Whether the reduction this allows is actually taken is
    # then decided per merge, by _reduction_is_supported.
    best_k, best_score, best_assignment = min(close, key=lambda row: row[0])

    if best_score < min_score:
        logger.info(
            f"offline recluster: best silhouette {best_score:.3f} < {min_score} "
            f"(k={best_k}) — keeping streaming labels"
        )
        return None

    logger.info(
        f"offline recluster: picked k={best_k} silhouette={best_score:.3f} "
        f"(peak={peak_score:.3f}, candidates={len(scores)})"
    )
    return best_k, best_score, best_assignment


def pick_speaker_count_by_distance_threshold(
    embeddings: np.ndarray,
    *,
    distance_threshold: float = 0.48,
) -> tuple[int, float, np.ndarray]:
    """
    Cluster embeddings using AgglomerativeClustering with distance_threshold
    (n_clusters=None). Automatically determines speaker count k based strictly
    on cosine embedding distance.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n < 2:
        return 1, 1.0, np.zeros(n, dtype=np.int32)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    ).fit(embeddings)

    assignment = clustering.labels_
    k = len(set(int(a) for a in assignment))
    score = 0.0
    if k > 1 and n > k:
        try:
            score = float(silhouette_score(embeddings, assignment, metric="cosine"))
        except Exception:
            score = 0.0

    return k, score, assignment


def _assignment_from_transcript(transcript: list[dict]) -> np.ndarray:
    """Map current speaker_label values to integer cluster ids (first-seen order)."""
    mapping: dict[str, int] = {}
    assignment: list[int] = []
    for entry in transcript:
        label = entry.get("speaker_label") or entry.get("speaker") or "Speaker_00"
        if label not in mapping:
            mapping[label] = len(mapping)
        assignment.append(mapping[label])
    return np.asarray(assignment, dtype=np.int32)


def _cluster_stats(
    embeddings: np.ndarray,
    assignment: np.ndarray,
    transcript: list[dict],
) -> dict[int, dict[str, Any]]:
    """Per-cluster centroid, line count, and talk-time seconds."""
    stats: dict[int, dict[str, Any]] = {}
    for idx, cluster_id in enumerate(assignment):
        cid = int(cluster_id)
        stats.setdefault(
            cid,
            {"indices": [], "seconds": 0.0},
        )
        stats[cid]["indices"].append(idx)
        start = float(transcript[idx].get("start_sec") or 0)
        end = float(transcript[idx].get("end_sec") or start)
        stats[cid]["seconds"] += max(end - start, 0.0)

    for cid, data in stats.items():
        idxs = data["indices"]
        stacked = embeddings[idxs]
        valid_stacked = stacked[np.linalg.norm(stacked, axis=1) > 1e-6]
        if len(valid_stacked) > 0:
            centroid = valid_stacked.mean(axis=0)
            norm = float(np.linalg.norm(centroid)) + 1e-9
            data["centroid"] = (centroid / norm).astype(np.float32)
        else:
            data["centroid"] = np.zeros(embeddings.shape[1], dtype=np.float32)
        data["count"] = len(idxs)
    return stats


def _dynamic_fragment_merge_distance(centroids: list[np.ndarray]) -> float:
    """
    Merge threshold derived from how far apart voices actually are in this
    meeting — not a fixed speaker count.
    """
    valid_centroids = [c for c in centroids if np.linalg.norm(c) > 1e-6]
    if len(valid_centroids) < 2:
        return CLUSTER_MERGE_DISTANCE
    dists = [
        float(cosine(valid_centroids[i], valid_centroids[j]))
        for i in range(len(valid_centroids))
        for j in range(i + 1, len(valid_centroids))
    ]
    if not dists:
        return CLUSTER_MERGE_DISTANCE
    # Tiny clusters that sit closer than the lower quartile of all voice pairs
    # are usually the same person over-split (e.g. one short reply → Speaker_06).
    return min(CLUSTER_MERGE_DISTANCE, float(np.percentile(dists, 30)))


def _reduction_is_supported(
    embeddings: np.ndarray,
    streaming_assignment: np.ndarray,
    offline_assignment: np.ndarray,
    transcript: list[dict],
) -> tuple[bool, str]:
    """
    Is the offline pass allowed to end up with fewer speakers than streaming?

    Both directions of getting this wrong are real, and they were each seen in
    a meeting. Streaming over-splits: one person pauses, restarts, and their
    second half is scored against a young cluster from a short noisy segment,
    so it opens Speaker_03 — the offline merge exists precisely to undo that.
    But the merge can also be wrong: two genuinely different people whose
    voices are somewhat alike get folded into one, and a whole participant
    disappears from the meeting.

    A blanket rule cannot serve both. Refusing every reduction (which is what
    "keep the streaming clusters whenever offline wants fewer" amounts to)
    disables the correction the offline pass exists to perform; accepting every
    reduction merges real people.

    So judge the reduction by what it actually merges. Look at each pair of
    streaming clusters this assignment would join and ask whether their voices
    agree:

      similarity >= WITHIN_MEETING_MERGE_SIMILARITY  -> the same voice, merge
      similarity <  INCONCLUSIVE_MERGE_SIMILARITY    -> different people, refuse
      in between                                     -> merge only if one side
                                                        is a fragment, i.e.
                                                        too little speech to
                                                        stand as a speaker

    The middle band is where the evidence genuinely does not decide, and there
    the cluster's size breaks the tie: a few seconds in one short segment is
    the shape of an over-split, not of a participant.
    """
    from config import INCONCLUSIVE_MERGE_SIMILARITY, WITHIN_MEETING_MERGE_SIMILARITY

    streaming_stats = _cluster_stats(embeddings, streaming_assignment, transcript)
    if len(streaming_stats) < 2:
        return True, "nothing_to_merge"

    total_lines = len(transcript)
    min_lines = max(2, total_lines // 25)
    min_seconds = max(3.0, sum(
        max(float(t.get("end_sec") or 0) - float(t.get("start_sec") or 0), 0)
        for t in transcript
    ) / max(total_lines * 2, 1))

    def is_fragment(cid: int) -> bool:
        data = streaming_stats[cid]
        return data["count"] < min_lines or data["seconds"] < min_seconds

    # Which streaming clusters each offline cluster would absorb.
    grouped: dict[int, set[int]] = {}
    for streaming_cid, offline_cid in zip(streaming_assignment, offline_assignment):
        grouped.setdefault(int(offline_cid), set()).add(int(streaming_cid))

    for offline_cid, streaming_cids in grouped.items():
        members = sorted(streaming_cids)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                cA = streaming_stats[a]["centroid"]
                cB = streaming_stats[b]["centroid"]
                if np.linalg.norm(cA) < 1e-6 or np.linalg.norm(cB) < 1e-6:
                    similarity = 0.5
                else:
                    similarity = 1.0 - float(cosine(cA, cB))

                if similarity >= WITHIN_MEETING_MERGE_SIMILARITY:
                    continue
                if similarity < INCONCLUSIVE_MERGE_SIMILARITY:
                    return False, f"voices_too_far:{similarity:.2f}"
                if not (is_fragment(a) or is_fragment(b)):
                    return False, f"inconclusive_and_both_established:{similarity:.2f}"

    return True, "voices_agree"


def merge_fragment_clusters(
    embeddings: np.ndarray,
    assignment: np.ndarray,
    transcript: list[dict],
) -> np.ndarray:
    """
    Fold one-line / very short clusters into their nearest neighbour when
    embeddings agree — fixes Speaker_06 with a single "What do you mean?".
    """
    if len(assignment) < 2:
        return assignment

    assignment = np.asarray(assignment, dtype=np.int32).copy()
    total_lines = len(transcript)
    # Dynamic: a cluster needs at least this much talk-time or this many lines.
    min_lines = max(2, total_lines // 25)
    min_seconds = max(3.0, sum(
        max(float(t.get("end_sec") or 0) - float(t.get("start_sec") or 0), 0)
        for t in transcript
    ) / max(total_lines * 2, 1))

    changed = True
    while changed:
        changed = False
        stats = _cluster_stats(embeddings, assignment, transcript)
        if len(stats) < 2:
            break

        centroids = [stats[cid]["centroid"] for cid in sorted(stats)]
        merge_dist = _dynamic_fragment_merge_distance(centroids)

        # Smallest / shortest clusters first.
        order = sorted(
            stats.keys(),
            key=lambda cid: (stats[cid]["seconds"], stats[cid]["count"]),
        )
        for cid in order:
            if cid not in stats:
                continue
            data = stats[cid]
            if data["count"] >= min_lines and data["seconds"] >= min_seconds:
                continue

            best_other = None
            best_dist = 1.0
            for other_id, other in stats.items():
                if other_id == cid:
                    continue
                cA = data["centroid"]
                cB = other["centroid"]
                if np.linalg.norm(cA) < 1e-6 or np.linalg.norm(cB) < 1e-6:
                    dist = 1.0
                else:
                    dist = float(cosine(cA, cB))
                if dist < best_dist:
                    best_dist = dist
                    best_other = other_id

            max_merge_dist = max(merge_dist, 0.65) if (data["seconds"] < 4.0 or data["count"] < 2) else merge_dist
            if best_other is None or best_dist > max_merge_dist:
                continue

            logger.info(
                f"offline merge: cluster {cid} ({data['count']} lines, "
                f"{data['seconds']:.1f}s) -> {best_other} (dist={best_dist:.3f})"
            )
            assignment[assignment == cid] = best_other
            changed = True
            break

    # Renumber cluster ids to 0..k-1 in first-appearance order.
    seen: dict[int, int] = {}
    renumbered = assignment.copy()
    next_id = 0
    for idx, cid in enumerate(assignment):
        cid = int(cid)
        if cid not in seen:
            seen[cid] = next_id
            next_id += 1
        renumbered[idx] = seen[cid]
    return renumbered


def _labels_equivalent(
    transcript: list[dict],
    assignment: np.ndarray,
) -> bool:
    """True when assignment groups lines the same way as current speaker_label."""
    current_groups: dict[str, set[int]] = {}
    for i, entry in enumerate(transcript):
        label = str(entry.get("speaker_label") or entry.get("speaker") or "")
        current_groups.setdefault(label, set()).add(i)

    new_groups: dict[int, set[int]] = {}
    for i, cluster_id in enumerate(assignment):
        new_groups.setdefault(int(cluster_id), set()).add(i)

    current_parts = frozenset(frozenset(g) for g in current_groups.values())
    new_parts = frozenset(frozenset(g) for g in new_groups.values())
    return current_parts == new_parts


def _labels_by_first_appearance(assignment: np.ndarray) -> dict[int, str]:
    """Map cluster ids to Speaker_00, Speaker_01, … in order of first line."""
    mapping: dict[int, str] = {}
    next_id = 0
    for cluster_id in assignment:
        cid = int(cluster_id)
        if cid not in mapping:
            mapping[cid] = f"Speaker_{next_id:02d}"
            next_id += 1
    return mapping


def apply_assignment_to_transcript(
    transcript: list[dict],
    assignment: np.ndarray,
) -> list[dict]:
    """
    Rewrite speaker_label / speaker on each line from the cluster assignment.
    Human names are cleared back to the new diarization id so finalize / identify
    can re-apply enrollment and greetings on the corrected clusters.
    """
    from models.speaker_matcher import UNKNOWN

    label_map = _labels_by_first_appearance(assignment)
    updated = []
    for entry, cluster_id in zip(transcript, assignment):
        new_label = label_map[int(cluster_id)]
        row = dict(entry)
        row["speaker_label"] = new_label
        row["speaker"] = new_label
        row["identified_as"] = UNKNOWN
        row["confidence"] = 0.0
        updated.append(row)
    return updated


def recluster_from_embeddings(
    transcript: list[dict],
    embeddings: list[np.ndarray] | np.ndarray,
    *,
    streaming_label_count: int | None = None,
    min_score: float | None = None,
    k_max: int = DEFAULT_K_MAX,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Re-cluster transcript lines from aligned embeddings.

    Returns (possibly updated transcript, info dict).
    info["applied"] is True only when labels changed.
    """
    info: dict[str, Any] = {"applied": False, "reason": "skipped"}
    if not transcript or embeddings is None:
        return transcript, info

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if len(emb) != len(transcript):
        info["reason"] = f"length_mismatch:{len(emb)}!={len(transcript)}"
        logger.warning(f"offline recluster: {info['reason']}")
        return transcript, info

    streaming_label_count = len(
        {
            t.get("speaker_label") or t.get("speaker")
            for t in transcript
            if t.get("speaker_label") or t.get("speaker")
        }
    )

    # Filter out zero / uncomputable vectors to prevent "Cosine affinity cannot be used when X contains zero vectors"
    norms = np.linalg.norm(emb, axis=1)
    valid_mask = norms > 1e-6
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 3:
        info["reason"] = "insufficient_valid_embeddings"
        info["streaming_k"] = streaming_label_count
        return transcript, info

    valid_emb = emb[valid_indices]

    # Solo recording guard — do not invent a second speaker from variation.
    if streaming_label_count > 1 and likely_single_speaker(valid_emb):
        new_transcript, solo_info = collapse_to_single_speaker(transcript)
        solo_info["streaming_k"] = streaming_label_count
        solo_info["source"] = "single_speaker"
        logger.info(
            f"offline recluster: collapsed {streaming_label_count} streaming labels -> 1 speaker"
        )
        return new_transcript, solo_info

    score_floor = SILHOUETTE_MIN_SCORE if min_score is None else min_score
    if streaming_label_count <= 1:
        score_floor = max(score_floor, SOLO_SPLIT_MIN_SILHOUETTE)

    streaming_assignment = _assignment_from_transcript(transcript)
    picked = pick_speaker_count(valid_emb, k_max=k_max, min_score=score_floor)

    if picked is not None:
        best_k, best_score, valid_assignment = picked
        assignment = streaming_assignment.copy()
        for idx, v_idx in enumerate(valid_indices):
            assignment[v_idx] = valid_assignment[idx]
        zero_indices = np.where(~valid_mask)[0]
        for z_idx in zero_indices:
            distances = np.abs(valid_indices - z_idx)
            closest_valid_idx = valid_indices[np.argmin(distances)]
            assignment[z_idx] = assignment[closest_valid_idx]

        supported, why = (
            _reduction_is_supported(emb, streaming_assignment, assignment, transcript)
            if best_k < streaming_label_count
            else (True, "no_reduction")
        )

        if not supported:
            logger.info(
                f"offline recluster: silhouette picked k={best_k} < streaming_k={streaming_label_count} "
                f"but the implied merge was rejected ({why}) — keeping streaming speaker clusters"
            )
            assignment = streaming_assignment.copy()
            info.update(
                {
                    "streaming_k": streaming_label_count,
                    "source": "streaming",
                    "reason": why,
                }
            )
        else:
            info.update(
                {
                    "k": best_k,
                    "silhouette": round(best_score, 3),
                    "streaming_k": streaming_label_count,
                    "source": "silhouette",
                }
            )
    else:
        assignment = streaming_assignment.copy()
        info.update(
            {
                "streaming_k": streaming_label_count,
                "source": "streaming",
                "reason": "weak_or_insufficient",
            }
        )

    merged = merge_fragment_clusters(emb, assignment, transcript)
    final_k = len(set(int(a) for a in merged))

    if _labels_equivalent(transcript, merged):
        info["reason"] = "already_matched"
        info["k"] = final_k
        info["applied"] = False
        return transcript, info

    new_transcript = apply_assignment_to_transcript(transcript, merged)
    info["applied"] = True
    info["reason"] = "reclustering"
    info["k"] = final_k
    info["merged_k"] = final_k
    logger.info(
        f"offline recluster applied: streaming_k={streaming_label_count} -> "
        f"k={final_k} (source={info.get('source')})"
    )
    return new_transcript, info


def _embedding_for_line(audio: np.ndarray, start: float, end: float) -> np.ndarray | None:
    """Extract L2-normalized ECAPA embedding for a single transcript line."""
    from config import SAMPLE_RATE
    from models.embedding import get_embedding

    if audio is None or len(audio) == 0:
        return None

    start_i = max(0, int(start * SAMPLE_RATE))
    end_i = min(len(audio), int(end * SAMPLE_RATE))

    if end_i <= start_i or (end_i - start_i) < int(0.1 * SAMPLE_RATE):
        return None

    clip = audio[start_i:end_i]
    rms = float(np.sqrt(np.mean(clip ** 2)))
    if rms < 1e-5:
        return None

    try:
        vec = get_embedding(clip)
        if vec is None or np.all(vec == 0) or np.isnan(vec).any():
            return None
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return (vec / norm).astype(np.float32)
    except Exception:
        return None


def embed_transcript_lines(
    audio_path: str,
    transcript: list[dict],
) -> list[np.ndarray] | None:
    """Compute an ECAPA embedding per transcript line from meeting audio."""
    from audio_utils import load_audio_file

    try:
        audio = load_audio_file(audio_path)
    except Exception:
        logger.exception("offline recluster: failed to load audio")
        return None

    embeddings: list[np.ndarray] = []
    for line in transcript:
        start = float(line.get("start_sec") or 0)
        end = float(line.get("end_sec") or start)
        emb = _embedding_for_line(audio, start, end)
        if emb is None:
            embeddings.append(np.zeros(192, dtype=np.float32))
        else:
            embeddings.append(emb)
    return embeddings


def recluster_transcript_from_audio(
    audio_path: str,
    transcript: list[dict],
    *,
    min_score: float | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Full path: embed each line from audio, then auto-k recluster."""
    embeddings = embed_transcript_lines(audio_path, transcript)
    if not embeddings:
        return transcript, {"applied": False, "reason": "embed_failed"}
    return recluster_from_embeddings(
        transcript, embeddings, min_score=min_score
    )
