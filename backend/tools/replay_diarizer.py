"""
Replay diarization over cached embeddings with different clustering parameters.

Run tools/extract_embeddings.py first. Everything here operates on the cached
vectors, so a full parameter sweep costs seconds instead of a Whisper pass per
configuration.

    python3 -m tools.replay_diarizer MTG-85407e28d5eb --expect 6

What it reports per configuration: how many clusters survive, how much talk
time each holds, and how many are "dust" (under 5 seconds total) — the
signature of fragmentation, where one real speaker is scattered across many
one-off labels.

It also runs offline agglomerative clustering as a reference. That is NOT a
proposed implementation — the pipeline is streaming by design, and offline
clustering needs the whole meeting up front. It answers a different question:
whether the embeddings separate the speakers at all. If offline clustering
cannot find the expected speaker count either, the fault is in segmentation or
the embeddings, and no threshold will fix it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import diarizer as diarizer_module  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage", "embeddings")


def load(meeting_id: str):
    path = os.path.join(CACHE_DIR, f"{meeting_id}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"no cache at {path} — run tools.extract_embeddings first")
    data = np.load(path)
    return data["embeddings"], data["starts"], data["ends"]


class MinDurationDiarizer:
    """
    Wraps SessionDiarizer with one rule: a short segment may not found a new
    speaker.

    ECAPA embeddings from ~1s of audio are noisy, and in the measured run every
    one of the 19 sub-5s clusters was born from a segment under 2s. Under this
    rule a short segment that matches nothing is attached to its nearest
    cluster instead, and — importantly — does NOT update that cluster's
    centroid, so a bad embedding cannot drag a good profile around.
    """

    def __init__(self, inner, min_new_cluster_seconds: float):
        self.inner = inner
        self.min_new_cluster_seconds = min_new_cluster_seconds

    def add_segment(self, start, end, embedding):
        duration = end - start
        if duration >= self.min_new_cluster_seconds or not self.inner.clusters:
            return self.inner.add_segment(start, end, embedding)

        best_label, best_dist, _ = self.inner._find_best_match(embedding)
        threshold = self.inner._effective_threshold(best_label)
        if best_dist < threshold:
            return self.inner.add_segment(start, end, embedding)

        # Too short to be trusted as a new voice: attach, do not learn from it.
        self.inner._last_assigned_label = best_label
        return best_label


def run(embeddings, starts, ends, *, threshold, merge_distance, merge_every,
        merge_on_new=False, min_new_cluster_seconds=0.0,
        growth=None, growth_cap=None, hysteresis=None):
    """
    One streaming diarization pass under the given parameters.

    The diarizer reads its constants as module globals, so they are patched on
    the module rather than on config — the same values config would have
    supplied, without a reimport.
    """
    saved = {}
    overrides = {
        "DIARIZATION_DISTANCE_THRESHOLD": threshold,
        "CLUSTER_MERGE_DISTANCE": merge_distance,
        "CLUSTER_MERGE_CHECK_EVERY": merge_every,
    }
    if growth is not None:
        overrides["THRESHOLD_GROWTH_PER_SEGMENT"] = growth
    if growth_cap is not None:
        overrides["THRESHOLD_GROWTH_CAP"] = growth_cap
    if hysteresis is not None:
        overrides["HYSTERESIS_BONUS"] = hysteresis

    for name, value in overrides.items():
        saved[name] = getattr(diarizer_module, name)
        setattr(diarizer_module, name, value)

    try:
        d = diarizer_module.SessionDiarizer()

        # Record every merge so earlier segments can be re-attributed to the
        # cluster their history ended up in. Without this, a merged-away label
        # would be counted as its own speaker and merging would look like it
        # changed nothing.
        merged_into: dict[str, str] = {}
        original_merge = d._merge

        def tracking_merge(keep_label, absorb_label):
            merged_into[absorb_label] = keep_label
            original_merge(keep_label, absorb_label)

        d._merge = tracking_merge
        front = MinDurationDiarizer(d, min_new_cluster_seconds) if min_new_cluster_seconds else d

        labels = []
        for embedding, start, end in zip(embeddings, starts, ends):
            label = front.add_segment(float(start), float(end), embedding)
            labels.append(label)
            if merge_on_new:
                # The shipped diarizer only considers merging after a MATCH, so
                # a run that keeps minting new clusters checks rarely. This
                # variant also counts the new-cluster path.
                d._maybe_merge_clusters()

        return [_resolve(lb, merged_into) for lb in labels], d
    finally:
        for name, value in saved.items():
            setattr(diarizer_module, name, value)


def _resolve(label: str, merged_into: dict[str, str]) -> str:
    """Follow a merge chain to the surviving label (A absorbed into B, B into C)."""
    seen = set()
    while label in merged_into and label not in seen:
        seen.add(label)
        label = merged_into[label]
    return label


def summarize(labels, starts, ends, dust_seconds=5.0):
    talk = {}
    for label, start, end in zip(labels, starts, ends):
        talk[label] = talk.get(label, 0.0) + float(end - start)
    ordered = sorted(talk.items(), key=lambda kv: -kv[1])
    dust = [lb for lb, secs in ordered if secs < dust_seconds]
    return ordered, dust


def auto_k(embeddings, starts, ends, k_max=10):
    """
    Pick the speaker count by silhouette, with no ground truth supplied.

    This is the question that decides whether a second offline pass is
    implementable: the pass is only useful if k can be chosen automatically.
    Reported per meeting so the choice can be checked against what is known
    about each recording.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    scores = []
    for k in range(2, min(k_max, len(embeddings) - 1) + 1):
        assignment = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(embeddings)
        scores.append((k, silhouette_score(embeddings, assignment, metric="cosine"), assignment))

    best_k, best_score, best_assignment = max(scores, key=lambda row: row[1])
    print("\nsilhouette by k (no ground truth used):")
    for k, score, _ in scores:
        bar = "#" * int(max(score, 0) * 100)
        print(f"  k={k:>2}  {score:+.3f}  {bar}{'   <-- picked' if k == best_k else ''}")

    seconds = {}
    for a, s, e in zip(best_assignment, starts, ends):
        seconds[int(a)] = seconds.get(int(a), 0.0) + float(e - s)
    spread = " ".join(f"{v:.0f}s" for _, v in sorted(seconds.items(), key=lambda kv: -kv[1]))
    print(f"  picked k={best_k} ({best_score:+.3f}) talk-time: {spread}")
    return best_k


def offline_reference(embeddings, starts, ends, expect):
    """Agglomerative clustering over the whole meeting — a ceiling, not a plan."""
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("sklearn not available — skipping offline reference")
        return

    print("\noffline agglomerative (average linkage, cosine) — is the signal even there?")
    for k in range(max(2, expect - 2), expect + 3):
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        assignment = model.fit_predict(embeddings)
        score = silhouette_score(embeddings, assignment, metric="cosine")
        seconds = {}
        for a, s, e in zip(assignment, starts, ends):
            seconds[int(a)] = seconds.get(int(a), 0.0) + float(e - s)
        spread = " ".join(f"{v:.0f}s" for _, v in sorted(seconds.items(), key=lambda kv: -kv[1]))
        print(f"  k={k}  silhouette={score:+.3f}  talk-time: {spread}")

    print("\n  distance_threshold sweep (what a cut would produce unsupervised):")
    for t in (0.30, 0.40, 0.50, 0.55, 0.60, 0.70):
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=t, metric="cosine", linkage="average"
        )
        assignment = model.fit_predict(embeddings)
        print(f"    threshold={t:.2f} -> {len(set(assignment))} clusters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_id")
    parser.add_argument("--expect", type=int, default=6, help="ground-truth speaker count")
    parser.add_argument("--auto-k-only", action="store_true",
                        help="only report the silhouette-picked speaker count")
    args = parser.parse_args()

    if args.auto_k_only:
        embeddings, starts, ends = load(args.meeting_id)
        print(f"{args.meeting_id}: {len(embeddings)} segments, {(ends - starts).sum():.1f}s speech")
        auto_k(embeddings, starts, ends)
        return

    embeddings, starts, ends = load(args.meeting_id)
    print(f"{len(embeddings)} segments, {(ends - starts).sum():.1f}s of speech")

    from config import (
        CLUSTER_MERGE_CHECK_EVERY,
        CLUSTER_MERGE_DISTANCE,
        DIARIZATION_DISTANCE_THRESHOLD,
    )

    print(
        f"\nshipped config: threshold={DIARIZATION_DISTANCE_THRESHOLD} "
        f"merge_distance={CLUSTER_MERGE_DISTANCE} merge_every={CLUSTER_MERGE_CHECK_EVERY}"
    )

    # Reference partition: offline agglomerative at the known speaker count.
    # This is a PSEUDO ground truth — nobody labelled this audio by hand. It is
    # useful because it is the best a clustering algorithm can do with these
    # embeddings when it can see the whole meeting, so a streaming config that
    # agrees with it is at least not fragmenting.
    reference = None
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import adjusted_rand_score

        reference = AgglomerativeClustering(
            n_clusters=args.expect, metric="cosine", linkage="average"
        ).fit_predict(embeddings)
    except ImportError:
        adjusted_rand_score = None

    configs = []
    # Baseline — must reproduce the live run.
    configs.append(dict(threshold=DIARIZATION_DISTANCE_THRESHOLD,
                        merge_distance=CLUSTER_MERGE_DISTANCE,
                        merge_every=CLUSTER_MERGE_CHECK_EVERY))
    # Merge distance alone, everything else untouched.
    for md in (0.45, 0.55, 0.65):
        configs.append(dict(threshold=DIARIZATION_DISTANCE_THRESHOLD, merge_distance=md,
                            merge_every=CLUSTER_MERGE_CHECK_EVERY))
    # Match threshold alone.
    for th in (0.65, 0.75, 0.85):
        configs.append(dict(threshold=th, merge_distance=CLUSTER_MERGE_DISTANCE,
                            merge_every=CLUSTER_MERGE_CHECK_EVERY))
    # Minimum duration to found a new cluster, at the shipped threshold.
    for mn in (1.5, 2.0, 2.5, 3.0):
        configs.append(dict(threshold=DIARIZATION_DISTANCE_THRESHOLD,
                            merge_distance=CLUSTER_MERGE_DISTANCE,
                            merge_every=CLUSTER_MERGE_CHECK_EVERY, min_new_cluster_seconds=mn))
    # Minimum duration combined with a usable merge distance and merge checks
    # on the new-cluster path.
    for mn, th, md in ((2.0, 0.55, 0.45), (2.0, 0.55, 0.55), (2.0, 0.65, 0.55),
                       (2.5, 0.60, 0.50), (2.5, 0.65, 0.55), (3.0, 0.65, 0.55)):
        configs.append(dict(threshold=th, merge_distance=md, merge_every=5,
                            merge_on_new=True, min_new_cluster_seconds=mn))

    header = (f"\n{'thresh':>6} {'merge':>6} {'every':>5} {'new?':>5} {'min_new':>7} | "
              f"{'clusters':>8} {'dust':>4} {'ARI':>6} | top talk-time")
    print(header)
    print("-" * len(header))
    for cfg in configs:
        labels, _ = run(embeddings, starts, ends, **cfg)
        ordered, dust = summarize(labels, starts, ends)
        top = " ".join(f"{secs:.0f}s" for _, secs in ordered[:7])
        ari = ""
        if reference is not None and adjusted_rand_score is not None:
            ari = f"{adjusted_rand_score(reference, labels):+.3f}"
        print(
            f"{cfg['threshold']:>6.2f} {cfg['merge_distance']:>6.2f} {cfg['merge_every']:>5} "
            f"{str(cfg.get('merge_on_new', False))[:5]:>5} "
            f"{cfg.get('min_new_cluster_seconds', 0):>7.1f} | "
            f"{len(ordered):>8} {len(dust):>4} {ari:>6} | {top}"
        )

    print("\nARI = agreement with offline clustering at the expected speaker count.")
    print("Cluster COUNT alone is not the score — 6 clusters holding 326s/42s/4s/2s/1s/1s")
    print("is one voice swallowing the meeting, not six speakers found.")

    offline_reference(embeddings, starts, ends, args.expect)


if __name__ == "__main__":
    main()
