"""
Resegmentation — re-cut speaker boundaries against known voices.

The problem this exists to solve
--------------------------------
models/scd.py finds a speaker change by sliding a window over a segment and
asking whether each window sounds different from *the one before it*. Both
windows are short, so both embeddings are noisy, and the comparison puts one
noisy measurement against another. The errors add.

That forces a choice with no good answer, measured on a 180s council meeting
with a known male->female switch 1.2s into a 6.5s segment:

    window   false cuts   found the real switch?
    2.0s              0   no
    1.5s              0   no
    1.0s              6   yes
    0.6s             12   yes

Wide enough to be reliable is too wide to see a short turn; narrow enough to
see it invents cuts everywhere else. No setting is good, because the question
is wrong.

What this does instead
----------------------
Ask "which of the people in this meeting does this second sound like?" rather
than "is this second different from the last one?".

A speaker centroid is averaged over every confident second that speaker owns,
so it is a far cleaner vector than any single window. Comparing a noisy window
to a clean centroid is roughly twice as accurate as comparing two noisy
windows, and it degrades gracefully: a window that matches nothing well is
marked unknown and inherits its neighbour instead of opening a boundary.

On the same segment, 1.0s windows scored against centroids separate cleanly
where 1.0s windows scored against each other produced six false cuts:

    t       vs male   vs female
    22.30     0.610       0.080
    22.80     0.238       0.063
    23.05     0.149       0.240   <- switch
    23.55     0.189       0.673

This runs after diarization, not instead of it: it needs the centroids that
diarization produces. SCD stays, deliberately conservative, because its job is
now only to keep a segment clean enough to build centroids from.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger("resegment")

# Window geometry. 1.0s is the shortest window that still separates speakers
# reliably WHEN compared against a centroid (measured above); the hop sets how
# precisely a boundary can be located, and 0.25s is well under the shortest
# turn worth resolving.
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.25

# A reference is only built from speech long enough to be a trustworthy sample
# of a voice. Short lines are the ones being corrected here; letting them vote
# on the reference would feed the error back in.
MIN_REFERENCE_SECONDS = 2.5

# Pieces shorter than this are absorbed into a neighbour rather than emitted —
# below it there is not enough audio to transcribe or to stand as a turn.
MIN_PIECE_SECONDS = 0.8

# A window whose best match beats its second best by less than this is not
# making a real choice; it inherits rather than votes.
MIN_MARGIN = 0.05

# Majority filter width over per-window labels, in windows. One window landing
# on a laugh or a cough should not open a turn.
SMOOTHING_WINDOWS = 3


def build_references(
    embeddings: list[np.ndarray],
    lines: list[dict],
    *,
    min_seconds: float = MIN_REFERENCE_SECONDS,
) -> dict[str, np.ndarray]:
    """
    One clean centroid per speaker, from their confident speech only.

    Falls back to every line of a speaker when none of them clear the length
    bar, because a noisy reference still beats having no reference at all —
    without one that speaker simply cannot win any window.
    """
    by_label: dict[str, list[np.ndarray]] = {}
    fallback: dict[str, list[np.ndarray]] = {}

    for emb, line in zip(embeddings, lines):
        label = line.get("speaker_label") or line.get("speaker")
        if not label or emb is None:
            continue
        vec = np.asarray(emb, dtype=np.float32)
        if np.linalg.norm(vec) < 1e-6:
            continue
        duration = float(line.get("end_sec") or 0) - float(line.get("start_sec") or 0)
        fallback.setdefault(label, []).append(vec)
        if duration >= min_seconds and not line.get("is_overlap"):
            by_label.setdefault(label, []).append(vec)

    references: dict[str, np.ndarray] = {}
    for label in fallback:
        vectors = by_label.get(label) or fallback[label]
        centroid = np.mean(np.stack(vectors), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-6:
            continue
        references[label] = (centroid / norm).astype(np.float32)
        logger.debug(
            f"reference {label}: {len(vectors)} sample(s)"
            f"{' (fallback, none long enough)' if label not in by_label else ''}"
        )
    return references


def _window_labels(
    audio: np.ndarray,
    start: float,
    end: float,
    references: dict[str, np.ndarray],
    sample_rate: int,
    embed,
) -> list[tuple[float, str | None]]:
    """(window_start_seconds, best_label_or_None) across one span."""
    window = int(WINDOW_SECONDS * sample_rate)
    hop = int(HOP_SECONDS * sample_rate)
    a, b = int(start * sample_rate), int(end * sample_rate)

    out: list[tuple[float, str | None]] = []
    pos = a
    while pos + window <= b:
        try:
            vec = embed(audio[pos : pos + window])
        except Exception as exc:  # one bad window must not lose the segment
            logger.debug(f"window embed failed at {pos / sample_rate:.2f}s: {exc}")
            out.append((pos / sample_rate, None))
            pos += hop
            continue

        scored = sorted(
            ((1.0 - float(cosine(vec, ref)), label) for label, ref in references.items()),
            reverse=True,
        )
        best_score, best_label = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        out.append((pos / sample_rate, best_label if best_score - runner_up >= MIN_MARGIN else None))
        pos += hop
    return out


def _smooth(labels: list[str | None], width: int = SMOOTHING_WINDOWS) -> list[str | None]:
    """Majority filter, then let unknowns inherit from the left."""
    if width < 2 or len(labels) < width:
        smoothed = list(labels)
    else:
        half = width // 2
        smoothed = []
        for i in range(len(labels)):
            votes = [l for l in labels[max(0, i - half) : i + half + 1] if l]
            smoothed.append(max(set(votes), key=votes.count) if votes else None)

    last: str | None = None
    for i, label in enumerate(smoothed):
        if label is None:
            smoothed[i] = last
        else:
            last = label
    # Anything before the first real vote inherits from the right.
    first = next((l for l in smoothed if l), None)
    return [l if l else first for l in smoothed]


def refine_span(
    audio: np.ndarray,
    start: float,
    end: float,
    references: dict[str, np.ndarray],
    *,
    sample_rate: int,
    embed,
) -> list[tuple[float, float, str]]:
    """
    Re-cut one transcript line into (start, end, label) pieces.

    Returns a single span unchanged when the whole thing matches one speaker,
    which is the common case and costs only the window scoring.
    """
    if not references or end - start < 2 * MIN_PIECE_SECONDS:
        return [(start, end, "")]

    scored = _window_labels(audio, start, end, references, sample_rate, embed)
    if len(scored) < 2:
        return [(start, end, "")]

    labels = _smooth([label for _, label in scored])
    times = [t for t, _ in scored]
    if not labels or labels[0] is None or len(set(labels)) == 1:
        return [(start, end, labels[0] or "")]

    # Runs of identical labels -> boundaries. A boundary sits at the start of
    # the first window that voted for the new speaker: that window is the
    # earliest audio the new voice is known to be present in.
    pieces: list[tuple[float, float, str]] = []
    run_label = labels[0]
    run_start = start
    for i in range(1, len(labels)):
        if labels[i] == run_label:
            continue
        pieces.append((run_start, times[i], run_label))
        run_start, run_label = times[i], labels[i]
    pieces.append((run_start, end, run_label))

    # Absorb pieces too short to stand on their own into the longer neighbour.
    merged: list[list] = []
    for piece_start, piece_end, label in pieces:
        if merged and piece_end - piece_start < MIN_PIECE_SECONDS:
            merged[-1][1] = piece_end
            continue
        merged.append([piece_start, piece_end, label])
    while len(merged) > 1 and merged[0][1] - merged[0][0] < MIN_PIECE_SECONDS:
        merged[1][0] = merged[0][0]
        merged.pop(0)

    # Collapse neighbours that ended up with the same label after absorption.
    collapsed: list[list] = []
    for piece in merged:
        if collapsed and collapsed[-1][2] == piece[2]:
            collapsed[-1][1] = piece[1]
        else:
            collapsed.append(piece)

    return [(s, e, l) for s, e, l in collapsed]
