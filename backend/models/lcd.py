"""
Language Change Detection (LCD).

VAD (models/vad.py) cuts on silence. SCD (models/scd.py) cuts where the
speaker changes. Neither cuts where the LANGUAGE changes, so a sentence like

    "The deadline is Friday म्हणजे उद्या"

— one breath, no pause, one speaker — arrives as a single segment, gets one
language and one engine, and half of it comes back mangled: Whisper forced to
English transliterates the Marathi, or IndicConformer renders the English as
approximate Devanagari.

This is the missing cut. It slides a short window across the segment, asks the
cheap language-ID model (models/lid.py) what each window sounds like, and
splits where the answer changes and stays changed.

It reports BOUNDARIES, not languages. The code it attaches to each piece is
advisory — used for logging and tests. The pipeline re-runs Whisper's
detect_language on each piece, because that is the better detector and each
piece is now language-homogeneous, which is exactly the condition detection
was always reliable under. A cheap model trusted only for "something changed
here" can be wrong about what the languages are and still be worth having.

Structured as a sibling of scd.py on purpose: same windowing, same
minimum-spacing idea, same "return the whole thing unchanged when in doubt".
"""

from __future__ import annotations

import logging

import numpy as np

from config import (
    LCD_ENABLED,
    LCD_HOP_SECONDS,
    LCD_MIN_PIECE_SECONDS,
    LCD_MIN_SEGMENT_SECONDS,
    LCD_MIN_WINDOW_CONFIDENCE,
    LCD_SMOOTHING_WINDOWS,
    LCD_SNAP_FRAME_MS,
    LCD_SNAP_RADIUS_SECONDS,
    LCD_WINDOW_SECONDS,
    SAMPLE_RATE,
)

logger = logging.getLogger("lcd")

WINDOW_SAMPLES = int(LCD_WINDOW_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(LCD_HOP_SECONDS * SAMPLE_RATE)
MIN_SEGMENT_SAMPLES = int(LCD_MIN_SEGMENT_SECONDS * SAMPLE_RATE)
MIN_PIECE_SAMPLES = int(LCD_MIN_PIECE_SECONDS * SAMPLE_RATE)
SNAP_RADIUS_SAMPLES = int(LCD_SNAP_RADIUS_SECONDS * SAMPLE_RATE)
SNAP_FRAME_SAMPLES = max(int(LCD_SNAP_FRAME_MS * SAMPLE_RATE / 1000), 1)

# The model is loaded once per process, so its absence is a single condition,
# not a per-segment one. Without this the warning repeats for every segment of
# every meeting and buries the log it is meant to stand out in.
_warned_unavailable = False


def split_on_language_change(
    audio: np.ndarray,
    *,
    posterior_fn=None,
) -> list[tuple[int, int, str | None]]:
    """
    audio: 1-D float32 PCM, 16kHz mono — one single-speaker segment, already
    through VAD and SCD.

    Returns [(start_sample, end_sample, language_code_or_None), ...]. A single
    element covering the whole segment means no change was found, which is the
    common and cheap case.

    posterior_fn: injection point for tests — anything with the signature of
    lid.language_posterior. Defaults to the real model.
    """
    n = len(audio)
    whole: list[tuple[int, int, str | None]] = [(0, n, None)]

    if not LCD_ENABLED:
        return whole

    # Too short to hold a switch plus two usable pieces on either side.
    if n < MIN_SEGMENT_SAMPLES or n < 2 * MIN_PIECE_SAMPLES or n < WINDOW_SAMPLES:
        return whole

    if posterior_fn is None:
        global _warned_unavailable
        from models import lid

        if not lid.is_available():
            # No model, no split. Deliberately NOT falling back to Whisper per
            # window: that is the cost this module exists to avoid, and a
            # sudden order-of-magnitude slowdown on every upload is a worse
            # failure than leaving segments unsplit, which is merely the
            # behaviour that shipped before this existed.
            if not _warned_unavailable:
                _warned_unavailable = True
                logger.warning(
                    "language-ID model unavailable — segments will not be split on language"
                )
            return whole
        posterior_fn = lid.language_posterior

    labels, starts = _window_labels(audio, posterior_fn)
    if len(labels) < 3:
        return whole

    smoothed = _mode_filter(labels, LCD_SMOOTHING_WINDOWS)
    boundaries = _run_boundaries(smoothed, starts, n)
    if not boundaries:
        return whole

    boundaries = [_snap_to_quietest(audio, b) for b in boundaries]
    pieces = _pieces_from_boundaries(smoothed, starts, boundaries, n)
    pieces = _merge_short_pieces(pieces)

    if len(pieces) < 2:
        return whole

    logger.info(
        "language change at "
        + str([round(p[0] / SAMPLE_RATE, 2) for p in pieces[1:]])
        + "s within segment -> "
        + str([p[2] for p in pieces])
    )
    return pieces


# ------------------------------------------------------------------ windowing


def _window_labels(audio: np.ndarray, posterior_fn) -> tuple[list[str | None], list[int]]:
    """
    One language label per analysis window, plus each window's start sample.

    A window whose top posterior is below LCD_MIN_WINDOW_CONFIDENCE is
    recorded as None — it genuinely does not know, and letting it vote would
    manufacture boundaries out of uninformative audio (a pause, a cough, a
    number read aloud). The mode filter fills those in from their neighbours.
    """
    labels: list[str | None] = []
    starts: list[int] = []

    pos = 0
    n = len(audio)
    while pos + WINDOW_SAMPLES <= n:
        posterior = posterior_fn(audio[pos : pos + WINDOW_SAMPLES]) or {}
        if posterior:
            code = max(posterior, key=posterior.get)
            labels.append(code if posterior[code] >= LCD_MIN_WINDOW_CONFIDENCE else None)
        else:
            labels.append(None)
        starts.append(pos)
        pos += HOP_SAMPLES

    return labels, starts


def _mode_filter(labels: list[str | None], width: int) -> list[str | None]:
    """
    Most common label in a sliding neighbourhood — a median filter for values
    that have no order.

    One window landing on a loanword, a name, or a number should not open a
    new language run; a real switch holds for several consecutive windows and
    survives this untouched.
    """
    if width < 2 or len(labels) < width:
        return _fill_gaps(labels)

    half = width // 2
    out: list[str | None] = []
    for i in range(len(labels)):
        window = [l for l in labels[max(0, i - half) : i + half + 1] if l is not None]
        if not window:
            out.append(None)
            continue
        out.append(max(set(window), key=window.count))
    return _fill_gaps(out)


def _fill_gaps(labels: list[str | None]) -> list[str | None]:
    """Carry the last known label forward, then the first known one backward."""
    filled = list(labels)
    last: str | None = None
    for i, label in enumerate(filled):
        if label is None:
            filled[i] = last
        else:
            last = label

    first_known = next((l for l in filled if l is not None), None)
    return [l if l is not None else first_known for l in filled]


# ----------------------------------------------------------------- boundaries


def _run_boundaries(labels: list[str | None], starts: list[int], n: int) -> list[int]:
    """
    Sample positions where the smoothed label changes.

    Window i is centred at starts[i] + WINDOW_SAMPLES/2. When the label flips
    between window i and i+1, window i still sounded like the old language and
    i+1 like the new one, so the switch is between their centres and the
    midpoint of the two centres is the best estimate available at this hop.
    """
    boundaries: list[int] = []
    for i in range(len(labels) - 1):
        if labels[i] is None or labels[i + 1] is None:
            continue
        if labels[i] == labels[i + 1]:
            continue
        centre_a = starts[i] + WINDOW_SAMPLES // 2
        centre_b = starts[i + 1] + WINDOW_SAMPLES // 2
        boundaries.append(min(max((centre_a + centre_b) // 2, 0), n))
    return boundaries


def _snap_to_quietest(audio: np.ndarray, boundary: int) -> int:
    """
    Nudge a boundary to the quietest instant nearby.

    The hop only locates the switch to within half a hop. People take a very
    short breath when changing language, so the true cut is almost always the
    local energy minimum — and landing on it keeps a word from being sliced in
    half, which is what actually degrades the ASR either side. Short-time RMS
    over a fraction of a second, so this costs nothing next to a forward pass.
    """
    lo = max(0, boundary - SNAP_RADIUS_SAMPLES)
    hi = min(len(audio), boundary + SNAP_RADIUS_SAMPLES)
    if hi - lo < SNAP_FRAME_SAMPLES * 2:
        return boundary

    best_pos, best_energy = boundary, None
    for pos in range(lo, hi - SNAP_FRAME_SAMPLES + 1, SNAP_FRAME_SAMPLES):
        frame = audio[pos : pos + SNAP_FRAME_SAMPLES]
        energy = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        if best_energy is None or energy < best_energy:
            best_pos, best_energy = pos + SNAP_FRAME_SAMPLES // 2, energy

    return best_pos


def _pieces_from_boundaries(
    labels: list[str | None],
    starts: list[int],
    boundaries: list[int],
    n: int,
) -> list[tuple[int, int, str | None]]:
    """Turn cut positions into (start, end, language) spans."""
    ordered = sorted({b for b in boundaries if 0 < b < n})
    edges = [0] + ordered + [n]

    pieces: list[tuple[int, int, str | None]] = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        pieces.append((start, end, _dominant_label(labels, starts, start, end)))
    return pieces


def _dominant_label(
    labels: list[str | None], starts: list[int], start: int, end: int
) -> str | None:
    """Most common window label whose centre falls inside this span."""
    inside = [
        label
        for label, window_start in zip(labels, starts)
        if label is not None and start <= window_start + WINDOW_SAMPLES // 2 < end
    ]
    if not inside:
        return None
    return max(set(inside), key=inside.count)


def _merge_short_pieces(
    pieces: list[tuple[int, int, str | None]],
) -> list[tuple[int, int, str | None]]:
    """
    Absorb any piece below LCD_MIN_PIECE_SECONDS into a neighbour.

    Not dropped — merged. A 0.6s piece handed to Whisper is mostly zero-padded
    silence, which is the exact input distribution that produces "Thank you
    for watching" (see MIN_SPEECH_SECONDS in config.py). Emitting it would
    trade a mangled clause for an invented sentence, and the invented one
    reads as perfectly fluent. Keeping it attached to its neighbour leaves the
    text imperfect but real.

    The survivor keeps its own language label: it is the longer side, so it is
    the side the label was actually derived from.
    """
    if len(pieces) < 2:
        return pieces

    merged = [list(p) for p in pieces]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, (start, end, _label) in enumerate(merged):
            if end - start >= MIN_PIECE_SAMPLES:
                continue
            # Merge into whichever neighbour is longer, so repeated merging
            # converges on the dominant span instead of chaining sideways.
            if i == 0:
                target = 1
            elif i == len(merged) - 1:
                target = i - 1
            else:
                before = merged[i - 1][1] - merged[i - 1][0]
                after = merged[i + 1][1] - merged[i + 1][0]
                target = i - 1 if before >= after else i + 1

            merged[target][0] = min(merged[target][0], start)
            merged[target][1] = max(merged[target][1], end)
            merged.pop(i)
            changed = True
            break

    return [(int(s), int(e), label) for s, e, label in merged]
