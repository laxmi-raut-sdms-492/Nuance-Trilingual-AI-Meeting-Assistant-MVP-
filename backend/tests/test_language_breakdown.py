"""
Meeting language breakdown — counted over ASR segments, not merged turns.

A speaker turn can contain more than one language (someone answers in English
and finishes in Marathi) but carries a single dominant label. Counting turns
charges the turn's whole duration to that one label, and the minority language
disappears from the breakdown even though its text is visible in the
transcript.
"""

from __future__ import annotations

from pipeline import MeetingSession


class _NullIdentifier:
    def identify(self, embedding):
        return "Unknown", 0.0

    def enroll(self, name, embedding, overwrite=False):
        return

    def refresh(self, force=True):
        return


def _line(start, end, lang, text="x", label="Speaker_00"):
    return {
        "start_sec": start,
        "end_sec": end,
        "speaker": label,
        "speaker_label": label,
        "language": lang,
        "text": text,
        "raw_text": text,
    }


def _session() -> MeetingSession:
    return MeetingSession("MTG-lang-test", _NullIdentifier())


def _by_code(breakdown):
    return {row["code"]: row["seconds"] for row in breakdown}


def test_breakdown_counts_segments_not_turns():
    """
    One 25s turn: 20s English then 5s Marathi. Reading the merged turn would
    report 25s of whichever language the turn is labelled with and 0s of the
    other. The segments are the truth.
    """
    session = _session()
    session.raw_segments = [
        _line(0, 20, "en"),
        _line(20, 25, "mr"),
    ]
    # What the turn looks like after the merge — a single line, dominant label.
    session.transcript = [_line(0, 25, "en", text="merged turn")]

    seconds = _by_code(session.language_breakdown())
    assert seconds == {"en": 20.0, "mr": 5.0}


def test_minority_language_does_not_vanish():
    """Speaker 1 en->mr and Speaker 2 hi->en. All three must survive."""
    session = _session()
    session.raw_segments = [
        _line(0, 20, "en", label="Speaker_00"),
        _line(20, 30, "mr", label="Speaker_00"),
        _line(30, 60, "hi", label="Speaker_01"),
        _line(60, 70, "en", label="Speaker_01"),
    ]
    session.transcript = [
        _line(0, 30, "en", label="Speaker_00"),
        _line(30, 70, "hi", label="Speaker_01"),
    ]

    seconds = _by_code(session.language_breakdown())
    assert seconds == {"en": 30.0, "mr": 10.0, "hi": 30.0}
    assert {row["code"] for row in session.language_breakdown()} == {"en", "hi", "mr"}


def test_percentages_are_over_segment_seconds():
    session = _session()
    session.raw_segments = [_line(0, 75, "en"), _line(75, 100, "hi")]
    session.transcript = [_line(0, 100, "en")]

    pcts = {row["code"]: row["pct"] for row in session.language_breakdown()}
    assert pcts == {"en": 75.0, "hi": 25.0}


def test_falls_back_to_transcript_before_turns_are_built():
    """
    raw_segments is only populated once build_and_apply_turns has run. During
    processing the transcript still holds the unmerged segments, so the upload
    progress sync must get the same answer from either source.
    """
    session = _session()
    assert session.raw_segments == []
    session.transcript = [_line(0, 20, "en"), _line(20, 25, "mr")]

    assert _by_code(session.language_breakdown()) == {"en": 20.0, "mr": 5.0}


def test_segments_without_a_language_are_skipped_not_counted_as_none():
    session = _session()
    session.raw_segments = [
        _line(0, 10, "en"),
        {**_line(10, 15, "mr"), "language": None},
    ]

    seconds = _by_code(session.language_breakdown())
    assert seconds == {"en": 10.0}
    assert None not in seconds


def test_empty_session_returns_empty_breakdown():
    assert _session().language_breakdown() == []
