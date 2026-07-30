"""
Smoke tests for `api.py`'s route functions.

These exist because of a real outage: a helper added for the summarization stage
was called `_summarize`, the same name as the pre-existing list-view trimmer, and
silently shadowed it. `GET /api/meetings` returned 500 on every call — the whole
app showed "Cannot reach the API" — while `GET /api/meetings/{id}` kept working,
because it does not use that helper. Nothing caught it: `api.py` had no tests,
and the browser reported it as a CORS failure, because a 500 raised inside a
route never reaches the CORS middleware that would have added the header.

The lesson these tests encode is narrow and cheap to keep: **call every list
endpoint once.** They run against SQLite through the same conftest fixtures as
the repository suite; the route functions are called directly, so no server and
no ASGI client is needed.
"""

from __future__ import annotations

from conftest import make_record

import api


def test_list_and_detail_helpers_are_distinct():
    """
    The two helpers must not share a name.

    This is the regression guard: module-level `def` silently rebinds, so the
    collision produced no import error, no warning, and a 500 at request time.
    """
    assert api._without_transcript is not api._generate_summary


def test_list_meetings_runs_and_omits_transcripts():
    """The endpoint that was returning 500. Exercised end to end."""
    record = make_record()
    api.store.add_meeting(record)

    payload = api.list_meetings()

    assert "meetings" in payload
    entry = next(m for m in payload["meetings"] if m["id"] == record["id"])
    # The transcript body is dropped for list views, replaced by its length.
    assert "transcript" not in entry
    assert entry["transcriptLineCount"] == len(record["transcript"])
    # Fields the list screens do render must survive the trim.
    assert "speakerStats" in entry
    assert "summary" in entry
    assert "summaryEngine" in entry


def test_get_meeting_detail_keeps_the_transcript():
    record = make_record()
    api.store.add_meeting(record)

    detail = api.get_meeting(record["id"])

    assert len(detail["transcript"]) == len(record["transcript"])


def test_generate_summary_is_disabled_cleanly(monkeypatch):
    """
    With the stage switched off it must return no fields at all, not empty ones.

    An empty dict means `update_meeting` is called with nothing to change; a dict
    of empty values would wipe a summary that already exists.
    """
    monkeypatch.setattr(api, "SUMMARY_ENABLED", False)
    assert api._generate_summary("MTG-whatever", make_record()["transcript"]) == {}


def test_generate_summary_survives_a_broken_summarizer(monkeypatch):
    """
    A transcript that cost two minutes of Whisper time must not be lost because
    the summarizer raised. The caller completes the meeting either way.
    """
    def explode(*args, **kwargs):
        raise RuntimeError("ollama fell over")

    monkeypatch.setattr(api.summarizer, "summarize", explode)
    assert api._generate_summary("MTG-whatever", make_record()["transcript"]) == {}
