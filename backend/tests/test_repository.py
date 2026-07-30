"""
Tests for db/repository.py — the only module the API talks to for persistence.

The properties worth protecting here are the ones that were true of the old
JSON store and must stay true now that a real database is underneath:

- the dict shapes crossing the boundary are byte-for-byte what the frontend
  already consumes (camelCase, same keys, same nesting);
- ordering is explicit, because speaker colours depend on first-appearance
  order and SQL guarantees no ordering without ORDER BY;
- `update_meeting(progress=...)` — called once per transcribed segment — must
  not touch the transcript.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest
from sqlalchemy import select

from conftest import make_record
from db import repository as repo
from db.models import Meeting, TranscriptLine
from db.session import session_scope


# ------------------------------------------------------------------ round trip


def test_add_and_get_round_trips_every_field():
    record = make_record()
    repo.add_meeting(record)

    stored = repo.get_meeting(record["id"])
    assert stored is not None

    for field in (
        "id",
        "title",
        "agenda",
        "fileName",
        "fileType",
        "fileSizeBytes",
        "fileSizeLabel",
        "status",
        "progress",
        "duration",
        "durationSeconds",
        "participants",
        "language",
    ):
        assert stored[field] == record[field], field

    assert stored["transcript"] == record["transcript"]
    assert stored["speakerStats"] == record["speakerStats"]
    assert stored["languages"] == record["languages"]


def test_get_missing_meeting_returns_none():
    assert repo.get_meeting("MTG-does-not-exist") is None


def test_summarization_fields_are_empty_not_absent():
    """The API contract returns these keys as empty lists; nothing populates
    them because no summarization stage exists."""
    repo.add_meeting(make_record())
    stored = repo.get_meeting("MTG-test000000")

    assert stored["decisions"] == []
    assert stored["actionItems"] == []
    assert stored["keywords"] == []
    assert stored["summary"] is None


def test_devanagari_survives_the_round_trip():
    repo.add_meeting(make_record())
    stored = repo.get_meeting("MTG-test000000")
    assert stored["transcript"][1]["text"] == "मच्छरो मुळे तक्रारी वाढल्या आहेत."
    assert stored["transcript"][1]["language"] == "mr"


def test_transcript_order_is_by_position_not_insertion():
    """Speaker colours are assigned in first-appearance order, so order is data."""
    record = make_record()
    repo.add_meeting(record)

    # Reverse the rows' primary keys' natural order by rewriting them out of
    # sequence; position must still drive what comes back.
    with session_scope() as session:
        lines = session.scalars(
            select(TranscriptLine).where(TranscriptLine.meeting_id == record["id"])
        ).all()
        for line in lines:
            line.position = 10 - line.position

    stored = repo.get_meeting(record["id"])
    texts = [t["text"] for t in stored["transcript"]]
    assert texts == [record["transcript"][1]["text"], record["transcript"][0]["text"]]


def test_list_meetings_is_newest_first():
    repo.add_meeting(make_record("MTG-old", uploadedAtISO="2026-01-01T00:00:00+00:00"))
    repo.add_meeting(make_record("MTG-new", uploadedAtISO="2026-07-01T00:00:00+00:00"))
    repo.add_meeting(make_record("MTG-mid", uploadedAtISO="2026-04-01T00:00:00+00:00"))

    assert [m["id"] for m in repo.list_meetings()] == ["MTG-new", "MTG-mid", "MTG-old"]


def test_naive_timestamp_is_read_as_utc():
    repo.add_meeting(make_record(uploadedAtISO="2026-07-01T10:00:00"))
    with session_scope() as session:
        meeting = session.get(Meeting, "MTG-test000000")
        assert meeting.uploaded_at.hour == 10
        # SQLite drops the tzinfo on the way back out; the point under test is
        # that a naive string was not rejected or shifted.
        assert meeting.uploaded_at.replace(tzinfo=dt.timezone.utc).utcoffset() == dt.timedelta(0)


def test_keywords_are_deduplicated_by_word():
    """A unique constraint on (meeting_id, word) would reject a duplicate; the
    repository sums the counts instead."""
    record = make_record(
        keywords=[{"word": "budget", "count": 3}, {"word": "budget", "count": 2}, {"word": "hiring", "count": 1}]
    )
    repo.add_meeting(record)

    stored = repo.get_meeting(record["id"])
    assert stored["keywords"] == [{"word": "budget", "count": 5}, {"word": "hiring", "count": 1}]


# ---------------------------------------------------------------------- update


def test_progress_update_does_not_touch_the_transcript():
    """This is the hot path — it fires once per transcribed segment."""
    record = make_record()
    repo.add_meeting(record)

    repo.update_meeting(record["id"], progress=42)

    stored = repo.get_meeting(record["id"])
    assert stored["progress"] == 42
    assert stored["transcript"] == record["transcript"]
    assert stored["speakerStats"] == record["speakerStats"]
    assert stored["languages"] == record["languages"]


def test_update_replaces_children_when_they_are_passed():
    record = make_record()
    repo.add_meeting(record)

    new_line = dict(record["transcript"][0], text="Replaced.")
    repo.update_meeting(record["id"], transcript=[new_line])

    stored = repo.get_meeting(record["id"])
    assert len(stored["transcript"]) == 1
    assert stored["transcript"][0]["text"] == "Replaced."
    # Untouched collections survive.
    assert len(stored["speakerStats"]) == 2


def test_reprocessing_replaces_children_that_collide_on_their_unique_keys():
    """Regression: transcript_lines is unique on (meeting_id, position) and
    speaker_stats on (meeting_id, name). Rewriting a meeting with the same
    positions and the same speaker names used to raise IntegrityError, because
    the INSERTs were emitted before the DELETEs they replace."""
    record = make_record()
    repo.add_meeting(record)

    rewritten = [dict(t, text=f"{t['text']} (take two)") for t in record["transcript"]]
    repo.update_meeting(
        record["id"],
        transcript=rewritten,
        speakerStats=record["speakerStats"],
        languages=record["languages"],
    )

    stored = repo.get_meeting(record["id"])
    assert [t["text"] for t in stored["transcript"]] == [t["text"] for t in rewritten]
    assert len(stored["speakerStats"]) == 2
    assert len(stored["languages"]) == 2


def test_update_can_clear_a_collection():
    record = make_record()
    repo.add_meeting(record)

    repo.update_meeting(record["id"], transcript=[])
    assert repo.get_meeting(record["id"])["transcript"] == []


def test_update_missing_meeting_returns_none():
    assert repo.update_meeting("MTG-nope", progress=10) is None


def test_failure_path_fields_are_writable():
    record = make_record(status="Processing", progress=30)
    repo.add_meeting(record)

    repo.update_meeting(record["id"], status="Failed", progress=0, error="Stored audio file is missing.")

    stored = repo.get_meeting(record["id"])
    assert stored["status"] == "Failed"
    assert stored["error"] == "Stored audio file is missing."


# ---------------------------------------------------------------------- delete


def test_soft_delete_hides_from_list_but_keeps_children(audio_dir):
    record = make_record()
    repo.add_meeting(record)

    assert repo.delete_meeting(record["id"]) is True
    assert repo.get_meeting(record["id"]) is None
    assert len(repo.list_trash()) == 1
    assert repo.list_trash()[0]["id"] == record["id"]

    with session_scope() as session:
        remaining = session.scalars(
            select(TranscriptLine).where(TranscriptLine.meeting_id == record["id"])
        ).all()
        assert len(remaining) == len(record["transcript"])


def test_restore_moves_meeting_back(audio_dir):
    record = make_record()
    repo.add_meeting(record)
    repo.delete_meeting(record["id"])

    restored = repo.restore_meeting(record["id"])
    assert restored is not None
    assert restored["id"] == record["id"]
    assert repo.get_meeting(record["id"]) is not None
    assert repo.list_trash() == []


def test_delete_missing_meeting_returns_false(audio_dir):
    assert repo.delete_meeting("MTG-nope") is False


def test_soft_delete_keeps_the_recording(audio_dir, tmp_path):
    record = make_record()
    repo.add_meeting(record)

    source = tmp_path / "upload.wav"
    source.write_bytes(b"RIFF")
    repo.save_audio(record["id"], "meeting.wav", str(source))
    assert repo.audio_path(record["id"]) is not None

    repo.delete_meeting(record["id"])
    assert repo.audio_path(record["id"]) is not None


def test_purge_removes_the_recording(audio_dir, tmp_path):
    record = make_record()
    repo.add_meeting(record)

    source = tmp_path / "upload.wav"
    source.write_bytes(b"RIFF")
    repo.save_audio(record["id"], "meeting.wav", str(source))

    repo.delete_meeting(record["id"])
    assert repo.purge_meeting(record["id"]) is True
    assert repo.audio_path(record["id"]) is None
    assert repo.list_trash() == []


def test_row_only_delete_keeps_the_recording(audio_dir, tmp_path):
    """Re-import replaces rows. The audio is the one thing that cannot be
    regenerated, so it must survive."""
    record = make_record()
    repo.add_meeting(record)

    source = tmp_path / "upload.wav"
    source.write_bytes(b"RIFF")
    repo.save_audio(record["id"], "meeting.wav", str(source))

    assert repo.delete_meeting_row_only(record["id"]) is True
    assert repo.get_meeting(record["id"]) is None
    assert repo.audio_path(record["id"]) is not None


# ----------------------------------------------------------------------- audio


def test_save_audio_keeps_the_file_inside_the_meeting_directory(audio_dir, tmp_path):
    """A hostile filename must not escape the storage directory."""
    record = make_record()
    repo.add_meeting(record)

    source = tmp_path / "upload.wav"
    source.write_bytes(b"RIFF")
    destination = repo.save_audio(record["id"], "../../etc/passwd", str(source))

    assert os.path.dirname(destination) == os.path.join(str(audio_dir), record["id"])
    assert os.path.basename(destination) == "passwd"


def test_audio_path_is_none_before_anything_is_saved(audio_dir):
    assert repo.audio_path("MTG-test000000") is None


# ---------------------------------------------------------------------- search
# SQLite takes the ILIKE branch. The Postgres to_tsvector branch is not
# exercised here — see conftest.


def test_search_finds_a_transcript_substring():
    repo.add_meeting(make_record())

    results = repo.search("quarterly planning meeting")
    assert [r["id"] for r in results] == ["MTG-test000000"]
    assert results[0]["matchCount"] == 1
    assert results[0]["matches"][0]["speaker"] == "Speaker 1"


def test_search_matches_devanagari_substring():
    """Whole-token full-text alone misses this: the transcript has "मच्छरो",
    the query is "मच्छर". Substring matching is what makes it hit."""
    repo.add_meeting(make_record())
    assert [r["id"] for r in repo.search("मच्छर")] == ["MTG-test000000"]


def test_search_matches_title_and_agenda():
    repo.add_meeting(make_record())
    assert repo.search("Quarterly")[0]["id"] == "MTG-test000000"
    assert repo.search("hiring")[0]["id"] == "MTG-test000000"


def test_search_on_title_only_reports_zero_transcript_matches():
    repo.add_meeting(make_record(title="Zzz unique title"))
    result = repo.search("Zzz unique")[0]
    assert result["matchCount"] == 0
    assert result["matches"] == []


@pytest.mark.parametrize("needle", ["", "   ", None])
def test_search_with_an_empty_needle_returns_nothing(needle):
    repo.add_meeting(make_record())
    assert repo.search(needle) == []


def test_search_treats_wildcards_as_literal_characters():
    """An unescaped '%' would match every meeting."""
    repo.add_meeting(make_record())
    assert repo.search("%") == []
    assert repo.search("_") == []


def test_search_respects_the_limit():
    for i in range(5):
        repo.add_meeting(make_record(f"MTG-{i}", uploadedAtISO=f"2026-07-0{i + 1}T00:00:00+00:00"))

    results = repo.search("quarterly", limit=2)
    assert len(results) == 2
    # Newest first, same as the list view.
    assert [r["id"] for r in results] == ["MTG-4", "MTG-3"]


def test_search_caps_matches_at_five_but_counts_them_all():
    line = make_record()["transcript"][0]
    transcript = [dict(line, text=f"budget line {i}") for i in range(8)]
    repo.add_meeting(make_record(transcript=transcript))

    result = repo.search("budget line")[0]
    assert result["matchCount"] == 8
    assert len(result["matches"]) == 5


def test_search_misses_return_an_empty_list():
    repo.add_meeting(make_record())
    assert repo.search("nothing here matches this") == []


# --------------------------------------------------------- speaker enrollment


def test_speaker_time_ranges_by_display_name_and_label():
    repo.add_meeting(make_record())
    by_name = repo.speaker_time_ranges("MTG-test000000", speaker="Speaker 1")
    assert by_name == [(0.0, 4.0)]

    by_label = repo.speaker_time_ranges("MTG-test000000", speaker_label="Speaker 2")
    assert by_label == [(4.0, 9.0)]

    either = repo.speaker_time_ranges(
        "MTG-test000000", speaker="Speaker 1", speaker_label="Speaker 2"
    )
    assert either == [(0.0, 4.0), (4.0, 9.0)]


def test_speaker_time_ranges_empty_without_keys():
    repo.add_meeting(make_record())
    assert repo.speaker_time_ranges("MTG-test000000") == []
    assert repo.speaker_time_ranges("MTG-missing", speaker="Speaker 1") == []


def test_rename_speaker_is_cosmetic_only():
    repo.add_meeting(make_record())
    updated = repo.rename_speaker("MTG-test000000", "Speaker 1", "Laxmi")
    assert updated is not None
    assert updated["transcript"][0]["speaker"] == "Laxmi"
    # Diarization id preserved so enrollment can still find the segments.
    assert updated["transcript"][0]["speaker_label"] == "Speaker 1"
    assert updated["speakerStats"][0]["name"] == "Laxmi"


def test_rename_merge_updates_participant_count():
    """Merging Speaker_XX into an existing name must drop the ghost participant."""
    repo.add_meeting(
        make_record(
            participants=3,
            speakerStats=[
                {"name": "anushka", "seconds": 13.0, "time": "13s", "pct": 48.0, "color": "#3b82f6"},
                {"name": "Lakshmi", "seconds": 11.0, "time": "11s", "pct": 41.0, "color": "#a855f7"},
                {"name": "Speaker_02", "seconds": 3.0, "time": "3s", "pct": 11.0, "color": "#10b981"},
            ],
            transcript=[
                {
                    "start_sec": 0.0,
                    "end_sec": 13.0,
                    "time": "00:00",
                    "speaker": "anushka",
                    "speaker_label": "Speaker_00",
                    "identified_as": "anushka",
                    "confidence": 0.9,
                    "color": "#3b82f6",
                    "language": "en",
                    "language_name": "English",
                    "language_prob": 0.99,
                    "language_detected": "en",
                    "language_fallback": False,
                    "text": "Hello Lakshmi.",
                },
                {
                    "start_sec": 13.0,
                    "end_sec": 24.0,
                    "time": "00:13",
                    "speaker": "Lakshmi",
                    "speaker_label": "Speaker_01",
                    "identified_as": "Lakshmi",
                    "confidence": 0.9,
                    "color": "#a855f7",
                    "language": "en",
                    "language_name": "English",
                    "language_prob": 0.99,
                    "language_detected": "en",
                    "language_fallback": False,
                    "text": "Hello Anushka.",
                },
                {
                    "start_sec": 24.0,
                    "end_sec": 27.0,
                    "time": "00:24",
                    "speaker": "Speaker_02",
                    "speaker_label": "Speaker_02",
                    "identified_as": None,
                    "confidence": 0.0,
                    "color": "#10b981",
                    "language": "en",
                    "language_name": "English",
                    "language_prob": 0.99,
                    "language_detected": "en",
                    "language_fallback": False,
                    "text": "So please, I am okay.",
                },
            ],
        )
    )
    updated = repo.rename_speaker("MTG-test000000", "Speaker_02", "anushka")
    assert updated["participants"] == 2
    names = {s["name"] for s in updated["speakerStats"]}
    assert names == {"anushka", "Lakshmi"}
    assert sum(s["seconds"] for s in updated["speakerStats"]) == 27.0


# ------------------------------------------------------------------ misc


def test_now_iso_is_utc_and_parseable():
    parsed = dt.datetime.fromisoformat(repo.now_iso())
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == dt.timedelta(0)
