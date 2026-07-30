"""
Meeting persistence, PostgreSQL-backed.

This module is a drop-in replacement for the old JSON `store.py`: same function
names, same arguments, same return shapes (camelCase dicts, exactly as the API
serialises them). `api.py` does not know which one it is talking to, which is
what makes the switch a one-line import change and the rollback equally cheap.

The dict boundary is deliberate. ORM objects are never returned, so no caller
can accidentally hold a detached instance or trigger lazy loading after the
session closes.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import shutil

from sqlalchemy import delete, func, inspect, select

from config import AUDIO_DIR
from db.models import (
    ActionItem,
    Decision,
    Keyword,
    Meeting,
    MeetingLanguage,
    SpeakerStat,
    TranscriptLine,
)
from db.session import session_scope

logger = logging.getLogger("store")


# ------------------------------------------------------------------ mapping
# The API speaks camelCase and the database speaks snake_case. Translation
# lives here and nowhere else.

_MEETING_FIELDS = {
    "title": "title",
    "agenda": "agenda",
    "fileName": "file_name",
    "fileType": "file_type",
    "fileSizeBytes": "file_size_bytes",
    "fileSizeLabel": "file_size_label",
    "status": "status",
    "progress": "progress",
    "error": "error",
    "failedSegments": "failed_segments",
    "duration": "duration",
    "durationSeconds": "duration_seconds",
    "participants": "participants",
    "language": "language",
    "summary": "summary",
    "summaryEngine": "summary_engine",
}


def _to_dict(meeting: Meeting) -> dict:
    """ORM row -> the exact JSON shape the frontend already consumes."""
    return {
        "id": meeting.id,
        "title": meeting.title,
        "agenda": meeting.agenda,
        "fileName": meeting.file_name,
        "fileType": meeting.file_type,
        "fileSizeBytes": meeting.file_size_bytes,
        "fileSizeLabel": meeting.file_size_label,
        "uploadedAtISO": meeting.uploaded_at.isoformat() if meeting.uploaded_at else None,
        "status": meeting.status,
        "progress": meeting.progress,
        "error": meeting.error,
        "failedSegments": meeting.failed_segments,
        "duration": meeting.duration,
        "durationSeconds": meeting.duration_seconds,
        "participants": meeting.participants,
        "language": meeting.language,
        "summary": meeting.summary,
        "summaryEngine": meeting.summary_engine,
        "languages": [
            {"code": l.code, "name": l.name, "seconds": l.seconds, "pct": l.pct}
            for l in meeting.languages
        ],
        "speakerStats": [
            {
                "name": s.name,
                "seconds": s.seconds,
                "time": s.time_label,
                "pct": s.pct,
                "color": s.color,
            }
            for s in meeting.speaker_stats
        ],
        "transcript": [
            {
                "start_sec": t.start_sec,
                "end_sec": t.end_sec,
                "time": t.time_label,
                "speaker": t.speaker,
                "speaker_label": t.speaker_label,
                "identified_as": t.identified_as,
                "confidence": t.confidence,
                "color": t.color,
                "language": t.language,
                "language_name": t.language_name,
                "language_prob": t.language_prob,
                "language_detected": t.language_detected,
                "language_fallback": t.language_fallback,
                "text": t.text,
            }
            for t in meeting.transcript_lines
        ],
        "decisions": [d.text for d in meeting.decisions],
        "actionItems": [
            {"title": a.title, "assignee": a.assignee, "due": a.due, "color": a.color}
            for a in meeting.action_items
        ],
        "keywords": [{"word": k.word, "count": k.count} for k in meeting.keywords],
    }


def _parse_iso(value) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str) and value:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Naive timestamps are assumed UTC — that is what now_iso() produces.
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    return dt.datetime.now(dt.timezone.utc)


def _apply_children(session, meeting: Meeting, record: dict):
    """
    Replace a meeting's child collections from a camelCase record.

    Replace rather than merge: the pipeline recomputes the whole transcript,
    speaker stats and language breakdown together, so a partial update has no
    meaning. Only keys actually present in `record` are touched, which is what
    lets update_meeting(progress=...) avoid rewriting the transcript.

    Replacement is two-phase for a reason. `transcript_lines` is unique on
    (meeting_id, position) and `speaker_stats` on (meeting_id, name), and
    SQLAlchemy's unit of work emits INSERTs before DELETEs within one flush.
    Assigning a new list in one go therefore inserts position 0 while the old
    position 0 is still present, and the constraint rejects it. Emptying the
    collections and flushing the DELETEs first is what makes re-transcribing an
    existing meeting work rather than raising IntegrityError.
    """
    pending: dict[str, list] = {}

    if "languages" in record:
        pending["languages"] = [
            MeetingLanguage(
                position=i,
                code=l.get("code", ""),
                name=l.get("name", ""),
                seconds=l.get("seconds", 0.0) or 0.0,
                pct=l.get("pct", 0.0) or 0.0,
            )
            for i, l in enumerate(record["languages"] or [])
        ]

    if "speakerStats" in record:
        pending["speaker_stats"] = [
            SpeakerStat(
                position=i,
                name=s.get("name", ""),
                seconds=s.get("seconds", 0.0) or 0.0,
                time_label=s.get("time"),
                pct=s.get("pct", 0.0) or 0.0,
                color=s.get("color"),
            )
            for i, s in enumerate(record["speakerStats"] or [])
        ]

    if "transcript" in record:
        pending["transcript_lines"] = [
            TranscriptLine(
                position=i,
                start_sec=t.get("start_sec", 0.0) or 0.0,
                end_sec=t.get("end_sec", 0.0) or 0.0,
                time_label=t.get("time"),
                speaker=t.get("speaker", ""),
                speaker_label=t.get("speaker_label"),
                identified_as=t.get("identified_as"),
                confidence=t.get("confidence", 0.0) or 0.0,
                color=t.get("color"),
                language=t.get("language"),
                language_name=t.get("language_name"),
                language_prob=t.get("language_prob", 0.0) or 0.0,
                language_detected=t.get("language_detected"),
                language_fallback=bool(t.get("language_fallback", False)),
                text=t.get("text", ""),
            )
            for i, t in enumerate(record["transcript"] or [])
        ]

    if "decisions" in record:
        pending["decisions"] = [
            Decision(position=i, text=d if isinstance(d, str) else str(d))
            for i, d in enumerate(record["decisions"] or [])
        ]

    if "actionItems" in record:
        pending["action_items"] = [
            ActionItem(
                position=i,
                title=a.get("title", ""),
                assignee=a.get("assignee"),
                due=a.get("due"),
                color=a.get("color"),
            )
            for i, a in enumerate(record["actionItems"] or [])
        ]

    if "keywords" in record:
        # Deduplicate: a unique constraint on (meeting_id, word) would
        # otherwise reject a keyword list containing the same word twice.
        seen: dict[str, int] = {}
        for k in record["keywords"] or []:
            word = k.get("word", "")
            seen[word] = seen.get(word, 0) + (k.get("count", 0) or 0)
        pending["keywords"] = [
            Keyword(position=i, word=w, count=c) for i, (w, c) in enumerate(seen.items())
        ]

    if not pending:
        return

    # Phase one: drop what is there. Skipped for a meeting that has not been
    # written yet — there is nothing to delete and the object is not flushable.
    if inspect(meeting).persistent:
        for attribute in pending:
            setattr(meeting, attribute, [])
        session.flush()

    # Phase two: insert the replacements.
    for attribute, rows in pending.items():
        setattr(meeting, attribute, rows)


# ------------------------------------------------------------------- public
# Same names and signatures as the old JSON store.


def list_meetings() -> list[dict]:
    with session_scope() as session:
        rows = session.scalars(select(Meeting).order_by(Meeting.uploaded_at.desc())).all()
        return [_to_dict(m) for m in rows]


def get_meeting(meeting_id: str) -> dict | None:
    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        return _to_dict(meeting) if meeting else None


def add_meeting(record: dict) -> dict:
    with session_scope() as session:
        meeting = Meeting(id=record["id"], uploaded_at=_parse_iso(record.get("uploadedAtISO")))
        for camel, column in _MEETING_FIELDS.items():
            if camel in record and record[camel] is not None:
                setattr(meeting, column, record[camel])
        _apply_children(session, meeting, record)
        session.add(meeting)
        session.flush()
        return _to_dict(meeting)


def update_meeting(meeting_id: str, **updates) -> dict | None:
    """
    Partial update. Only the keys passed are written.

    This is the hot path: the pipeline calls it once per processed segment to
    move the progress bar, so it must not rewrite the transcript. Scalar-only
    updates touch a single row and never load the children.
    """
    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None:
            return None

        for camel, column in _MEETING_FIELDS.items():
            if camel in updates:
                setattr(meeting, column, updates[camel])

        if "uploadedAtISO" in updates:
            meeting.uploaded_at = _parse_iso(updates["uploadedAtISO"])

        _apply_children(session, meeting, updates)
        session.flush()
        return _to_dict(meeting)


def rename_speaker(meeting_id: str, old_name: str, new_name: str) -> dict | None:
    """
    Rename a speaker label across one meeting: every transcript line and the
    speaker_stats row.

    `new_name` collapsing onto an existing speaker (e.g. renaming Speaker_04
    to a name already used by Speaker_01) merges their stats rather than
    violating the (meeting_id, name) uniqueness constraint on speaker_stats.
    """
    new_name = new_name.strip()
    if not new_name:
        return None

    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None:
            return None

        session.execute(
            TranscriptLine.__table__.update()
            .where(TranscriptLine.meeting_id == meeting_id, TranscriptLine.speaker == old_name)
            .values(speaker=new_name)
        )

        old_stat = session.scalars(
            select(SpeakerStat).where(
                SpeakerStat.meeting_id == meeting_id, SpeakerStat.name == old_name
            )
        ).first()
        if old_stat is not None:
            existing = session.scalars(
                select(SpeakerStat).where(
                    SpeakerStat.meeting_id == meeting_id, SpeakerStat.name == new_name
                )
            ).first()
            if existing is not None and existing.id != old_stat.id:
                # target name already exists on another speaker in this meeting
                # -- merge seconds/pct into it and drop the old row.
                existing.seconds += old_stat.seconds
                existing.pct += old_stat.pct
                session.delete(old_stat)
            else:
                old_stat.name = new_name

        session.flush()
        meeting = session.get(Meeting, meeting_id)
        return _to_dict(meeting)


def delete_meeting(meeting_id: str) -> bool:
    """Delete the meeting AND its recording. This is what the API exposes."""
    with session_scope() as session:
        # ON DELETE CASCADE removes every child row; no manual cleanup.
        result = session.execute(delete(Meeting).where(Meeting.id == meeting_id))
        if result.rowcount == 0:
            return False

    _delete_audio(meeting_id)
    return True


def delete_meeting_row_only(meeting_id: str) -> bool:
    """
    Delete the database rows but leave the audio on disk.

    Exists for re-import: replacing a meeting's rows must not destroy the
    recording, which is the one piece of data that cannot be regenerated.
    Not exposed by the API.
    """
    with session_scope() as session:
        result = session.execute(delete(Meeting).where(Meeting.id == meeting_id))
        return result.rowcount > 0


def search(needle: str, limit: int = 50) -> list[dict]:
    """
    Full-text search across transcripts, titles and agendas.

    Two matchers, OR'd together:

    1. Full-text over the 'simple' configuration. Not 'english' — that applies
       English stemming and an English stopword list, which is meaningless for
       Hindi and Marathi and would mangle Devanagari tokens. 'simple' only
       lowercases and tokenizes, treating all three languages identically.

    2. Substring (ILIKE), backed by a trigram index.

    The second is not redundant. 'simple' matches whole tokens, and Devanagari
    is heavily inflected: a transcript containing "मच्छरो" would not match a
    search for "मच्छर" on full-text alone. Measured on the real sample — this
    exact case returned zero results before substring matching was added.
    English has the same problem in milder form ("meeting" vs "meetings").

    Falls back to ILIKE alone when the dialect has no full-text support, so
    the same call works against SQLite in tests.
    """
    needle = (needle or "").strip()
    if not needle:
        return []

    with session_scope() as session:
        dialect = session.bind.dialect.name
        # Escape LIKE wildcards so a literal % or _ in a query is not treated
        # as a pattern.
        pattern = "%" + needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"

        if dialect == "postgresql":
            tsquery = func.plainto_tsquery("simple", needle)
            line_match = func.to_tsvector("simple", TranscriptLine.text).op("@@")(
                tsquery
            ) | TranscriptLine.text.ilike(pattern)
            meeting_match = func.to_tsvector(
                "simple", func.coalesce(Meeting.title, "") + " " + func.coalesce(Meeting.agenda, "")
            ).op("@@")(tsquery) | Meeting.title.ilike(pattern) | Meeting.agenda.ilike(pattern)
        else:
            line_match = TranscriptLine.text.ilike(pattern)
            meeting_match = Meeting.title.ilike(pattern) | Meeting.agenda.ilike(pattern)

        hit_ids = set(
            session.scalars(
                select(TranscriptLine.meeting_id).where(line_match).distinct()
            ).all()
        ) | set(session.scalars(select(Meeting.id).where(meeting_match)).all())

        if not hit_ids:
            return []

        meetings = session.scalars(
            select(Meeting)
            .where(Meeting.id.in_(hit_ids))
            .order_by(Meeting.uploaded_at.desc())
            .limit(limit)
        ).all()

        results = []
        for meeting in meetings:
            matches = session.scalars(
                select(TranscriptLine)
                .where(TranscriptLine.meeting_id == meeting.id, line_match)
                .order_by(TranscriptLine.position)
            ).all()
            results.append(
                {
                    "id": meeting.id,
                    "title": meeting.title,
                    "uploadedAtISO": meeting.uploaded_at.isoformat(),
                    "status": meeting.status,
                    "matchCount": len(matches),
                    "matches": [
                        {
                            "time": m.time_label,
                            "speaker": m.speaker,
                            "color": m.color,
                            "language": m.language,
                            "language_name": m.language_name,
                            "text": m.text,
                        }
                        for m in matches[:5]
                    ],
                }
            )
        return results


# ------------------------------------------------------------- audio on disk
# Recordings stay on the filesystem. A 300 MB blob in a row buys nothing and
# costs every query that touches the table.


def audio_path(meeting_id: str) -> str | None:
    directory = os.path.join(AUDIO_DIR, meeting_id)
    if not os.path.isdir(directory):
        return None
    for name in os.listdir(directory):
        return os.path.join(directory, name)
    return None


def save_audio(meeting_id: str, filename: str, source_path: str) -> str:
    """
    Move an uploaded temp file into permanent storage under the meeting's id.

    The original filename is kept inside a per-meeting directory rather than
    used as the key, so two uploads called "recording.m4a" don't collide and a
    hostile filename can't escape the storage directory.
    """
    directory = os.path.join(AUDIO_DIR, meeting_id)
    os.makedirs(directory, exist_ok=True)
    safe_name = os.path.basename(filename) or "recording"
    destination = os.path.join(directory, safe_name)
    shutil.move(source_path, destination)
    return destination


def _delete_audio(meeting_id: str):
    directory = os.path.join(AUDIO_DIR, meeting_id)
    if os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
