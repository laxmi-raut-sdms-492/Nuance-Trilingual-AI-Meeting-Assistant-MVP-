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
import re
import shutil

from sqlalchemy import delete, func, inspect, select

from config import AUDIO_DIR, DEFAULT_PROCESSING_MODE
from models.summarizer import _shorten_action_title
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
    "audioQualityWarning": "audio_quality_warning",
    "failedSegments": "failed_segments",
    "duration": "duration",
    "durationSeconds": "duration_seconds",
    "participants": "participants",
    "language": "language",
    "summary": "summary",
    "summaryEngine": "summary_engine",
    "processingMode": "processing_mode",
    "sttProvider": "stt_provider",
    "insights": "insights",
}


# transcript_lines.language_mix is stored comma-joined ("en,mr") rather than as
# JSON, so SQLite (the test database) and Postgres behave identically. The API
# shape is a list, as the pipeline produces and the UI expects.


def _join_language_mix(codes) -> str | None:
    """['en', 'mr'] -> 'en,mr'. None for a single-language or unmarked line."""
    if not codes:
        return None
    if isinstance(codes, str):
        codes = [c for c in codes.split(",")]
    cleaned = [str(c).strip() for c in codes if str(c).strip()]
    # One language is not a mix; storing it would make every line look
    # code-switched in the UI.
    if len(cleaned) < 2:
        return None
    return ",".join(cleaned)


def _split_language_mix(value: str | None) -> list[str] | None:
    """'en,mr' -> ['en', 'mr']. None when the line was single-language."""
    if not value:
        return None
    codes = [c.strip() for c in value.split(",") if c.strip()]
    return codes or None


def _clean_speaker_stats(raw_stats: list[dict]) -> list[dict]:
    if not raw_stats:
        return []
    # If no composite overlap names exist, return raw_stats untouched to preserve exact DB schema
    has_overlap = any("+" in (s.get("name") or "") or " & " in (s.get("name") or "") for s in raw_stats)
    if not has_overlap:
        return raw_stats

    totals: dict[str, float] = {}
    colors: dict[str, str] = {}

    for s in raw_stats:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        secs = float(s.get("seconds") or 0.0)
        if secs <= 0:
            continue

        if "+" in name or " & " in name:
            parts = [p.strip() for p in re.split(r"\s*(?:\+|\&)\s*", name) if p.strip()]
            share = secs / max(len(parts), 1)
            for p in parts:
                totals[p] = totals.get(p, 0.0) + share
                if p not in colors and s.get("color"):
                    colors[p] = s["color"]
        else:
            totals[name] = totals.get(name, 0.0) + secs
            if s.get("color"):
                colors.setdefault(name, s["color"])

    grand_total = sum(totals.values())
    if grand_total <= 0:
        return []

    from config import SPEAKER_COLORS
    from pipeline import format_duration
    sorted_items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    res = []
    for idx, (spk, secs) in enumerate(sorted_items):
        color = colors.get(spk) or SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
        res.append(
            {
                "name": spk,
                "seconds": round(secs, 1),
                "time": format_duration(secs),
                "pct": round(secs / grand_total * 100, 1),
                "color": color,
            }
        )
    return res


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
        "audioQualityWarning": meeting.audio_quality_warning,
        "failedSegments": meeting.failed_segments,
        "duration": meeting.duration,
        "durationSeconds": meeting.duration_seconds,
        "participants": meeting.participants,
        "language": meeting.language,
        "summary": meeting.summary,
        "summaryEngine": meeting.summary_engine,
        # NULL means "not explicitly set" at the DB level; exposed to
        # callers already resolved to what the pipeline will actually use,
        # so the frontend/API never needs its own copy of the default.
        "processingMode": meeting.processing_mode or DEFAULT_PROCESSING_MODE,
        "sttProvider": meeting.stt_provider,
        "insights": meeting.insights,
        "languages": [
            {"code": l.code, "name": l.name, "seconds": l.seconds, "pct": l.pct}
            for l in meeting.languages
        ],
        "speakerStats": _clean_speaker_stats([
            {
                "name": s.name,
                "seconds": s.seconds,
                "time": s.time_label,
                "pct": s.pct,
                "color": s.color,
            }
            for s in meeting.speaker_stats
        ]),
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
                "language_mix": _split_language_mix(t.language_mix),
                "language_mixed_suspected": t.language_mixed_suspected,
                "language_margin": t.language_margin,
                "is_overlap": bool(t.is_overlap),
                "candidate_speakers": _split_language_mix(t.candidate_speakers),
                "candidate_labels": _split_language_mix(t.candidate_labels),
                "is_separated_overlap": bool(t.is_separated_overlap),
                "separation_confidence": t.separation_confidence,
                "attributed_spans": json.loads(t.attributed_spans) if t.attributed_spans else None,
                "raw_text": t.raw_text,
                "cleaned_text": t.cleaned_text,
                "text": t.cleaned_text or t.text,
            }
            for t in meeting.transcript_lines
        ],
        # The two halves of this are not in tension: the title is what the card
        # shows and is shortened for it, while quote/sourceTime are the verbatim
        # transcript line the item was drawn from. Shortening the title is only
        # safe *because* the untouched quote is carried alongside it — a reader
        # who doubts a terse action item can still see what was actually said.
        "decisions": [
            {
                "text": _shorten_action_title(d.text, max_words=14),
                "quote": d.quote,
                "sourceTime": d.source_time,
            }
            for d in meeting.decisions
        ],
        "actionItems": [
            {
                "title": _shorten_action_title(a.title, max_words=10),
                "assignee": a.assignee,
                "due": a.due,
                "color": a.color,
                "quote": a.quote,
                "sourceTime": a.source_time,
            }
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
                language_mix=_join_language_mix(t.get("language_mix")),
                language_mixed_suspected=bool(t.get("language_mixed_suspected", False)),
                language_margin=(
                    1.0 if t.get("language_margin") is None else float(t["language_margin"])
                ),
                is_overlap=bool(t.get("is_overlap", False)),
                candidate_speakers=_join_language_mix(t.get("candidate_speakers")),
                candidate_labels=_join_language_mix(t.get("candidate_labels")),
                is_separated_overlap=bool(t.get("is_separated_overlap", False)),
                separation_confidence=t.get("separation_confidence"),
                attributed_spans=json.dumps(t["attributed_spans"]) if isinstance(t.get("attributed_spans"), (list, dict)) else (t["attributed_spans"] if isinstance(t.get("attributed_spans"), str) else None),
                raw_text=t.get("raw_text") or t.get("text", ""),
                cleaned_text=t.get("cleaned_text") or t.get("text", ""),
                text=t.get("cleaned_text") or t.get("text", ""),
            )
            for i, t in enumerate(record["transcript"] or [])
        ]

    if "decisions" in record:
        # Accepts either the current {"text", "quote", "sourceTime"} shape or a
        # bare string, so a caller that only ever produced plain text (an old
        # tool script, a hand-written fixture) still writes cleanly.
        pending["decisions"] = [
            Decision(
                position=i,
                text=d.get("text", "") if isinstance(d, dict) else str(d),
                quote=d.get("quote") if isinstance(d, dict) else None,
                source_time=d.get("sourceTime") if isinstance(d, dict) else None,
            )
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
                quote=a.get("quote"),
                source_time=a.get("sourceTime"),
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
        rows = session.scalars(
            select(Meeting)
            .where(Meeting.deleted_at.is_(None))
            .order_by(Meeting.uploaded_at.desc())
        ).all()
        return [_to_dict(m) for m in rows]


def get_meeting(meeting_id: str) -> dict | None:
    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None or meeting.deleted_at is not None:
            return None
        return _to_dict(meeting)


def list_trash() -> list[dict]:
    with session_scope() as session:
        rows = session.scalars(
            select(Meeting)
            .where(Meeting.deleted_at.is_not(None))
            .order_by(Meeting.deleted_at.desc())
        ).all()
        return [_to_dict(m) for m in rows]


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

    Always refreshes `participants` from the surviving speaker set so the UI
    never shows "3 Speakers" after two voices were merged into one.
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
                existing.seconds += old_stat.seconds
                session.delete(old_stat)
            else:
                old_stat.name = new_name

        session.flush()
        # Bulk UPDATE bypasses the identity map — reload lines before rebuilding
        # stats so merged names are reflected in participants / talk-time bars.
        session.expire(meeting)
        meeting = session.get(Meeting, meeting_id)
        _refresh_speaker_stats_from_transcript(session, meeting)
        session.expire(meeting)
        meeting = session.get(Meeting, meeting_id)
        return _to_dict(meeting)


def _format_talk_time(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _refresh_speaker_stats_from_transcript(session, meeting: Meeting) -> None:
    """
    Rebuild talk-time bars and participants from transcript lines.

    Keeps colours from the first line of each display name. Called after
    renames/merges so orphan Speaker_XX rows and stale participant counts
    cannot linger.
    """
    totals: dict[str, float] = {}
    colors: dict[str, str | None] = {}
    order: list[str] = []
    for line in sorted(meeting.transcript_lines, key=lambda t: t.position):
        name = line.speaker
        if name not in totals:
            order.append(name)
            colors[name] = line.color
        totals[name] = totals.get(name, 0.0) + max(0.0, float(line.end_sec) - float(line.start_sec))

    grand = sum(totals.values())
    for stat in list(meeting.speaker_stats):
        session.delete(stat)
    session.flush()

    for position, name in enumerate(order):
        seconds = totals[name]
        session.add(
            SpeakerStat(
                meeting_id=meeting.id,
                position=position,
                name=name,
                seconds=round(seconds, 1),
                time_label=_format_talk_time(seconds),
                pct=round(seconds / grand * 100, 1) if grand > 0 else 0.0,
                color=colors.get(name),
            )
        )

    meeting.participants = len(order)
    session.flush()


def speaker_time_ranges(
    meeting_id: str,
    *,
    speaker: str | None = None,
    speaker_label: str | None = None,
) -> list[tuple[float, float]]:
    """
    (start_sec, end_sec) ranges for one speaker in a meeting.

    Match on display `speaker` and/or diarization `speaker_label` so enrollment
    still works after a cosmetic rename (rename updates `speaker` only).
    """
    if not speaker and not speaker_label:
        return []

    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None:
            return []

        ranges: list[tuple[float, float]] = []
        for line in meeting.transcript_lines:
            matched = False
            if speaker and line.speaker == speaker:
                matched = True
            if speaker_label and line.speaker_label == speaker_label:
                matched = True
            if matched:
                ranges.append((float(line.start_sec), float(line.end_sec)))
        return ranges


def set_speaker_identification(
    meeting_id: str,
    *,
    speaker: str,
    speaker_label: str | None,
    identified_as: str,
    confidence: float,
) -> dict | None:
    """Update identified_as / confidence on lines for one speaker (after rename)."""
    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None:
            return None

        for line in meeting.transcript_lines:
            if line.speaker != speaker and (
                not speaker_label or line.speaker_label != speaker_label
            ):
                continue
            if line.speaker == speaker or (
                speaker_label and line.speaker_label == speaker_label
            ):
                line.identified_as = identified_as
                line.confidence = float(confidence)

        session.flush()
        session.expire(meeting)
        meeting = session.get(Meeting, meeting_id)
        return _to_dict(meeting)


def reassign_transcript_lines(
    meeting_id: str,
    changes: list[dict],
) -> dict | None:
    """
    Reassign individual transcript lines to a different display speaker.

    Used when diarization collapsed two people onto one label and greeting
    turn-taking (or embedding split) tells us which later lines belong to
    the other person. Each change needs start_sec + new_speaker; optionally
    new_speaker_label and confidence.
    """
    if not changes:
        return get_meeting(meeting_id)

    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None:
            return None

        # Next free Speaker_XX id if we need distinct diarization labels.
        used_ids: set[int] = set()
        for line in meeting.transcript_lines:
            label = line.speaker_label or ""
            m = re.match(r"(?i)^speaker[_\s]?(\d+)$", label.strip())
            if m:
                used_ids.add(int(m.group(1)))
        next_id = (max(used_ids) + 1) if used_ids else 0

        by_start = {round(float(c["start_sec"]), 2): c for c in changes if "start_sec" in c}
        for line in meeting.transcript_lines:
            key = round(float(line.start_sec), 2)
            change = by_start.get(key)
            if change is None:
                continue
            new_name = (change.get("new_speaker") or "").strip()
            if not new_name:
                continue
            line.speaker = new_name
            line.identified_as = new_name
            if change.get("confidence") is not None:
                line.confidence = float(change["confidence"])
            # Give the reply its own diarization label so future enroll/match
            # treats the two voices separately.
            new_label = change.get("new_speaker_label")
            if not new_label:
                new_label = f"Speaker_{next_id:02d}"
                next_id += 1
            line.speaker_label = new_label

        session.flush()
        session.expire(meeting)
        meeting = session.get(Meeting, meeting_id)
        _refresh_speaker_stats_from_transcript(session, meeting)
        session.expire(meeting)
        meeting = session.get(Meeting, meeting_id)
        return _to_dict(meeting)


def delete_meeting(meeting_id: str) -> bool:
    """Soft delete: mark the meeting trashed."""
    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None or meeting.deleted_at is not None:
            return False
        meeting.deleted_at = dt.datetime.now(dt.timezone.utc)
    return True


def restore_meeting(meeting_id: str) -> dict | None:
    """Move a meeting out of Trash and back into the active list."""
    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None or meeting.deleted_at is None:
            return None
        meeting.deleted_at = None
        session.flush()
        return _to_dict(meeting)


def purge_meeting(meeting_id: str) -> bool:
    """Permanently delete a trashed meeting AND its recording. No undo."""
    with session_scope() as session:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None or meeting.deleted_at is None:
            return False
        session.execute(delete(Meeting).where(Meeting.id == meeting_id))

    _delete_audio(meeting_id)
    return True


def purge_all_trash() -> int:
    """Permanently delete every trashed meeting and its recording. No undo."""
    with session_scope() as session:
        meeting_ids = list(
            session.scalars(
                select(Meeting.id).where(Meeting.deleted_at.is_not(None))
            ).all()
        )
        if not meeting_ids:
            return 0
        session.execute(delete(Meeting).where(Meeting.deleted_at.is_not(None)))

    for meeting_id in meeting_ids:
        _delete_audio(meeting_id)
    return len(meeting_ids)


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
                select(TranscriptLine.meeting_id)
                .join(Meeting, Meeting.id == TranscriptLine.meeting_id)
                .where(line_match, Meeting.deleted_at.is_(None))
                .distinct()
            ).all()
        ) | set(
            session.scalars(
                select(Meeting.id).where(meeting_match, Meeting.deleted_at.is_(None))
            ).all()
        )

        if not hit_ids:
            return []

        meetings = session.scalars(
            select(Meeting)
            .where(Meeting.id.in_(hit_ids), Meeting.deleted_at.is_(None))
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


def clear_meeting_audio(meeting_id: str) -> None:
    """Remove all stored audio files for a meeting before replacing them."""
    _delete_audio(meeting_id)


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
