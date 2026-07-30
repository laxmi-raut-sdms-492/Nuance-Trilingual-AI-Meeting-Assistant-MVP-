"""
REST API for uploaded meetings — the path the React frontend actually uses.

The WebSocket endpoint in main.py handles *live* meetings. This handles the
other half: someone uploads or records a file, we transcribe it through the
exact same pipeline, and every screen reads the result back from here instead
of from the browser's localStorage.

Processing is deliberately serialized behind a single lock. Whisper and
SpeechBrain are not thread-safe, and on CPU two concurrent transcriptions are
slower than the same two run back to back. Uploads return immediately with
status "Processing"; the frontend polls until it flips to "Completed".
"""

import logging
import os
import re
import tempfile
import threading
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

# PostgreSQL-backed persistence. Bound to the name `store` because it is a
# drop-in replacement for the old JSON module of that name — same functions,
# same arguments, same dict shapes. To roll back to file storage, change this
# one line to `import store`; nothing else in this file depends on which is
# in use. The legacy module is kept for exactly that reason.
from db import repository as store
from audio_utils import load_audio_file
from config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    LANGUAGE_NAMES,
    MAX_UPLOAD_MB,
    SAMPLE_RATE,
    SUMMARY_ENABLED,
    SUMMARY_KEYWORD_IDF_MEETINGS,
)
from models import summarizer
from models.identifier import SpeakerIdentifier
from pipeline import MeetingSession, format_duration

logger = logging.getLogger("api")

router = APIRouter(prefix="/api")

# One transcription at a time — see module docstring.
_processing_lock = threading.Lock()

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_COPY_CHUNK = 1024 * 1024


def _identifier() -> SpeakerIdentifier:
    """
    Resolved lazily so enrolled speakers added through /enroll are visible to
    uploads without restarting, and so importing this module doesn't pull in
    the speaker DB.
    """
    import main

    return main.identifier


# ---------------------------------------------------------------- helpers


def _format_bytes(size: int) -> str:
    mb = size / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def _title_from_filename(name: str) -> str:
    stem = re.sub(r"\.[^./\\]+$", "", os.path.basename(name))
    cleaned = re.sub(r"[_-]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else "Untitled Meeting"


def _extension(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename or ""))[1].lower()


def _save_upload_to_temp(upload: UploadFile) -> tuple[str, int]:
    """
    Stream the upload to a temp file, enforcing the size cap as we go.

    Reading `await upload.read()` in one shot would let a large upload sit
    entirely in memory before the size check could reject it, so the cap is
    checked incrementally and the partial file removed on breach.
    """
    suffix = _extension(upload.filename) or ".bin"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = upload.file.read(UPLOAD_COPY_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File is larger than the {MAX_UPLOAD_MB} MB limit.")
                out.write(chunk)
    except Exception:
        os.path.exists(tmp_path) and os.remove(tmp_path)
        raise

    if size == 0:
        os.remove(tmp_path)
        raise HTTPException(400, "Uploaded file is empty.")

    return tmp_path, size


# ---------------------------------------------------------------- endpoints


def _without_transcript(meeting: dict) -> dict:
    """
    List view without the transcript body.

    Named for what it does. It was `_summarize`, which collided with the
    summarization stage's `_generate_summary` below and silently shadowed it —
    `/api/meetings` returned 500 on every call while the detail endpoint, which
    does not use this helper, kept working.

    An hour-long meeting's transcript is a few hundred entries; the list
    screens (dashboard, all meetings, analytics) never render it, so shipping
    every transcript on every list call would grow the payload without bound.
    Speaker stats and language breakdown stay — the charts do use those.
    """
    trimmed = {k: v for k, v in meeting.items() if k != "transcript"}
    trimmed["transcriptLineCount"] = len(meeting.get("transcript") or [])
    return trimmed


@router.get("/meetings")
def list_meetings():
    return {"meetings": [_without_transcript(m) for m in store.list_meetings()]}


@router.get("/meetings/trash")
def get_trash():
    return {"meetings": [_without_transcript(m) for m in store.list_trash()]}


@router.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    meeting = store.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(404, "Meeting not found.")
    return meeting


@router.post("/meetings", status_code=201)
def create_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(""),
    agenda: str = Form(""),
):
    extension = _extension(file.filename)
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            415,
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Accepted: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
        )

    tmp_path, size = _save_upload_to_temp(file)

    meeting_id = f"MTG-{uuid.uuid4().hex[:12]}"
    stored_name = os.path.basename(file.filename) or f"recording{extension}"

    try:
        store.save_audio(meeting_id, stored_name, tmp_path)
    except Exception:
        os.path.exists(tmp_path) and os.remove(tmp_path)
        raise

    record = {
        "id": meeting_id,
        "title": title.strip() or _title_from_filename(stored_name),
        "agenda": agenda.strip() or None,
        "fileName": stored_name,
        "fileType": file.content_type or "unknown",
        "fileSizeBytes": size,
        "fileSizeLabel": _format_bytes(size),
        "uploadedAtISO": store.now_iso(),
        "status": "Processing",
        "progress": 0,
        "error": None,
        # Filled in by the pipeline once processing finishes.
        "duration": None,
        "durationSeconds": None,
        "participants": None,
        "language": None,
        "languages": [],
        "transcript": [],
        "speakerStats": [],
        # Still empty by design: summary / decisions / action items / keywords
        # require an LLM pass, which is a separate piece of work. They stay
        # null rather than being faked.
        "summary": None,
        "decisions": [],
        "actionItems": [],
        "keywords": [],
    }
    store.add_meeting(record)

    background_tasks.add_task(process_meeting, meeting_id)
    return record


@router.delete("/meetings/{meeting_id}")
def delete_meeting(meeting_id: str):
    if not store.delete_meeting(meeting_id):
        raise HTTPException(404, "Meeting not found.")
    return {"status": "trashed", "id": meeting_id}


@router.post("/meetings/{meeting_id}/restore")
def restore_meeting(meeting_id: str):
    record = store.restore_meeting(meeting_id)
    if record is None:
        raise HTTPException(404, "Meeting not found in trash.")
    return record


@router.delete("/meetings/{meeting_id}/purge")
def purge_meeting(meeting_id: str):
    if not store.purge_meeting(meeting_id):
        raise HTTPException(404, "Meeting not found in trash.")
    return {"status": "purged", "id": meeting_id}

@router.patch("/meetings/{meeting_id}/speakers/{speaker_label}")
def rename_speaker(
    meeting_id: str,
    speaker_label: str,
    name: str = Form(...),
    remember: bool = Form(True),
    overwrite: bool = Form(False),
):
    """
    Rename a diarized speaker label (e.g. Speaker_00) to a human name across
    one meeting's transcript and speaker stats.

    By default also permanently enrolls this speaker's voice from the meeting
    audio (remember=true) so every future meeting auto-labels them. Pass
    remember=false for a one-meeting-only cosmetic rename. overwrite=true
    replaces an existing profile instead of blending.
    """
    new_name = name.strip()
    if not new_name:
        raise HTTPException(422, "Name cannot be empty.")

    # Capture time ranges before rename so we still find segments if the
    # client passed the display name that is about to change.
    ranges = store.speaker_time_ranges(
        meeting_id, speaker=speaker_label, speaker_label=speaker_label
    )

    meeting = store.rename_speaker(meeting_id, speaker_label, new_name)
    if meeting is None:
        raise HTTPException(404, "Meeting not found.")

    enrollment = None
    if remember:
        path = store.audio_path(meeting_id)
        if path is None:
            raise HTTPException(400, "No audio stored for this meeting; cannot enroll voice.")
        if not ranges:
            ranges = store.speaker_time_ranges(
                meeting_id, speaker=new_name, speaker_label=speaker_label
            )
        if not ranges:
            raise HTTPException(
                400,
                "No transcript segments found for this speaker to build a voice profile.",
            )
        try:
            from models.speaker_enrollment import enroll_from_meeting_audio

            enrollment = enroll_from_meeting_audio(
                _identifier(),
                audio_path=path,
                segments=ranges,
                name=new_name,
                overwrite=overwrite,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception(f"[{meeting_id}] enroll-from-meeting failed")
            raise HTTPException(400, f"Could not enroll voice: {exc}") from exc

    if enrollment is not None:
        return {**meeting, "enrollment": enrollment}
    return meeting


@router.post("/meetings/{meeting_id}/speakers/{speaker_label}/enroll")
def enroll_speaker_from_meeting(
    meeting_id: str,
    speaker_label: str,
    name: str = Form(...),
    overwrite: bool = Form(False),
):
    """
    Enroll a reusable voice profile from this meeting's audio for one speaker.

    Separate from rename: call after renaming (or with the desired display
    name) when the user opts into "Remember this speaker for future meetings".
    """
    new_name = name.strip()
    if not new_name:
        raise HTTPException(422, "Name cannot be empty.")

    meeting = store.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(404, "Meeting not found.")

    path = store.audio_path(meeting_id)
    if path is None:
        raise HTTPException(400, "No audio stored for this meeting; cannot enroll voice.")

    ranges = store.speaker_time_ranges(
        meeting_id, speaker=speaker_label, speaker_label=speaker_label
    )
    if not ranges:
        ranges = store.speaker_time_ranges(meeting_id, speaker=new_name)
    if not ranges:
        raise HTTPException(
            400,
            "No transcript segments found for this speaker to build a voice profile.",
        )

    try:
        from models.speaker_enrollment import enroll_from_meeting_audio

        return enroll_from_meeting_audio(
            _identifier(),
            audio_path=path,
            segments=ranges,
            name=new_name,
            overwrite=overwrite,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception(f"[{meeting_id}] enroll-from-meeting failed")
        raise HTTPException(400, f"Could not enroll voice: {exc}") from exc


@router.post("/meetings/{meeting_id}/identify-speakers")
def identify_speakers(meeting_id: str, background_tasks: BackgroundTasks):
    """
    Automatically label speakers on an already-processed meeting.

    1) Self-intros / mutual greetings in the transcript
    2) Match against enrolled voice profiles
    3) Merge same-meeting fragments (Speaker_02 ≈ already-named Anushka)

    Voice enrollment for resolved names runs so future meetings stay permanent.
    """
    meeting = store.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(404, "Meeting not found.")

    transcript = meeting.get("transcript") or []
    path = store.audio_path(meeting_id)

    from models.speaker_enrollment import (
        identify_speakers_in_meeting,
        introduction_labels_from_transcript,
        enroll_from_meeting_audio,
        same_meeting_fragment_merges,
    )

    matches: list[dict] = []

    for intro in introduction_labels_from_transcript(transcript):
        matches.append(intro)

    claimed = {m["speaker_label"] for m in matches}
    if path and _identifier().list_speakers():
        try:
            for voice in identify_speakers_in_meeting(
                _identifier(),
                audio_path=path,
                transcript=transcript,
            ):
                if voice.get("matched") and voice["speaker_label"] not in claimed:
                    matches.append(voice)
                    claimed.add(voice["speaker_label"])
        except Exception:
            logger.exception(f"[{meeting_id}] voice identify-speakers failed")

    # Include humans already shown on the transcript (e.g. prior rename).
    named_labels: dict[str, str] = {}
    for line in transcript:
        label = line.get("speaker_label") or line.get("speaker")
        display = line.get("speaker") or label
        if not label or not display:
            continue
        if re.match(r"(?i)^speaker[_\s]?\d+$", str(display).strip()):
            continue
        named_labels.setdefault(label, display)
    for match in matches:
        if match.get("matched"):
            named_labels[match["speaker_label"]] = match["identified_as"]

    if path is not None or named_labels:
        try:
            for frag in same_meeting_fragment_merges(
                audio_path=path,
                transcript=transcript,
                named_labels=named_labels,
            ):
                if frag["speaker_label"] not in claimed:
                    matches.append(frag)
                    claimed.add(frag["speaker_label"])
        except Exception:
            logger.exception(f"[{meeting_id}] fragment merge failed")

    applied = []
    for match in matches:
        if not match.get("matched"):
            continue
        old_name = match["old_name"]
        new_name = match["identified_as"]
        if old_name != new_name:
            updated = store.rename_speaker(meeting_id, old_name, new_name)
            if updated is None:
                continue
        store.set_speaker_identification(
            meeting_id,
            speaker=new_name,
            speaker_label=match["speaker_label"],
            identified_as=new_name,
            confidence=match.get("confidence") or 0.0,
        )
        if path and match.get("source") in (
            "introduction",
            "greeting",
            "fragment_merge",
            "fragment_merge_weak",
            "turn_taking",
            "continuity",
        ):
            ranges = store.speaker_time_ranges(
                meeting_id, speaker=new_name, speaker_label=match["speaker_label"]
            )
            if ranges:

                def _enroll(name=new_name, segs=ranges):
                    try:
                        enroll_from_meeting_audio(
                            _identifier(),
                            audio_path=path,
                            segments=segs,
                            name=name,
                            overwrite=False,
                        )
                    except Exception:
                        logger.exception(
                            f"[{meeting_id}] auto-enroll after label failed"
                        )

                background_tasks.add_task(_enroll)
        applied.append(match)

    meeting = store.get_meeting(meeting_id)
    return {
        "matches": matches,
        "applied": applied,
        "meeting": meeting,
    }


@router.get("/meetings/{meeting_id}/audio")
def get_meeting_audio(meeting_id: str):
    path = store.audio_path(meeting_id)
    if path is None or not os.path.exists(path):
        raise HTTPException(404, "No audio stored for this meeting.")
    return FileResponse(path, filename=os.path.basename(path))


@router.get("/search")
def search(q: str = Query("", min_length=0), limit: int = Query(50, ge=1, le=200)):
    """
    Full-text search across meeting titles, agendas, and transcript lines.

    Runs in the database now rather than by loading every meeting and scanning
    transcripts in Python. Indexed by GIN over to_tsvector plus a trigram index
    for substring matches — the latter matters because Devanagari is inflected
    and token matching alone misses "मच्छर" inside "मच्छरो".
    """
    return {"query": q, "results": store.search(q, limit=limit)}


@router.get("/languages")
def supported_languages():
    """The three languages ASR is constrained to. Drives UI labels/filters."""
    return {"languages": [{"code": c, "name": n} for c, n in LANGUAGE_NAMES.items()]}


# ---------------------------------------------------------------- processing


def _generate_summary(meeting_id: str, transcript: list[dict]) -> dict:
    """
    Run the summarization stage and return the fields to persist.

    Isolated behind its own try/except: a transcript that took two minutes of
    Whisper time must not be thrown away because a local LLM was unreachable or
    slow. On any failure the meeting still completes, with the empty summary
    panels the UI already explains honestly.
    """
    if not SUMMARY_ENABLED:
        return {}

    try:
        # IDF corpus for keyword ranking: the transcripts of other recent
        # meetings. Bounded, because this loads their transcript lines.
        background = []
        for other in store.list_meetings()[:SUMMARY_KEYWORD_IDF_MEETINGS]:
            if other["id"] == meeting_id:
                continue
            background.append(" ".join(l.get("text") or "" for l in other.get("transcript") or []))

        result = summarizer.summarize(transcript, background_documents=background)
        logger.info(
            f"[{meeting_id}] summary via {result.get('summaryEngine')} — "
            f"{len(result.get('decisions') or [])} decisions, "
            f"{len(result.get('actionItems') or [])} action items, "
            f"{len(result.get('keywords') or [])} keywords"
        )
        return result
    except Exception:
        logger.exception(f"[{meeting_id}] summarization failed; completing without it")
        return {}


def process_meeting(meeting_id: str):
    """
    Transcribe one uploaded meeting. Runs in FastAPI's background threadpool
    after the upload response has already been sent.
    """
    path = store.audio_path(meeting_id)
    if path is None:
        store.update_meeting(meeting_id, status="Failed", error="Stored audio file is missing.")
        return

    with _processing_lock:
        try:
            # Reclaim the GPU before Whisper asks for it. On a 4 GB card an
            # Ollama model left resident by an earlier summary takes essentially
            # all of it, and every segment then dies on
            # CUBLAS_STATUS_ALLOC_FAILED while the meeting reports "Failed" with
            # an empty transcript. Cheap, idempotent, and a no-op without Ollama.
            if SUMMARY_ENABLED:
                summarizer.release_vram()

            logger.info(f"[{meeting_id}] decoding {os.path.basename(path)}")
            audio = load_audio_file(path)
            wall_clock_seconds = len(audio) / SAMPLE_RATE

            session = MeetingSession(meeting_id, _identifier())

            def on_progress(fraction: float):
                store.update_meeting(meeting_id, progress=int(fraction * 100))

            logger.info(f"[{meeting_id}] transcribing {wall_clock_seconds:.1f}s of audio")
            session.process_audio(audio, on_progress=on_progress)

            # A transcript that is empty *because every segment threw* is a
            # failure, not a silent recording, and must not be reported as a
            # completed meeting — that is how a broken dependency hides.
            if not session.transcript and session.failed_segments:
                message = (
                    f"All {session.failed_segments} speech segments failed to process. "
                    f"Last error: {session.last_error}"
                )
                logger.error(f"[{meeting_id}] {message}")
                store.update_meeting(meeting_id, status="Failed", progress=0, error=message)
                return

            speaker_stats = session.speaker_stats()
            languages = session.language_breakdown()
            dominant = languages[0]["name"] if languages else None

            # Written while the status is still "Processing" on purpose. The
            # frontend polls only while a meeting is processing, and
            # summarization takes tens of seconds after the last segment — so
            # the transcript lands here and becomes visible immediately, and the
            # summary fills in under the same poll loop instead of needing a
            # manual refresh. "Completed" is set once everything is in.
            store.update_meeting(
                meeting_id,
                progress=100,
                error=None,
                # Surfaced even on success: a partially-degraded transcript is
                # worth knowing about rather than quietly shipping.
                failedSegments=session.failed_segments,
                transcript=session.transcript,
                speakerStats=speaker_stats,
                languages=languages,
                language=dominant,
                participants=len(speaker_stats),
                durationSeconds=round(wall_clock_seconds, 1),
                duration=format_duration(wall_clock_seconds),
            )
            logger.info(
                f"[{meeting_id}] transcribed — {len(session.transcript)} lines, "
                f"{len(speaker_stats)} speakers, languages={[l['code'] for l in languages]}"
            )

            summary_fields = _generate_summary(meeting_id, session.transcript)

            store.update_meeting(meeting_id, status="Completed", **summary_fields)
            logger.info(f"[{meeting_id}] done")

        except Exception as e:
            logger.exception(f"[{meeting_id}] processing failed")
            store.update_meeting(meeting_id, status="Failed", progress=0, error=str(e))
