"""
Meeting Intelligence backend.

Two ways in, one pipeline behind both:

  Uploaded meetings (what the React frontend uses) — see api.py
    POST   /api/meetings                 -> upload a recording, transcribe in the background
    GET    /api/meetings                 -> list every stored meeting
    GET    /api/meetings/{id}            -> one meeting, full transcript + speaker stats
    DELETE /api/meetings/{id}            -> delete a meeting and its audio
    GET    /api/meetings/{id}/audio      -> stream the stored recording
    GET    /api/search?q=                -> full-text search across transcripts
    GET    /api/languages                -> the three supported languages

  Live meetings
    WS   /ws/meeting/{session_id}        -> stream live PCM16 audio, get transcript entries back
    GET  /meeting/{session_id}/transcript -> full transcript + speaking-time summary so far

  Speaker enrollment (shared by both)
    POST   /enroll                       -> register a known speaker's voice
    GET    /speakers                     -> list enrolled speaker names
    DELETE /speakers/{name}              -> remove an enrolled speaker

Run with:  uvicorn main:app --reload --port 8000
"""

import logging
import tempfile
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import LOG_LEVEL, CORS_ORIGINS
from audio_utils import load_audio_file
from models.embedding import get_embedding
from models.identifier import SpeakerIdentifier
from pipeline import MeetingSession

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI(title="Meeting Intelligence API")

app.add_middleware(
    CORSMiddleware,
    # Explicit origins, not "*". This API has no authentication in front of
    # it — a wildcard means any page the user visits can enroll voices and
    # read every meeting transcript. Set CORS_ORIGINS for deployment.
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

identifier = SpeakerIdentifier()
sessions: dict[str, MeetingSession] = {}

# Imported after `identifier` exists — api.py resolves it lazily at request
# time, but keeping the order explicit avoids a confusing partial-import.
from api import router as api_router  # noqa: E402

app.include_router(api_router)


@app.post("/enroll")
async def enroll_speaker(name: str = Form(...), audio: UploadFile = None):
    """
    Upload a short (5-10s) clean voice sample to register a known speaker.

    Decoded with load_audio_file, not wav_bytes_to_float32: the enrollment UI
    records through the browser's MediaRecorder, which emits .webm/Opus (or
    .mp4 on Safari) and never WAV. torchaudio cannot open those containers, so
    the WAV-only loader failed on every browser-recorded sample. The
    ffmpeg-backed loader handles all of them and is already a hard dependency.
    """
    if audio is None:
        raise HTTPException(400, "No audio file provided.")

    name = name.strip()
    if not name:
        raise HTTPException(400, "Speaker name is required.")

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        waveform = load_audio_file(tmp_path)
        if waveform.size == 0:
            raise HTTPException(400, "That recording contained no audio.")
        embedding = get_embedding(waveform)
        identifier.enroll(name, embedding)
    except HTTPException:
        raise
    except Exception as exc:
        logging.getLogger("enroll").exception("enrollment failed")
        raise HTTPException(400, f"Could not read that recording: {exc}")
    finally:
        os.remove(tmp_path)

    return {"status": "enrolled", "name": name}


@app.get("/speakers")
def list_speakers():
    return {"speakers": identifier.list_speakers()}


@app.delete("/speakers/{name}")
def delete_speaker(name: str):
    identifier.remove(name)
    return {"status": "removed", "name": name}


@app.websocket("/ws/meeting/{session_id}")
async def meeting_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in sessions:
        sessions[session_id] = MeetingSession(session_id, identifier)
    session = sessions[session_id]

    while True:
        try:
            pcm_bytes = await websocket.receive_bytes()
        except WebSocketDisconnect:
            # Normal end of session. Drain the segment still sitting in the
            # VAD buffer — the last thing anyone says never gets the trailing
            # silence that would otherwise close it out, so without this the
            # final utterance of every live meeting is lost. The socket is
            # already gone, so these entries land in the transcript only.
            try:
                session.finish()
            except Exception as e:
                logging.getLogger("main").error(f"final flush failed for session {session_id}: {e}")
            break

        try:
            new_entries = session.process_chunk(pcm_bytes)
            for entry in new_entries:
                await websocket.send_json(entry)
        except Exception as e:
            # process_chunk already guards individual segments, but this is
            # the last line of defense: whatever happens here, the WebSocket
            # loop keeps running for the rest of the meeting instead of dying.
            logging.getLogger("main").error(f"chunk processing failed for session {session_id}: {e}")


@app.get("/meeting/{session_id}/transcript")
def get_transcript(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")
    return sessions[session_id].full_transcript()
