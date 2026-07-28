"""
Meeting Intelligence backend.

Endpoints:
  POST /enroll                       -> register a known speaker's voice
  GET  /speakers                     -> list enrolled speaker names
  WS   /ws/meeting/{session_id}      -> stream live PCM16 audio, get transcript entries back
  GET  /meeting/{session_id}/transcript  -> full transcript + speaking-time summary so far

Run with:  uvicorn main:app --reload --port 8000
"""

import logging
import tempfile
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import LOG_LEVEL
from audio_utils import wav_bytes_to_float32
from models.embedding import get_embedding
from models.identifier import SpeakerIdentifier
from pipeline import MeetingSession

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI(title="Meeting Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only — restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

identifier = SpeakerIdentifier()
sessions: dict[str, MeetingSession] = {}


@app.post("/enroll")
async def enroll_speaker(name: str = Form(...), audio: UploadFile = None):
    """Upload a short (5-10s) clean voice sample WAV file to register a known speaker."""
    if audio is None:
        raise HTTPException(400, "No audio file provided.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        waveform = wav_bytes_to_float32(tmp_path)
        embedding = get_embedding(waveform)
        identifier.enroll(name, embedding)
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
            break  # normal end of session — data stays in memory, /transcript still works

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
