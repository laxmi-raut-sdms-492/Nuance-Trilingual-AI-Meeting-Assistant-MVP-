# Meeting Intelligence — Live Diarization, Identification & Transcript

A working real-time system: capture audio from a browser mic, stream it to a
Python backend, and get back a live transcript labeled with **who spoke
what**, matched against known ("enrolled") speakers where possible.

```
Browser mic
    │  (raw 16-bit PCM, 16kHz, mono, streamed over WebSocket)
    ▼
FastAPI backend
    │
    ├── Silence check (RMS)          — skip empty audio cheaply
    ├── Voice embedding (SpeechBrain) — turn 3s of audio into a 192-d vector
    ├── Diarization (clustering)      — assign a stable Speaker_XX label
    ├── Identification (cosine sim)   — match against enrolled voices
    └── Transcription (Whisper)       — speech-to-text for this chunk
    │
    ▼
Live transcript entry -> sent back to the browser over the same WebSocket
```

## Project structure

```
project/
├── backend/
│   ├── main.py              FastAPI app: /enroll, /speakers, /ws/meeting/{id}
│   ├── pipeline.py          Per-session processing loop
│   ├── config.py            All tunable constants
│   ├── audio_utils.py       PCM/WAV conversion helpers
│   ├── requirements.txt
│   └── models/
│       ├── embedding.py     SpeechBrain ECAPA-TDNN wrapper
│       ├── identifier.py    Enrolled-speaker database + matching
│       ├── diarizer.py      Incremental clustering with stable labels
│       └── asr.py           Whisper transcription wrapper
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js                Mic capture, WAV encoding, WebSocket streaming
```

## 1. Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need **ffmpeg** installed system-wide (Whisper uses it internally):

```bash
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
# Windows: download from ffmpeg.org and add to PATH
```

Run the server:

```bash
uvicorn main:app --reload --port 8000
```

**First run will be slow** — it downloads the SpeechBrain embedding model
(~80MB) and the Whisper `base` model (~150MB) from the internet and caches
them locally. After that, startup is fast.

You should see:
```
Uvicorn running on http://127.0.0.1:8000
```

## 2. Frontend setup

Mic access requires a proper HTTP origin (not just double-clicking the HTML
file), so serve it with a simple local server:

```bash
cd frontend
python3 -m http.server 5500
```

Open **http://localhost:5500** in Chrome or Edge (best `getUserMedia` +
AudioContext support).

## 3. Using it

1. **(Optional) Enroll speakers** — type a name, click "Record 6s sample,"
   and talk for 6 seconds. Repeat for each person you want recognized by
   name. Skip this entirely and everyone just appears as `Speaker_00`,
   `Speaker_01`, etc. — diarization still works without identification.
2. **Start meeting** — grants mic permission, opens a WebSocket session,
   and starts streaming audio. Transcript entries appear every ~3 seconds
   as the backend finishes processing each chunk.
3. **Stop** — ends the stream and fetches a speaking-time summary per
   speaker.

## Tuning

All the knobs that affect accuracy live in `backend/config.py`:

| Setting | Effect |
|---|---|
| `CHUNK_SECONDS` | Smaller = lower latency but less context for ASR/embeddings; larger = better accuracy but slower feedback |
| `SILENCE_RMS_THRESHOLD` | Raise if silence is being transcribed as noise; lower if quiet speech is being skipped |
| `DIARIZATION_DISTANCE_THRESHOLD` | Lower = more speakers detected (stricter separation); higher = fewer, more merged speakers |
| `IDENTIFICATION_SIMILARITY_THRESHOLD` | Lower = more voices get matched to enrolled names (more false positives); higher = stricter matching (more "Unknown") |
| `WHISPER_MODEL_SIZE` | `tiny`/`base` = fast, less accurate. `small`/`medium` = slower, more accurate. Bigger models need more RAM/VRAM. |

## Known limitations (read before treating this as production-ready)

- **Fixed 3-second windows**, not proper VAD-aligned segments — a sentence
  can get cut mid-word at a chunk boundary. This is the same simplification
  called out in the earlier standalone script; production systems (e.g.
  Pyannote's own pipeline) use adaptive segmentation instead.
- **CPU-only by default.** Whisper + SpeechBrain both run noticeably faster
  on GPU. If you have an NVIDIA GPU, installing the CUDA build of `torch`
  will speed things up substantially without any code changes.
- **In-memory session storage** — restarting the backend loses all live
  meeting transcripts (enrolled speakers persist to `backend/storage/speakers.json`,
  but per-meeting data does not). Fine for a demo; swap in a real database
  before shipping.
- **Diarization accuracy** on real multi-speaker, overlapping audio will be
  noticeably worse than on clean, one-speaker-at-a-time test audio — this
  matches the DER figures discussed earlier (roughly 75–90% depending on
  conditions), not 100%.
- **`ScriptProcessorNode`** (used for mic capture) is deprecated in favor of
  `AudioWorklet`, but is used here for broader browser compatibility and
  simpler code. Fine for a demo; worth migrating for a production build.

## Natural next steps

- Swap fixed-window chunking for real VAD-based segmentation (Silero VAD)
- Move Whisper inference to GPU or a hosted ASR API for lower latency
- Replace in-memory sessions with Redis/Postgres so meetings survive a restart
- Add authentication before exposing `/enroll` publicly — right now anyone
  who can reach the API can enroll a voice under any name
