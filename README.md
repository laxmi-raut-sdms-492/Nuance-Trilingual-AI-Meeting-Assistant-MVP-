# Nuance — Trilingual AI Meeting Assistant

Upload or record a meeting and get back a transcript labeled with **who spoke
what, in which language** — English, Hindi, or Marathi, detected per utterance
so a code-switching conversation transcribes correctly instead of being forced
into one language.

```
Recording (upload or browser mic)
    │
    ▼
FastAPI backend
    │
    ├── Silero VAD          — cut on real speech/pause boundaries, not fixed windows
    ├── Speaker change det.  — split a segment that contains two back-to-back talkers
    ├── Voice embedding      — SpeechBrain ECAPA-TDNN, 192-d, L2-normalized
    ├── Diarization          — incremental clustering into stable Speaker_XX labels
    ├── Identification       — cosine match against enrolled voices
    └── Whisper ASR          — language detected per segment, restricted to en/hi/mr
    │
    ▼
Transcript + speaker talk-time + language breakdown, stored in PostgreSQL
    │
    ▼
React frontend (dashboard, meeting list, detail, analytics)
```

## Two ways in, one pipeline

| Path | Entry point | Use |
|---|---|---|
| **Upload** | `POST /api/meetings` | The React app. File is stored, transcribed in the background, polled until done. |
| **Live** | `WS /ws/meeting/{id}` | Streaming PCM16 from a mic, transcript entries pushed back as they're produced. |

Both run the identical `MeetingSession` pipeline, so a live meeting and an
uploaded one produce the same shape of result.

## Project structure

```
backend/
├── main.py              FastAPI app: enrollment, WebSocket, mounts the REST router
├── api.py               /api/* REST endpoints used by the frontend
├── db/
│   ├── models.py        SQLAlchemy schema — meetings, transcript lines, speakers
│   ├── repository.py    The only module the API talks to for persistence
│   ├── session.py       Engine + transactional session scope
│   └── import_json.py   One-shot migration of the old JSON store into Postgres
├── alembic/             Schema migrations (`alembic upgrade head`)
├── tests/               pytest suite for the repository, runs on SQLite
├── pipeline.py          MeetingSession — drives both the live and upload paths
├── config.py            All tunable constants
├── audio_utils.py       PCM/WAV/any-container decoding helpers
└── models/
    ├── vad.py           Silero VAD streaming segmenter
    ├── scd.py           Speaker change detection within a segment
    ├── embedding.py     SpeechBrain ECAPA-TDNN wrapper
    ├── diarizer.py      Incremental clustering with stable labels
    ├── identifier.py    Enrolled-speaker database + matching
    └── asr.py           Whisper + per-segment language detection

frontend/                React 18 + Vite + Tailwind
└── src/
    ├── services/api.js  Axios client — the single place the backend URL lives
    ├── context/         MeetingsContext: fetches, polls, uploads, deletes
    ├── pages/           Dashboard, Meetings, MeetingDetails, Upload, Analytics, Settings
    └── components/
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/meetings` | POST | Upload a recording (multipart: `file`, `title`, `agenda`); queues transcription |
| `/api/meetings` | GET | List meetings (transcript bodies omitted) |
| `/api/meetings/{id}` | GET | One meeting, full transcript + speaker stats |
| `/api/meetings/{id}` | DELETE | Delete the meeting, its audio, and its transcript |
| `/api/meetings/{id}/audio` | GET | Stream the stored recording |
| `/api/search?q=` | GET | Full-text search across titles, agendas, transcripts |
| `/api/languages` | GET | The three supported languages |
| `/enroll` | POST | Register a known speaker's voice (multipart: `name`, `audio` WAV) |
| `/speakers` | GET / DELETE | List / remove enrolled speakers |
| `/ws/meeting/{id}` | WS | Live PCM16 stream in, transcript entries out |

## 1. Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**ffmpeg** must be installed system-wide — it decodes every uploaded container
format (`.mp3`, `.mp4`, `.m4a`, `.webm`, …):

```bash
sudo apt install ffmpeg     # Ubuntu/Debian
brew install ffmpeg         # macOS
```

**PostgreSQL** holds the meetings. Create the role and database once:

```bash
sudo -u postgres psql -c "CREATE ROLE nuance LOGIN PASSWORD 'nuance_dev_pw';" \
                     -c "CREATE DATABASE nuance OWNER nuance;"
```

Then apply the schema:

```bash
cd backend && alembic upgrade head
```

Override the connection with `DATABASE_URL` (note the driver — psycopg 3, so
`postgresql+psycopg://`, not `postgresql://`):

```bash
export DATABASE_URL=postgresql+psycopg://user:pass@host:5432/nuance
```

`./startup.sh` does the connection check and the `alembic upgrade` for you, and
prints these commands if the database is missing.

Run it:

```bash
uvicorn main:app --reload --port 8000
```

**First run is slow.** It downloads the SpeechBrain embedding model (~80 MB),
Silero VAD (~2 MB), and the Whisper `small` model (~460 MB), then caches them.

### Storage

| What | Where |
|---|---|
| Meetings, transcripts, speaker stats, language breakdown | PostgreSQL |
| Enrolled voice embeddings | PostgreSQL (`speakers`) |
| Audio recordings | `backend/storage/audio/<meeting-id>/<original-name>` |
| Profile, members, theme, integrations | Browser `localStorage` |

`db/repository.py` is the only module the API talks to for persistence. It
returns plain camelCase dicts — never ORM objects — so nothing downstream can
hold a detached instance or trigger a lazy load after the session closes.

Search is a real database query: a GIN index over `to_tsvector('simple', …)`
plus a trigram index for substring matches. `'simple'`, not `'english'` —
English stemming and stopwords are meaningless for Hindi and Marathi. The
trigram half is not redundant: Devanagari is heavily inflected, and a search
for `मच्छर` must find `मच्छरो`, which whole-token matching misses.

### Migrating from the old JSON store

Earlier builds kept everything in `backend/storage/meetings.json` and
`speakers.json`. If you have one of those:

```bash
cd backend
python3 -m db.import_json            # import; existing ids are skipped
python3 -m db.import_json --replace  # overwrite meetings that already exist
python3 -m db.import_json --verify   # compare counts, write nothing
```

It writes through the repository rather than issuing raw SQL, so anything it
imports is by construction readable by the running app. Audio files are left
alone — they already live on disk.

### Tests

```bash
cd backend && pytest
```

The suite covers `db/repository.py` and runs against SQLite, so no Postgres
server is needed. What that does *not* cover is the `to_tsvector` branch of
search, which is Postgres-only; changes there need a real database.

## 2. Frontend setup

```bash
cd frontend
npm install
cp .env.example .env       # VITE_API_BASE_URL=http://localhost:8000/api
npm run dev
```

Open **http://localhost:5173**.

## 3. Using it

1. **(Optional) Enroll speakers** — `POST /enroll` with a name and a clean 5-10s
   WAV sample. Enrolling the same name repeatedly blends samples into a more
   robust profile. Without enrollment everyone is `Speaker_00`, `Speaker_01`, …
   and diarization still works, just without names.
2. **Create a meeting** — give it a title and agenda, then upload a file or
   record one live in the browser.
3. **Watch it process** — the meeting appears immediately as `Processing` with
   a real progress bar. The list and detail pages poll until it flips to
   `Completed`; no reload needed.
4. **Read the transcript** — each line carries speaker, timestamp, and detected
   language. Download it as a text file, or search across every meeting.

## The trilingual part

Language is detected **per segment**, not per meeting, and the candidate set is
restricted to English, Hindi, and Marathi (`ALLOWED_LANGUAGES` in `config.py`).

Both halves of that matter:

- **Per segment**, because a real meeting code-switches. Detecting once for the
  whole file locks every later segment to whichever language was spoken first,
  and Whisper then transliterates the rest into that language's script.
- **Restricted**, because Whisper scores all ~99 languages it knows, and on a
  2-4 second segment its unrestricted top guess is often a language nobody in
  the room speaks (Urdu or Nepali for Hindi, Sanskrit for Marathi). A 3-way
  choice is far more reliable at that segment length than a 99-way one.

When a segment's own detection is near chance (`LANGUAGE_DETECT_MIN_PROB`), it
falls back to the meeting's dominant language so far. The transcript entry then
carries `language_fallback: true`, and `language_prob` always describes
`language_detected` — the detector's own top choice — not the language finally
used.

**Measured caveat: keep that threshold low.** On a mixed en/hi/mr recording,
correct detections on clean speech score 0.97-1.00, but correct detections on
short or noisy Devanagari segments score as low as 0.52 — overlapping the range
where wrong guesses live. There is no clean separating value. Raising the
threshold to 0.85 was tried and made results strictly worse: real Hindi and
Marathi segments were discarded and re-transcribed as English gibberish
(`ये मख्षर` became `Yeah, mutther`).

The cause is the fallback *target*. "Dominant language so far" is biased toward
whoever spoke first and longest, which in a trilingual meeting is actively
wrong. Trusting a mediocre detection beats falling back to a confident wrong
answer, so the threshold should only ever catch near-chance guesses. A better
fallback (nearest-neighbour by speaker, since a given person usually keeps to
one language) is the obvious improvement and isn't built yet.

## Hallucination guards

Whisper is an autoregressive decoder with no alignment constraint — nothing
ties its output to the input the way CTC does. On uninformative audio it falls
back on its language prior and invents fluent stock phrases learned from
subtitle tracks over video outros. Four guards in `asr.py`, because no single
signal catches every case:

1. **Minimum segment length** (`MIN_SPEECH_SECONDS`, 1.0s). The encoder always
   consumes exactly 30s of mel, so a 0.7s clip is zero-padded to 98% silence —
   precisely the distribution that maps to outro captions. Cheapest defence is
   never handing Whisper the clip. This one change removed
   `"Thank you for watching. Be safe, and I'll see you next time. Bye."` and
   the phantom fifth speaker that segment had created.
2. **Whisper's own doubt**, `no_speech_prob` and `avg_logprob` together.
3. **Speech rate.** Human speech peaks near 7 words/sec; the hallucination
   above was 18.6 w/s. Needs no model internals and a confident decoder can't
   fool it — `avg_logprob` measures *fluency*, not correctness, so an invented
   stock phrase scores better than honestly transcribed accented Marathi.
4. **Script consistency.** `language=mr` returning Latin text means the decode
   failed whatever confidence it claims. Also catches repetition loops
   (`"slippery"` ×12) for free.

**Measured: do not retry a script-mismatched segment in another language.**
It was tried. A failed Marathi decode retried as English returned
`"Khokhla, this is the end of today's episode"` — a fluent hallucination that
passes every guard above, because English is where Whisper's outro-caption
prior lives. Obvious garbage is safer than convincing garbage: a reader spots
`comme c.o-pid Jay` as broken instantly and cannot spot the other one at all.
The retry also drained Marathi from 28.6% of the meeting to 11.7% by
reassigning its segments to whichever language happened to decode fluently.
Failed decodes are dropped.

## Tuning

All knobs live in `backend/config.py`:

| Setting | Effect |
|---|---|
| `WHISPER_MODEL_SIZE` | `small` is the practical floor for Hindi/Marathi; `base` is much worse on Devanagari. `medium` is better still and much slower. Also settable via env var. |
| `ALLOWED_LANGUAGES` | The candidate set for per-segment detection |
| `LANGUAGE_DETECT_MIN_PROB` | Below this, fall back to the meeting's dominant language. **Keep it low.** Chance is 0.33 (probabilities are renormalized over 3 languages). Raising it to 0.85 was measured and made results strictly worse — see below. |
| `MAX_SEGMENT_SECONDS` | Force-cut for long uninterrupted speech — bounds latency |
| `MIN_SILENCE_MS` | How long a pause must be before a segment is considered ended |
| `MIN_SPEECH_SECONDS` | Segments shorter than this never reach Whisper. **Don't lower below ~1.0s** — short clips are where hallucinations come from |
| `SILENCE_RMS_THRESHOLD` | Raise if silence is transcribed as noise; lower if quiet speech is skipped |
| `DIARIZATION_DISTANCE_THRESHOLD` | Lower = more speakers detected; higher = more merged |
| `IDENTIFICATION_SIMILARITY_THRESHOLD` | Lower = more matches to enrolled names (more false positives); higher = stricter |
| `ASR_MAX_NO_SPEECH_PROB` / `ASR_MIN_AVG_LOGPROB` | Hallucination guard 1 — both required together |
| `ASR_STANDALONE_NO_SPEECH_PROB` | Guard 2 — fires alone when Whisper is near-certain there was no speech, catching *confident* hallucinations the AND above misses |
| `ASR_MAX_WORDS_PER_SECOND` | Guard 3 — physically impossible speech rate. The strongest check, and it needs no model internals |
| `ASR_MIN_DEVANAGARI_RATIO` | Guard 4 — a `hi`/`mr` decode returning Latin script failed. Kept low enough that quoting an English word or a number doesn't trip it |
| `CORS_ORIGINS` | Allowed browser origins. Env var. Do not set to `*`. |
| `MAX_UPLOAD_MB` | Upload size cap |
| `SUMMARY_ENABLED` | Turns the whole summarization stage off. Transcription is unaffected. Env var. |
| `SUMMARY_MODEL` | Ollama model for summary/decisions/action items. Env var. Falls back to the extractive engine if it isn't pulled. |
| `SUMMARY_NUM_CTX` | **Must stay explicit.** Ollama's default of 4096 truncates a longer prompt silently — a fabricated summary with no error to show for it. |
| `SUMMARY_WINDOW_SECONDS` | Map-reduce window. Lower it if a window overflows `SUMMARY_NUM_CTX`. |
| `SUMMARY_CITATION_THRESHOLD` | How closely a quoted line must match the transcript to count as cited. Raise toward 1.0 to demand verbatim quotes; items below it are dropped. |
| `SUMMARY_KEYWORD_COUNT` / `SUMMARY_KEYWORD_MIN_OCCURRENCES` | How many keywords to keep, and how often a word must appear to qualify |

## Known limitations (read before treating this as production-ready)

- **No authentication.** Anyone who can reach the API can upload meetings, read
  every transcript, and enroll a voice under any name. CORS is restricted to
  localhost by default, which is a guardrail, not a security control. Put real
  auth in front of this before exposing it.
- **Recordings are files on disk, not rows.** Everything else lives in
  PostgreSQL, but the audio stays under `backend/storage/audio/<meeting-id>/`.
  A 300 MB blob in a row costs every query that touches the table. The
  consequence is that a database dump alone is not a complete backup — the
  audio directory has to be copied with it.
- **`speakers.centroid` is biometric data, stored unencrypted.** It is a voice
  embedding behind no authentication. See the point above about auth.
- **Transcription is serialized.** One meeting at a time, by design: Whisper
  and SpeechBrain are not thread-safe, and on CPU two concurrent runs are
  slower than the same two back to back. A long upload delays the next one.
- **Summaries are generated prose and labelled as such.** Decisions and action
  items are dropped unless the transcript line they quote can be found, so what
  survives is traceable. A summary paraphrase has no line to match against, so
  it cannot be verified the same way — the API returns `summaryEngine` and the
  UI names the engine under the text. There is no sentiment analysis anywhere.
- **Summarization needs a local model to be at its best.** With Ollama
  unreachable the extractive engine still runs, but its summary is stitched
  from real transcript lines rather than written, and it finds items by cue
  phrase only.
- **CPU-only by default.** Whisper and SpeechBrain are both much faster on GPU;
  installing the CUDA build of `torch` speeds this up with no code changes.
- **Diarization accuracy** on real overlapping multi-speaker audio is
  materially worse than on clean one-at-a-time audio. Expect roughly 75-90%
  depending on conditions, not 100%.
- **`ScriptProcessorNode`** in the browser recorder is deprecated in favour of
  `AudioWorklet`; kept for broader compatibility and simpler code.

## Next steps

- A job queue in place of the in-process transcription lock
- Authentication and per-user meeting ownership
- Move Whisper inference to GPU or a hosted ASR API for lower latency
