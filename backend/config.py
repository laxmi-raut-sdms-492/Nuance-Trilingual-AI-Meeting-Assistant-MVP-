"""Central configuration for the meeting intelligence backend."""

import os

# --- Audio ---

SAMPLE_RATE = 16000            # all audio is normalized to this rate
SCD_WINDOW_SECONDS = 1.5
SCD_HOP_SECONDS = 0.5
SCD_CHANGE_THRESHOLD = 0.35
SCD_MIN_SEGMENT_SECONDS = 2.0
SCD_MIN_SUBSEGMENT_SECONDS = 1.0
SILENCE_RMS_THRESHOLD = 0.01   # below this average amplitude, treat a segment as silence

# --- Voice Activity Detection (speech segmentation) ---
# Segments are now bounded by real speech start/pause events (Silero VAD),
# not a fixed window — this replaces the old CHUNK_SECONDS approach.
MAX_SEGMENT_SECONDS = 8.0      # force-cut very long uninterrupted speech so latency stays bounded
MIN_SILENCE_MS = 400           # how long a pause must be before a segment is considered "ended"
# Discard segments shorter than this. Raised from 0.3 after a 0.7s segment
# produced "Thank you for watching. Be safe, and I'll see you next time. Bye."
# — a pure hallucination. Whisper's encoder always consumes exactly 30s of mel,
# so a 0.7s clip is zero-padded to 98% silence, which is precisely the input
# distribution that maps to YouTube outro captions in its training data. Under
# ~1s there is not enough audio to transcribe reliably anyway, so the cheapest
# defence is to never hand Whisper the clip at all.
MIN_SPEECH_SECONDS = 1.0

# --- Models ---
EMBEDDING_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
# tiny | base | small | medium | large — bigger = more accurate, slower.
# `small` is the practical floor for Hindi/Marathi; `base` is noticeably worse
# on Devanagari-script languages than it is on English.

# --- Languages (the "trilingual" part) ---
# Whisper's raw auto-detect can return any of ~99 languages, and on a short,
# noisy meeting segment it routinely guesses something absurd (Welsh, Nepali,
# Urdu for Hindi). We constrain detection to the three languages this product
# actually supports and pick the most likely one among them.
ALLOWED_LANGUAGES = ["en", "hi", "mr"]
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
DEFAULT_LANGUAGE = "en"  # used when detection is too weak to trust

# Below this, stop trusting the detector and fall back (see below). Chance is
# 0.33 here, since probabilities are renormalized over three languages.
#
# Keep this LOW, and resist the temptation to raise it. Measured on a mixed
# en/hi/mr recording: correct detections on clean speech score 0.97-1.00, but
# correct detections on short or noisy Hindi/Marathi segments score as low as
# 0.52, overlapping the range where wrong guesses also live. There is no clean
# separating value. Raising this to 0.85 was tried and made results strictly
# worse — real Devanagari segments got discarded and re-transcribed as English
# gibberish ("ये मख्षर" became "Yeah, mutther").
#
# The reason is the fallback target: it is the meeting's dominant language so
# far, which in a trilingual meeting is biased toward whoever spoke first and
# most. Falling back is therefore usually worse than trusting a mediocre
# detection. This threshold should only catch near-chance guesses.
LANGUAGE_DETECT_MIN_PROB = 0.40

# --- ASR quality guards ---
#
# Whisper is an autoregressive decoder with no alignment constraint: nothing
# ties its output to the input the way CTC does. Given uninformative audio it
# falls back on its language prior and invents fluent stock phrases ("Thank
# you for watching", "Please subscribe") learned from subtitle tracks over
# video outros. Four independent guards, because no single signal catches
# every case.
#
# Guard 1 — Whisper's own doubt, both signals together.
# `no_speech_prob` says "this probably isn't speech"; `avg_logprob` says "that
# decode was a struggle". Required together, this catches garbled output over
# noise without touching clean speech.
ASR_MAX_NO_SPEECH_PROB = 0.6
ASR_MIN_AVG_LOGPROB = -1.0

# Guard 2 — overwhelming non-speech evidence on its own.
# The AND above misses the most damaging case: a *confident* hallucination.
# "Thank you for watching" is a very high-probability token sequence, so
# avg_logprob comes back healthy (~-0.3) while the model invents text. Fluency
# is not correctness. This standalone bar is set well above guard 1's so it
# only fires when the model is nearly certain there was no speech at all.
ASR_STANDALONE_NO_SPEECH_PROB = 0.85

# Guard 3 — physically impossible speech rate.
# The strongest check needs no model internals. Human speech peaks around 7
# words/sec; the hallucination above was 14 words in 0.7s = 20 w/s. A confident
# decoder cannot fool arithmetic.
ASR_MAX_WORDS_PER_SECOND = 8.0

# Guard 4 — script consistency.
# Forcing `language=mr` and getting Latin text back ("comme c.o-pid Jay,
# feste ?") means the decode failed, whatever confidence it reports. Hindi and
# Marathi are written in Devanagari (U+0900-U+097F). Below this ratio of
# Devanagari to total letters, the result is retried once with the detector's
# next-ranked language rather than discarded — an Indian-language speaker
# quoting an English word or a number is normal and must not trip this.
ASR_DEVANAGARI_LANGUAGES = {"hi", "mr"}
ASR_MIN_DEVANAGARI_RATIO = 0.35

# --- Diarization ---
# Cosine distance between a new segment and a cluster's centroid.
# 0 = identical voice, ~1 = totally different.
DIARIZATION_DISTANCE_THRESHOLD = 0.55       # base threshold for a brand-new/young cluster
DIARIZATION_EMA_ALPHA = 0.15                # centroid update rate — constant, doesn't freeze over a long meeting
HYSTERESIS_BONUS = 0.05                     # discount applied to "the speaker who just spoke" to prevent flip-flopping
THRESHOLD_GROWTH_PER_SEGMENT = 0.01         # mature clusters get a looser (more forgiving) threshold
THRESHOLD_GROWTH_CAP = 0.15                 # ...up to this much looser
CLUSTER_MERGE_DISTANCE = 0.25               # if two clusters' centroids end up this close, merge them
CLUSTER_MERGE_CHECK_EVERY = 10              # check for mergeable clusters every N segments

# --- Identification ---
# Cosine similarity (not distance) between a segment and an enrolled voice.
IDENTIFICATION_SIMILARITY_THRESHOLD = 0.55

# --- Summarization ---
# Summary, decisions, action items and keywords. Local Ollama when it is
# reachable, extractive otherwise, so the demo never depends on a model being
# resident. See models/summarizer.py for what each engine guarantees.
SUMMARY_ENABLED = os.getenv("SUMMARY_ENABLED", "1").lower() in ("1", "true", "yes")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen2.5:7b")

# num_ctx MUST be set explicitly. Ollama defaults to 4096 and truncates a
# longer prompt SILENTLY — no error, no warning, just a confident summary of
# whatever fraction of the transcript survived. That is a fabrication bug
# wearing the disguise of a working feature. 8192 covers one window with the
# instructions and the model's own output.
SUMMARY_NUM_CTX = int(os.getenv("SUMMARY_NUM_CTX", "8192"))

# Unload the model the moment it has answered. **This is not a tuning knob.**
#
# Ollama's default keep_alive is 5 minutes, and on this 4 GB card qwen2.5:7b
# holds ~4 GB of it. Measured: a summarization run left the model resident, an
# upload arrived 4 minutes later, and Whisper got 58 MiB — every one of the 63
# speech segments failed with CUBLAS_STATUS_ALLOC_FAILED / CUFFT_INTERNAL_ERROR
# and the meeting completed as "Failed" with an empty transcript.
#
# Raising this trades the transcription pipeline for a warm model. Transcription
# is the product; summarization is downstream of it. Reload cost is seconds.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "0")

# Map-reduce window. Measured on real transcripts: ~20k tokens per hour of
# English and 2-3x that for Hindi/Marathi, so an hour-long meeting does not fit
# any context this machine can afford. CPU prompt ingestion is the bottleneck
# (Whisper holds the 4 GB of VRAM), so windows are summarized separately and
# then merged.
SUMMARY_WINDOW_SECONDS = 600.0
SUMMARY_TIMEOUT_SECONDS = int(os.getenv("SUMMARY_TIMEOUT_SECONDS", "1800"))

# Fuzzy-match floor for "does this quote actually appear in the transcript".
# Not 1.0: a model that repairs a typo or drops a filler word is still citing a
# real line, and rejecting that throws away good items. Below this, the quote is
# treated as invented and the item is dropped.
SUMMARY_CITATION_THRESHOLD = 0.85

# Keywords are ranked by TF-IDF, where the IDF corpus is the transcripts of the
# most recent N other meetings. IDF needs a background corpus — within a single
# document it would penalise exactly the words the meeting is about. With no
# other meetings stored the ranking degrades to term frequency, which is the
# correct single-document answer. The count shown in the UI is always the real
# number of occurrences either way.
SUMMARY_KEYWORD_COUNT = 12
SUMMARY_KEYWORD_IDF_MEETINGS = 20
SUMMARY_KEYWORD_MIN_OCCURRENCES = 2

# --- Logging ---
LOG_LEVEL = "INFO"  # set to "DEBUG" for more verbose diarization/identification reasoning

# --- Storage ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SPEAKERS_DB_PATH = os.path.join(STORAGE_DIR, "speakers.json")
MEETINGS_DB_PATH = os.path.join(STORAGE_DIR, "meetings.json")
AUDIO_DIR = os.path.join(STORAGE_DIR, "audio")

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- Database ---
# PostgreSQL. Override with DATABASE_URL in the environment; the default is a
# local development role and is not a secret worth protecting, but do not reuse
# it anywhere real.
#
# The driver is psycopg 3 — note "postgresql+psycopg", not "postgresql" (which
# resolves to psycopg2) and not "postgresql+psycopg2".
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://nuance:nuance_dev_pw@localhost:5432/nuance",
)

# Set SQL_ECHO=1 to log every statement. Useful when a query looks slow;
# extremely noisy during transcription, which updates progress continuously.
SQL_ECHO = os.getenv("SQL_ECHO", "").lower() in ("1", "true", "yes")

# --- Uploads ---
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "300"))
ALLOWED_UPLOAD_EXTENSIONS = {".mp3", ".wav", ".mp4", ".m4a", ".webm", ".ogg", ".flac", ".aac"}

# --- HTTP ---
# Vite dev server (5173) and `vite preview` (4173) by default. Override with
# a comma-separated CORS_ORIGINS env var in any real deployment — do not put
# "*" back here, the API has no authentication in front of it.
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if o.strip()
]

# --- Presentation ---
# Stable per-speaker colours, assigned in first-appearance order so a given
# speaker keeps the same colour across the transcript, chart, and stats bar.
# These are the design's speaker-1 .. speaker-10 tokens, in order. They must
# stay in sync with the speaker-N values in frontend/src/styles/tokens.css.
# Assignment stays server-side on purpose: the same colour
# has to identify a person across the transcript, the speaker bars, and the
# charts, and only the backend knows first-appearance order. If the design
# palette changes, change it here — never re-map colours in the client.
SPEAKER_COLORS = [
    "#3b82f6",  # speaker-1
    "#a855f7",  # speaker-2
    "#10b981",  # speaker-3
    "#ec4899",  # speaker-4
    "#06b6d4",  # speaker-5
    "#8b5cf6",  # speaker-6
    "#14b8a6",  # speaker-7
    "#f472b6",  # speaker-8
    "#6366f1",  # speaker-9
    "#2dd4bf",  # speaker-10
]
