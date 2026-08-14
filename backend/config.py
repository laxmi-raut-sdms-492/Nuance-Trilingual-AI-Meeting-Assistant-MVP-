"""Central configuration for the meeting intelligence backend."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- STT Adapter Architecture ---
STT_ADAPTER = os.getenv("STT_ADAPTER", "local").lower()            # local | cloud
CLOUD_STT_PROVIDER = os.getenv("CLOUD_STT_PROVIDER", "sarvam").lower()  # sarvam | google

# Sarvam STT Config
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v3")

# Google Cloud STT Config
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# --- Audio ---

SAMPLE_RATE = 16000            # all audio is normalized to this rate
SCD_WINDOW_SECONDS = 0.8
SCD_HOP_SECONDS = 0.2
SCD_CHANGE_THRESHOLD = 0.18
SCD_MIN_SEGMENT_SECONDS = 1.0
SCD_MIN_SUBSEGMENT_SECONDS = 0.5
SILENCE_RMS_THRESHOLD = 0.01   # below this average amplitude, treat a segment as silence

# --- Voice Activity Detection (speech segmentation) ---
# Segments are now bounded by real speech start/pause events (Silero VAD),
# not a fixed window — this replaces the old CHUNK_SECONDS approach.
MAX_SEGMENT_SECONDS = 25.0      # allow complete sentences to finish at natural pause events
MIN_SILENCE_MS = 400           # how long a pause must be before a segment is considered "ended"
# When Silero finds no speech but the file is non-silent, process uploads up to
# this length as one segment so Whisper sees full context. Longer files are
# split into MAX_SEGMENT_SECONDS windows.
VAD_FALLBACK_WHOLE_FILE_MAX_SECONDS = 60.0
# Batch Silero VAD threshold used when streaming VAD finds nothing. Lower than
# the library default (0.5) to recover speech buried under steady hum.
VAD_BATCH_FALLBACK_THRESHOLD = 0.15
# Uploads whose per-second RMS barely varies are usually a steady tone/hum with
# speech drowned out — run a notch filter before VAD/Whisper.
FLAT_TONE_RMS_STD_MAX = 0.005
HUM_NOTCH_FREQ_HZ = 220.0
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
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
# tiny | base | small | medium | large — used for English transcription and
# per-segment language detection across en/hi/mr. Hindi/Marathi decode is
# handled by Indic Conformer when INDIC_CONFORMER_ENABLED=1.

# --- Indic Conformer (Hindi + Marathi ASR, local via HuggingFace) ---
INDIC_CONFORMER_ENABLED = os.getenv("INDIC_CONFORMER_ENABLED", "1").lower() in (
    "1", "true", "yes"
)
INDIC_CONFORMER_MODEL = os.getenv(
    "INDIC_CONFORMER_MODEL", "ai4bharat/indic-conformer-600m-multilingual"
)
# ctc is faster on CPU; rnnt is often slightly more accurate.
INDIC_CONFORMER_DECODE = os.getenv("INDIC_CONFORMER_DECODE", "ctc").lower()
WHISPER_DECODE_LANGUAGES = ["en"]
INDIC_DECODE_LANGUAGES = ["hi", "mr"]

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
# When detection is weak and there is no meeting context yet, still prefer the
# 3-way guess over blind English if it clears this floor (below 3-way chance).
LANGUAGE_DETECT_WEAK_FLOOR = 0.25

# --- Code-switch (mixed-language segment) detection ---
#
# Language is decided once per segment, so a segment that contains a switch
# with no pause in it gets ONE language and ONE engine, and the half in the
# other language comes back mangled. Nothing splits a segment on language the
# way scd.py splits it on speaker.
#
# This does not fix that. It detects it, so a mangled line is flagged instead
# of being presented as a confident transcription — and so a future language
# change detector has a cheap gate telling it which segments are worth the
# expensive windowed pass.
#
# The signal is free: language_ranking() already computes the full ranking and
# only the winner was ever read. Clean single-language speech separates hugely
# (0.97 vs 0.02); a segment holding two languages lands near 0.45 vs 0.40,
# because the detector really did hear both.
#
# UNMEASURED. Unlike the ASR thresholds above, this number has not been tuned
# against recordings with known switch points — it is a starting value. Measure
# before trusting it, and see ASR_MIXED_REQUIRES_SCRIPT_BOUNDARY below.
LANGUAGE_AMBIGUITY_MARGIN = float(os.getenv("LANGUAGE_AMBIGUITY_MARGIN", "0.15"))

# Require the top two candidates to straddle the Latin/Devanagari boundary
# before calling a segment mixed.
#
# A close hi-vs-mr margin means almost nothing on its own: the two are closely
# related and the detector confuses them constantly on short segments — which
# is why asr.py already carries a hi<->mr retry. Flagging those would bury the
# real signal in noise. English against a Devanagari language is different:
# they are not acoustically confusable, so both scoring highly is good evidence
# that both were actually spoken. Set to 0 to flag every close call.
ASR_MIXED_REQUIRES_SCRIPT_BOUNDARY = os.getenv(
    "ASR_MIXED_REQUIRES_SCRIPT_BOUNDARY", "1"
).lower() in ("1", "true", "yes")

# --- Language Change Detection (splitting a segment on a code-switch) ---
#
# scd.py splits a VAD segment where the SPEAKER changes. This splits it where
# the LANGUAGE changes, so "The deadline is Friday म्हणजे उद्या" — one breath,
# no pause, one segment — stops being decoded end to end by a single engine
# with half the words coming back mangled.
#
# Division of labour, and it matters: the LID model below finds WHERE the
# switch is; it does not decide what the languages are. Whisper's
# detect_language re-runs on each resulting piece and decides that, because it
# is markedly better at hi vs mr — VoxLingua107 confuses closely related
# Indic languages. A cheap model that is only trusted for boundaries can be
# wrong about the label and still be useful.
LCD_ENABLED = os.getenv("LCD_ENABLED", "1").lower() in ("1", "true", "yes")

# Spoken-language ID model. ECAPA-TDNN, the same architecture already loaded
# for speaker embeddings, so this adds a second small model rather than a new
# class of dependency.
LID_MODEL_SOURCE = os.getenv(
    "LID_MODEL_SOURCE", "speechbrain/lang-id-voxlingua107-ecapa"
)

# Window geometry. Deliberately close to SCD's (1.5s/0.5s): that pass already
# runs an ECAPA forward per window over every segment ≥2s, so this costs about
# the same as a stage the pipeline has always paid for. 2.0s rather than 1.5s
# because language ID needs more phonetic material than speaker ID does.
LCD_WINDOW_SECONDS = float(os.getenv("LCD_WINDOW_SECONDS", "2.0"))
LCD_HOP_SECONDS = float(os.getenv("LCD_HOP_SECONDS", "0.5"))

# Below this there is no room for a switch plus two usable pieces.
LCD_MIN_SEGMENT_SECONDS = float(os.getenv("LCD_MIN_SEGMENT_SECONDS", "3.0"))

# Pieces shorter than this are merged back into a neighbour instead of being
# emitted. MIN_SPEECH_SECONDS (1.0) is the floor below which Whisper is handed
# mostly-silence and starts inventing outro captions, so cutting a 0.6s piece
# out would trade a mangled word for a hallucinated sentence. Keeping it
# attached leaves the text imperfect but real.
LCD_MIN_PIECE_SECONDS = float(os.getenv("LCD_MIN_PIECE_SECONDS", "1.5"))

# Mode filter width over the per-window language labels. One window landing on
# a loanword or a name should not open a new language run.
LCD_SMOOTHING_WINDOWS = int(os.getenv("LCD_SMOOTHING_WINDOWS", "3"))

# A window whose top posterior is below this says nothing useful; it inherits
# its predecessor's label rather than voting.
LCD_MIN_WINDOW_CONFIDENCE = float(os.getenv("LCD_MIN_WINDOW_CONFIDENCE", "0.45"))

# The hop only locates a boundary to within ±hop/2. Speakers pause very
# briefly when switching language, so the true cut is almost always the
# quietest instant nearby — search this far either side for it. Nearly free
# (short-time RMS over a fraction of a second) and worth more than a smaller
# hop, which would cost a forward pass per window.
LCD_SNAP_RADIUS_SECONDS = float(os.getenv("LCD_SNAP_RADIUS_SECONDS", "0.75"))
LCD_SNAP_FRAME_MS = int(os.getenv("LCD_SNAP_FRAME_MS", "20"))

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
# Passed through to Whisper before our post-decode guards run. Defaults (0.6,
# -1.0) make Whisper return zero segments on quiet or noisy speech whose
# no_speech_prob is high even when the decode is real — measured on a 30s
# upload where Silero VAD also found nothing. Loosening these lets Whisper
# attempt a decode; guards 1-4 below still drop fluent hallucinations.
ASR_WHISPER_NO_SPEECH_THRESHOLD = 1.0
ASR_WHISPER_LOGPROB_THRESHOLD = -2.0
# Guard 1 skips decodes at or below this logprob — extremely uncertain output
# is more likely quiet real speech than garbled noise (measured at -1.66 on a
# 30s upload Silero missed entirely).
ASR_UNCERTAIN_LOGPROB_CUTOFF = -1.5

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
# Pad short segments with neighbouring audio so Whisper sees sentence context.
# Critical for Hindi/Marathi — 2s clips without context produce garbled words.
ASR_CONTEXT_PADDING_SEC = float(os.getenv("ASR_CONTEXT_PADDING_SEC", "8.0"))
# Wider beam search for Devanagari languages (slower but more accurate).
ASR_BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "5"))

# --- Diarization ---
# Cosine distance between a new segment and a cluster's centroid.
# 0 = identical voice, ~1 = totally different.
DIARIZATION_DISTANCE_THRESHOLD = 0.55       # base threshold for a brand-new/young cluster
DIARIZATION_EMA_ALPHA = 0.15                # centroid update rate — constant, doesn't freeze over a long meeting
HYSTERESIS_BONUS = 0.05                     # discount for the speaker who just spoke (anti flip-flop)
THRESHOLD_GROWTH_PER_SEGMENT = 0.01         # mature clusters get a looser (more forgiving) threshold
THRESHOLD_GROWTH_CAP = 0.15                 # ...up to this much looser
# Evidence (tools/DIARIZATION_FINDINGS.md): 0.55 is the largest merge distance
# safe on both over-split and correctly-split meetings. Uploads also get an
# offline auto-k recluster pass that fixes remaining fragmentation.
CLUSTER_MERGE_DISTANCE = float(os.getenv("CLUSTER_MERGE_DISTANCE", "0.55"))
CLUSTER_MERGE_CHECK_EVERY = 10              # check for mergeable clusters every N segments

# Short clips may still be a NEW speaker (brief reply). Only force them onto
# an existing cluster when the voice is also close enough.
# 2.0s matches the evidenced config that hits ground truth on clean multi-party.
MIN_NEW_CLUSTER_SECONDS = float(os.getenv("MIN_NEW_CLUSTER_SECONDS", "2.0"))
SHORT_FORCE_MATCH_MAX_DISTANCE = float(
    os.getenv("SHORT_FORCE_MATCH_MAX_DISTANCE", "0.70")
)
# Rolling ECAPA samples kept per speaker cluster for profile matching.
CLUSTER_PROFILE_MAX_SAMPLES = int(os.getenv("CLUSTER_PROFILE_MAX_SAMPLES", "24"))
# Outlier segments assigned tentatively until this many consistent outliers appear.
NEW_CLUSTER_EVIDENCE_COUNT = int(os.getenv("NEW_CLUSTER_EVIDENCE_COUNT", "3"))
# Distance margin above threshold still treated as same speaker (variation, not new voice).
NEW_CLUSTER_MARGIN = float(os.getenv("NEW_CLUSTER_MARGIN", "0.08"))
# When only one cluster exists, dist must exceed spread * this to split.
SINGLE_CLUSTER_SPLIT_MULTIPLIER = float(
    os.getenv("SINGLE_CLUSTER_SPLIT_MULTIPLIER", "0.1")
)
# Offline: collapse to 1 speaker when p90 pairwise embedding distance is below this.
SINGLE_SPEAKER_P90_DISTANCE = float(os.getenv("SINGLE_SPEAKER_P90_DISTANCE", "0.38"))

# --- Speaker turns (transcript readability) ---
TURN_MERGE_MAX_GAP_SEC = float(os.getenv("TURN_MERGE_MAX_GAP_SEC", "2.5"))
TURN_MAX_DURATION_SEC = float(os.getenv("TURN_MAX_DURATION_SEC", "120.0"))

# --- Transcript cleanup (separate from summarization) ---
TRANSCRIPT_CLEANUP_ENABLED = os.getenv("TRANSCRIPT_CLEANUP_ENABLED", "0").lower() in (
    "1", "true", "yes"
)
TRANSCRIPT_CLEANUP_TIMEOUT_SECONDS = int(
    os.getenv("TRANSCRIPT_CLEANUP_TIMEOUT_SECONDS", "5")
)

# Offline auto-k recluster (file uploads): ignore silhouette below this.
OFFLINE_RECLUSTER_MIN_SILHOUETTE = float(
    os.getenv("OFFLINE_RECLUSTER_MIN_SILHOUETTE", "0.18")
)
# When two k values score within this silhouette band, prefer the smaller k.
OFFLINE_SILHOUETTE_TIE_EPSILON = float(
    os.getenv("OFFLINE_SILHOUETTE_TIE_EPSILON", "0.001")
)

# Same-meeting fragment repair (over-split): only merge when embeddings agree.
WITHIN_MEETING_MERGE_SIMILARITY = float(
    os.getenv("WITHIN_MEETING_MERGE_SIMILARITY", "0.78")
)
SHORT_LEFTOVER_SECONDS = float(os.getenv("SHORT_LEFTOVER_SECONDS", "4.0"))
INCONCLUSIVE_MERGE_SIMILARITY = float(
    os.getenv("INCONCLUSIVE_MERGE_SIMILARITY", "0.40")
)
# Collapsed dialogue repair (under-split): split consecutive same-label lines
# when their embeddings differ by at least this cosine distance.
COLLAPSED_SPLIT_DISTANCE = float(os.getenv("COLLAPSED_SPLIT_DISTANCE", "0.40"))

# --- Identification ---
# Cosine similarity (not distance) between a segment and an enrolled voice.
# Auto-label only when similarity >= this value. Override with
# IDENTIFICATION_SIMILARITY_THRESHOLD.
IDENTIFICATION_SIMILARITY_THRESHOLD = float(
    os.getenv("IDENTIFICATION_SIMILARITY_THRESHOLD", "0.72")
)
# If the top two enrolled matches are this close, treat the result as
# ambiguous and keep the generic diarization label instead of guessing.
IDENTIFICATION_AMBIGUITY_MARGIN = float(
    os.getenv("IDENTIFICATION_AMBIGUITY_MARGIN", "0.03")
)

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
SUMMARY_TIMEOUT_SECONDS = int(os.getenv("SUMMARY_TIMEOUT_SECONDS", "15"))

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

# --- STT processing mode (local vs. cloud) ---
# Per-meeting choice (Meeting.processing_mode) of which engine transcribes
# that meeting. This env var is only the fallback used when a meeting's own
# column is NULL/missing (old rows created before this existed) — see
# stt/resolver.py. Local stays the default everywhere; cloud is opt-in.
DEFAULT_PROCESSING_MODE = os.getenv("DEFAULT_PROCESSING_MODE", "local").strip().lower()

# Which cloud provider stt/resolver.py uses when a meeting has
# processing_mode="cloud" but no stt_provider of its own recorded (or when
# the caller passes stt_provider=None). Sarvam is the only provider wired up
# today; the resolver's provider registry is what makes adding another
# (Google/Azure/AWS) not require touching pipeline.py.
CLOUD_STT_DEFAULT_PROVIDER = os.getenv("CLOUD_STT_DEFAULT_PROVIDER", "sarvam").strip().lower()

# --- Sarvam (cloud STT provider) ---
# API key is never hardcoded — required only when a meeting actually uses
# processing_mode=cloud with provider=sarvam; local-only usage never reads
# this. See stt/sarvam_adapter.py.
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
# saaras:v3 (default/recommended) or saaras:v4 (broader language coverage).
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
# transcribe | translate | verbatim | translit | codemix — only applies to
# saaras:v3. "transcribe" matches what the local pipeline does (original
# language, normalized).
SARVAM_STT_MODE = os.getenv("SARVAM_STT_MODE", "transcribe")
SARVAM_TIMEOUT_SECONDS = float(os.getenv("SARVAM_TIMEOUT_SECONDS", "30"))

# --- Database ---
# PostgreSQL. Override with DATABASE_URL in the environment; the default is a
# local development role and is not a secret worth protecting, but do not reuse
# it anywhere real.
#
# The driver is psycopg 3 — note "postgresql+psycopg", not "postgresql" (which
# resolves to psycopg2) and not "postgresql+psycopg2".
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://nuance:nuance_dev@localhost:5432/nuance",
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
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:4173,http://127.0.0.1:4173",
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
