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
MIN_SPEECH_SECONDS = 0.3       # discard segments shorter than this (likely noise blips, not speech)

# --- Models ---
EMBEDDING_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
WHISPER_MODEL_SIZE = "base"    # tiny | base | small | medium | large — bigger = more accurate, slower

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

# --- Logging ---
LOG_LEVEL = "INFO"  # set to "DEBUG" for more verbose diarization/identification reasoning

# --- Storage ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SPEAKERS_DB_PATH = os.path.join(STORAGE_DIR, "speakers.json")

os.makedirs(STORAGE_DIR, exist_ok=True)
