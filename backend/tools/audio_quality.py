"""
Measure recording quality for a stored meeting.

Written because a diarization result is only as trustworthy as the audio under
it. Acoustic loopback — playing a video through speakers and recording it on
the laptop microphone — adds microphone AGC, room reflection and clipping, all
of which move a speaker's ECAPA embedding around and can look exactly like
"the diarizer fragments".

    python3 -m tools.audio_quality MTG-85407e28d5eb MTG-de4df0ad891a
    python3 -m tools.audio_quality /path/to/new-recording.wav   # before uploading

Reports, per meeting:
  clipping        share of samples pinned at full scale — the direct signature
                  of an overdriven microphone
  crest factor    peak / RMS in dB. Speech recorded cleanly sits around 12-20dB;
                  much lower means the peaks have been squashed (AGC/limiting)
  HF rolloff      frequency below which 95% of the energy sits. Speech through
                  a laptop speaker and back loses its top end
  level drift     spread of per-second RMS across the recording — AGC riding
                  the gain shows up here
  intra/inter     distance between segments of the same cluster vs different
                  clusters, from the cached embeddings. This is the number that
                  decides whether diarization is even solvable: if the two
                  distributions overlap, no threshold separates them
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file  # noqa: E402
from config import SAMPLE_RATE  # noqa: E402
from db import repository as repo  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage", "embeddings")


def clipping_ratio(audio: np.ndarray, threshold: float = 0.985) -> float:
    return float(np.mean(np.abs(audio) >= threshold))


def crest_factor_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    if rms <= 0 or peak <= 0:
        return 0.0
    return 20.0 * np.log10(peak / rms)


def spectral_rolloff(audio: np.ndarray, fraction: float = 0.95) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
    cumulative = np.cumsum(spectrum)
    if cumulative[-1] <= 0:
        return 0.0
    return float(freqs[np.searchsorted(cumulative, cumulative[-1] * fraction)])


def level_drift_db(audio: np.ndarray) -> float:
    """Spread between the loud and quiet ends of the per-second RMS."""
    per_second = [
        float(np.sqrt(np.mean(audio[i : i + SAMPLE_RATE] ** 2)))
        for i in range(0, len(audio) - SAMPLE_RATE, SAMPLE_RATE)
    ]
    voiced = [r for r in per_second if r > 0.005]
    if len(voiced) < 4:
        return 0.0
    loud, quiet = np.percentile(voiced, 90), np.percentile(voiced, 10)
    return float(20.0 * np.log10(loud / quiet)) if quiet > 0 else 0.0


def embedding_separation(meeting_id: str):
    """
    Same-speaker vs different-speaker distances, using the stored transcript's
    labels as the grouping. Those labels are the diarizer's own output, so this
    is not an accuracy measure — it shows how far apart the embeddings sit,
    which is what any clustering algorithm has to work with.
    """
    path = os.path.join(CACHE_DIR, f"{meeting_id}.npz")
    if not os.path.exists(path):
        return None

    from scipy.spatial.distance import cosine

    data = np.load(path)
    embeddings = data["embeddings"]

    meeting = repo.get_meeting(meeting_id)
    labels = [line["speaker"] for line in (meeting or {}).get("transcript", [])]
    if len(labels) != len(embeddings):
        # Extraction and the stored transcript can differ by segments ASR
        # dropped. Compare only the common prefix rather than silently
        # mismatching embeddings to labels.
        n = min(len(labels), len(embeddings))
        labels, embeddings = labels[:n], embeddings[:n]

    same, different = [], []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = cosine(embeddings[i], embeddings[j])
            (same if labels[i] == labels[j] else different).append(d)

    if not same or not different:
        return None
    return float(np.mean(same)), float(np.mean(different)), len(same), len(different)


def report(target: str):
    """
    Report on a stored meeting id, or on a bare audio file.

    The file form exists so a new recording can be vetted BEFORE it is uploaded:
    the acoustic metrics need no transcript, and a recording whose separation is
    hopeless is not worth a Whisper pass. Only `separation` needs a stored
    meeting, because it reads the cached embeddings keyed by meeting id.
    """
    if os.path.isfile(target):
        path, meeting_id, title = target, None, os.path.basename(target)
    else:
        meeting_id = target
        path = repo.audio_path(meeting_id)
        if not path:
            print(f"{meeting_id}: no audio stored, and no file at that path")
            return
        title = (repo.get_meeting(meeting_id) or {}).get("title", "?")

    audio = load_audio_file(path)

    print(f"\n{meeting_id or path}  \"{title}\"")
    print(f"  duration        {len(audio) / SAMPLE_RATE:.1f}s")
    print(f"  clipping        {clipping_ratio(audio) * 100:.3f}% of samples at full scale")
    print(f"  crest factor    {crest_factor_db(audio):.1f} dB   (clean speech ~12-20 dB)")
    print(f"  HF rolloff      {spectral_rolloff(audio):.0f} Hz  (95% of energy below this)")
    print(f"  level drift     {level_drift_db(audio):.1f} dB   (p90/p10 of per-second RMS)")

    separation = embedding_separation(meeting_id) if meeting_id else None
    if separation:
        same, different, n_same, n_diff = separation
        print(f"  same-speaker    mean cosine distance {same:.3f}  ({n_same} pairs)")
        print(f"  cross-speaker   mean cosine distance {different:.3f}  ({n_diff} pairs)")
        print(f"  separation      {different - same:+.3f}  (bigger is easier to cluster)")
    elif meeting_id:
        print("  separation      no cached embeddings — run tools.extract_embeddings first")
    else:
        print("  separation      needs an uploaded meeting (acoustic metrics only for a file)")


if __name__ == "__main__":
    ids = sys.argv[1:]
    if not ids:
        ids = [m["id"] for m in repo.list_meetings()]
    for meeting_id in ids:
        report(meeting_id)
