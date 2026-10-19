"""
Measure how many voices the stored meetings really hold, and compare what
the offline recluster decides with a duration-aware alternative.

For every stored meeting (or the ids given) this embeds each transcript line
from the audio, caches the vectors under storage/embeddings/<id>_lines.npz,
and prints:

  - the speaker count the meeting currently shows
  - what models.offline_diarizer.recluster_from_embeddings would pick
  - a threshold clustering at several cut-offs, with the centroid distances
    between the clusters it finds, so one can see whether those clusters are
    different people (centroids ~0.5+ apart) or one person split (~0.2-0.35)

    python3 -m tools.diagnose_speaker_count            # every meeting
    python3 -m tools.diagnose_speaker_count MTG-xxxx   # one meeting
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)

from db import repository as repo  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage", "embeddings")
CHUNK_SECONDS = 6.0
MAX_LINES = 400


def _line_embeddings(meeting_id: str, transcript: list[dict]):
    """(embeddings, starts, ends, labels) per stored line, cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{meeting_id}_lines.npz")
    starts = np.array([float(t.get("start_sec") or 0) for t in transcript])
    ends = np.array([float(t.get("end_sec") or 0) for t in transcript])
    if os.path.exists(cache):
        data = np.load(cache)
        if len(data["starts"]) == len(starts) and np.allclose(data["starts"], starts):
            return data["embeddings"], data["chunk_emb"], data["chunk_line"], data["chunk_dur"]

    from audio_utils import load_audio_file
    from models.offline_diarizer import _embedding_for_line

    audio = load_audio_file(repo.audio_path(meeting_id))
    embs, chunk_emb, chunk_line, chunk_dur = [], [], [], []
    for i, (s, e) in enumerate(zip(starts, ends)):
        v = _embedding_for_line(audio, s, e)
        embs.append(v if v is not None else np.zeros(192, dtype=np.float32))
        # Also embed fixed-length chunks inside the line: this is what the
        # pipeline sees before lines are merged into turns.
        t = s
        while t < e:
            u = min(e, t + CHUNK_SECONDS)
            if u - t >= 0.8:
                cv = _embedding_for_line(audio, t, u)
                if cv is not None:
                    chunk_emb.append(cv)
                    chunk_line.append(i)
                    chunk_dur.append(u - t)
            t = u
    embs = np.stack(embs)
    chunk_emb = np.stack(chunk_emb) if chunk_emb else np.zeros((0, 192), dtype=np.float32)
    np.savez(
        cache,
        embeddings=embs,
        starts=starts,
        ends=ends,
        chunk_emb=chunk_emb,
        chunk_line=np.array(chunk_line, dtype=np.int32),
        chunk_dur=np.array(chunk_dur, dtype=np.float32),
    )
    return embs, chunk_emb, np.array(chunk_line), np.array(chunk_dur)


def threshold_clusters(emb: np.ndarray, dur: np.ndarray, cut: float):
    """Average-linkage cosine clustering at a distance cut-off; returns labels, centroids."""
    from sklearn.cluster import AgglomerativeClustering

    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int), [emb.mean(axis=0)] if len(emb) else []
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=cut, metric="cosine", linkage="average"
    ).fit_predict(emb)
    cents = []
    for k in sorted(set(labels)):
        m = labels == k
        c = (emb[m] * dur[m, None]).sum(axis=0)
        cents.append(c / (np.linalg.norm(c) + 1e-9))
    return labels, cents


def _dist_matrix(cents):
    from scipy.spatial.distance import cosine

    n = len(cents)
    out = []
    for i in range(n):
        out.append(" ".join(f"{cosine(cents[i], cents[j]):.2f}" for j in range(n)))
    return out


def diagnose(meeting_id: str) -> None:
    from models.offline_diarizer import recluster_from_embeddings

    m = repo.get_meeting(meeting_id) or next((t for t in repo.list_trash() if t["id"] == meeting_id), None)
    if not m or not repo.audio_path(meeting_id):
        print(f"{meeting_id}: no audio/meeting")
        return
    transcript = [t for t in (m.get("transcript") or []) if float(t.get("end_sec") or 0) > float(t.get("start_sec") or 0)]
    if len(transcript) < 2:
        print(f"{meeting_id}: {len(transcript)} line(s), skip")
        return
    embs, chunk_emb, chunk_line, chunk_dur = _line_embeddings(meeting_id, transcript)
    dur = np.array([float(t["end_sec"]) - float(t["start_sec"]) for t in transcript])
    shown = {t.get("speaker") for t in transcript}
    print("=" * 78)
    print(f"{meeting_id}  '{(m.get('title') or '')[:30]}'  {int(sum(dur))}s speech, {len(transcript)} lines, "
          f"shows {len(shown)} speakers: {sorted(shown)[:8]}")

    valid = np.linalg.norm(embs, axis=1) > 1e-6
    _, info = recluster_from_embeddings(transcript, embs)
    print(f"  current recluster on stored lines -> k={info.get('k')} ({info.get('source')}, {info.get('reason')})")

    for name, E, D in (("turn-level", embs[valid], dur[valid]), ("6s-chunks", chunk_emb, chunk_dur)):
        if len(E) < 2:
            continue
        for cut in (0.40, 0.50, 0.60):
            labels, cents = threshold_clusters(E, D, cut)
            secs = [float(D[labels == k].sum()) for k in sorted(set(labels))]
            big = [s for s in secs if s >= 8.0]
            print(f"  {name:10s} cut={cut:.2f}: k={len(cents):2d} (>=8s: {len(big)})  talk={' '.join(f'{s:.0f}' for s in sorted(secs, reverse=True)[:8])}")
            if cut == 0.50 and len(cents) <= 8:
                for row in _dist_matrix(cents):
                    print(f"      {row}")


if __name__ == "__main__":
    ids = sys.argv[1:] or [x["id"] for x in repo.list_meetings() if x.get("status") == "Completed"]
    for mid in ids:
        try:
            diagnose(mid)
        except Exception as exc:  # keep going over the rest
            print(f"{mid}: ERROR {exc!r}")
