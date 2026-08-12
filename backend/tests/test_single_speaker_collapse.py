"""Single-speaker collapse in offline recluster."""

from __future__ import annotations

import numpy as np

from models.offline_diarizer import likely_single_speaker, recluster_from_embeddings


def _solo_embeddings(n=10, dim=16, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    center = np.pad([1.0, 0.0, 0.0], (0, dim - 3))
    center = center / np.linalg.norm(center)
    embs = []
    for _ in range(n):
        v = center + rng.normal(0, noise, dim)
        embs.append((v / np.linalg.norm(v)).astype(np.float32))
    return np.stack(embs)


def test_likely_single_speaker_true_for_tight_cluster():
    embs = _solo_embeddings(n=12, noise=0.05)
    assert likely_single_speaker(embs) is True


def test_likely_single_speaker_false_for_two_voices():
    a = _solo_embeddings(n=6, seed=1)
    b_center = np.pad([0.0, 1.0, 0.0], (0, 13))
    b_center = b_center / np.linalg.norm(b_center)
    b = np.stack([(b_center + np.random.default_rng(2).normal(0, 0.04, 16)).astype(np.float32) for _ in range(6)])
    for row in b:
        row /= np.linalg.norm(row)
    embs = np.vstack([a, b])
    assert likely_single_speaker(embs) is False


def test_recluster_collapses_false_two_speaker_split():
    embs = _solo_embeddings(n=8, noise=0.07)
    transcript = []
    for i in range(len(embs)):
        label = "Speaker_00" if i % 2 == 0 else "Speaker_01"
        transcript.append(
            {
                "start_sec": i * 2.0,
                "end_sec": i * 2.0 + 1.5,
                "speaker_label": label,
                "speaker": label,
                "text": f"line {i}",
            }
        )
    assert len({t["speaker_label"] for t in transcript}) == 2

    new_transcript, info = recluster_from_embeddings(transcript, embs)
    assert info.get("source") == "single_speaker" or info.get("k") == 1
    labels = {t["speaker_label"] for t in new_transcript}
    assert len(labels) == 1
