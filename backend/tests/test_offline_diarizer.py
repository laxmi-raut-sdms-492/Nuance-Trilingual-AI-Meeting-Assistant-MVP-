"""Offline auto-k recluster — accurate dynamic speaker count."""

from __future__ import annotations

import numpy as np

from models.offline_diarizer import (
    pick_speaker_count,
    recluster_from_embeddings,
    apply_assignment_to_transcript,
)


def _make_speakers(n_per=(8, 8, 8), dim=16, seed=0):
    """Synthetic well-separated speaker embeddings."""
    rng = np.random.default_rng(seed)
    centers = [
        np.pad([1.0, 0.0, 0.0], (0, dim - 3)),
        np.pad([0.0, 1.0, 0.0], (0, dim - 3)),
        np.pad([0.0, 0.0, 1.0], (0, dim - 3)),
        np.pad([0.7, 0.7, 0.0], (0, dim - 3)),
        np.pad([0.0, 0.7, 0.7], (0, dim - 3)),
    ]
    embs = []
    transcript = []
    t = 0.0
    for speaker_i, count in enumerate(n_per):
        center = centers[speaker_i]
        center = center / np.linalg.norm(center)
        for j in range(count):
            noise = rng.normal(0, 0.05, size=dim)
            vec = center + noise
            vec = (vec / np.linalg.norm(vec)).astype(np.float32)
            embs.append(vec)
            # Pretend streaming over-split every line to its own label.
            label = f"Speaker_{len(embs) - 1:02d}"
            transcript.append(
                {
                    "start_sec": t,
                    "end_sec": t + 2.0,
                    "speaker": label,
                    "speaker_label": label,
                    "text": f"line {len(transcript)}",
                }
            )
            t += 2.5
    return transcript, np.stack(embs)


def test_pick_speaker_count_finds_three_voices():
    _, embs = _make_speakers(n_per=(10, 10, 10))
    picked = pick_speaker_count(embs, k_max=8, min_score=0.1)
    assert picked is not None
    best_k, score, _ = picked
    assert best_k == 3
    assert score > 0.2


def test_recluster_collapses_over_split():
    transcript, embs = _make_speakers(n_per=(10, 10, 10))
    assert len({t["speaker_label"] for t in transcript}) == 30

    new_transcript, info = recluster_from_embeddings(
        transcript, embs, min_score=0.1
    )
    assert info["applied"] is True
    assert info["k"] == 3
    labels = {t["speaker_label"] for t in new_transcript}
    assert len(labels) == 3


def test_merge_fragment_folds_single_line_speaker():
    """One short line split as its own speaker merges into the nearest voice."""
    from models.offline_diarizer import merge_fragment_clusters

    dim = 16
    center_a = np.pad([1.0, 0.0, 0.0], (0, dim - 3))
    center_a = center_a / np.linalg.norm(center_a)
    center_b = np.pad([0.0, 1.0, 0.0], (0, dim - 3))
    center_b = center_b / np.linalg.norm(center_b)
    # Fragment: almost identical to speaker A (one brief reply).
    frag = (center_a + np.random.default_rng(0).normal(0, 0.03, dim)).astype(np.float32)
    frag = frag / np.linalg.norm(frag)

    embs = np.stack([center_a, center_a, center_b, center_b, frag]).astype(np.float32)
    assignment = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    transcript = [
        {"start_sec": 0, "end_sec": 2, "speaker_label": "Speaker_00", "text": "a"},
        {"start_sec": 2, "end_sec": 4, "speaker_label": "Speaker_00", "text": "b"},
        {"start_sec": 4, "end_sec": 6, "speaker_label": "Speaker_01", "text": "c"},
        {"start_sec": 6, "end_sec": 8, "speaker_label": "Speaker_01", "text": "d"},
        {"start_sec": 8, "end_sec": 9, "speaker_label": "Speaker_02", "text": "What do you mean?"},
    ]
    merged = merge_fragment_clusters(embs, assignment, transcript)
    assert len(set(int(a) for a in merged)) == 2
    assert int(merged[-1]) == int(merged[0])


def test_recluster_skips_when_already_correct_k():
    transcript, embs = _make_speakers(n_per=(8, 8))
    # Assign correct streaming labels (2 speakers).
    for i, row in enumerate(transcript):
        label = "Speaker_00" if i < 8 else "Speaker_01"
        row["speaker"] = label
        row["speaker_label"] = label

    new_transcript, info = recluster_from_embeddings(
        transcript, embs, min_score=0.1
    )
    assert info["applied"] is False
    assert info["reason"] == "already_matched"
    assert new_transcript is transcript


def test_apply_assignment_renumbers_by_first_appearance():
    transcript = [
        {"start_sec": 0, "speaker": "A", "speaker_label": "A", "text": "x"},
        {"start_sec": 1, "speaker": "A", "speaker_label": "A", "text": "y"},
        {"start_sec": 2, "speaker": "A", "speaker_label": "A", "text": "z"},
    ]
    assignment = np.array([1, 0, 1])
    out = apply_assignment_to_transcript(transcript, assignment)
    assert out[0]["speaker_label"] == "Speaker_00"
    assert out[1]["speaker_label"] == "Speaker_01"
    assert out[2]["speaker_label"] == "Speaker_00"


def test_recluster_handles_zero_vectors():
    """Verify that embeddings containing zero vectors do not cause ValueError in cosine distance."""
    transcript, embs = _make_speakers(n_per=(6, 6, 6))
    # Inject zero vectors at random indices (e.g. short/silent lines)
    embs[2] = np.zeros_like(embs[2])
    embs[7] = np.zeros_like(embs[7])

    # Should not raise ValueError: Cosine affinity cannot be used when X contains zero vectors
    new_transcript, info = recluster_from_embeddings(
        transcript, embs, min_score=0.1
    )
    assert info is not None
    assert len(new_transcript) == len(transcript)

