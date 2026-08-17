"""Tests for evidence-based online diarization — false split prevention."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine

from models.diarizer import SessionDiarizer


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def test_single_speaker_variation_stays_one_cluster():
    """Volume/style shifts should not mint Speaker_01 on solo speech."""
    d = SessionDiarizer()
    center = _norm([1.0, 0.2, 0.0] + [0.0] * 13)
    rng = np.random.default_rng(42)

    label = None
    for i in range(12):
        noise = rng.normal(0, 0.08, size=center.shape)
        emb = _norm(center + noise)
        label = d.add_segment(float(i * 2), float(i * 2 + 1.5), emb)

    assert len(d.clusters) == 1
    assert label == "Speaker_00"


def test_two_distinct_voices_create_two_clusters():
    d = SessionDiarizer()
    voice_a = _norm([1.0, 0.0, 0.0] + [0.0] * 13)
    voice_b = _norm([0.0, 1.0, 0.0] + [0.0] * 13)
    rng = np.random.default_rng(0)

    for i in range(6):
        emb = _norm(voice_a + rng.normal(0, 0.04, voice_a.shape))
        d.add_segment(float(i * 2), float(i * 2 + 1.5), emb)

    labels = set()
    for i in range(6, 12):
        emb = _norm(voice_b + rng.normal(0, 0.04, voice_b.shape))
        labels.add(d.add_segment(float(i * 2), float(i * 2 + 1.5), emb))

    assert len(d.clusters) == 2


def test_cluster_profile_stores_multiple_embeddings():
    d = SessionDiarizer()
    emb = _norm(np.ones(16))
    d.add_segment(0, 1, emb)
    d.add_segment(2, 3, _norm(emb + 0.01))
    assert d.clusters["Speaker_00"]["count"] == 2
    assert len(d.clusters["Speaker_00"]["embeddings"]) == 2


def test_merge_recover_premature_split():
    d = SessionDiarizer()
    base = _norm([1.0, 0.1, 0.0] + [0.0] * 13)
    close = _norm(base + 0.05)
    far = _norm([0.0, 1.0, 0.0] + [0.0] * 13)

    d.add_segment(0, 2, base)
    d.add_segment(2, 4, close)  # might defer or merge later
    d.add_segment(4, 6, base)
    # Force periodic merge check
    for _ in range(10):
        d._maybe_merge_clusters()

    # After merges, similar voices should not stay as 3+ clusters for same person
    if len(d.clusters) >= 2:
        dists = [
            cosine(d.clusters[a]["centroid"], d.clusters[b]["centroid"])
            for i, a in enumerate(d.clusters)
            for b in list(d.clusters)[i + 1 :]
        ]
        assert min(dists) > 0.2 or len(d.clusters) <= 2


def test_overlapping_speech_is_flagged_and_does_not_mint_spurious_cluster():
    """
    Overlapping speech mixture embedding must be flagged with is_overlap=True,
    list both candidate speaker labels, and NOT mint a spurious third cluster or poll centroids.
    """
    d = SessionDiarizer()
    voice_a = _norm([1.0, 0.0, 0.0] + [0.0] * 189)
    voice_b = _norm([0.0, 1.0, 0.0] + [0.0] * 189)

    # Establish Speaker_00
    label_a = d.add_segment(0.0, 3.0, voice_a)
    assert label_a == "Speaker_00"
    assert d.last_segment_info["is_overlap"] is False

    # Establish Speaker_01
    label_b = d.add_segment(3.5, 6.5, voice_b)
    assert label_b == "Speaker_01"
    assert d.last_segment_info["is_overlap"] is False

    # Save original centroids
    c0_orig = d.clusters["Speaker_00"]["centroid"].copy()
    c1_orig = d.clusters["Speaker_01"]["centroid"].copy()

    # Mixed overlapping segment
    emix = _norm(voice_a + voice_b)
    label_mix = d.add_segment(7.0, 10.0, emix)

    # Overlap assertions
    info = d.last_segment_info
    assert info["is_overlap"] is True
    assert set(info["candidate_labels"]) == {"Speaker_00", "Speaker_01"}
    assert "Speaker_00" in label_mix and "Speaker_01" in label_mix

    # Ensure cluster count stays exactly 2 (no spurious Speaker_02 created)
    assert len(d.clusters) == 2

    # Ensure centroids were NOT updated/corrupted by the mixed vector
    assert np.allclose(d.clusters["Speaker_00"]["centroid"], c0_orig)
    assert np.allclose(d.clusters["Speaker_01"]["centroid"], c1_orig)


def test_normal_noisy_segment_does_not_false_positive_overlap():
    """
    A single-speaker segment with standard acoustic noise/drift must match its cluster
    and NOT trigger a false positive overlap detection.
    """
    d = SessionDiarizer()
    voice_a = _norm([1.0, 0.0, 0.0] + [0.0] * 189)
    voice_b = _norm([0.0, 1.0, 0.0] + [0.0] * 189)

    d.add_segment(0.0, 3.0, voice_a)
    d.add_segment(3.5, 6.5, voice_b)

    rng = np.random.default_rng(123)
    # Speaker A speaks with realistic acoustic variation/noise
    noisy_a = _norm(voice_a + rng.normal(0, 0.06, size=voice_a.shape))
    label = d.add_segment(7.0, 10.0, noisy_a)

    assert label == "Speaker_00"
    assert d.last_segment_info["is_overlap"] is False
    assert d.last_segment_info["candidate_labels"] == ["Speaker_00"]


