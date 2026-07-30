"""Unit tests for cosine matching and ambiguity rejection."""

from __future__ import annotations

import numpy as np
import pytest

from models.speaker_matcher import UNKNOWN, cosine_similarity, match_speaker, rank_matches


def _unit(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_cosine_similarity_identical():
    v = _unit([1.0, 0.0, 0.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)


def test_rank_matches_orders_best_first():
    profiles = {
        "A": {"centroid": _unit([1.0, 0.0, 0.0])},
        "B": {"centroid": _unit([0.0, 1.0, 0.0])},
    }
    ranked = rank_matches(_unit([0.9, 0.1, 0.0]), profiles)
    assert ranked[0][0] == "A"
    assert ranked[0][1] > ranked[1][1]


def test_match_below_threshold_returns_unknown():
    profiles = {"A": {"centroid": _unit([1.0, 0.0, 0.0])}}
    name, score = match_speaker(
        _unit([0.0, 1.0, 0.0]),
        profiles,
        threshold=0.95,
        ambiguity_margin=0.03,
    )
    assert name == UNKNOWN
    assert score < 0.95


def test_match_above_threshold():
    profiles = {"Laxmi": {"centroid": _unit([1.0, 0.0, 0.0])}}
    query = _unit([1.0, 0.01, 0.0])
    name, score = match_speaker(query, profiles, threshold=0.95, ambiguity_margin=0.03)
    assert name == "Laxmi"
    assert score >= 0.95


def test_ambiguous_close_matches_rejected():
    # Two enrolled voices nearly equally similar to the query.
    profiles = {
        "A": {"centroid": _unit([1.0, 0.05, 0.0])},
        "B": {"centroid": _unit([1.0, 0.04, 0.0])},
    }
    query = _unit([1.0, 0.045, 0.0])
    name, score = match_speaker(query, profiles, threshold=0.5, ambiguity_margin=0.05)
    assert name == UNKNOWN
    assert score > 0.5


def test_clear_winner_not_ambiguous():
    profiles = {
        "A": {"centroid": _unit([1.0, 0.0, 0.0])},
        "B": {"centroid": _unit([0.0, 1.0, 0.0])},
    }
    name, score = match_speaker(
        _unit([1.0, 0.0, 0.0]),
        profiles,
        threshold=0.95,
        ambiguity_margin=0.03,
    )
    assert name == "A"
    assert score == pytest.approx(1.0, abs=1e-5)


def test_empty_profiles():
    name, score = match_speaker(_unit([1.0, 0.0]), {}, threshold=0.95)
    assert name == UNKNOWN
    assert score == 0.0
