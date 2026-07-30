"""Pipeline finalize_labels — remaps display names without touching ASR text."""

from __future__ import annotations

import numpy as np

from pipeline import MeetingSession


class _FakeIdentifier:
    def __init__(self, mapping: dict[str, tuple[str, float]] | None = None):
        # keyed by a fingerprint of the centroid so tests stay simple
        self._mapping = mapping or {}
        self.enrolled: dict[str, dict] = {}
        self.calls = 0

    def identify(self, embedding):
        self.calls += 1
        key = tuple(np.round(embedding, 3).tolist())
        if key in self._mapping:
            return self._mapping[key]
        # Also match against anything enrolled during finalize.
        for name, profile in self.enrolled.items():
            if np.allclose(profile["centroid"], embedding, atol=1e-3):
                return name, 0.99
        return "Unknown", 0.0

    def enroll(self, name, embedding, overwrite=False):
        self.enrolled[name] = {"centroid": np.asarray(embedding, dtype=np.float32), "sample_count": 1}

    def refresh(self, force=True):
        return


def test_finalize_labels_remaps_all_lines_for_a_cluster():
    centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    identifier = _FakeIdentifier({(1.0, 0.0, 0.0): ("Laxmi", 0.98)})
    session = MeetingSession("MTG-test", identifier)
    session.diarizer.clusters["Speaker_00"] = {"centroid": centroid, "count": 5}
    session.transcript = [
        {
            "speaker": "Speaker_00",
            "speaker_label": "Speaker_00",
            "identified_as": "Unknown",
            "confidence": 0.4,
            "text": "hello",
        },
        {
            "speaker": "Speaker_00",
            "speaker_label": "Speaker_00",
            "identified_as": "Unknown",
            "confidence": 0.5,
            "text": "world",
        },
    ]

    session.finalize_labels()

    for entry in session.transcript:
        assert entry["speaker"] == "Laxmi"
        assert entry["identified_as"] == "Laxmi"
        assert entry["confidence"] == 0.98
        assert entry["speaker_label"] == "Speaker_00"
        assert entry["text"] in ("hello", "world")
    assert "Laxmi" in identifier.enrolled


def test_finalize_labels_keeps_generic_label_when_unknown():
    centroid = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    identifier = _FakeIdentifier({})
    session = MeetingSession("MTG-test", identifier)
    session.diarizer.clusters["Speaker_01"] = {"centroid": centroid, "count": 2}
    session.transcript = [
        {
            "speaker": "Speaker_01",
            "speaker_label": "Speaker_01",
            "identified_as": "Unknown",
            "confidence": 0.2,
            "text": "hi",
        }
    ]

    session.finalize_labels()

    assert session.transcript[0]["speaker"] == "Speaker_01"
    assert session.transcript[0]["identified_as"] == "Unknown"


def test_finalize_merges_same_voice_split_across_clusters():
    """Speaker_02 that is really Anushka (close centroid) must become Anushka."""
    anushka = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Slightly drifted but still the same voice (> 0.50 cosine similarity).
    fragment = np.array([0.95, 0.1, 0.0], dtype=np.float32)
    fragment = fragment / np.linalg.norm(fragment)

    identifier = _FakeIdentifier({(1.0, 0.0, 0.0): ("Anushka", 0.97)})
    session = MeetingSession("MTG-test", identifier)
    session.diarizer.clusters["Speaker_00"] = {"centroid": anushka, "count": 4}
    session.diarizer.clusters["Speaker_02"] = {"centroid": fragment, "count": 1}
    session.transcript = [
        {
            "speaker": "Speaker_00",
            "speaker_label": "Speaker_00",
            "identified_as": "Unknown",
            "confidence": 0.0,
            "text": "Good morning Lakshmi.",
        },
        {
            "speaker": "Speaker_02",
            "speaker_label": "Speaker_02",
            "identified_as": "Unknown",
            "confidence": 0.0,
            "text": "So please, I am okay.",
        },
    ]

    session.finalize_labels()

    assert session.transcript[0]["speaker"] == "Anushka"
    assert session.transcript[1]["speaker"] == "Anushka"
    assert session.transcript[1]["speaker_label"] == "Speaker_02"
