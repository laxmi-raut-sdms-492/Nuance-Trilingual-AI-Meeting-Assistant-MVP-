"""
When the offline pass is allowed to end up with MORE speakers than streaming.

tests/test_offline_reduction_gate.py pins the merge direction. Nothing pinned
this one, and the gap was not theoretical: because the support check ran only
when silhouette asked for a smaller k, a larger k was applied unchecked.

Measured on a 3-speaker, 180s recording the offline pass took four streaming
clusters to seven, and every speaker it "found" was a sliver peeled off a real
one — 0.6s off a 42s speaker, 0.4s off a 10s speaker, two short pieces off a
24s speaker.

The similarity test the merge gate uses cannot catch this. Each peeled part's
centroid sat 0.06-0.20 similarity from the cluster it left, which that test
reads as a different person; a sub-second ECAPA vector is noise, and noise is
far from everything. What separates a discovered speaker from a promoted
over-split is whether there is enough speech to be a participant at all, so
that is what this gate asks and what these tests pin.
"""

from __future__ import annotations

import numpy as np

from models.offline_diarizer import _split_is_supported


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / (np.linalg.norm(vector) + 1e-9)


def _voice(angle: float) -> np.ndarray:
    return _unit([np.cos(angle), np.sin(angle), 0.0])


def _lines(spans: list[tuple[float, float]]) -> list[dict]:
    return [{"start_sec": a, "end_sec": b, "speaker_label": "S0"} for a, b in spans]


def test_peeling_a_subsecond_sliver_off_a_real_speaker_is_refused():
    """
    The reported bug, at its clearest: one line of 0.4s is separated from a
    speaker holding 40s. The sliver's embedding is deliberately far from the
    cluster it leaves, which is exactly what made the similarity test wave
    this through.
    """
    spans = [(i * 4.0, i * 4.0 + 4.0) for i in range(10)] + [(40.0, 40.4)]
    transcript = _lines(spans)
    embeddings = np.array([_voice(0.0) for _ in range(10)] + [_voice(1.4)], dtype=np.float32)
    streaming = np.array([0] * 11)
    offline = np.array([0] * 10 + [1])

    supported, why = _split_is_supported(embeddings, streaming, offline, transcript)

    assert supported is False
    assert why.startswith("split_peels_fragment")


def test_two_established_halves_are_allowed_to_split():
    """
    The opposite failure. Streaming lumped two real people into one cluster;
    both halves carry plenty of speech, so the split is the correction the
    offline pass should be making and must not be blocked.
    """
    spans = [(i * 4.0, i * 4.0 + 4.0) for i in range(12)]
    transcript = _lines(spans)
    embeddings = np.array(
        [_voice(0.0) for _ in range(6)] + [_voice(1.2) for _ in range(6)],
        dtype=np.float32,
    )
    streaming = np.array([0] * 12)
    offline = np.array([0] * 6 + [1] * 6)

    supported, why = _split_is_supported(embeddings, streaming, offline, transcript)

    assert supported is True
    assert why == "split_parts_established"


def test_assignment_that_splits_nothing_is_untouched():
    """A pure relabel/merge must not be judged by this gate."""
    spans = [(i * 4.0, i * 4.0 + 4.0) for i in range(6)]
    transcript = _lines(spans)
    embeddings = np.array([_voice(0.0) for _ in range(6)], dtype=np.float32)
    streaming = np.array([0, 0, 0, 1, 1, 1])
    offline = np.array([0, 0, 0, 0, 0, 0])

    supported, why = _split_is_supported(embeddings, streaming, offline, transcript)

    assert supported is True
    assert why == "split_parts_established"


def test_a_single_suspect_part_rejects_the_whole_assignment():
    """
    One cluster splits defensibly, another peels a sliver. The clustering is
    judged as a whole — matching _reduction_is_supported — because these are
    two halves of one decision about whether to trust it.
    """
    spans = (
        [(i * 4.0, i * 4.0 + 4.0) for i in range(12)]   # cluster 0: 2 real halves
        + [(60.0, 70.0), (70.0, 70.5)]                   # cluster 1: speaker + sliver
    )
    transcript = _lines(spans)
    embeddings = np.array(
        [_voice(0.0) for _ in range(6)]
        + [_voice(1.2) for _ in range(6)]
        + [_voice(2.4), _voice(2.9)],
        dtype=np.float32,
    )
    streaming = np.array([0] * 12 + [1, 1])
    offline = np.array([0] * 6 + [1] * 6 + [2, 3])

    supported, why = _split_is_supported(embeddings, streaming, offline, transcript)

    assert supported is False
    assert why.startswith("split_peels_fragment")
