"""
When the offline pass is allowed to end up with fewer speakers than streaming.

Two real failures pull in opposite directions and share one switch.

  Over-split: one person pauses mid-turn and restarts. The restart is scored
  against a young cluster built from a short noisy segment, misses, and opens
  Speaker_03. The transcript shows one person as two, and a sentence spanning
  the pause is attributed to both.

  Over-merge: two genuinely different people with somewhat similar voices are
  folded together, and a participant disappears from the meeting.

Refusing every reduction turns off the correction the offline pass exists for.
Accepting every reduction merges strangers. So the decision is made per merge,
on whether the voices being joined actually agree — which is what these tests
pin, from both sides.
"""

from __future__ import annotations

import numpy as np

from config import INCONCLUSIVE_MERGE_SIMILARITY, WITHIN_MEETING_MERGE_SIMILARITY
from models.offline_diarizer import _reduction_is_supported


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / (np.linalg.norm(vector) + 1e-9)


def _voice(angle: float) -> np.ndarray:
    """A point on the unit circle — cosine similarity is cos(angle gap)."""
    return _unit([np.cos(angle), np.sin(angle), 0.0])


def _transcript(n: int, seconds: float = 4.0) -> list[dict]:
    return [
        {"start_sec": i * seconds, "end_sec": (i + 1) * seconds, "speaker_label": f"S{i}"}
        for i in range(n)
    ]


def _gap_for(similarity: float) -> float:
    return float(np.arccos(np.clip(similarity, -1.0, 1.0)))


def test_one_speaker_split_by_a_pause_is_allowed_to_merge():
    """
    The reported bug. Same person either side of a pause, so the two streaming
    clusters sit almost on top of each other. Merging them is the whole point
    of the offline pass.
    """
    same = _gap_for(0.97)
    embeddings = np.array(
        [_voice(0.0), _voice(0.01), _voice(0.02)] + [_voice(same), _voice(same + 0.01)],
        dtype=np.float32,
    )
    streaming = np.array([0, 0, 0, 1, 1])
    offline = np.array([0, 0, 0, 0, 0])

    supported, why = _reduction_is_supported(embeddings, streaming, offline, _transcript(5))

    assert supported, why


def test_two_different_people_are_never_merged():
    """
    Anushka's bug, and the reason a blanket 'always reduce' is wrong. Far
    apart voices must survive the reduction as separate speakers however
    confident the silhouette score is.
    """
    far = _gap_for(0.20)
    embeddings = np.array(
        [_voice(0.0)] * 4 + [_voice(far)] * 4,
        dtype=np.float32,
    )
    streaming = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    offline = np.zeros(8, dtype=np.int32)

    supported, why = _reduction_is_supported(embeddings, streaming, offline, _transcript(8))

    assert not supported
    assert "voices_too_far" in why


def test_an_ambiguous_pair_of_established_speakers_is_left_alone():
    """
    In the middle band the voices do not decide. Two clusters that each hold a
    real share of the meeting are participants until something proves
    otherwise — merging them would delete one.
    """
    middling = _gap_for((WITHIN_MEETING_MERGE_SIMILARITY + INCONCLUSIVE_MERGE_SIMILARITY) / 2)
    embeddings = np.array(
        [_voice(0.0)] * 6 + [_voice(middling)] * 6,
        dtype=np.float32,
    )
    streaming = np.array([0] * 6 + [1] * 6)
    offline = np.zeros(12, dtype=np.int32)

    supported, why = _reduction_is_supported(embeddings, streaming, offline, _transcript(12))

    assert not supported
    assert "both_established" in why


def test_an_ambiguous_pair_is_merged_when_one_side_is_a_fragment():
    """
    Same uncertain similarity, different shape. A single short reply is not a
    participant — that is what an over-split looks like — so size breaks the
    tie the voices could not.
    """
    middling = _gap_for((WITHIN_MEETING_MERGE_SIMILARITY + INCONCLUSIVE_MERGE_SIMILARITY) / 2)
    embeddings = np.array([_voice(0.0)] * 11 + [_voice(middling)], dtype=np.float32)
    streaming = np.array([0] * 11 + [1])
    offline = np.zeros(12, dtype=np.int32)

    supported, why = _reduction_is_supported(embeddings, streaming, offline, _transcript(12))

    assert supported, why


def test_a_rejection_anywhere_rejects_the_whole_reduction():
    """
    One bad merge is enough. The assignment is taken or left as a unit, so a
    reduction that would fold a stranger in cannot be half-applied.
    """
    far = _gap_for(0.15)
    embeddings = np.array(
        [_voice(0.0), _voice(0.01)] + [_voice(far)] * 3,
        dtype=np.float32,
    )
    streaming = np.array([0, 0, 1, 1, 1])
    offline = np.zeros(5, dtype=np.int32)

    supported, _why = _reduction_is_supported(embeddings, streaming, offline, _transcript(5))

    assert not supported


def test_keeping_speakers_apart_is_always_supported():
    """A reduction that merges nothing has nothing to justify."""
    embeddings = np.array([_voice(0.0)] * 3 + [_voice(_gap_for(0.1))] * 3, dtype=np.float32)
    streaming = np.array([0, 0, 0, 1, 1, 1])

    supported, _why = _reduction_is_supported(embeddings, streaming, streaming.copy(), _transcript(6))

    assert supported


def test_a_single_streaming_cluster_has_nothing_to_merge():
    embeddings = np.array([_voice(0.0)] * 3, dtype=np.float32)
    streaming = np.zeros(3, dtype=np.int32)

    supported, why = _reduction_is_supported(embeddings, streaming, streaming.copy(), _transcript(3))

    assert supported
    assert why == "nothing_to_merge"
