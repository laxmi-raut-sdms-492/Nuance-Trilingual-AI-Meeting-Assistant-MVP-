"""
The speaker count of an uploaded meeting comes from decide_speakers.

The failure these tests pin: one person's recording shown as two or three
speakers, usually because their short lines ("hi", "okay") carry scattered
voice prints. A single voice must come out as one speaker, and short lines
must never found a speaker of their own.
"""

from __future__ import annotations

import numpy as np

from models.offline_diarizer import decide_speakers, recluster_from_embeddings


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _voice(seed: int, dim: int = 32):
    return _unit(np.random.default_rng(seed).normal(size=dim))


def _lines(voice, n, noise, seed, dim=32):
    rng = np.random.default_rng(seed)
    return np.stack([_unit(voice + rng.normal(0, noise, dim)) for _ in range(n)])


def test_one_voice_with_short_scattered_lines_is_one_speaker():
    voice = _voice(1)
    long_lines = _lines(voice, 6, noise=0.12, seed=2)   # proper sentences
    short_lines = _lines(voice, 6, noise=0.3, seed=3)   # "hi", "okay": prints scatter
    emb = np.vstack([long_lines, short_lines])
    dur = np.array([4.0] * 6 + [0.5] * 6)

    assignment = decide_speakers(emb, dur)

    assert len(set(assignment.tolist())) == 1


def test_two_voices_stay_two_speakers():
    a, b = _voice(10), _voice(11)
    emb = np.vstack([_lines(a, 6, 0.12, 1), _lines(b, 6, 0.12, 2)])
    dur = np.full(12, 3.0)

    assignment = decide_speakers(emb, dur)

    assert len(set(assignment.tolist())) == 2
    assert len(set(assignment[:6].tolist())) == 1
    assert len(set(assignment[6:].tolist())) == 1


def test_two_short_greetings_are_one_speaker_not_two():
    """'hi' then 'hello' from one person, prints far apart: still one speaker."""
    voice = _voice(5)
    emb = _lines(voice, 2, noise=0.3, seed=7)
    dur = np.array([0.6, 0.7])

    assignment = decide_speakers(emb, dur)

    assert len(set(assignment.tolist())) == 1


def test_brief_interjection_joins_the_nearest_voice():
    a, b = _voice(20), _voice(21)
    emb = np.vstack([_lines(a, 5, 0.15, 1), _lines(b, 5, 0.15, 2), _lines(b, 1, 0.5, 3)])
    dur = np.array([5.0] * 10 + [0.8])

    assignment = decide_speakers(emb, dur)

    assert len(set(assignment.tolist())) == 2
    assert assignment[-1] == assignment[6]


def test_short_line_follows_the_turn_it_interrupts_when_close():
    """A short 'hmm' equally like both voices belongs to whoever was talking."""
    a, b = _voice(30), _voice(31)
    mid = _unit(a + b)
    emb = np.vstack([_lines(a, 4, 0.1, 1), [mid], _lines(b, 4, 0.1, 2)])
    dur = np.array([4.0] * 4 + [0.4] + [4.0] * 4)

    assignment = decide_speakers(emb, dur)

    assert assignment[4] == assignment[3]


def test_k_max_caps_the_count():
    voices = [_voice(40 + i) for i in range(5)]
    emb = np.vstack([_lines(v, 4, 0.1, i) for i, v in enumerate(voices)])
    dur = np.full(20, 3.0)

    assignment = decide_speakers(emb, dur, k_max=3)

    assert len(set(assignment.tolist())) == 3


def test_recluster_reports_single_speaker():
    voice = _voice(50)
    emb = _lines(voice, 5, 0.12, 1)
    transcript = [
        {"start_sec": i * 3.0, "end_sec": i * 3.0 + 2.5, "speaker_label": f"Speaker_0{i % 2}", "speaker": f"Speaker_0{i % 2}", "text": "x"}
        for i in range(5)
    ]

    new_transcript, info = recluster_from_embeddings(transcript, emb)

    assert info["applied"] is True
    assert info["k"] == 1
    assert {t["speaker_label"] for t in new_transcript} == {"Speaker_00"}
