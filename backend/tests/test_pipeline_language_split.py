"""
Pipeline wiring for the mid-segment language split.

lcd.py finding a boundary is only half the job — the pieces then have to reach
the right ASR engine. These tests cover the handoff, which is the part that can
break silently: the split still happens, the transcript still looks plausible,
and the minority-language piece is quietly mangled exactly as it was before.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from config import SAMPLE_RATE
from pipeline import MeetingSession


class _NullIdentifier:
    def identify(self, _embedding):
        return "Unknown", 0.0

    def enroll(self, *_args, **_kwargs):
        return

    def refresh(self, force=True):
        return


def _speech(seconds: float) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def _asr_result(language: str, text: str) -> dict:
    return {
        "text": text,
        "language": language,
        "language_name": {"en": "English", "hi": "Hindi", "mr": "Marathi"}[language],
        "language_prob": 0.9,
        "language_detected": language,
        "language_fallback": False,
        "language_margin": 0.5,
        "language_mixed_suspected": False,
    }


def _session() -> MeetingSession:
    session = MeetingSession("MTG-split", _NullIdentifier())
    session._full_audio = _speech(30.0)
    return session


def _english_then_marathi(audio, start_sample_of_switch):
    """Two pieces: English up to the switch, Marathi after."""
    return [
        (0, start_sample_of_switch, "en"),
        (start_sample_of_switch, len(audio), "mr"),
    ]


def test_a_split_produces_one_transcript_entry_per_piece():
    session = _session()
    audio = _speech(10.0)

    with (
        patch(
            "pipeline.split_on_language_change",
            side_effect=lambda a, **k: _english_then_marathi(a, len(a) // 2),
        ),
        patch("pipeline.get_embedding", return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        patch("pipeline.transcribe", side_effect=[_asr_result("en", "the deadline is Friday"),
                                                  _asr_result("mr", "म्हणजे उद्या")]),
    ):
        entries = session._process_subsegment(0.0, 10.0, audio)

    assert len(entries) == 2
    assert [e["language"] for e in entries] == ["en", "mr"]
    assert entries[0]["text"] == "the deadline is Friday"
    assert entries[1]["text"] == "म्हणजे उद्या"


def test_split_pieces_are_transcribed_without_surrounding_context():
    """
    The regression this guards: transcribe_with_context detects the language
    over a window padded by seconds on each side. Either side of a language
    boundary IS the other language, and usually the longer side — so a padded
    detection hands the minority piece back to the majority engine and the
    split achieves nothing.
    """
    session = _session()
    audio = _speech(10.0)

    with (
        patch(
            "pipeline.split_on_language_change",
            side_effect=lambda a, **k: _english_then_marathi(a, len(a) // 2),
        ),
        patch("pipeline.get_embedding", return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        patch("pipeline.transcribe_with_context") as with_context,
        patch("pipeline.transcribe", side_effect=[_asr_result("en", "a"), _asr_result("mr", "ब")]),
    ):
        session._process_subsegment(0.0, 10.0, audio)

    with_context.assert_not_called()


def test_an_unsplit_segment_still_uses_context():
    """Context is a real accuracy win everywhere else — only splits opt out."""
    session = _session()
    audio = _speech(6.0)

    with (
        patch("pipeline.split_on_language_change", side_effect=lambda a, **k: [(0, len(a), None)]),
        patch("pipeline.get_embedding", return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        patch("pipeline.transcribe_with_context", return_value=_asr_result("en", "hello")) as ctx,
    ):
        entries = session._process_subsegment(2.0, 8.0, audio)

    assert len(entries) == 1
    ctx.assert_called_once()


def test_each_piece_is_hinted_with_its_own_language_not_the_meetings():
    """
    The meeting-dominant hint rescues weak detection normally; on a split piece
    it pulls toward the language the split just proved this piece is not.
    """
    session = _session()
    session._language_counts = {"en": 20}  # dominant_language -> 'en'
    audio = _speech(10.0)
    hints = []

    def record(_audio, hint_language=None):
        hints.append(hint_language)
        return _asr_result("mr" if len(hints) > 1 else "en", "x")

    with (
        patch(
            "pipeline.split_on_language_change",
            side_effect=lambda a, **k: _english_then_marathi(a, len(a) // 2),
        ),
        patch("pipeline.get_embedding", return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        patch("pipeline.transcribe", side_effect=record),
    ):
        session._process_subsegment(0.0, 10.0, audio)

    assert hints == ["en", "mr"]


def test_piece_timestamps_are_absolute_and_contiguous():
    session = _session()
    audio = _speech(10.0)

    with (
        patch(
            "pipeline.split_on_language_change",
            side_effect=lambda a, **k: _english_then_marathi(a, len(a) // 2),
        ),
        patch("pipeline.get_embedding", return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        patch("pipeline.transcribe", side_effect=[_asr_result("en", "a"), _asr_result("mr", "ब")]),
    ):
        entries = session._process_subsegment(12.0, 22.0, audio)

    assert entries[0]["start_sec"] == 12.0
    assert entries[-1]["end_sec"] == 22.0
    assert entries[0]["end_sec"] == entries[1]["start_sec"]


def test_every_piece_gets_an_embedding_so_the_recluster_pass_stays_aligned():
    """
    offline_diarizer.recluster_from_embeddings indexes embeddings by
    transcript line. Two lines from one sub-segment need two embeddings — the
    same voice, recorded twice — or every later line is attributed to the
    wrong speaker.
    """
    session = _session()
    audio = _speech(10.0)

    with (
        patch(
            "pipeline.split_on_language_change",
            side_effect=lambda a, **k: _english_then_marathi(a, len(a) // 2),
        ),
        patch("pipeline.get_embedding", return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        patch("pipeline.transcribe", side_effect=[_asr_result("en", "a"), _asr_result("mr", "ब")]),
    ):
        session._process_subsegment(0.0, 10.0, audio)

    assert len(session._embeddings) == len(session.transcript) == 2
    assert np.allclose(session._embeddings[0], session._embeddings[1])


def test_both_pieces_keep_the_same_speaker():
    """The speaker did not change — only the language did."""
    session = _session()
    audio = _speech(10.0)

    with (
        patch(
            "pipeline.split_on_language_change",
            side_effect=lambda a, **k: _english_then_marathi(a, len(a) // 2),
        ),
        patch("pipeline.get_embedding", return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        patch("pipeline.transcribe", side_effect=[_asr_result("en", "a"), _asr_result("mr", "ब")]),
    ):
        entries = session._process_subsegment(0.0, 10.0, audio)

    assert entries[0]["speaker_label"] == entries[1]["speaker_label"]


def test_a_failing_boundary_search_costs_only_the_split():
    """LCD is an enhancement. If it throws, the segment still transcribes."""
    session = _session()
    audio = _speech(6.0)

    with (
        patch("pipeline.split_on_language_change", side_effect=RuntimeError("model exploded")),
        patch("pipeline.get_embedding", return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        patch("pipeline.transcribe_with_context", return_value=_asr_result("en", "hello")),
    ):
        entries = session._process_subsegment(0.0, 6.0, audio)

    assert len(entries) == 1
    assert entries[0]["text"] == "hello"
