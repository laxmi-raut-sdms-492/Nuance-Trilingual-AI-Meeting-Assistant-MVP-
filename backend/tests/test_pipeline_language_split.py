"""
Pipeline wiring for the mid-segment language split.

lcd.py finding a boundary is only half the job — the pieces then have to reach
the right STT engine. These tests cover the handoff, which is the part that can
break silently: the split still happens, the transcript still looks plausible,
and the minority-language piece is quietly mangled exactly as it was before.

Transcription goes through the STT adapter (stt/base.py), so these drive a fake
adapter rather than patching models.asr — the split has to survive whichever
provider is configured, local or cloud.
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


class _FakeAdapter:
    """Records how it was called; returns scripted results in order."""

    adapter_type = "local"
    provider_name = "fake"
    model_name = "fake-1"

    def __init__(self, results: list[dict] | None = None):
        self._results = list(results or [])
        self.hints: list[str | None] = []
        self.context_calls = 0
        self.plain_calls = 0

    def _next(self) -> dict:
        return self._results.pop(0) if self._results else _asr_result("en", "x")

    def transcribe(self, audio, language=None, hint_language=None):
        self.plain_calls += 1
        self.hints.append(hint_language)
        return self._next()

    def transcribe_with_context(self, full_audio, start, end, hint_language=None):
        self.context_calls += 1
        self.hints.append(hint_language)
        return self._next()


def _speech(seconds: float) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def _session(adapter: _FakeAdapter) -> MeetingSession:
    session = MeetingSession("MTG-split", _NullIdentifier(), stt_adapter=adapter)
    session._full_audio = _speech(30.0)
    return session


def _english_then_marathi(audio, switch_sample):
    """Two pieces: English up to the switch, Marathi after."""
    return [(0, switch_sample, "en"), (switch_sample, len(audio), "mr")]


def _split_in_half(audio, **_kwargs):
    return _english_then_marathi(audio, len(audio) // 2)


def _no_split(audio, **_kwargs):
    return [(0, len(audio), None)]


_EMBEDDING = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_a_split_produces_one_transcript_entry_per_piece():
    adapter = _FakeAdapter(
        [_asr_result("en", "the deadline is Friday"), _asr_result("mr", "म्हणजे उद्या")]
    )
    session = _session(adapter)
    audio = _speech(10.0)

    with (
        patch("pipeline.split_on_language_change", side_effect=_split_in_half),
        patch("pipeline.get_embedding", return_value=_EMBEDDING),
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
    adapter = _FakeAdapter([_asr_result("en", "a"), _asr_result("mr", "ब")])
    session = _session(adapter)

    with (
        patch("pipeline.split_on_language_change", side_effect=_split_in_half),
        patch("pipeline.get_embedding", return_value=_EMBEDDING),
    ):
        session._process_subsegment(0.0, 10.0, _speech(10.0))

    assert adapter.context_calls == 0
    assert adapter.plain_calls == 2


def test_an_unsplit_segment_still_uses_context():
    """Context is a real accuracy win everywhere else — only splits opt out."""
    adapter = _FakeAdapter([_asr_result("en", "hello")])
    session = _session(adapter)

    with (
        patch("pipeline.split_on_language_change", side_effect=_no_split),
        patch("pipeline.get_embedding", return_value=_EMBEDDING),
    ):
        entries = session._process_subsegment(2.0, 8.0, _speech(6.0))

    assert len(entries) == 1
    assert adapter.context_calls == 1
    assert adapter.plain_calls == 0


def test_each_piece_is_hinted_with_its_own_language_not_the_meetings():
    """
    The meeting-dominant hint rescues weak detection normally; on a split piece
    it pulls toward the language the split just proved this piece is not.
    """
    adapter = _FakeAdapter([_asr_result("en", "a"), _asr_result("mr", "ब")])
    session = _session(adapter)
    session._language_counts = {"en": 20}  # dominant_language -> 'en'

    with (
        patch("pipeline.split_on_language_change", side_effect=_split_in_half),
        patch("pipeline.get_embedding", return_value=_EMBEDDING),
    ):
        session._process_subsegment(0.0, 10.0, _speech(10.0))

    assert adapter.hints == ["en", "mr"]


def test_piece_timestamps_are_absolute_and_contiguous():
    adapter = _FakeAdapter([_asr_result("en", "a"), _asr_result("mr", "ब")])
    session = _session(adapter)

    with (
        patch("pipeline.split_on_language_change", side_effect=_split_in_half),
        patch("pipeline.get_embedding", return_value=_EMBEDDING),
    ):
        entries = session._process_subsegment(12.0, 22.0, _speech(10.0))

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
    adapter = _FakeAdapter([_asr_result("en", "a"), _asr_result("mr", "ब")])
    session = _session(adapter)

    with (
        patch("pipeline.split_on_language_change", side_effect=_split_in_half),
        patch("pipeline.get_embedding", return_value=_EMBEDDING),
    ):
        session._process_subsegment(0.0, 10.0, _speech(10.0))

    assert len(session._embeddings) == len(session.transcript) == 2
    assert np.allclose(session._embeddings[0], session._embeddings[1])


def test_both_pieces_keep_the_same_speaker():
    """The speaker did not change — only the language did."""
    adapter = _FakeAdapter([_asr_result("en", "a"), _asr_result("mr", "ब")])
    session = _session(adapter)

    with (
        patch("pipeline.split_on_language_change", side_effect=_split_in_half),
        patch("pipeline.get_embedding", return_value=_EMBEDDING),
    ):
        entries = session._process_subsegment(0.0, 10.0, _speech(10.0))

    assert entries[0]["speaker_label"] == entries[1]["speaker_label"]


def test_pieces_are_attributed_to_the_configured_provider():
    """Provenance is per line, and a split must not lose it on either piece."""
    adapter = _FakeAdapter([_asr_result("en", "a"), _asr_result("mr", "ब")])
    session = _session(adapter)

    with (
        patch("pipeline.split_on_language_change", side_effect=_split_in_half),
        patch("pipeline.get_embedding", return_value=_EMBEDDING),
    ):
        entries = session._process_subsegment(0.0, 10.0, _speech(10.0))

    for entry in entries:
        assert entry["adapter"] == "local"
        assert entry["provider"] == "fake"
        assert entry["model"] == "fake-1"


def test_a_failing_boundary_search_costs_only_the_split():
    """LCD is an enhancement. If it throws, the segment still transcribes."""
    adapter = _FakeAdapter([_asr_result("en", "hello")])
    session = _session(adapter)

    with (
        patch("pipeline.split_on_language_change", side_effect=RuntimeError("model exploded")),
        patch("pipeline.get_embedding", return_value=_EMBEDDING),
    ):
        entries = session._process_subsegment(0.0, 6.0, _speech(6.0))

    assert len(entries) == 1
    assert entries[0]["text"] == "hello"
