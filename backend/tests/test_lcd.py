"""
Language change detection — cutting a segment where the language changes.

The model itself is not exercised here. split_on_language_change takes an
injectable posterior_fn, so these tests drive the windowing, smoothing,
boundary estimation and merging logic with a scripted "language ID" and no
weights on disk. What the real model does to real audio is a measurement, not
a unit test — see the tuning notes in config.py.
"""

from __future__ import annotations

import numpy as np

from config import SAMPLE_RATE
from models.lcd import split_on_language_change


def _speech(seconds: float, amplitude: float = 0.1) -> np.ndarray:
    """Non-silent audio. Content is irrelevant — the posterior is scripted."""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _switch_at(seconds: float, first: str = "en", second: str = "mr", confidence=0.9):
    """
    A posterior_fn that reports `first` before `seconds` and `second` after.

    lcd.py passes it one window at a time and never tells it where that window
    sits, so the boundary is inferred from window length: the fake keeps a
    running position, exactly as the real sliding loop advances.
    """
    state = {"pos": 0}

    def posterior(window: np.ndarray) -> dict[str, float]:
        centre = state["pos"] + len(window) / 2
        state["pos"] += int(0.5 * SAMPLE_RATE)  # LCD_HOP_SECONDS
        code = first if centre < seconds * SAMPLE_RATE else second
        other = second if code == first else first
        return {code: confidence, other: 1.0 - confidence}

    return posterior


def _constant(code: str, confidence: float = 0.95):
    def posterior(_window):
        other = "mr" if code != "mr" else "en"
        return {code: confidence, other: 1.0 - confidence}

    return posterior


# ------------------------------------------------------------- no-split cases


def test_single_language_segment_is_not_split():
    audio = _speech(8.0)
    pieces = split_on_language_change(audio, posterior_fn=_constant("en"))
    assert len(pieces) == 1
    assert pieces[0][0] == 0
    assert pieces[0][1] == len(audio)


def test_short_segment_is_never_split():
    """Below LCD_MIN_SEGMENT_SECONDS there is no room for two usable pieces."""
    audio = _speech(2.0)
    pieces = split_on_language_change(audio, posterior_fn=_switch_at(1.0))
    assert len(pieces) == 1


def test_low_confidence_windows_do_not_manufacture_a_boundary():
    """
    A window that does not know must not vote. Uninformative audio — a pause,
    a number, a cough — would otherwise open a language run of its own.
    """

    def unsure(_window):
        return {"en": 0.34, "hi": 0.33, "mr": 0.33}

    pieces = split_on_language_change(_speech(10.0), posterior_fn=unsure)
    assert len(pieces) == 1


def test_single_stray_window_is_smoothed_away():
    """One window on a loanword should not split the segment."""
    calls = {"n": 0}

    def one_blip(_window):
        calls["n"] += 1
        # Window 4 of many disagrees; everything else is English.
        code = "mr" if calls["n"] == 4 else "en"
        return {code: 0.9, "hi": 0.1}

    pieces = split_on_language_change(_speech(12.0), posterior_fn=one_blip)
    assert len(pieces) == 1


# ---------------------------------------------------------------- split cases


def test_code_switch_is_split_into_two_pieces():
    audio = _speech(12.0)
    pieces = split_on_language_change(audio, posterior_fn=_switch_at(6.0))

    assert len(pieces) == 2
    # Contiguous and complete — no audio invented, none dropped.
    assert pieces[0][0] == 0
    assert pieces[-1][1] == len(audio)
    assert pieces[0][1] == pieces[1][0]


def test_split_lands_near_the_real_switch():
    audio = _speech(12.0)
    pieces = split_on_language_change(audio, posterior_fn=_switch_at(6.0))

    boundary_sec = pieces[0][1] / SAMPLE_RATE
    # Window 2.0s / hop 0.5s cannot do better than about a window either way.
    assert 4.5 < boundary_sec < 7.5


def test_pieces_carry_their_advisory_language():
    audio = _speech(12.0)
    pieces = split_on_language_change(audio, posterior_fn=_switch_at(6.0, "en", "mr"))

    assert pieces[0][2] == "en"
    assert pieces[1][2] == "mr"


def test_boundary_snaps_to_a_quiet_gap():
    """
    People pause briefly when switching language. Given a real silence near
    the estimated boundary, the cut should land in it rather than mid-word.
    """
    audio = _speech(12.0)
    gap_start = int(6.2 * SAMPLE_RATE)
    gap_end = int(6.4 * SAMPLE_RATE)
    audio[gap_start:gap_end] = 0.0

    pieces = split_on_language_change(audio, posterior_fn=_switch_at(6.3))
    assert len(pieces) == 2
    boundary = pieces[0][1]
    assert gap_start <= boundary <= gap_end


# ------------------------------------------------------------- short pieces


def test_short_trailing_piece_is_merged_not_emitted():
    """
    A sub-second piece handed to Whisper is mostly zero-padded silence, which
    is where hallucinated outro captions come from. Merging keeps the text
    imperfect but real; emitting it would risk replacing it with fiction.
    """
    audio = _speech(10.0)
    # Switch 0.4s before the end — far too short to stand alone.
    pieces = split_on_language_change(audio, posterior_fn=_switch_at(9.6))

    assert len(pieces) == 1
    assert pieces[0][0] == 0
    assert pieces[0][1] == len(audio)


def test_every_emitted_piece_clears_the_minimum_length():
    from config import LCD_MIN_PIECE_SECONDS

    audio = _speech(14.0)
    pieces = split_on_language_change(audio, posterior_fn=_switch_at(7.0))

    for start, end, _lang in pieces:
        assert (end - start) / SAMPLE_RATE >= LCD_MIN_PIECE_SECONDS


def test_pieces_always_tile_the_segment_exactly():
    """No gaps, no overlaps, nothing lost — whatever the split decides."""
    audio = _speech(14.0)
    for switch in (4.0, 7.0, 10.0):
        pieces = split_on_language_change(audio, posterior_fn=_switch_at(switch))
        assert pieces[0][0] == 0
        assert pieces[-1][1] == len(audio)
        for i in range(len(pieces) - 1):
            assert pieces[i][1] == pieces[i + 1][0]


def test_disabled_flag_returns_the_whole_segment():
    import models.lcd as lcd

    original = lcd.LCD_ENABLED
    try:
        lcd.LCD_ENABLED = False
        pieces = split_on_language_change(_speech(12.0), posterior_fn=_switch_at(6.0))
        assert len(pieces) == 1
    finally:
        lcd.LCD_ENABLED = original
