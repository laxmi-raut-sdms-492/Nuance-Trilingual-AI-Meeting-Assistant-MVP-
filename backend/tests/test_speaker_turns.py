"""Tests for speaker turn construction and transcript readability."""

from models.speaker_turns import (
    build_speaker_turns,
    rule_based_cleanup,
    should_merge_turn,
)


def _seg(start, end, speaker_label, text, lang="hi"):
    return {
        "start_sec": start,
        "end_sec": end,
        "speaker_label": speaker_label,
        "speaker": speaker_label,
        "text": text,
        "raw_text": text,
        "language": lang,
        "language_name": "Hindi" if lang == "hi" else "English",
        "time": f"0:{int(start):02d}",
    }


def test_one_speaker_continuous_turn_merged():
    segments = [
        _seg(0, 2, "Speaker_00", "मेरा साथी"),
        _seg(2.5, 4, "Speaker_00", "और हरिश्चंद्र खेले"),
        _seg(4.5, 7, "Speaker_00", "और मैं इस खेल का एक पक्का खिलाड़ी हूँ"),
    ]
    turns = build_speaker_turns(segments)
    assert len(turns) == 1
    assert turns[0]["speaker_label"] == "Speaker_00"
    assert "मेरा साथी" in turns[0]["raw_text"]
    assert "खिलाड़ी" in turns[0]["raw_text"]
    assert turns[0]["segment_count"] == 3


def test_language_change_does_not_split_speaker():
    segments = [
        _seg(0, 3, "Speaker_00", "We need to finish this.", lang="en"),
        _seg(3.5, 6, "Speaker_00", "आपण हे आज पूर्ण करूया.", lang="mr"),
    ]
    turns = build_speaker_turns(segments)
    assert len(turns) == 1
    assert "finish" in turns[0]["raw_text"]
    assert "पूर्ण" in turns[0]["raw_text"]
    assert turns[0].get("language_mix") == ["en", "mr"]


def test_two_speakers_produce_two_turns():
    segments = [
        _seg(0, 2, "Speaker_00", "Hello"),
        _seg(2.5, 4, "Speaker_01", "Hi there"),
    ]
    turns = build_speaker_turns(segments)
    assert len(turns) == 2
    labels = {t["speaker_label"] for t in turns}
    assert labels == {"Speaker_00", "Speaker_01"}


def test_pause_splits_turn_when_gap_large():
    segments = [
        _seg(0, 2, "Speaker_00", "First part"),
        _seg(10, 12, "Speaker_00", "After long pause"),
    ]
    turns = build_speaker_turns(segments, max_gap_sec=2.0)
    assert len(turns) == 2


def test_rule_based_cleanup_adds_danda_for_hindi():
    cleaned = rule_based_cleanup("मेरा साथी")
    assert cleaned.endswith("।")


def test_should_merge_same_speaker_small_gap():
    cur = _seg(0, 2, "Speaker_00", "a")
    nxt = _seg(2.2, 4, "Speaker_00", "b")
    assert should_merge_turn(cur, nxt, max_gap_sec=2.5) is True


def test_should_not_merge_different_speakers():
    cur = _seg(0, 2, "Speaker_00", "a")
    nxt = _seg(2.2, 4, "Speaker_01", "b")
    assert should_merge_turn(cur, nxt) is False
