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


def test_turn_language_is_the_majority_not_the_last_segment():
    """
    Speaker starts in English and finishes in Marathi. The turn is English —
    that is where 20 of its 25 seconds went. Labelling it by whichever segment
    merged last called this Marathi and then charged all 25s to Marathi in the
    meeting's language breakdown.
    """
    segments = [
        _seg(0, 20, "Speaker_00", "We need to ship the release this week.", lang="en"),
        _seg(20, 25, "Speaker_00", "म्हणजे उद्या सुरू करूया.", lang="mr"),
    ]
    turns = build_speaker_turns(segments)
    assert len(turns) == 1
    assert turns[0]["language"] == "en"
    assert turns[0]["language_name"] == "English"
    # The minority language is reported, not discarded.
    assert turns[0]["language_mix"] == ["en", "mr"]
    assert "म्हणजे" in turns[0]["raw_text"]


def test_turn_language_majority_when_minority_spoke_first():
    """Same rule in the other direction — first-spoken must not win either."""
    segments = [
        _seg(0, 4, "Speaker_01", "Quick update.", lang="en"),
        _seg(4, 30, "Speaker_01", "बाकी काम आज पूर्ण होईल.", lang="hi"),
    ]
    turns = build_speaker_turns(segments)
    assert turns[0]["language"] == "hi"
    assert turns[0]["language_mix"] == ["en", "hi"]


def test_single_language_turn_has_no_language_mix():
    segments = [
        _seg(0, 3, "Speaker_00", "One", lang="en"),
        _seg(3.2, 6, "Speaker_00", "Two", lang="en"),
    ]
    turns = build_speaker_turns(segments)
    assert turns[0]["language"] == "en"
    assert "language_mix" not in turns[0]


def test_turn_carries_the_winning_languages_probability():
    """language_prob must describe the language actually shown, not another."""
    segments = [
        {**_seg(0, 20, "Speaker_00", "Long English part.", lang="en"), "language_prob": 0.91},
        {**_seg(20, 24, "Speaker_00", "थोडं मराठी.", lang="mr"), "language_prob": 0.44},
    ]
    turns = build_speaker_turns(segments)
    assert turns[0]["language"] == "en"
    assert turns[0]["language_prob"] == 0.91


def test_turn_is_flagged_mixed_if_any_segment_was():
    """
    One suspect segment makes the whole turn suspect. A reader deciding whether
    to trust a turn needs to know some of it may be mistranscribed, and the
    turn is the unit they actually read.
    """
    segments = [
        {**_seg(0, 5, "Speaker_00", "Clean English.", lang="en"), "language_margin": 0.95},
        {
            **_seg(5, 9, "Speaker_00", "Friday manje udya", lang="en"),
            "language_margin": 0.03,
            "language_mixed_suspected": True,
        },
    ]
    turns = build_speaker_turns(segments)
    assert len(turns) == 1
    assert turns[0]["language_mixed_suspected"] is True
    # Narrowest margin seen — the most ambiguous moment in the turn.
    assert turns[0]["language_margin"] == 0.03


def test_turn_of_clean_segments_is_not_flagged():
    segments = [
        {**_seg(0, 5, "Speaker_00", "One.", lang="en"), "language_margin": 0.95},
        {**_seg(5, 9, "Speaker_00", "Two.", lang="en"), "language_margin": 0.91},
    ]
    turns = build_speaker_turns(segments)
    assert turns[0]["language_mixed_suspected"] is False
    assert turns[0]["language_margin"] == 0.91


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
