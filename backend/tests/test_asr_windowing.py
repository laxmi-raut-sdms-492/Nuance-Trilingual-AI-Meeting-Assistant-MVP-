"""
Attributing a padded decode back to the segment that was actually spoken.

English segments are decoded together with several seconds of surrounding
audio, because context makes Whisper better. The text then has to be split
back out. Getting that split wrong is not a subtle quality loss — it tears
sentences in half across two speakers and deletes clauses outright.

Both failures below were observed in a real meeting transcript:

    01:11  SPEAKER_03  ... We haven't even released an album.
    01:19  SPEAKER_02  software version yet As I said earlier, ...

one sentence — "we haven't even released an alpha software version yet" —
split across two speakers, with the words at the join lost.
"""

from __future__ import annotations

from models.asr import _hallucination_reason, _result_within_window, _text_for_time_range


def _word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


def _segment(text: str, start: float, end: float, words=None, **stats) -> dict:
    seg = {"text": text, "start": start, "end": end, **stats}
    if words is not None:
        seg["words"] = words
    return seg


# ------------------------------------------------------- word attribution


def test_a_sentence_spanning_two_windows_is_split_at_the_right_word():
    """
    Whisper emitted one sentence; the pipeline cut a segment boundary through
    the middle of it. Each half must land in its own window, whole.
    """
    words = [
        _word("we", 0.0, 0.2), _word("haven't", 0.2, 0.6),
        _word("even", 0.6, 0.9), _word("released", 0.9, 1.4),
        _word("an", 1.4, 1.5), _word("alpha", 1.5, 2.0),
        _word("software", 2.2, 2.7), _word("version", 2.7, 3.1),
        _word("yet", 3.1, 3.3),
    ]
    result = {"segments": [_segment("we haven't even released an alpha software version yet", 0.0, 3.3, words)]}

    first = _text_for_time_range(result, 0.0, 2.1, clip_start=0.0)
    second = _text_for_time_range(result, 2.1, 3.4, clip_start=0.0)

    assert first == "we haven't even released an alpha"
    assert second == "software version yet"


def test_no_word_is_lost_across_touching_windows():
    """
    The deletion bug, in the geometry it actually happened in.

    SCD and LCD cut a VAD segment into sub-segments that touch exactly — one
    ends where the next begins. Under the old rule a Whisper segment spanning
    that cut went whole to one side, so the other side lost its words even
    though no audio was missing. Every word must come back, once.
    """
    words = [_word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate("the first delivery is tomorrow".split())]
    result = {"segments": [_segment("the first delivery is tomorrow", 0.0, 2.5, words)]}

    first = _text_for_time_range(result, 0.0, 1.2, clip_start=0.0)
    second = _text_for_time_range(result, 1.2, 2.6, clip_start=0.0)

    assert f"{first} {second}".split() == ["the", "first", "delivery", "is", "tomorrow"]


def test_words_inside_a_vad_silence_gap_are_deliberately_not_imported():
    """
    Not every word belongs to somebody. Between two VAD segments is audio the
    detector judged to be silence; the padded decode still runs over it, and
    what Whisper "hears" there is the invented filler the guards exist to
    reject. Attributing it to whichever segment is nearest would quietly
    reintroduce exactly that.
    """
    words = [_word("phantom", 1.0, 1.4)]
    result = {"segments": [_segment("phantom", 1.0, 1.4, words)]}

    before = _text_for_time_range(result, 0.0, 0.9, clip_start=0.0)
    after = _text_for_time_range(result, 1.6, 2.5, clip_start=0.0)

    assert before == ""
    assert after == ""


def test_no_word_is_claimed_by_both_windows():
    words = [_word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate("one two three four".split())]
    result = {"segments": [_segment("one two three four", 0.0, 2.0, words)]}

    first = _text_for_time_range(result, 0.0, 1.0, clip_start=0.0).split()
    second = _text_for_time_range(result, 1.0, 2.0, clip_start=0.0).split()

    assert not (set(first) & set(second))


def test_clip_start_offsets_word_times_into_absolute_time():
    """The clip begins partway through the meeting; word times are relative."""
    words = [_word("hello", 0.0, 0.4), _word("there", 0.4, 0.8)]
    result = {"segments": [_segment("hello there", 0.0, 0.8, words)]}

    assert _text_for_time_range(result, 10.0, 10.9, clip_start=10.0) == "hello there"
    assert _text_for_time_range(result, 0.0, 0.9, clip_start=10.0) == ""


def test_a_neighbours_speech_is_never_returned_as_this_segments_text():
    """
    The old fallback returned the ENTIRE padded clip when nothing matched —
    up to eight seconds of other people talking, attributed to this segment.
    Empty is the honest answer.
    """
    result = {
        "text": "everything anyone said anywhere near here",
        "segments": [_segment("someone else entirely", 20.0, 24.0, [_word("elsewhere", 20.0, 24.0)])],
    }

    assert _text_for_time_range(result, 0.0, 2.0, clip_start=0.0) == ""


def test_segment_midpoints_are_still_used_when_there_are_no_word_timings():
    """Older Whisper output, or a decode that produced no word timings."""
    result = {"segments": [_segment("kept", 0.0, 2.0), _segment("dropped", 30.0, 32.0)]}

    assert _text_for_time_range(result, 0.0, 2.0, clip_start=0.0) == "kept"


# --------------------------------------------------- guards see the window


def test_guards_are_computed_from_the_windows_own_segments():
    """
    The "Hello." bug. A fabricated line on trailing silence survived because
    the guards were fed the whole padded decode: the confident, silent segment
    that should have tripped them was averaged against unrelated audio
    seconds away.
    """
    result = {
        "segments": [
            # Real speech earlier in the padded clip — struggling decode.
            _segment("actual discussion", 0.0, 8.0, no_speech_prob=0.10, avg_logprob=-1.80),
            # The window itself: silence, decoded with high confidence.
            _segment("Hello.", 9.0, 11.0, no_speech_prob=0.97, avg_logprob=-0.20),
        ]
    }

    # Whole clip: min(avg_logprob) = -1.80 pulls it under the threshold, so
    # the "confident despite silence" condition cannot hold. Guard misses.
    assert _hallucination_reason(result, "Hello.", 2.0) is None

    # This segment's own evidence: silent and confident. Guard fires.
    narrowed = _result_within_window(result, 9.0, 11.0, clip_start=0.0)
    assert _hallucination_reason(narrowed, "Hello.", 2.0) is not None


def test_window_narrowing_keeps_partially_overlapping_segments():
    result = {
        "segments": [
            _segment("straddles the start", 1.0, 3.0, no_speech_prob=0.2, avg_logprob=-0.5),
            _segment("well outside", 40.0, 42.0, no_speech_prob=0.9, avg_logprob=-0.1),
        ]
    }

    narrowed = _result_within_window(result, 2.0, 5.0, clip_start=0.0)
    assert len(narrowed["segments"]) == 1
    assert narrowed["segments"][0]["text"] == "straddles the start"


def test_window_narrowing_falls_back_rather_than_removing_all_evidence():
    """
    An empty segment list reads to the guards as "nothing suspicious", which
    would pass every hallucination through. Keep the original instead.
    """
    result = {"segments": [_segment("nowhere near", 100.0, 102.0, no_speech_prob=0.9, avg_logprob=-0.1)]}

    narrowed = _result_within_window(result, 0.0, 2.0, clip_start=0.0)
    assert narrowed["segments"] == result["segments"]


def test_real_quiet_speech_still_survives_the_narrowed_guards():
    """
    The guards exist alongside a measured counter-example: genuinely quiet
    speech at no_speech_prob=0.94 with a struggling avg_logprob=-1.66 must be
    kept. Narrowing must not turn into over-dropping.
    """
    result = {"segments": [_segment("I'm not sure if you can hear me.", 0.0, 30.0, no_speech_prob=0.94, avg_logprob=-1.66)]}

    narrowed = _result_within_window(result, 0.0, 30.0, clip_start=0.0)
    assert _hallucination_reason(narrowed, "I'm not sure if you can hear me.", 30.0) is None
