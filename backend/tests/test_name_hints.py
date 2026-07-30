"""Tests for self-introduction name extraction."""

from models.name_hints import extract_self_introduction_name
from models.speaker_enrollment import introduction_labels_from_transcript


def test_extracts_my_name_is():
    assert extract_self_introduction_name("My name is Anushka.") == "Anushka"


def test_extracts_case_insensitive():
    assert extract_self_introduction_name("hi, my name is laxmi") == "Laxmi"


def test_ignores_unrelated_text():
    assert extract_self_introduction_name("Let us begin the meeting.") is None


def test_mutual_greetings_swap_names():
    from models.name_hints import resolve_names_from_greetings

    transcript = [
        {
            "speaker": "Speaker_00",
            "speaker_label": "Speaker_00",
            "text": "Good morning Lakshmi. Let's review the progress.",
        },
        {
            "speaker": "Speaker_01",
            "speaker_label": "Speaker_01",
            "text": "Good morning Anushka. I have completed the dashboard.",
        },
    ]
    assert resolve_names_from_greetings(transcript) == {
        "Speaker_00": "Anushka",
        "Speaker_01": "Lakshmi",
    }


def test_introduction_labels_from_generic_speaker():
    transcript = [
        {
            "speaker": "Speaker_00",
            "speaker_label": "Speaker_00",
            "text": "My name is Anushka.",
        }
    ]
    labels = introduction_labels_from_transcript(transcript)
    assert len(labels) == 1
    assert labels[0]["identified_as"] == "Anushka"
    assert labels[0]["matched"] is True


def test_greeting_labels_from_transcript():
    transcript = [
        {
            "speaker": "Speaker_00",
            "speaker_label": "Speaker_00",
            "text": "Good morning Lakshmi. Let's review.",
        },
        {
            "speaker": "Speaker_01",
            "speaker_label": "Speaker_01",
            "text": "Good morning Anushka. I finished the UI.",
        },
        {
            "speaker": "Speaker_02",
            "speaker_label": "Speaker_02",
            "text": "So please, I am okay.",
        },
    ]
    labels = {row["speaker_label"]: row["identified_as"] for row in introduction_labels_from_transcript(transcript)}
    assert labels["Speaker_00"] == "Anushka"
    assert labels["Speaker_01"] == "Lakshmi"
    assert "Speaker_02" not in labels


def test_turn_taking_merges_short_leftover_after_other_speaker():
    """Speaker_02 after Lakshmi in a 2-person meeting → Anushka."""
    from models.speaker_enrollment import same_meeting_fragment_merges

    transcript = [
        {
            "start_sec": 0.0,
            "end_sec": 8.0,
            "speaker": "Anushka",
            "speaker_label": "Speaker_00",
            "text": "Good morning Lakshmi.",
        },
        {
            "start_sec": 14.0,
            "end_sec": 25.0,
            "speaker": "Lakshmi",
            "speaker_label": "Speaker_01",
            "text": "Good morning Anushka.",
        },
        {
            "start_sec": 25.9,
            "end_sec": 28.6,
            "speaker": "Speaker_02",
            "speaker_label": "Speaker_02",
            "text": "So please, I am okay.",
        },
    ]
    named = {"Speaker_00": "Anushka", "Speaker_01": "Lakshmi"}
    # No audio — turn-taking path only.
    merges = same_meeting_fragment_merges(
        audio_path=None, transcript=transcript, named_labels=named
    )
    assert len(merges) == 1
    assert merges[0]["speaker_label"] == "Speaker_02"
    assert merges[0]["identified_as"] == "Anushka"
    assert merges[0]["source"] == "turn_taking"


def test_introduction_skips_already_named_speakers():
    transcript = [
        {
            "speaker": "Anushka",
            "speaker_label": "Speaker_00",
            "text": "My name is Anushka.",
        }
    ]
    assert introduction_labels_from_transcript(transcript) == []
