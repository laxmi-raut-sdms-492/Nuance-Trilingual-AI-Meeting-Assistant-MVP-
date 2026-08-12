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


def test_collapsed_greeting_turn_assigns_addressee_reply():
    from models.name_hints import repair_collapsed_greeting_turns

    transcript = [
        {
            "start_sec": 0.0,
            "end_sec": 5.0,
            "speaker": "Vaishnavi",
            "speaker_label": "Speaker_00",
            "text": "Good morning Lakshmi. Avaayu.",
        },
        {
            "start_sec": 5.0,
            "end_sec": 16.0,
            "speaker": "Vaishnavi",
            "speaker_label": "Speaker_00",
            "text": "I am fine. How about you?",
        },
    ]
    fixes = repair_collapsed_greeting_turns(transcript)
    assert len(fixes) == 1
    assert fixes[0]["new_speaker"] == "Lakshmi"
    assert fixes[0]["old_speaker"] == "Vaishnavi"
    assert fixes[0]["start_sec"] == 5.0


def test_collapsed_greeting_chain_three_people():
    """A greets B, B replies, A greets C, C replies — all on one label."""
    from models.name_hints import repair_collapsed_greeting_turns

    transcript = [
        {
            "start_sec": 0.0,
            "end_sec": 2.0,
            "speaker": "Anushka",
            "speaker_label": "Speaker_00",
            "text": "Good morning Lakshmi.",
        },
        {
            "start_sec": 2.5,
            "end_sec": 4.0,
            "speaker": "Anushka",
            "speaker_label": "Speaker_00",
            "text": "I am fine, thank you.",
        },
        {
            "start_sec": 5.0,
            "end_sec": 7.0,
            "speaker": "Anushka",
            "speaker_label": "Speaker_00",
            "text": "Good morning Priya.",
        },
        {
            "start_sec": 7.5,
            "end_sec": 9.0,
            "speaker": "Anushka",
            "speaker_label": "Speaker_00",
            "text": "Hello, ready to start.",
        },
    ]
    fixes = repair_collapsed_greeting_turns(transcript)
    by_start = {round(f["start_sec"], 1): f["new_speaker"] for f in fixes}
    assert by_start[2.5] == "Lakshmi"
    assert by_start[7.5] == "Priya"
    assert len(fixes) == 2


def test_mutual_greetings_devanagari():
    from models.name_hints import resolve_names_from_greetings

    transcript = [
        {
            "speaker": "Speaker_00",
            "speaker_label": "Speaker_00",
            "text": "नमस्ते लक्ष्मी, चला सुरुवात करूया.",
        },
        {
            "speaker": "Speaker_01",
            "speaker_label": "Speaker_01",
            "text": "नमस्कार अनुष्का, मी तयार आहे.",
        },
    ]
    assert resolve_names_from_greetings(transcript) == {
        "Speaker_00": "अनुष्का",
        "Speaker_01": "लक्ष्मी",
    }


def test_extracts_marathi_self_introduction():
    assert extract_self_introduction_name("माझं नाव प्रिया आहे.") == "प्रिया"


def test_extracts_hindi_greeted_name():
    from models.name_hints import extract_greeted_name

    assert extract_greeted_name("सुप्रभात राहुल.") == "राहुल"
    transcript = [
        {
            "speaker": "Anushka",
            "speaker_label": "Speaker_00",
            "text": "My name is Anushka.",
        }
    ]
    assert introduction_labels_from_transcript(transcript) == []
