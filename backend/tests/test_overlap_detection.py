"""Overlap is re-decided after the recluster, from short windows of each line."""

import numpy as np

from config import SAMPLE_RATE
from models.overlap_detection import detect_overlaps, window_overlap

A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
B = np.array([0.0, 1.0, 0.0], dtype=np.float32)
C = np.array([0.0, 0.0, 1.0], dtype=np.float32)
BLEND = (A + B) / np.linalg.norm(A + B)

# Fake audio: each sample's value says whose voice it is. The fake voice-print
# model reads the window's mean value.
VOICE = {1.0: A, 2.0: B, 3.0: C, 1.5: BLEND}


def NO_MODEL(audio):
    """The pyannote model is unavailable: the voice-print fallback runs."""
    return None


def fake_embed(window):
    return VOICE[round(float(np.mean(window)) * 2) / 2]


def line(label, start, end, **extra):
    return {"speaker_label": label, "speaker": label, "start_sec": start, "end_sec": end, **extra}


def build(spans, total):
    audio = np.zeros(int(total * SAMPLE_RATE), dtype=np.float32)
    for start, end, value in spans:
        audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = value
    return audio


def test_blend_window_scores_above_either_voice():
    centroids = {"Speaker_00": A, "Speaker_01": B}
    assert window_overlap(BLEND, "Speaker_00", centroids)[0] > 0.1
    assert window_overlap(A, "Speaker_00", centroids)[0] < 0


def test_interruption_inside_a_line_is_found():
    transcript = [
        line("Speaker_00", 0, 3),
        line("Speaker_01", 3, 6),
        line("Speaker_00", 6, 10.5),  # Speaker_01 joins in at 7.5-9.0
    ]
    audio = build([(0, 3, 1.0), (3, 6, 2.0), (6, 7.5, 1.0), (7.5, 9.0, 1.5), (9.0, 10.5, 1.0)], 10.5)
    out, info = detect_overlaps(transcript, [A, B, A], audio, embed=fake_embed, find_regions=NO_MODEL)

    assert info["overlaps"] == 1
    assert [l.get("is_overlap", False) for l in out] == [False, False, True]
    assert out[2]["candidate_labels"] == ["Speaker_00", "Speaker_01"]
    assert out[2]["speaker"] == "Speaker_00 + Speaker_01"


def test_clean_lines_are_not_flagged():
    transcript = [line("Speaker_00", 0, 3), line("Speaker_01", 3, 6)]
    audio = build([(0, 3, 1.0), (3, 6, 2.0)], 6)
    out, info = detect_overlaps(transcript, [A, B], audio, embed=fake_embed, find_regions=NO_MODEL)
    assert info["overlaps"] == 0
    assert not any(l.get("is_overlap", False) for l in out)


def test_stale_streaming_overlap_is_re_decided_with_final_speakers():
    """Streaming named clusters the recluster later renamed; they must not survive."""
    transcript = [
        line("Speaker_00", 0, 3),
        line("Speaker_01", 3, 6),
        line(
            "Speaker_07 + Speaker_09", 6, 9,
            is_overlap=True, candidate_labels=["Speaker_07", "Speaker_09"],
        ),
    ]
    audio = build([(0, 3, 1.0), (3, 6, 2.0), (6, 9, 2.0)], 9)
    out, _ = detect_overlaps(transcript, [A, B, B], audio, embed=fake_embed, find_regions=NO_MODEL)
    assert out[2]["is_overlap"] is False
    assert out[2]["speaker_label"] == "Speaker_01"


def test_separated_overlap_is_left_alone():
    sep = line(
        "Speaker_00 + Speaker_01", 3, 6, is_overlap=True, is_separated_overlap=True,
        candidate_labels=["Speaker_00", "Speaker_01"],
    )
    transcript = [line("Speaker_00", 0, 3), sep, line("Speaker_01", 6, 9)]
    audio = build([(0, 3, 1.0), (3, 6, 1.5), (6, 9, 2.0)], 9)
    out, _ = detect_overlaps(transcript, [A, BLEND, B], audio, embed=fake_embed, find_regions=NO_MODEL)
    assert out[1] == sep


def test_one_speaker_meeting_has_no_overlap():
    transcript = [
        line("Speaker_00", 0, 3),
        line("Speaker_00 + Speaker_03", 3, 6, is_overlap=True, candidate_labels=["Speaker_00", "Speaker_03"]),
    ]
    audio = build([(0, 6, 1.0)], 6)
    out, _ = detect_overlaps(transcript, [A, A], audio, embed=fake_embed, find_regions=NO_MODEL)
    assert not any(l.get("is_overlap", False) for l in out)
    assert out[1]["speaker_label"] == "Speaker_00"


def test_model_regions_mark_lines_and_voice_prints_name_the_joiner():
    transcript = [
        line("Speaker_00", 0, 3),
        line("Speaker_01", 3, 6),
        line("Speaker_00", 6, 10),
        line("Speaker_02", 10, 13),
    ]
    # The joiner's voice is in the audio of the overlap region.
    audio = build([(0, 3, 1.0), (3, 6, 2.0), (6, 8, 1.0), (8, 9, 2.0), (9, 10, 1.0), (10, 13, 3.0)], 13)
    regions = [(8.2, 8.8), (12.0, 12.1)]  # the second is too short to count
    out, info = detect_overlaps(
        transcript, [A, B, A, C], audio, embed=fake_embed, find_regions=lambda a: regions
    )
    assert info["source"] == "model"
    assert [l.get("is_overlap", False) for l in out] == [False, False, True, False]
    assert out[2]["candidate_labels"] == ["Speaker_00", "Speaker_01"]


def test_model_unavailable_without_token(monkeypatch):
    import models.overlap_detection as od

    monkeypatch.setattr(od, "_pipeline", None)
    monkeypatch.setattr(od, "_load_error", RuntimeError("gated"))
    assert od.overlap_regions(np.zeros(SAMPLE_RATE, dtype=np.float32)) is None
