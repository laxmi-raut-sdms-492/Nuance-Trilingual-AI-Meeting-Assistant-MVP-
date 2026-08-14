"""
Tests for the summarization stage.

No model and no server: every test here exercises the parts that decide what
reaches the screen — citation verification, assignee verification, keyword
counting, and the extractive fallback. Those are the project's "never fabricate
data" rule expressed as code, so they are the parts worth pinning.

The Ollama path itself is not tested against a live model. What *is* tested is
that `_collect_items` throws away whatever a model returns if the quote is not
in the transcript, which is the only guarantee that matters regardless of which
model is configured.
"""

from __future__ import annotations

import json

from models import summarizer


def line(text, speaker="Speaker 1", start=0.0, color="#3b82f6", **extra):
    return {
        "text": text,
        "speaker": speaker,
        "start_sec": start,
        "end_sec": start + 3.0,
        "color": color,
        **extra,
    }


# ------------------------------------------------------------- citations


def test_exact_quote_verifies():
    lines = ["We will ship on Friday.", "मच्छरों की समस्या बढ़ रही है"]
    assert summarizer.verify_quote("We will ship on Friday.", lines)
    assert summarizer.verify_quote("मच्छरों की समस्या बढ़ रही है", lines)


def test_near_quote_verifies():
    """A repaired typo or dropped filler is still a citation of a real line."""
    lines = ["So we will, uh, ship on Friday."]
    assert summarizer.verify_quote("So we will ship on Friday.", lines)


def test_invented_quote_fails():
    lines = ["We will ship on Friday."]
    assert not summarizer.verify_quote("The budget was approved unanimously.", lines)


def test_empty_quote_fails():
    """An item with no citation is unverifiable, not trivially verified."""
    assert not summarizer.verify_quote("", ["anything"])
    assert not summarizer.verify_quote(None, ["anything"])


# ------------------------------------------------- item collection & drops


def test_uncited_items_are_dropped():
    transcript = [line("We agreed to postpone the launch.")]
    windows = [
        {
            "decisions": [
                {"text": "Launch postponed", "quote": "We agreed to postpone the launch."},
                {"text": "Budget doubled", "quote": "We doubled the budget."},
            ],
            "action_items": [
                {"title": "Tell the vendor", "quote": "Nobody said this line."},
            ],
        }
    ]
    decisions, actions = summarizer._collect_items(windows, transcript)
    assert [d["text"] for d in decisions] == ["Launch postponed"]
    # The stored quote is the real transcript line, not the model's copy of it.
    assert decisions[0]["quote"] == "We agreed to postpone the launch."
    assert actions == []


def test_duplicate_items_across_windows_collapse():
    transcript = [line("We agreed to postpone the launch.")]
    window = {
        "decisions": [{"text": "Launch postponed", "quote": "We agreed to postpone the launch."}],
        "action_items": [],
    }
    decisions, _ = summarizer._collect_items([window, dict(window)], transcript)
    assert [d["text"] for d in decisions] == ["Launch postponed"]


def test_unknown_assignee_is_dropped_but_item_survives():
    """The verified quote makes the item real; the name is inference on top."""
    transcript = [line("Someone needs to send the report.")]
    windows = [
        {
            "action_items": [
                {
                    "title": "Send the report",
                    "assignee": "Priyanka",
                    "quote": "Someone needs to send the report.",
                }
            ]
        }
    ]
    _, actions = summarizer._collect_items(windows, transcript)
    assert len(actions) == 1
    assert actions[0]["assignee"] is None
    assert actions[0]["quote"] == "Someone needs to send the report."


def test_spoken_assignee_is_kept_and_coloured():
    transcript = [
        line("Rohan will send the report.", speaker="Speaker 2", color="#a855f7"),
        line("Rohan", speaker="Rohan", color="#10b981", start=5.0),
    ]
    windows = [
        {
            "action_items": [
                {
                    "title": "Send the report",
                    "assignee": "Rohan",
                    "quote": "Rohan will send the report.",
                }
            ]
        }
    ]
    _, actions = summarizer._collect_items(windows, transcript)
    assert actions[0]["assignee"] == "Rohan"
    # Colour is the speaker's, taken from the transcript the backend coloured.
    assert actions[0]["color"] == "#10b981"


def test_null_like_due_becomes_none():
    """A model that writes the string "null" must not reach the UI as text."""
    transcript = [line("We need to file the form.")]
    windows = [
        {
            "action_items": [
                {"title": "File the form", "due": "null", "quote": "We need to file the form."},
                {"title": "File it again", "due": "  ", "quote": "We need to file the form."},
            ]
        }
    ]
    _, actions = summarizer._collect_items(windows, transcript)
    assert [a["due"] for a in actions] == [None, None]


# ---------------------------------------------------------------- keywords


def test_keyword_counts_are_real_occurrences():
    transcript = [
        line("The drainage problem near the market is serious."),
        line("Drainage work must start before the monsoon."),
        line("Monsoon drainage is the priority."),
    ]
    result = summarizer.keywords(transcript)
    counts = {k["word"]: k["count"] for k in result}
    assert counts["drainage"] == 3
    assert counts["monsoon"] == 2


def test_stop_words_are_excluded():
    transcript = [line("The and for are but not you all the and for")]
    assert summarizer.keywords(transcript) == []


def test_devanagari_keywords_are_counted():
    transcript = [
        line("मच्छरों की समस्या बढ़ रही है"),
        line("मच्छरों के लिए दवा छिड़कनी है"),
    ]
    words = {k["word"] for k in summarizer.keywords(transcript)}
    assert "मच्छरों" in words
    # A postposition in the stop list must not survive as a keyword.
    assert "की" not in words


def test_idf_downweights_words_common_to_other_meetings():
    """
    Ranking uses a background corpus; counts do not.

    "drainage" and "budget" appear equally often here, but every other stored
    meeting talks about budget, so drainage is the more distinguishing keyword.
    """
    transcript = [
        line("drainage drainage budget budget"),
        line("drainage tender budget tender"),
    ]
    background = ["budget budget budget review", "budget approval budget again"]
    ranked = summarizer.keywords(transcript, background_documents=background)
    order = [k["word"] for k in ranked]
    assert order.index("drainage") < order.index("budget")
    assert {k["word"]: k["count"] for k in ranked}["budget"] == 3


# ------------------------------------------------------- extractive engine


def test_extractive_summary_is_verbatim_transcript_text():
    transcript = [
        line("The drainage problem near the market is serious and needs attention."),
        line("Drainage work must start before the monsoon arrives in the city."),
        line("Okay."),
    ]
    result = summarizer._extractive(transcript, summarizer.keywords(transcript))
    assert result["summaryEngine"] == "extractive"
    # Every sentence in the summary must be a line that was actually spoken.
    texts = [l["text"] for l in transcript]
    assert summarizer.verify_quote(result["summary"].split(".")[0], texts)


def test_extractive_finds_cued_decisions_and_actions():
    transcript = [
        line("We decided to postpone the launch until August."),
        line("I will send the revised timeline tomorrow.", speaker="Speaker 2"),
        line("The weather is nice today."),
    ]
    result = summarizer._extractive(transcript, summarizer.keywords(transcript))
    assert [d["text"] for d in result["decisions"]] == [
        "We decided to postpone the launch until August."
    ]
    # Extractive quote == text: it only ever shows a line as itself.
    assert result["decisions"][0]["quote"] == "We decided to postpone the launch until August."
    assert len(result["actionItems"]) == 1
    # This engine knows who talked and claims nothing more.
    assert result["actionItems"][0]["assignee"] == "Speaker 2"
    assert result["actionItems"][0]["quote"] == "I will send the revised timeline tomorrow."


def test_extractive_invents_nothing_on_a_meeting_with_no_cues():
    transcript = [line("Good morning everyone, the weather is pleasant.")]
    result = summarizer._extractive(transcript, summarizer.keywords(transcript))
    assert result["decisions"] == []
    assert result["actionItems"] == []


# -------------------------------------------------------------- map-reduce


def test_windows_split_on_wall_clock_not_line_count():
    transcript = [line("a", start=0.0), line("b", start=100.0), line("c", start=700.0)]
    windows = summarizer._windows(transcript, 600.0)
    assert [len(w) for w in windows] == [2, 1]


def test_empty_transcript_returns_empty_fields_not_a_summary():
    result = summarizer.summarize([])
    assert result == {
        "summary": None,
        "decisions": [],
        "actionItems": [],
        "keywords": [],
        "summaryEngine": None,
    }


def test_generate_requests_immediate_unload(monkeypatch):
    """
    Every generate call must carry keep_alive so the VRAM is released.

    Regression guard for a real outage: Ollama's default keep_alive is 5 minutes,
    a resident 7B model takes essentially all of a 4 GB card, and four uploads in
    a row failed with CUBLAS_STATUS_ALLOC_FAILED because Whisper had no room.
    """
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"response": "{}"}'

    def fake_urlopen(request, timeout=None):
        captured["payload"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr(summarizer.urllib.request, "urlopen", fake_urlopen)
    summarizer._ollama_generate("prompt", "some-model")

    assert captured["payload"]["keep_alive"] == summarizer.OLLAMA_KEEP_ALIVE
    # num_ctx must always be explicit — Ollama's 4096 default truncates silently.
    assert captured["payload"]["options"]["num_ctx"] == summarizer.SUMMARY_NUM_CTX


def test_release_vram_is_quiet_when_ollama_is_absent(monkeypatch):
    """Called before every transcription; an unreachable Ollama is not an error."""
    def refuse(*args, **kwargs):
        raise summarizer.urllib.error.URLError("connection refused")

    monkeypatch.setattr(summarizer.urllib.request, "urlopen", refuse)
    assert summarizer.release_vram("some-model") is False


def test_summarize_falls_back_to_extractive_when_no_model(monkeypatch):
    """An unreachable Ollama must degrade, not raise — the transcript is done."""
    monkeypatch.setattr(summarizer, "model_available", lambda *a, **k: False)
    transcript = [line("We decided to postpone the launch until August.")]
    result = summarizer.summarize(transcript)
    assert result["summaryEngine"] == "extractive"
    assert [d["text"] for d in result["decisions"]] == [
        "We decided to postpone the launch until August."
    ]
