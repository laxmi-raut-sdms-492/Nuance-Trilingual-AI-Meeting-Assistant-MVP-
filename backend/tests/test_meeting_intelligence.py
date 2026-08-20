import pytest
from models.summarizer import summarize, verify_quote


def test_explicit_task_extraction():
    transcript = [
        {"start_sec": 0.0, "time": "00:05", "speaker": "John", "text": "I will prepare the final sales report by Friday."}
    ]
    res = summarize(transcript)
    actions = res.get("actionItems") or []
    assert len(actions) > 0
    act = actions[0]
    assert "john" in act["assignee"].lower() or act["assignee"] == "John"
    assert "friday" in act["due"].lower()


def test_implicit_commitment_extraction():
    transcript = [
        {"start_sec": 0.0, "time": "00:10", "speaker": "Sarah", "text": "I'll speak to the customer today regarding the invoice."}
    ]
    res = summarize(transcript)
    actions = res.get("actionItems") or []
    assert len(actions) > 0
    act = actions[0]
    assert "sarah" in act["assignee"].lower() or act["assignee"] == "Sarah"
    assert "today" in act["due"].lower()


def test_follow_up_detection():
    transcript = [
        {"start_sec": 0.0, "time": "00:01", "speaker": "Alex", "text": "Can you let me know how the client demo goes?"},
        {"start_sec": 4.0, "time": "00:04", "speaker": "Ben", "text": "Yes, I will follow up with you right after."}
    ]
    res = summarize(transcript)
    actions = res.get("actionItems") or []
    assert len(actions) > 0
    assert any("ben" in a["assignee"].lower() or a["assignee"] == "Ben" for a in actions)


def test_confirmed_decision_extraction():
    transcript = [
        {"start_sec": 0.0, "time": "00:15", "speaker": "Leader", "text": "Let's use the new system from next Monday."}
    ]
    res = summarize(transcript)
    decisions = res.get("decisions") or []
    assert len(decisions) > 0
    assert "new system" in decisions[0]["text"].lower()


def test_possibility_gating():
    """Statements containing 'might', 'maybe', 'could' must NOT be tagged as confirmed decisions."""
    transcript = [
        {"start_sec": 0.0, "time": "00:20", "speaker": "Dave", "text": "We might launch the mobile app next month if testing passes."}
    ]
    res = summarize(transcript)
    decisions = res.get("decisions") or []
    assert len(decisions) == 0


def test_quartz_power_group_transcript():
    """
    Quartz Power Group multi-turn transcript test:
    Verifies multi-turn extraction, pronoun mapping, implicit tasks, decision identification,
    and possibility gating without hardcoding names.
    """
    transcript = [
        {"start_sec": 0.0, "time": "00:05", "speaker": "Paul", "text": "I'm going to get back to you when I've spoken to Anna about the call-center training attendance."},
        {"start_sec": 10.0, "time": "00:15", "speaker": "Maria", "text": "You're going to speak to Matt and Lucy about the intranet training this week."},
        {"start_sec": 20.0, "time": "00:25", "speaker": "Maria", "text": "You're going to proceed with caution with David on the school's events."},
        {"start_sec": 30.0, "time": "00:35", "speaker": "Paul", "text": "I'm interviewing Monica with Maya tomorrow at 1:30."},
        {"start_sec": 40.0, "time": "00:45", "speaker": "Maria", "text": "Should the school event come under the training budget?"},
        {"start_sec": 50.0, "time": "00:55", "speaker": "Paul", "text": "No, the school event should come under the Marketing budget rather than the Training budget."},
        {"start_sec": 60.0, "time": "01:05", "speaker": "Maria", "text": "We might sponsor another school event later this year."}
    ]

    res = summarize(transcript)
    actions = res.get("actionItems") or []
    decisions = res.get("decisions") or []

    # 1. Action Items verification
    assert len(actions) >= 3

    # Check Paul -> Anna / Call center training
    paul_anna = [a for a in actions if "anna" in (a.get("quote") or "").lower() or "call-center" in (a.get("quote") or "").lower()]
    assert len(paul_anna) > 0
    assert paul_anna[0]["assignee"].lower() in ("paul", "unassigned")

    # Check Matt / Lucy / Intranet training
    intranet_task = [a for a in actions if "matt" in (a.get("quote") or "").lower() or "intranet" in (a.get("quote") or "").lower()]
    assert len(intranet_task) > 0

    # Check Monica interview tomorrow at 1:30
    monica_task = [a for a in actions if "monica" in (a.get("quote") or "").lower()]
    assert len(monica_task) > 0

    # 2. Decision verification (Marketing budget decision)
    assert len(decisions) >= 1
    budget_decision = [d for d in decisions if "marketing" in d["text"].lower() or "budget" in d["text"].lower()]
    assert len(budget_decision) > 0

    # 3. Possibility gating (We might sponsor another event -> NOT a decision)
    might_sponsor = [d for d in decisions if "might sponsor" in d["text"].lower()]
    assert len(might_sponsor) == 0


def test_quote_verification_tolerance():
    transcript_lines = [
        "Paul: I'm going to get back to you when I've spoken to Anna about call center training."
    ]
    # Punctuation / ASR minor variation
    quote = "I'm going to get back to you when I've spoken to Anna about call-center training"
    assert verify_quote(quote, transcript_lines) is True
