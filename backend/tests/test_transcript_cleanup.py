"""Transcript cleanup faithfulness checks."""

from models.transcript_cleanup import _output_is_faithful, cleanup_turn_text


def test_rejects_hallucinated_llm_output():
    raw = "मेरा साथी और हरिश्चंद्र"
    invented = "मेरा साथी और हरिश्चंद्र ने कल बैठक की और निर्णय लिया"
    assert _output_is_faithful(raw, invented) is False


def test_accepts_punctuation_only_change():
    raw = "Hello world"
    cleaned = "Hello world."
    assert _output_is_faithful(raw, cleaned) is True


def test_rule_based_fallback_when_llm_disabled():
    text, engine = cleanup_turn_text("test phrase", use_llm=False)
    assert engine == "rules"
    assert "test phrase" in text
