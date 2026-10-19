"""Garbled / wrong-language STT output must never reach the transcript."""

from types import SimpleNamespace

import numpy as np

from stt.sarvam_adapter import SarvamSTTAdapter
from stt.script_guard import has_foreign_script, is_garbled, strip_foreign_script


def test_urdu_script_is_foreign():
    assert has_foreign_script("یہ ایک ٹیسٹ ہے")
    assert is_garbled("یہ ایک ٹیسٹ ہے")


def test_english_hindi_marathi_are_kept():
    for text in ("Good morning team", "नमस्ते, कैसे हैं आप?", "आपण अभिप्राय पाठवला", "Hindustanच्या PMPL 430"):
        assert not has_foreign_script(text)
        assert strip_foreign_script(text) == text


def test_few_stray_foreign_letters_are_stripped_not_dropped():
    text = "The release is ready ی today"
    assert has_foreign_script(text) and not is_garbled(text)
    assert strip_foreign_script(text) == "The release is ready today"


def test_other_scripts_are_foreign():
    for text in ("привет мир", "你好世界", "வணக்கம்", "ধন্যবাদ"):
        assert has_foreign_script(text)


class _FakeSTT:
    def __init__(self, replies):
        self.replies = list(replies)
        self.codes = []

    def transcribe(self, **kw):
        self.codes.append(kw["language_code"])
        return self.replies.pop(0)


def _adapter(replies):
    a = SarvamSTTAdapter.__new__(SarvamSTTAdapter)
    fake = _FakeSTT(replies)
    a._client = SimpleNamespace(speech_to_text=fake)
    return a, fake


def _resp(text, lang, prob=0.9):
    return SimpleNamespace(transcript=text, language_code=lang, language_probability=prob)


AUDIO = np.zeros(16000, dtype=np.float32)


def test_urdu_answer_is_retried_with_forced_meeting_language():
    adapter, fake = _adapter([_resp("یہ ایک ٹیسٹ ہے", "ur-IN"), _resp("हा एक टेस्ट आहे", "mr-IN")])
    out = adapter.transcribe(AUDIO, hint_language="mr")
    assert fake.codes == ["unknown", "mr-IN"]
    assert out["text"] == "हा एक टेस्ट आहे"
    assert out["language"] == "mr"


def test_clean_supported_answer_is_not_retried():
    adapter, fake = _adapter([_resp("Good morning everyone", "en-IN")])
    out = adapter.transcribe(AUDIO, hint_language="mr")
    assert fake.codes == ["unknown"]
    assert out["text"] == "Good morning everyone" and out["language"] == "en"


def test_still_garbled_after_retry_is_dropped():
    adapter, _ = _adapter([_resp("یہ ایک ٹیسٹ ہے", "ur-IN"), _resp("یہ ایک ٹیسٹ ہے", "hi-IN")])
    out = adapter.transcribe(AUDIO, hint_language="hi")
    assert out["text"] == ""


def test_unclear_marks_junk_but_keeps_every_line():
    from models.clarity import mark_unclear

    def line(text, lang, prob, start):
        return {"text": text, "language": lang, "language_prob": prob, "start_sec": start, "end_sec": start + 20}

    lines = [
        line("आपण अभिप्राय पाठवला आणि तो आला आहे आता पुढे जाऊ", "mr", 0.99, 0),
        line("Hi Visha.", "en", 0.30, 20),
        line("तो तो तो तो तो", "mr", 0.95, 40),
        line("माननीय सदस्य यांनी हा प्रश्न विचारला आहे", "mr", 0.95, 60),
    ]
    out = mark_unclear(lines)
    assert len(out) == 4
    assert [t["is_unclear"] for t in out] == [False, True, True, False]


# ------------------------------------------------------------ retry language


def _patch_lid(monkeypatch, code):
    import stt.sarvam_adapter as sarvam

    monkeypatch.setattr(sarvam, "_spoken_language", lambda audio: code)


def test_english_clip_misheard_is_retried_as_english(monkeypatch):
    """The old rule forced the meeting's top language, turning English into Hindi."""
    _patch_lid(monkeypatch, "en")
    adapter, fake = _adapter([_resp("یہ ایک ٹیسٹ ہے", "ur-IN"), _resp("This is a test", "en-IN")])
    out = adapter.transcribe(AUDIO, hint_language="hi")
    assert fake.codes == ["unknown", "en-IN"]
    assert out["text"] == "This is a test" and out["language"] == "en"


def test_kannada_answer_retried_as_the_indic_language_already_heard(monkeypatch):
    _patch_lid(monkeypatch, None)
    adapter, fake = _adapter([
        _resp("आपण अभिप्राय पाठवला", "mr-IN"),
        _resp("ಈಗೋ", "kn-IN"),
        _resp("आता", "mr-IN"),
    ])
    adapter.transcribe(AUDIO, hint_language="en")
    out = adapter.transcribe(AUDIO, hint_language="en")
    assert fake.codes == ["unknown", "unknown", "mr-IN"]
    assert out["text"] == "आता"


def test_retry_without_meeting_context_follows_the_wrong_answer(monkeypatch):
    _patch_lid(monkeypatch, None)
    adapter, fake = _adapter([_resp("یہ", "ur-IN"), _resp("यह", "hi-IN")])
    adapter.transcribe(AUDIO, hint_language="en")
    assert fake.codes[-1] == "hi-IN"

    adapter, fake = _adapter([_resp("હા", "gu-IN"), _resp("हो", "mr-IN")])
    adapter.transcribe(AUDIO, hint_language="en")
    assert fake.codes[-1] == "mr-IN"


def test_too_short_or_silent_clip_skips_language_id():
    from stt.sarvam_adapter import _spoken_language

    assert _spoken_language(np.zeros(8000, dtype=np.float32)) is None
    assert _spoken_language(np.zeros(32000, dtype=np.float32)) is None


# ------------------------------------------------------------ final gate


def test_foreign_vowel_signs_and_digits_are_caught():
    assert strip_foreign_script("રૂમ ઓપન. आहे") == ". आहे"
    assert has_foreign_script("੧੨")  # Gurmukhi digits
    for text in ("कि", "क्षेत्र", "café", "नमस्ते 2026", "१२३"):
        assert not has_foreign_script(text), text


def test_clean_text_drops_leftover_punctuation():
    from stt.script_guard import clean_text

    assert clean_text("હા.") == ""
    assert clean_text("Good morning") == "Good morning"


def test_clean_transcript_removes_and_strips_lines():
    from stt.script_guard import clean_transcript

    lines = [
        {"text": "ಈಗೋ", "raw_text": "ಈಗೋ"},
        {"text": "રૂમ ઓપન. रेफरन्स घ्यायचा आहे आणि तो पर्यंत माझ्याकडे पण आहे", "raw_text": "x"},
        {"text": "Good morning", "attributed_spans": [{"speaker": "A", "text": "hello یہ"}]},
    ]
    out = clean_transcript(lines)
    assert len(out) == 2
    assert out[0]["text"] == ". रेफरन्स घ्यायचा आहे आणि तो पर्यंत माझ्याकडे पण आहे"
    assert out[1]["attributed_spans"][0]["text"] == "hello"
    assert lines[0]["text"] == "ಈಗೋ"  # input is not mutated
