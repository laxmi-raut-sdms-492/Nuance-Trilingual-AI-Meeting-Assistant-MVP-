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
