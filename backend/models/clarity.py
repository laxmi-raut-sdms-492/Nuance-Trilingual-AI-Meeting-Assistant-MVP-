"""
Mark transcript lines that are probably misheard, without deleting any of them.

A wrong-script answer is removed upstream (stt/script_guard.py). What is left
are lines in the right script but the wrong words: a two-word fragment tagged
as a different language, a low-confidence scrap, a word repeated in a loop.
A real short reply ("हो", "ठीक आहे") is indistinguishable from those by text
alone, so these lines are flagged `is_unclear` for the reader to judge rather
than dropped.
"""

from __future__ import annotations

import re
from collections import defaultdict

SHORT_LINE_WORDS = 4
LOW_LANGUAGE_PROB = 0.6
REPEAT_RUN = 4

_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def dominant_language(transcript: list[dict]) -> str | None:
    """Language with the most spoken seconds across reliable lines."""
    seconds: dict[str, float] = defaultdict(float)
    for t in transcript or []:
        lang = t.get("language")
        if lang and not t.get("language_fallback"):
            seconds[lang] += max(0.0, float(t.get("end_sec") or 0) - float(t.get("start_sec") or 0))
    return max(seconds, key=seconds.get) if seconds else None


def _repeats(words: list[str]) -> bool:
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        if run >= REPEAT_RUN:
            return True
    return False


def unclear_reason(line: dict, dominant: str | None) -> str | None:
    """Why this line looks misheard, or None when it looks fine."""
    text = (line.get("cleaned_text") or line.get("text") or "").strip()
    words = _words(text)
    if not words:
        return "no words"
    if _repeats(words):
        return "repeated word loop"

    short = len(words) <= SHORT_LINE_WORDS
    other_language = bool(dominant) and line.get("language") not in (None, dominant)
    prob = line.get("language_prob")
    weak = bool(line.get("language_fallback")) or (prob is not None and float(prob) < LOW_LANGUAGE_PROB)

    if short and other_language:
        return f"short fragment tagged {line.get('language')}, meeting is {dominant}"
    if short and weak:
        return "short fragment with low language confidence"
    return None


def mark_unclear(transcript: list[dict]) -> list[dict]:
    """Add is_unclear / unclear_reason to every line (in place) and return it."""
    dominant = dominant_language(transcript)
    for line in transcript or []:
        reason = unclear_reason(line, dominant)
        line["is_unclear"] = reason is not None
        line["unclear_reason"] = reason
    return transcript
