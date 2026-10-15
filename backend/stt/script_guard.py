"""
Keep transcripts to the scripts this product supports: Latin (English) and
Devanagari (Hindi/Marathi).

A cloud engine that mishears a clip can answer in Urdu, Bengali, Tamil, Arabic,
Cyrillic or CJK. Relabelling the language is not enough — the garbled text is
still in the transcript — so the text itself is checked by script.
"""

from __future__ import annotations

import re
import unicodedata

_KEEP_PREFIXES = ("LATIN", "DEVANAGARI")
# Zero-width joiners are part of correct Devanagari conjuncts.
_KEEP_CHARS = {"‌", "‍"}
# More than this share of letters in other scripts means the whole clip is
# garbage, not a few stray characters worth cleaning.
GARBLED_LETTER_RATIO = 0.4


def _script_of(ch: str) -> str:
    try:
        return unicodedata.name(ch).split(" ", 1)[0]
    except ValueError:
        return "UNKNOWN"


def _is_foreign_letter(ch: str) -> bool:
    if ch in _KEEP_CHARS or not ch.isalpha():
        return False
    return _script_of(ch) not in _KEEP_PREFIXES


def foreign_letter_ratio(text: str) -> float:
    """Share of letters that are neither Latin nor Devanagari (0.0 - 1.0)."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(_is_foreign_letter(c) for c in letters) / len(letters)


def has_foreign_script(text: str) -> bool:
    return any(_is_foreign_letter(c) for c in (text or ""))


def is_garbled(text: str) -> bool:
    return foreign_letter_ratio(text) > GARBLED_LETTER_RATIO


def strip_foreign_script(text: str) -> str:
    """Remove letters outside Latin/Devanagari and tidy the leftover spacing."""
    cleaned = "".join(c for c in (text or "") if not _is_foreign_letter(c))
    return re.sub(r"\s+", " ", cleaned).strip()
