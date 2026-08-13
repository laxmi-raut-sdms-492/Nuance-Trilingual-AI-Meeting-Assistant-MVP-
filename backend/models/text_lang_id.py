"""
Text-based Hindi / Marathi / English disambiguation.

Root cause this module fixes:

  Whisper's `detect_language` is an *acoustic* classifier. Hindi and Marathi
  are close enough in phonetics and share Devanagari script, and Whisper was
  never trained with much Marathi audio, so on real meeting audio it lands on
  "hi" for Marathi speech far more often than chance would predict — even at
  high reported confidence. Restricting the candidate set (ALLOWED_LANGUAGES)
  makes the *English* vs *Hindi/Marathi* call reliable, but does nothing for
  the Hindi-vs-Marathi call, because the ambiguity is acoustic, not about
  which languages are in the candidate pool.

  There was previously no independent signal to catch this: the only
  post-decode check was `_is_script_mismatch`, which only asks "is this
  Devanagari at all", not "is this Devanagari *Marathi* or *Hindi*". A
  confident, fluent Hindi-shaped decode of Marathi audio passes that check
  cleanly and was kept as Hindi.

  This module adds that missing signal: a lexical classifier over the
  *decoded text itself* (which is a much stronger signal than raw audio,
  once decoded correctly) using grammar markers that reliably distinguish
  Hindi from Marathi — the copula (Marathi "आहे"/"ahe" vs Hindi "है"/"hai"),
  first-person pronoun ("मी"/"mi" vs "मैं"/"main"/"mein"), and other closed-class
  function words / verb-ending markers. It works on both Devanagari and
  romanized (Latin-script) text, which also covers code-switched Romanized
  Marathi/Hindi that never had a Devanagari decode at all.

  This is deliberately NOT a lookup table of the bug-report example
  sentences. It is a bag of ~30-40 closed-class function words per language
  (pronouns, copulas, common verb endings, postpositions) — the same kind of
  small, closed, high-frequency vocabulary that classical language-ID systems
  are built on, so it generalizes to unseen sentences using the same words in
  new combinations.
"""

from __future__ import annotations

import re

# Devanagari, whole-word markers. These are function words / inflections
# that differ between the two languages regardless of topic.
_MR_DEVANAGARI = {
    "आहे", "आहेस", "आहात", "आहोत", "आहेत",  # copula "to be" (all persons)
    "नाही",  # negation "is not"
    "मी", "माझा", "माझी", "माझे", "माझं", "तुझा", "तुझी", "आम्ही", "आपण", "तू",
    "त्याला", "तिला", "त्यांना",
    "मला", "तुला", "आम्हाला",
    "काय", "कसं", "कशी", "कोण", "कुठे", "कधी",
    "आणि", "पण", "किंवा",
    "करतो", "करते", "करतोय", "करतेय", "करत", "बोलतो", "बोलते", "बोलतोय",
    "जातो", "जाते", "येतो", "येते",
    "होता", "होती", "होते",
}

_HI_DEVANAGARI = {
    "है", "हैं", "था", "थी", "थे",  # copula
    "नहीं",  # negation
    "मैं", "मेरा", "मेरी", "मेरे", "तुम", "तुम्हारा", "हम", "आप", "तू",
    "उसे", "उसको", "उन्हें",
    "मुझे", "तुझे", "हमें",
    "क्या", "कैसे", "कौन", "कहाँ", "कब",
    "और", "लेकिन", "या",
    "करता", "करती", "करते", "बोलता", "बोलती", "बोलते",
    "जाता", "जाती", "जाते", "आता", "आती", "आते",
    "रहा", "रही", "रहे", "हूँ", "हूं",
}

# Romanized (Latin-script) equivalents of the same closed-class words, for
# code-switched speech that Whisper transcribes phonetically instead of in
# Devanagari (e.g. "Me Saloni ahe", "Majha nav Saloni ahe"). Matched as whole
# words, case-insensitive, so this does not fire on substrings inside
# unrelated English words.
_MR_ROMAN = {
    "ahe", "aahe", "ahes", "aahet", "nahi", "nahin",
    "mi", "majha", "majhi", "maze", "majhe", "tuza", "tuzi", "amhi", "aapan",
    "mala", "tula", "amhala", "tyala", "tila",
    "kay", "kasa", "kashi", "kon", "kuthe", "kadhi",
    "ani", "pan",
    "karto", "karte", "kartoy", "kartey", "bolto", "bolte",
    "jato", "jate", "yeto", "yete", "hota", "hoti",
    "nav", "naav",
}

_HI_ROMAN = {
    "hai", "hain", "tha", "thi", "the", "nahi", "nahin",
    "main", "mein", "mera", "meri", "mere", "tum", "tumhara", "hum", "aap",
    "usse", "usko", "unhe",
    "mujhe", "tujhe", "hume",
    "kya", "kaise", "kaun", "kahan", "kab",
    "aur", "lekin", "ya",
    "karta", "karti", "karte", "bolta", "bolti", "bolte",
    "jata", "jati", "jate", "aata", "aati", "aate",
    "raha", "rahi", "rahe", "hoon", "hu",
    "naam",
}

# Romanized markers that are also ordinary English words.
#
# These have to come back out. Romanization collides with English, and the
# collisions are not rare words: "the" was carried in _HI_ROMAN for Hindi थे,
# so a plain English sentence containing "the" scored Hindi 1 / Marathi 0 —
# a unanimous vote, confidence 1.0 — and callers then re-decoded that line
# with Indic Conformer, which returns Devanagari for whatever it is handed.
# "the deadline is Friday" came back as "द डेडलाइन इज फ्रायडे".
#
# A marker word only carries information if seeing it makes one language more
# likely than the other. These do not, so they cannot be evidence.
_ENGLISH_COLLISIONS = {"the", "main", "mere", "hum", "hu", "ya", "maze", "pan"}

_MR_ROMAN -= _ENGLISH_COLLISIONS
_HI_ROMAN -= _ENGLISH_COLLISIONS

# Note what is NOT done here: no minimum number of hits. Every romanized marker
# that survives the subtraction above is a word English does not have, so one
# of them really is evidence — and demanding two would throw away the short
# self-introductions this module was written for ("Me Saloni ahe" carries
# exactly one). The defect was ambiguous words, not insufficient words.

# Split on whitespace/punctuation only. `\w` under Python's Unicode regex
# excludes Devanagari combining vowel signs (category Mn, e.g. the ी in
# "मी"), which would otherwise fracture words like "आहे" into "आह" + stray
# marks and silently break every Devanagari lookup.
_SPLIT_RE = re.compile(r"[\s,.!?;:'\"()\[\]{}|/\\<>]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return [w for w in _SPLIT_RE.split(text.lower()) if w]


def classify_text_language(text: str) -> tuple[str | None, float]:
    """
    Lexical hi/mr/en guess from decoded text alone.

    Returns (language, confidence) where language is "hi", "mr", "en", or
    None if the text has no recognizable marker words (too short / all
    proper nouns / numbers). confidence is (winning_hits - runner_up_hits)
    scaled by total hits — 0 when the vote is tied or empty.

    Only hi vs mr are scored against curated marker sets; "en" is returned
    when the text has letters but zero Devanagari characters and zero hits
    against either Indic marker set (i.e. plain English words only).
    """
    if not text:
        return None, 0.0

    words = _words(text)
    if not words:
        return None, 0.0

    has_devanagari = any("ऀ" <= ch <= "ॿ" for ch in text)

    mr_hits = 0
    hi_hits = 0
    for w in words:
        if has_devanagari:
            if w in _MR_DEVANAGARI:
                mr_hits += 1
            if w in _HI_DEVANAGARI:
                hi_hits += 1
        else:
            if w in _MR_ROMAN:
                mr_hits += 1
            if w in _HI_ROMAN:
                hi_hits += 1

    total = mr_hits + hi_hits
    if total == 0:
        if has_devanagari:
            return None, 0.0
        return "en", 0.0

    if mr_hits == hi_hits:
        return None, 0.0

    winner = "mr" if mr_hits > hi_hits else "hi"
    confidence = abs(mr_hits - hi_hits) / total
    return winner, confidence


# Below this confidence, the lexical vote is too close to call and the
# caller should keep the acoustic/existing result rather than overriding it.
TEXT_LANG_MIN_CONFIDENCE = 0.34
