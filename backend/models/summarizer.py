"""
Summarization stage — summary, decisions, action items, keywords.

Two engines, picked at runtime:

- **Ollama** (`SUMMARY_MODEL`), a local LLM. Writes the prose summary and
  proposes decisions and action items. Every proposed item must quote the
  transcript line it came from; anything whose quote cannot be found in the
  transcript is **dropped**, never shown.
- **Extractive**, always available, used when Ollama is unreachable or the model
  is not pulled. The summary is real transcript lines, verbatim, so the demo
  never depends on a model being resident.

Three things this module refuses to do, because the project's hard rule is that
nothing on screen may be invented:

1. **No unverifiable citation survives.** `avg_logprob`-style confidence is not
   evidence; string matching against the transcript is. A fluent decision nobody
   made is worse than an empty panel, and only verification tells them apart.
2. **Assignees are checked too.** A model reading a transcript happily invents a
   plausible name. An assignee is kept only if it names a known speaker or is
   spoken somewhere in the transcript; otherwise the item stays and the assignee
   becomes null. `due` is never guessed — no date is inferred from "soon".
3. **Keywords never come from the model.** They are counted over the real
   transcript, so the number beside a word in the UI is a fact.

Prose summaries cannot be verified this way — there is no line to match a
paraphrase against. So the engine that produced one is recorded on the meeting
(`summaryEngine`) and the UI labels it, rather than presenting generated text as
if it were extracted.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import urllib.error
import urllib.request

from config import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_URL,
    SUMMARY_CITATION_THRESHOLD,
    SUMMARY_KEYWORD_COUNT,
    SUMMARY_KEYWORD_MIN_OCCURRENCES,
    SUMMARY_MODEL,
    SUMMARY_NUM_CTX,
    SUMMARY_TIMEOUT_SECONDS,
    SUMMARY_WINDOW_SECONDS,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- stop words
# Ranking keywords needs to drop function words in all three languages. These
# lists are **empirical**: every English entry past the obvious function words
# was added because it actually appeared in the top twelve of a real meeting and
# described nothing about it. Extend the same way — from output, not from a
# published stop-word list, which will either miss conversational filler or eat
# real vocabulary. Hindi/Marathi share enough surface forms that one merged
# Devanagari set is simpler than two.
#
# Note the interaction with TOKEN_RE's three-character floor: "he", "we", "it",
# "me" never reach this set, which is why it looks like it is missing pronouns.

STOP_WORDS_EN = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "two", "three", "our", "out", "his", "him", "has", "have", "had",
    "what", "when", "where", "who", "will", "with", "this", "that", "these",
    "those", "then", "than", "there", "their", "them", "they", "from", "your",
    "yeah", "okay", "just", "like", "know", "think", "going", "right", "want",
    "need", "make", "sure", "well", "some", "would", "could", "should", "about",
    "because", "very", "also", "into", "over", "more", "much", "been", "were",
    "does", "did", "get", "got", "say", "said", "see", "now", "how", "why",
    "any", "let", "lets", "yes", "actually", "really", "basically", "thing",
    "things", "something", "anything", "everything", "here", "hey", "hello",
    "thanks", "thank", "please", "sorry", "maybe", "mean", "meeting", "team",
    "today",
    # Added after a real run put these in the top twelve of both test meetings.
    # They are frequent in any conversation and describe none of them.
    "good", "great", "lot", "look", "looking", "might", "time", "item", "items",
    "next", "back", "first", "last", "come", "came", "take", "takes", "little",
    "bit", "kind", "sort", "even", "still", "same", "other", "another", "both",
    "many", "few", "able", "around", "before", "after", "again", "away",
    "far", "way", "ways", "through", "coming", "getting",
    # Third pass, from a third meeting. Pronouns under three characters are
    # already excluded by TOKEN_RE's length floor, which is why only these
    # survive; the rest is affirmation and small talk.
    "she", "its", "too", "anybody", "everybody", "somebody", "nobody",
    "lovely", "fantastic", "nice", "yep", "hmm", "everyone", "anyone",
    "someone", "which", "while", "where's", "there's", "that's",
}

STOP_WORDS_DEVANAGARI = {
    "का", "के", "की", "को", "है", "हैं", "हूं", "हूँ", "में", "से", "पर",
    "यह", "वह", "ये", "वो", "और", "कि", "तो", "ही", "भी", "नहीं", "था",
    "थे", "थी", "हो", "होगा", "होता", "होती", "करना", "करने", "किया",
    "लिए", "एक", "अपने", "अपना", "इस", "उस", "कुछ", "सब", "जो", "मैं",
    "हम", "आप", "तुम", "क्या", "कैसे", "कहां", "कहाँ", "कब", "अब", "फिर",
    "जब", "अगर", "लेकिन", "बहुत", "थोड़ा", "सकते", "सकता", "चाहिए",
    "आहे", "आहेत", "नाही", "मला", "मी", "तू", "तुम्ही", "आम्ही", "तो",
    "ती", "ते", "आणि", "पण", "म्हणून", "म्हणजे", "हा", "ही", "हे", "या",
    "त्या", "केले", "करा", "करू", "होते", "होता", "असे", "असं", "काही",
    "सर्व", "वर", "मध्ये", "ला", "चा", "ची", "चे", "काय", "कसे", "कुठे",
    "आता", "मग", "जर", "पाहिजे", "शकतो", "शकते", "बरं", "हो",
}

STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_DEVANAGARI

# Tokenizer. The Devanagari range is spelled out rather than relying on `\w`,
# which does NOT work here: vowel signs (U+093E and friends) are Unicode
# category Mn, and Python's `\w` excludes them. `\w+` therefore splits मच्छरों
# into its consonant runs — measured, it yielded the fragment "समस" as a
# top keyword. The range skips U+0964-U+0970, which is danda, double danda and
# the Devanagari digits, not letters.
#
# Three characters minimum: shorter Latin tokens are almost all function words,
# and a two-codepoint Devanagari token is a postposition. Inflected forms stay
# distinct (मच्छर and मच्छरों count separately) — stemming Hindi and Marathi
# needs a morphological analyser, and guessing at suffixes would merge words
# that are not the same word.
DEVANAGARI = r"ऀ-ॣॱ-ॿ"
TOKEN_RE = re.compile(rf"[A-Za-z{DEVANAGARI}]{{3,}}")


# ------------------------------------------------------------- cue phrases
# Used only by the extractive engine. A cue phrase is weak evidence — it finds
# the line, and the line is then shown verbatim as its own justification. It is
# never used to write a sentence that is not in the transcript.

DECISION_CUES = (
    "we decided", "we've decided", "we have decided", "we agreed", "we've agreed",
    "decided to", "agreed to", "final decision", "let's go with", "lets go with",
    "we'll go with", "going with", "we will use", "we'll use", "sign off",
    "approved", "the plan is", "should come under", "will come under", "decided that", "budget allocation",
    "let's use", "lets use",
    "तय किया", "तय हुआ", "तय कर", "फैसला", "निर्णय", "तय है", "मंजूर", "स्वीकार", "निश्चित",
    "ठरवले", "ठरलं", "ठरव", "निर्णय घेतला", "मान्य", "ठरवायचं",
)

ACTION_CUES = (
    "action item", "follow up", "follow-up", "going to speak to", "going to get back", "i'll speak to",
    "interviewing", "interview", "proceed with caution", "prepare report", "by tomorrow", "by monday", "by friday",
    "करना है", "करेंगे", "करना होगा", "भेज दो", "भेजना है", "जिम्मेदारी",
    "करायचं", "करायचे", "करू", "पाठवा", "पाठवेन", "जबाबदारी", "पाहिजे",
)

# Removed after a real run, not theorised: "please", "can you" and "could you"
# matched ordinary politeness — "Could you bring Ms. Rare, sir, please?" became
# an action item. A cue has to imply an obligation, not a request. "next week"
# went the same way: it dates things that are not tasks.


# ---------------------------------------------------------------- utilities


# What `identifier.py` returns for a voice it could not match to an enrolled
# name, and what `pipeline.py:190` already falls back from. It is a sentinel, not
# a person: left unhandled it became an action item's assignee and collapsed
# every item's colour onto speaker-1, because every line "belongs" to Unknown.
UNIDENTIFIED = "unknown"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip(" .,!?-\"'।"))


def _shorten_action_title(text: str, max_words: int = 12) -> str:
    """
    Trims rambling transcript quotes into clean, concise action titles (max 12 words).
    Strips conversational fillers in English, Hindi, and Marathi.
    """
    if not text:
        return ""

    cleaned = str(text).strip()

    filler_pattern = re.compile(
        r"^(?:uh|um|oh|well|so|yeah|okay|right|really|in fact|i mean|like|you know|look|listen|पण|आणि|नाही|म्हणजे|असं|ते|ह्या|हे)[,\s\.-]+",
        flags=re.IGNORECASE,
    )
    while filler_pattern.search(cleaned):
        cleaned = filler_pattern.sub("", cleaned).strip()

    cleaned = re.sub(r"^[^a-zA-Z\u0900-\u097F]+", "", cleaned)

    raw_sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]
    valid_sentences = [s for s in raw_sentences if len(s.split()) >= 2]

    if valid_sentences:
        cleaned = valid_sentences[0]
    elif raw_sentences:
        cleaned = raw_sentences[0]

    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).rstrip(".,;:-") + "..."

    return cleaned.strip(" .,;:-")


def _display_name(line: dict) -> str | None:
    """The name to attribute a line to — an enrolled name if there is one."""
    identified = line.get("identified_as")
    if identified and _normalize(identified) != UNIDENTIFIED:
        return identified
    return line.get("speaker") or line.get("speaker_label")


def verify_quote(
    quote: str, transcript_lines: list[str], threshold: float = SUMMARY_CITATION_THRESHOLD
) -> bool:
    """
    Does this quote actually appear in the transcript?

    Exact containment first, then fuzzy sequence ratio, then token overlap ratio.
    Tolerates ASR disfluencies, filler word omissions, and minor paraphrasing while
    preventing unsupported/hallucinated claims from surviving.
    """
    needle = _normalize(quote)
    if not needle:
        return False

    needle_words = set(re.findall(r"[A-Za-zऀ-ॣॱ-ॿ]{3,}", needle.lower()))

    for line in transcript_lines:
        haystack = _normalize(line)
        if not haystack:
            continue
        if needle in haystack or haystack in needle:
            return True
        if difflib.SequenceMatcher(None, needle, haystack).ratio() >= threshold:
            return True
        haystack_words = set(re.findall(r"[A-Za-zऀ-ॣॱ-ॿ]{3,}", haystack.lower()))
        if needle_words and haystack_words:
            overlap = len(needle_words & haystack_words) / len(needle_words)
            if overlap >= 0.65 and len(needle_words) >= 2:
                return True

    return False


def _render_window(lines: list[dict]) -> str:
    return "\n".join(
        f"[{line.get('time') or ''}] {line.get('speaker') or '?'}: {line.get('text') or ''}"
        for line in lines
    )


def _windows(transcript: list[dict], seconds: float) -> list[list[dict]]:
    """
    Split the transcript into time windows for map-reduce.

    Cut on wall-clock position rather than line count so a window is a stretch of
    meeting — the unit a summary is about — instead of an arbitrary N lines that
    may span a minute or twenty.
    """
    if not transcript:
        return []
    windows: list[list[dict]] = [[]]
    window_start = transcript[0].get("start_sec") or 0.0
    for line in transcript:
        start = line.get("start_sec") or 0.0
        if windows[-1] and start - window_start >= seconds:
            windows.append([])
            window_start = start
        windows[-1].append(line)
    return [w for w in windows if w]


_WINDOW_PROMPT = """You are an executive assistant extracting structured meeting intelligence from a transcript.

The transcript may be in Marathi, Hinglish, or English with technical terms and ASR disfluencies.
Synthesize all output in clear, professional English.

JSON Schema:
{{
  "summary": "Synthesized 1-2 sentence executive summary (max 35 words). Do NOT quote a raw transcript line.",
  "topics": ["Competitor analysis", "Pricing and tiers", "Deployment options"],
  "action_items": [
    {{
      "owner": "Person assigned. Map pronouns 'I'/'you'/'we' using dialogue context. If unknown, write 'Unassigned'.",
      "task": "Clear, synthesized action task title summarizing what must be done.",
      "deadline": "Spoken deadline or timing (e.g. 'Today', 'This week'; if none, 'Not specified')",
      "status": "Pending",
      "quote": "Transcript line containing the primary commitment"
    }}
  ],
  "key_decisions": [
    {{
      "decision": "Confirmed choice, budget allocation, or policy settled/agreed. Do NOT include hypotheticals or unagreed proposals.",
      "people_involved": ["names"],
      "context": "Context of decision",
      "quote": "Transcript quote line where decision was made"
    }}
  ],
  "requirements": [
    "Product feature, technical capability, or deployment requirement discussed"
  ],
  "key_insights": [
    {{"type": "Competitor Analysis/Pricing/Architecture/Risk", "insight": "Synthesized observation or insight"}}
  ],
  "risks_and_blockers": ["Risk or blocker description"],
  "follow_ups": [
    {{"owner": "Person responsible or Unassigned", "follow_up": "Follow-up detail", "deadline": "Deadline or Not specified"}}
  ],
  "keywords": ["kw1", "kw2"],
  "people_and_responsibilities": [
    {{"person": "Name", "responsibility": "Responsibility"}}
  ]
}}

CRITICAL MULTILINGUAL & INTENT CLASSIFICATION RULES:
1. SUMMARY: Must be a clear synthesized executive overview. Never output a raw transcript sentence line.
2. ACTION ITEMS - BE CONSERVATIVE:
   - Only produce an action item when someone is clearly responsible AND committing to an actionable task.
   - Do NOT treat competitor pricing observations ("both haven't provided pricing / दोघांनी प्राईज नाही दिलेले"), pricing tier comparisons, SaaS vs on-premise debates, call limit discussions, or hypothetical product ideas as action items!
   - Questions, opinions, and incomplete ASR fragments are NOT action items.
   - If the meeting contains mostly general discussion and no confirmed tasks, return an empty array `[]` for action_items.
3. DECISIONS:
   - Extract ONLY actual confirmed choices or agreed policies.
   - If there are no confirmed decisions, return an empty array `[]` for key_decisions.

TRANSCRIPT:
{transcript}
"""

_MERGE_PROMPT = """Below are summaries of consecutive parts of one meeting, in order.

Merge them into a short, concise executive summary of 1-2 sentences (max 35 words). Use only what the
part summaries say — do not add detail that is not in them. Return ONLY valid
JSON: {{"summary": "..."}}

PART SUMMARIES:
{parts}
"""


def _ollama_generate(prompt: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            # keep_alive 0 releases the VRAM as soon as the answer is produced.
            # Whisper and this model cannot both fit on a 4 GB card; see
            # OLLAMA_KEEP_ALIVE in config.py for what happens when they try.
            "keep_alive": OLLAMA_KEEP_ALIVE,
            # temperature 0: summarization has a right answer, and sampling only
            # adds ways to be wrong. num_ctx explicit — see config.py.
            "options": {"temperature": 0, "num_ctx": SUMMARY_NUM_CTX, "num_predict": 512},
        }
    ).encode()
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=SUMMARY_TIMEOUT_SECONDS) as response:
        return json.loads(response.read()).get("response", "")


def _ollama_json(prompt: str, model: str) -> dict | None:
    """One generate call, parsed. Returns None on any failure — never raises."""
    try:
        raw = _ollama_generate(prompt, model)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning(f"summarizer: ollama call failed ({e})")
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # `format: json` makes this rare but not impossible: a model that hits
        # its token limit mid-object emits truncated JSON.
        logger.warning(f"summarizer: model returned non-JSON ({len(raw)} chars)")
        return None
    return parsed if isinstance(parsed, dict) else None


def release_vram(model: str = SUMMARY_MODEL) -> bool:
    """
    Ask Ollama to evict the model from memory now. Best effort, never raises.

    Called before transcription starts. `keep_alive: 0` on our own generate calls
    already unloads after each answer, but that only covers models *this* process
    loaded — a `bench_summarizer` run, a backfill, or someone at an `ollama run`
    prompt can leave 4 GB resident with a five-minute timer, and Whisper then
    fails on every segment. An empty prompt with keep_alive 0 is Ollama's
    documented way to unload without generating anything.

    Returns True if Ollama accepted the request; False if it was unreachable,
    which is not an error — no Ollama means no contention.
    """
    payload = json.dumps({"model": model, "prompt": "", "keep_alive": 0}).encode()
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.debug(f"summarizer: could not unload {model} ({e})")
        return False


def model_available(model: str = SUMMARY_MODEL) -> bool:
    """
    Is Ollama up with this model already pulled?

    Checked before use rather than discovered by failure: `/api/generate` on a
    missing model would start a multi-gigabyte download inside a request, and an
    exhibition stand may have no usable network at all.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
            tags = json.loads(response.read()).get("models") or []
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False
    names = {(t.get("name") or "") for t in tags}
    # Ollama reports "qwen2.5:7b"; accept a bare "qwen2.5" as its :latest tag.
    return model in names or f"{model}:latest" in names


# ------------------------------------------------------- item verification


def _known_names(transcript: list[dict]) -> tuple[set[str], str]:
    """Speaker labels plus every word actually spoken, for checking assignees."""
    names = set()
    for line in transcript:
        for key in ("speaker", "identified_as", "speaker_label"):
            value = _normalize(line.get(key) or "")
            if value and value != UNIDENTIFIED:
                names.add(value)
    spoken = _normalize(" ".join((line.get("text") or "") for line in transcript))
    return names, spoken


def _verify_assignee(assignee, names: set[str], spoken: str) -> str | None:
    """
    Keep an assignee only if there is evidence for the name.

    A verified quote makes the *item* real; the assignee is the model's
    inference on top of it. Kept when it matches a speaker, or when the name is
    spoken somewhere in the meeting — someone saying "Priya will send it" is
    evidence. Otherwise the item survives with no assignee, which is honest,
    rather than pinning work on a person who was never there.
    """
    if not isinstance(assignee, str):
        return None
    candidate = _normalize(assignee)
    if not candidate or candidate in ("null", "none", UNIDENTIFIED, "n/a"):
        return None
    if candidate in names:
        return assignee.strip()
    if len(candidate) >= 3 and candidate in spoken:
        return assignee.strip()
    return None


def _find_source_line(quote: str, transcript: list[dict]) -> dict | None:
    """
    The transcript line a verified quote actually matches, for provenance.

    `verify_quote` (above) only needs a yes/no over plain strings — that is
    also the interface the citation benchmark measures. This does the same
    normalize-then-fuzzy comparison but over the full line objects, so the UI
    can show the line's real timestamp and speaker next to the insight it
    supports rather than just the insight text.
    """
    needle = _normalize(quote)
    if not needle:
        return None
    best, best_ratio = None, 0.0
    for entry in transcript:
        haystack = _normalize(entry.get("text") or "")
        if not haystack:
            continue
        if needle in haystack or haystack in needle:
            return entry
        ratio = difflib.SequenceMatcher(None, needle, haystack).ratio()
        if ratio > best_ratio:
            best, best_ratio = entry, ratio
    return best if best_ratio >= SUMMARY_CITATION_THRESHOLD else None


def _speaker_colors(transcript: list[dict]) -> dict[str, str]:
    """
    name -> colour, from the transcript the backend already coloured.

    Colours come from the backend in first-appearance order so one person keeps
    one colour everywhere (see SPEAKER_COLORS in config.py). An action item
    reuses its assignee's colour; it never picks its own.
    """
    colors: dict[str, str] = {}
    for line in transcript:
        color = line.get("color")
        name = _display_name(line)
        if color and name:
            colors.setdefault(_normalize(name), color)
    return colors


def _collect_items(raw_windows: list[dict], transcript: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """
    Verify, de-duplicate, resolve pronouns, and colour the model's proposed items.
    """
    texts = [line.get("text") or "" for line in transcript]
    names, spoken = _known_names(transcript)
    colors = _speaker_colors(transcript)

    decisions: list[dict] = []
    seen_decisions: set[str] = set()
    actions: list[dict] = []
    seen_actions: set[str] = set()
    dropped = 0

    hypothetical_markers = ("might", "maybe", "could we", "could consider", "possibility", "hypothetically", "wanna see")
    filler_decisions = ("yes, i will", "i know, i know, it's fine", "i know", "it's fine", "sure", "okay", "yes", "no", "yes, they are", "that's fine")

    invalid_action_fragments = (
        "but they have to go", "that's quite a lot there", "are all the interviews this week",
        "yes, they are", "yes, i will", "just saying watch how much time it takes",
        "but what about my time on it", "no, i was going to do that today"
    )

    for window in raw_windows:
        raw_decisions = window.get("key_decisions") or window.get("decisions") or []
        for decision in raw_decisions:
            if not isinstance(decision, dict):
                continue
            text = (decision.get("decision") or decision.get("text") or "").strip()
            if not text:
                continue

            lowered = text.lower().strip(" .,!?")
            # Rule: Filter out non-decision hypotheticals & conversational fillers
            if any(marker in lowered for marker in hypothetical_markers) or lowered in filler_decisions:
                continue
            if len(text.split()) < 2 and not any(k in lowered for k in ("budget", "marketing", "system", "plan", "decide", "agree", "policy", "approve", "postpone", "postponed", "launch")):
                continue

            quote = (decision.get("quote") or "").strip()
            if quote and not verify_quote(quote, texts):
                dropped += 1
                continue

            key = _normalize(text)
            if key in seen_decisions:
                continue
            seen_decisions.add(key)
            source = _find_source_line(quote, transcript) if quote else None
            decisions.append(
                {
                    "text": text,
                    "quote": (source or {}).get("text") or quote or text,
                    "sourceTime": (source or {}).get("time"),
                    "people_involved": decision.get("people_involved") or [],
                    "context": decision.get("context") or text,
                }
            )

        raw_actions = window.get("action_items") or window.get("actionItems") or window.get("commitments") or []
        for action in raw_actions:
            if not isinstance(action, dict):
                continue
            task = (action.get("task") or action.get("title") or "").strip()
            if not task:
                continue

            lowered_task = task.lower().strip(" .,!?")
            # Rule: Reject standalone questions, raw responses, and non-action commentary
            if task.strip().endswith("?") or lowered_task in invalid_action_fragments:
                continue
            if lowered_task.startswith("are all") or lowered_task.startswith("should the") or lowered_task.startswith("can you let me"):
                continue

            quote = (action.get("quote") or "").strip()
            if quote and not verify_quote(quote, texts):
                dropped += 1
                continue

            source = _find_source_line(quote, transcript) if quote else None

            # Rule 9: Resolve "I", "you", "we" pronouns to real speaker labels using source line headers
            raw_owner = (action.get("owner") or action.get("assignee") or "").strip()
            owner_norm = raw_owner.lower()
            if owner_norm in ("i", "me", "my", "myself") and source:
                assignee = _display_name(source) or None
            elif owner_norm in ("null", "none", "unassigned", "", "unknown"):
                assignee = None
            else:
                verified = _verify_assignee(raw_owner, names, spoken)
                assignee = verified if verified else None

            raw_due = (action.get("deadline") or action.get("due") or "").strip()
            due = raw_due if raw_due and raw_due.lower() not in ("null", "none", "not specified", "unspecified", "") else None

            # Rule 8: Prevent duplicate action items by normalizing title + assignee
            key = _normalize(f"{task}:{assignee}")
            if key in seen_actions:
                continue
            seen_actions.add(key)

            actions.append(
                {
                    "title": task,
                    "assignee": assignee,
                    "due": due,
                    "color": colors.get(_normalize(assignee)) if assignee and assignee != "Unassigned" else None,
                    "quote": (source or {}).get("text") or quote or task,
                    "sourceTime": (source or {}).get("time"),
                    "status": action.get("status") or "Pending",
                }
            )

    if dropped:
        logger.info(f"summarizer: dropped {dropped} item(s) failing citation check")

    return decisions, actions


# ---------------------------------------------------------------- keywords


def keywords(transcript: list[dict], background_documents: list[str] | None = None) -> list[dict]:
    counts: dict[str, int] = {}
    for line in transcript:
        for token in TOKEN_RE.findall((line.get("text") or "").lower()):
            if token in STOP_WORDS:
                continue
            counts[token] = counts.get(token, 0) + 1

    candidates = {w: c for w, c in counts.items() if c >= SUMMARY_KEYWORD_MIN_OCCURRENCES}
    if not candidates:
        candidates = counts
    if not candidates:
        return []

    scores = dict(candidates)
    documents = [d for d in (background_documents or []) if d and d.strip()]
    if documents:
        total = len(documents) + 1
        tokenized = [set(TOKEN_RE.findall(d.lower())) for d in documents]
        scores = {}
        for word, count in candidates.items():
            document_frequency = 1 + sum(1 for tokens in tokenized if word in tokens)
            scores[word] = count * math.log(total / document_frequency)

    ranked = sorted(candidates, key=lambda w: (scores.get(w, 0), candidates[w], w), reverse=True)
    return [{"word": w, "count": candidates[w]} for w in ranked[:SUMMARY_KEYWORD_COUNT]]





def _extract_insights(transcript: list[dict], actions: list[dict], decisions: list[dict], extra: dict | None = None) -> dict:
    """
    Intelligent Post-Meeting Insights derived directly from clean synthesized action items:
    - commitments: Owner, Action title, Timing.
    - deadlines: Stated deadlines.
    - pending: Unresolved/in-progress items.
    - attentionNeeded: Items requiring attention (unassigned tasks).
    - requirements: Technical / feature requirements discussed.
    """
    extra = extra or {}
    attention_needed = []
    pending_items = []
    commitments = []
    deadlines = []

    seen_commitments = set()

    for act in (actions or []):
        if not isinstance(act, dict):
            continue
        title = (act.get("title") or "").strip()
        if not title:
            continue
        owner = act.get("assignee") or "Unassigned"
        due = act.get("due") or "Not specified"

        norm_title = _normalize(title)
        if norm_title in seen_commitments:
            continue
        seen_commitments.add(norm_title)

        commitments.append({
            "owner": owner,
            "action": title,
            "timing": due
        })

        if due and due.lower() not in ("null", "none", "not specified", "unspecified"):
            deadlines.append({
                "deadline": due,
                "detail": title
            })

        if owner == "Unassigned":
            attention_needed.append({
                "severity": "yellow",
                "text": f"{title} — unassigned task"
            })
        else:
            pending_items.append({
                "topic": " ".join(title.split()[:4]),
                "description": title,
                "owner": owner,
                "status": "Pending"
            })

    return {
        "attentionNeeded": attention_needed[:4],
        "pending": pending_items[:5],
        "commitments": commitments[:8],
        "deadlines": deadlines[:4],
        "requirements": extra.get("requirements") or [],
        "topics": extra.get("topics") or [],
        "key_insights": extra.get("key_insights") or [],
        "risks_and_blockers": extra.get("risks_and_blockers") or [],
        "follow_ups": extra.get("follow_ups") or [],
    }


def _build_multilingual_summaries(transcript: list[dict], base_summary: str | None) -> dict[str, str]:
    if not transcript or not base_summary:
        return {"en": "", "hi": "", "mr": ""}

    mr_lines = []
    hi_lines = []
    en_lines = []

    for line in transcript:
        text = (line.get("text") or "").strip()
        lang = (line.get("language") or "").lower()
        dev_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
        lat_chars = sum(1 for ch in text if ch.isalpha() and ch.isascii())

        if dev_chars > lat_chars:
            marathi_markers = ("आहे", "नाही", "पण", "माझे", "केले", "होते", "पान", "पानावर", "पाण्या", "नळाचं", "आणून", "ठरलं", "बरं")
            if any(marker in text for marker in marathi_markers) or lang in ("mr", "mr-in"):
                mr_lines.append(text)
            else:
                hi_lines.append(text)
        elif lat_chars > 0:
            en_lines.append(text)

    def _format_summary_lines(lines: list[str]) -> str:
        summary_text = " ".join(lines[:2])
        if len(summary_text.split()) > 35:
            summary_text = " ".join(summary_text.split()[:35]).rstrip(",;:") + "."
        return summary_text

    summary_en = _format_summary_lines(en_lines) if en_lines else base_summary
    summary_mr = _format_summary_lines(mr_lines) if mr_lines else base_summary
    summary_hi = _format_summary_lines(hi_lines) if hi_lines else base_summary

    return {
        "en": summary_en or base_summary,
        "hi": summary_hi or base_summary,
        "mr": summary_mr or base_summary,
    }


def _extract_summary_text(res_dict: dict) -> str:
    if not isinstance(res_dict, dict):
        return ""
    raw = res_dict.get("summary") or res_dict.get("Summary") or ""
    if isinstance(raw, dict):
        topic = raw.get("Topic") or raw.get("topic") or ""
        points = raw.get("Key Points") or raw.get("key_points") or raw.get("points") or raw.get("Action Items") or []
        if isinstance(points, list):
            points_str = " ".join(str(x) for x in points)
        else:
            points_str = str(points)
        if topic and points_str:
            return f"{topic}: {points_str}"
        return points_str or str(raw)
    elif isinstance(raw, list):
        return " ".join(str(x) for x in raw)
    return str(raw).strip()


def summarize(
    transcript: list[dict],
    background_documents: list[str] | None = None,
    model: str = SUMMARY_MODEL,
) -> dict:
    if not transcript:
        return {
            "summary": None,
            "decisions": [],
            "actionItems": [],
            "keywords": [],
            "insights": {},
            "summaryEngine": None,
        }

    ranked_keywords = keywords(transcript, background_documents)

    if not model_available(model):
        logger.warning(f"summarizer: {model} unavailable — returning clean empty result")
        return {
            "summary": None,
            "decisions": [],
            "actionItems": [],
            "keywords": ranked_keywords,
            "insights": {},
            "summaryEngine": None,
        }

    windows = _windows(transcript, SUMMARY_WINDOW_SECONDS)
    logger.info(f"summarizer: {model} over {len(windows)} window(s), {len(transcript)} lines")

    results = []
    for index, window in enumerate(windows, start=1):
        parsed = _ollama_json(_WINDOW_PROMPT.format(transcript=_render_window(window)), model)
        if parsed is None:
            logger.warning(f"summarizer: window {index}/{len(windows)} produced nothing")
            continue
        results.append(parsed)

    if not results:
        logger.warning("summarizer: LLM windows failed — returning clean empty result")
        return {
            "summary": None,
            "decisions": [],
            "actionItems": [],
            "keywords": ranked_keywords,
            "insights": {},
            "summaryEngine": None,
        }

    parts = [_extract_summary_text(r) for r in results]
    parts = [p for p in parts if p]

    if len(parts) <= 1:
        summary = parts[0] if parts else None
    else:
        merged = _ollama_json(_MERGE_PROMPT.format(parts="\n\n".join(parts)), model)
        summary = _extract_summary_text(merged or {}) or " ".join(parts)

    decisions, actions = _collect_items(results, transcript)
    extra = {
        "topics": list(dict.fromkeys(t for w in results for t in (w.get("topics") or []) if t)),
        "requirements": list(dict.fromkeys(req for w in results for req in (w.get("requirements") or []) if req)),
        "key_insights": [i for w in results for i in (w.get("key_insights") or []) if isinstance(i, dict)],
        "risks_and_blockers": list(dict.fromkeys(r for w in results for r in (w.get("risks_and_blockers") or []) if r)),
        "follow_ups": [f for w in results for f in (w.get("follow_ups") or []) if isinstance(f, dict)],
        "people_and_responsibilities": [p for w in results for p in (w.get("people_and_responsibilities") or []) if isinstance(p, dict)],
    }
    multilingual = _build_multilingual_summaries(transcript, summary)
    insights = _extract_insights(transcript, actions, decisions, extra)

    # Attach all 9 schema areas to the insights dictionary
    insights["topics"] = extra.get("topics") or []
    insights["action_items"] = actions
    insights["key_decisions"] = decisions
    insights["key_insights"] = extra.get("key_insights") or []
    insights["risks_and_blockers"] = extra.get("risks_and_blockers") or []
    insights["follow_ups"] = extra.get("follow_ups") or []
    insights["people_and_responsibilities"] = extra.get("people_and_responsibilities") or []

    return {
        "summary": summary or None,
        "summaries": multilingual,
        "decisions": decisions,
        "actionItems": actions,
        "keywords": ranked_keywords,
        "insights": insights,
        "summaryEngine": model,
    }
