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
    "approved", "the plan is",
    "तय किया", "तय हुआ", "तय कर", "फैसला", "निर्णय", "तय है", "मंजूर", "स्वीकार",
    "ठरवले", "ठरलं", "ठरव", "निर्णय घेतला", "मान्य", "निश्चित", "ठरवायचं",
)

ACTION_CUES = (
    "i will", "i'll", "we need to", "we must", "you should", "you need to",
    "action item", "follow up", "follow-up", "take care of", "let's make sure",
    "by tomorrow", "by monday", "by friday", "deadline", "assign",
    "करना है", "करेंगे", "करना होगा", "कर दो", "कर दें", "भेज दो", "भेजना है", "जिम्मेदारी",
    "करायचं", "करायचे", "करू", "पाठवा", "पाठवेन", "जबाबदारी", "पाहिजे", "करा",
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

    Exact containment first, then a fuzzy ratio. Not an equality test: a model
    that repairs a typo or drops a filler word is still citing a real line, and
    rejecting that discards good items. Anything below the threshold is treated
    as invented.

    Also the citation check used by `tools/bench_summarizer.py`, deliberately —
    the benchmark must measure the same bar production enforces, or its numbers
    describe a system nobody ships.
    """
    needle = _normalize(quote)
    if not needle:
        return False
    for line in transcript_lines:
        haystack = _normalize(line)
        if not haystack:
            continue
        if needle in haystack or haystack in needle:
            return True
        if difflib.SequenceMatcher(None, needle, haystack).ratio() >= threshold:
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


# ------------------------------------------------------------------- Ollama

# Defines what qualifies BEFORE warning against invention. Measured, both ways,
# on both clean meetings: a prompt built only from prohibitions made the model
# withhold real items. Warnings-only returned 0 decisions and 0 action items on
# a meeting that contains three, and adding the positive definition recovered
# them with zero fabricated citations. Precision was never the problem here —
# nothing was being dropped by the citation check, because nothing was being
# proposed. If this is edited, re-measure both meetings; the failure mode is
# silent, and an empty panel looks identical to an honest one.
_WINDOW_PROMPT = """You are summarizing part of a meeting transcript. Work ONLY from the lines given.

What counts as an action item: any point where someone commits to doing something, is asked to do something, or a task is named as needing to be done. It is still an action item if no date and no owner were stated.
What counts as a decision: any point where a choice is settled, agreed, approved, or stated as the plan going forward.

Rules:
- Every action item and every decision MUST quote, word for word, the transcript line it came from. Copy that line exactly.
- Report what is there. If this part genuinely contains none, return an empty list — but do not withhold a real item because it lacks an owner or a date.
- Do not invent anything that was not said out loud. Do not guess a due date; leave it null unless a date was spoken.
- The transcript may mix English, Hindi and Marathi. Write the summary in the same language as most of this part (English, Hindi, or Marathi). Keep quotes in their original language and script.

Return ONLY valid JSON, no prose around it, in this exact shape:
{{
  "summary": "2-4 sentence summary of this part",
  "action_items": [{{"title": "what must be done", "assignee": "who, or null", "due": "only if a date was stated, else null", "quote": "the exact transcript line"}}],
  "decisions": [{{"text": "what was decided", "quote": "the exact transcript line"}}]
}}

TRANSCRIPT:
{transcript}
"""

_MERGE_PROMPT = """Below are summaries of consecutive parts of one meeting, in order.

Merge them into a single executive summary of 2-4 sentences. Use only what the
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
            "options": {"temperature": 0, "num_ctx": SUMMARY_NUM_CTX},
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


def _collect_items(raw_windows: list[dict], transcript: list[dict]) -> tuple[list[str], list[dict]]:
    """Verify, de-duplicate and colour the model's proposed items."""
    texts = [line.get("text") or "" for line in transcript]
    names, spoken = _known_names(transcript)
    colors = _speaker_colors(transcript)

    decisions: list[str] = []
    seen_decisions: set[str] = set()
    actions: list[dict] = []
    seen_actions: set[str] = set()
    dropped = 0

    for window in raw_windows:
        for decision in window.get("decisions") or []:
            if not isinstance(decision, dict):
                continue
            text = (decision.get("text") or "").strip()
            if not text:
                continue
            if not verify_quote(decision.get("quote") or "", texts):
                dropped += 1
                continue
            key = _normalize(text)
            if key in seen_decisions:
                continue
            seen_decisions.add(key)
            decisions.append(text)

        for action in window.get("action_items") or []:
            if not isinstance(action, dict):
                continue
            title = (action.get("title") or "").strip()
            if not title:
                continue
            if not verify_quote(action.get("quote") or "", texts):
                dropped += 1
                continue
            key = _normalize(title)
            if key in seen_actions:
                continue
            seen_actions.add(key)
            assignee = _verify_assignee(action.get("assignee"), names, spoken)
            due = action.get("due")
            actions.append(
                {
                    "title": title,
                    "assignee": assignee,
                    # Never normalised into a date. A stated "by Friday" is
                    # shown as said; anything else stays empty.
                    "due": due.strip() if isinstance(due, str) and due.strip().lower() not in ("", "null", "none") else None,
                    "color": colors.get(_normalize(assignee or "")) if assignee else None,
                }
            )

    if dropped:
        logger.info(f"summarizer: dropped {dropped} item(s) whose citation was not in the transcript")
    return decisions, actions


# ---------------------------------------------------------------- keywords


def keywords(transcript: list[dict], background_documents: list[str] | None = None) -> list[dict]:
    """
    Term frequency over the real transcript, ranked by TF-IDF when there is a
    corpus to compute IDF from.

    The corpus is other meetings' transcripts. IDF within a single document is
    meaningless — it would penalise exactly the words the meeting is about — so
    with nothing to compare against the ranking degrades to plain frequency,
    which is the correct single-document answer. `count` is the true number of
    occurrences in both cases; it is never a score.
    """
    counts: dict[str, int] = {}
    for line in transcript:
        for token in TOKEN_RE.findall((line.get("text") or "").lower()):
            if token in STOP_WORDS:
                continue
            counts[token] = counts.get(token, 0) + 1

    candidates = {w: c for w, c in counts.items() if c >= SUMMARY_KEYWORD_MIN_OCCURRENCES}
    # A short meeting may not say anything twice. Better to show single mentions
    # than an empty panel that implies extraction failed.
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


# -------------------------------------------------------- extractive engine


def _extractive(transcript: list[dict], ranked_keywords: list[dict]) -> dict:
    """
    Fallback that cannot fabricate, because it only ever copies.

    Summary = the transcript lines that carry the most of the meeting's own
    keywords, verbatim and in their original order. Decisions and action items
    = lines containing a cue phrase, also verbatim. Cue phrases are weak
    evidence, which is exactly why the line is shown as itself rather than
    paraphrased into a claim about what was decided.
    """
    weights = {k["word"]: k["count"] for k in ranked_keywords}

    scored = []
    for index, line in enumerate(transcript):
        text = (line.get("text") or "").strip()
        tokens = TOKEN_RE.findall(text.lower())
        # Six, not four: a transcript line is not a sentence, and short ones are
        # fragments ("which we need to be aware of.") that read as broken when
        # stitched into a summary.
        if len(tokens) < 6:
            continue
        # Length-normalised, or one rambling line wins every meeting.
        score = sum(weights.get(t, 0) for t in tokens) / (len(tokens) ** 0.5)
        scored.append((score, index, text))

    chosen = sorted(sorted(scored, reverse=True)[:4], key=lambda s: s[1])
    summary = " ".join(text for _, _, text in chosen) or None

    decisions: list[str] = []
    actions: list[dict] = []
    colors = _speaker_colors(transcript)
    for line in transcript:
        text = (line.get("text") or "").strip()
        # A cue inside a fragment is not a finding. "which we need to be aware
        # of." matched the obligation cue and read as a broken action item.
        if not text or len(TOKEN_RE.findall(text.lower())) < 5:
            continue
        lowered = text.lower()
        if any(cue in lowered for cue in DECISION_CUES) and len(decisions) < 8:
            decisions.append(text)
        elif any(cue in lowered for cue in ACTION_CUES) and len(actions) < 8:
            speaker = _display_name(line)
            actions.append(
                {
                    "title": text,
                    # The speaker of the line, not an inferred owner: this engine
                    # knows who talked and nothing more.
                    "assignee": speaker,
                    "due": None,
                    "color": colors.get(_normalize(speaker or "")) if speaker else None,
                }
            )

    return {
        "summary": summary,
        "decisions": decisions,
        "actionItems": actions,
        "keywords": ranked_keywords,
        "summaryEngine": "extractive",
    }


# -------------------------------------------------------------- entry point


def summarize(
    transcript: list[dict],
    background_documents: list[str] | None = None,
    model: str = SUMMARY_MODEL,
) -> dict:
    """
    Produce summary / decisions / actionItems / keywords for one meeting.

    Returns camelCase keys ready to hand straight to `repository.update_meeting`.
    Never raises: a failed summary must leave a completed transcript alone, so
    every failure path degrades to the extractive engine, and the worst case is
    empty fields plus the UI's honest empty states.
    """
    if not transcript:
        return {
            "summary": None,
            "decisions": [],
            "actionItems": [],
            "keywords": [],
            "summaryEngine": None,
        }

    ranked_keywords = keywords(transcript, background_documents)

    if not model_available(model):
        logger.info(f"summarizer: {model} unavailable — using the extractive engine")
        return _extractive(transcript, ranked_keywords)

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
        logger.warning("summarizer: every window failed — falling back to extractive")
        return _extractive(transcript, ranked_keywords)

    parts = [(r.get("summary") or "").strip() for r in results]
    parts = [p for p in parts if p]

    if len(parts) <= 1:
        summary = parts[0] if parts else None
    else:
        # Reduce step. If the merge call fails, concatenating the part summaries
        # is a worse summary but not a wrong one — every sentence still came
        # from a window of this meeting.
        merged = _ollama_json(_MERGE_PROMPT.format(parts="\n\n".join(parts)), model)
        summary = ((merged or {}).get("summary") or "").strip() or " ".join(parts)

    decisions, actions = _collect_items(results, transcript)

    return {
        "summary": summary or None,
        "decisions": decisions,
        "actionItems": actions,
        "keywords": ranked_keywords,
        "summaryEngine": model,
    }
