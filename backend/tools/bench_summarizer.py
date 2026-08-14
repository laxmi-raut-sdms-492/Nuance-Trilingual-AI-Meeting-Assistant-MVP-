"""
Benchmark local LLMs at summarizing a real stored meeting transcript.

    python3 -m tools.bench_summarizer MTG-1c443bab406f --models qwen2.5:3b qwen2.5:7b
    python3 -m tools.bench_summarizer MTG-1c443bab406f MTG-de4df0ad891a --models qwen2.5:7b

This drives `models/summarizer.py` itself rather than re-implementing the
prompt and the citation check. That is the point: a benchmark with its own copy
of the prompt measures a system nobody ships, and the two drift the moment
either is tuned. Everything reported here — the window split, the quote
verification, the assignee verification, the keyword counts — is exactly what
the API will persist.

What to read in the output:

- **dropped** is the headline. It counts items the model proposed whose quote
  could not be found in the transcript. A model with a high drop rate is
  fabricating and being caught; one with zero is either honest or silent, and
  the item counts tell you which.
- **time** is wall clock for the whole stage, map-reduce included. Prompt
  ingestion on CPU dominates, so it scales with transcript length, not with the
  length of the summary.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import repository as repo  # noqa: E402
from models import summarizer  # noqa: E402


def run(model: str, meeting: dict, background: list[str]) -> dict | None:
    transcript = meeting.get("transcript") or []
    if not transcript:
        print(f"  {meeting['id']} has no transcript")
        return None

    if not summarizer.model_available(model):
        print(f"\n{model}: not available in Ollama — skipped (pull it first)")
        return None

    windows = summarizer._windows(transcript, summarizer.SUMMARY_WINDOW_SECONDS)
    approximate_tokens = sum(len(summarizer._render_window(w)) for w in windows) // 4

    print(f"\n{'=' * 78}")
    print(
        f"{model}  ({len(transcript)} lines, {len(windows)} window(s), "
        f"~{approximate_tokens} prompt tokens, num_ctx={summarizer.SUMMARY_NUM_CTX})"
    )
    print("=" * 78)

    # Count the drops rather than only logging them: the ratio of proposed to
    # verified items is the whole measurement.
    proposed = {"decisions": 0, "actions": 0}
    original_collect = summarizer._collect_items

    def counting_collect(raw_windows, transcript_lines):
        for window in raw_windows:
            proposed["decisions"] += len(window.get("decisions") or [])
            proposed["actions"] += len(window.get("action_items") or [])
        return original_collect(raw_windows, transcript_lines)

    summarizer._collect_items = counting_collect
    try:
        started = time.monotonic()
        result = summarizer.summarize(transcript, background_documents=background, model=model)
        elapsed = time.monotonic() - started
    finally:
        summarizer._collect_items = original_collect

    engine = result.get("summaryEngine")
    if engine != model:
        # summarize() degrades to the extractive engine rather than raising. If
        # that happened, the numbers below are not this model's numbers.
        print(f"  !! fell back to '{engine}' — the model produced nothing usable")

    decisions = result.get("decisions") or []
    actions = result.get("actionItems") or []
    summary = result.get("summary") or ""
    dropped = (proposed["decisions"] - len(decisions)) + (proposed["actions"] - len(actions))

    print(f"  time            {elapsed:.1f}s")
    print(f"  summary         {len(summary.split())} words")
    print(f"    {summary[:400]}")
    print(f"  decisions       {len(decisions)} kept / {proposed['decisions']} proposed")
    for d in decisions:
        print(f"    - {d['text'][:88]}")
    print(f"  action items    {len(actions)} kept / {proposed['actions']} proposed")
    for a in actions:
        owner = a.get("assignee") or "unassigned"
        due = f", due {a['due']}" if a.get("due") else ""
        print(f"    - {a['title'][:70]}  ({owner}{due})")
    print(f"  dropped         {dropped} item(s) whose citation was not in the transcript")
    print(f"  keywords        {', '.join(k['word'] for k in (result.get('keywords') or []))}")

    return {
        "model": model,
        "engine": engine,
        "seconds": elapsed,
        "summary_words": len(summary.split()),
        "decisions": len(decisions),
        "decisions_proposed": proposed["decisions"],
        "actions": len(actions),
        "actions_proposed": proposed["actions"],
        "dropped": dropped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_ids", nargs="+")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:3b", "qwen2.5:7b"])
    parser.add_argument("--quiet", action="store_true", help="hide the summarizer's own logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    meetings = []
    for meeting_id in args.meeting_ids:
        meeting = repo.get_meeting(meeting_id)
        if not meeting:
            raise SystemExit(f"no meeting {meeting_id}")
        meetings.append(meeting)

    # Same IDF corpus the API builds: the other stored meetings.
    all_meetings = repo.list_meetings()

    rows = []
    for meeting in meetings:
        print(f"\n### {meeting['title']} ({meeting['id']}) — {meeting.get('duration')}")
        background = [
            " ".join(l.get("text") or "" for l in (other.get("transcript") or []))
            for other in all_meetings
            if other["id"] != meeting["id"]
        ]
        for model in args.models:
            row = run(model, meeting, background)
            if row:
                rows.append({**row, "meeting": meeting["title"]})

    if rows:
        print(f"\n{'meeting':<12} {'model':<14} {'time':>8} {'summary':>8} {'dec':>9} {'act':>9} {'dropped':>8}")
        print("-" * 76)
        for r in rows:
            print(
                f"{r['meeting'][:11]:<12} {r['model']:<14} {r['seconds']:>7.1f}s "
                f"{r['summary_words']:>7}w {r['decisions']:>4}/{r['decisions_proposed']:<4} "
                f"{r['actions']:>4}/{r['actions_proposed']:<4} {r['dropped']:>8}"
            )
        print("\nkept/proposed. dropped = citation not found in the transcript.")


if __name__ == "__main__":
    main()
