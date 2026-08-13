"""
Summarize meetings that were transcribed before the summarization stage existed.

    python3 -m tools.backfill_summaries                  # every meeting missing a summary
    python3 -m tools.backfill_summaries MTG-abc123        # named meetings only
    python3 -m tools.backfill_summaries --force           # redo ones already summarized
    python3 -m tools.backfill_summaries --dry-run         # print, write nothing
    python3 -m tools.backfill_summaries --keywords-only   # recount keywords, no model call

`--keywords-only` exists because keywords need no LLM — they are counted from the
transcript — while the stop-word list is empirical and grows every time a new
meeting exposes filler in the top twelve. Re-ranking every stored meeting after
editing it costs seconds, where `--force` would spend minutes regenerating
summaries that would come back identical (temperature is 0).

No audio is touched and nothing is re-transcribed: this reads the stored
transcript and runs the same `api._generate_summary` the pipeline calls, so a
backfilled meeting is indistinguishable from a freshly processed one.

Skips meetings whose status is not "Completed" — a meeting still processing will
be summarized by the pipeline itself when it finishes, and writing to it from
here would race the worker.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import _generate_summary  # noqa: E402
from config import SUMMARY_KEYWORD_IDF_MEETINGS  # noqa: E402
from db import repository as store  # noqa: E402
from models import summarizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_ids", nargs="*", help="default: all meetings missing a summary")
    parser.add_argument("--force", action="store_true", help="re-summarize even if one exists")
    parser.add_argument("--dry-run", action="store_true", help="print the result, write nothing")
    parser.add_argument(
        "--keywords-only", action="store_true", help="recount keywords; no model call"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.meeting_ids:
        meetings = []
        for meeting_id in args.meeting_ids:
            meeting = store.get_meeting(meeting_id)
            if not meeting:
                raise SystemExit(f"no meeting {meeting_id}")
            meetings.append(meeting)
    else:
        meetings = store.list_meetings()

    for meeting in meetings:
        label = f"{meeting['id']} \"{meeting['title']}\""

        if meeting["status"] != "Completed":
            print(f"skip {label}: status is {meeting['status']}")
            continue
        if not meeting.get("transcript"):
            print(f"skip {label}: no transcript to summarize")
            continue
        if meeting.get("summary") and not (args.force or args.keywords_only):
            print(f"skip {label}: already summarized (--force to redo)")
            continue

        print(f"\n=== {label} — {len(meeting['transcript'])} lines")

        if args.keywords_only:
            # Same IDF corpus api.py builds: the other stored meetings.
            background = [
                " ".join(l.get("text") or "" for l in (other.get("transcript") or []))
                for other in store.list_meetings()[:SUMMARY_KEYWORD_IDF_MEETINGS]
                if other["id"] != meeting["id"]
            ]
            fields = {
                "keywords": summarizer.keywords(
                    meeting["transcript"], background_documents=background
                )
            }
        else:
            fields = _generate_summary(meeting["id"], meeting["transcript"])
        if not fields:
            print("  summarization returned nothing; left untouched")
            continue

        if not args.keywords_only:
            print(f"  engine    {fields.get('summaryEngine')}")
            print(f"  summary   {fields.get('summary')}")
            for decision in fields.get("decisions") or []:
                print(f"  decision  {decision['text']}")
            for action in fields.get("actionItems") or []:
                owner = action.get("assignee") or "unassigned"
                print(f"  action    {action['title']}  ({owner})")
        print(f"  keywords  {', '.join(k['word'] for k in fields.get('keywords') or [])}")

        if args.dry_run:
            print("  (dry run — nothing written)")
            continue

        store.update_meeting(meeting["id"], **fields)
        print("  written")


if __name__ == "__main__":
    main()
