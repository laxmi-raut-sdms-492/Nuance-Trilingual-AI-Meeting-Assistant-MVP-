"""
Re-run the full pipeline over an already-uploaded meeting.

    python3 -m tools.reprocess MTG-abc123 MTG-def456
    python3 -m tools.reprocess --failed          # every meeting with status Failed

The audio is already on disk under `storage/audio/<meeting-id>/`, so a failed run
does not need re-uploading: that would burn the bytes again and mint a new
meeting id, orphaning whatever you had already written down. This calls the same
`api.process_meeting` the upload endpoint schedules, so the result is identical
to a fresh upload — transcript, diarization, and summary.

Written after four uploads failed at once for a reason that had nothing to do
with the recordings: an Ollama model was still resident in VRAM, so Whisper had
no room and every segment raised. The fix belongs in the pipeline (see
`OLLAMA_KEEP_ALIVE`), but the meetings still needed re-running.

Run this with the API server **stopped**, or at least idle. It transcribes in
this process, and two concurrent Whisper runs on one GPU is the failure mode
this tool exists to clean up after.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api  # noqa: E402
from db import repository as store  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_ids", nargs="*")
    parser.add_argument("--failed", action="store_true", help="every meeting with status Failed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )

    if args.failed:
        ids = [m["id"] for m in store.list_meetings() if m["status"] == "Failed"]
        if not ids:
            print("no failed meetings")
            return
    elif args.meeting_ids:
        ids = args.meeting_ids
    else:
        raise SystemExit("pass meeting ids or --failed")

    print(f"reprocessing {len(ids)} meeting(s): {', '.join(ids)}\n")

    for meeting_id in ids:
        meeting = store.get_meeting(meeting_id)
        if not meeting:
            print(f"{meeting_id}: no such meeting")
            continue

        # Reset the visible state first. Leaving "Failed" up while a run is in
        # flight is a lie to anyone watching the UI, which polls on status.
        store.update_meeting(meeting_id, status="Processing", progress=0, error=None)

        api.process_meeting(meeting_id)

        result = store.get_meeting(meeting_id)
        print(
            f"\n{meeting_id} \"{result['title']}\" -> {result['status']}"
            f"  speakers={result['participants']}"
            f"  lines={len(result.get('transcript') or [])}"
            f"  failedSegments={result.get('failedSegments')}"
        )
        if result.get("error"):
            print(f"  error: {result['error']}")
        if result.get("summary"):
            print(f"  summary via {result.get('summaryEngine')}: {result['summary'][:120]}")


if __name__ == "__main__":
    main()
