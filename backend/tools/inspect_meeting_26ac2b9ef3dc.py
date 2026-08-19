"""
Inspect meeting MTG-26ac2b9ef3dc from database store
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import repository as store


def inspect():
    meeting_id = "MTG-26ac2b9ef3dc"
    meeting = store.get_meeting(meeting_id)
    if not meeting:
        print(f"Meeting {meeting_id} not found in DB store.")
        all_m = store.list_meetings()
        print(f"Available meetings in DB ({len(all_m)}):")
        for m in all_m[:10]:
            print(f"  {m['id']}: {m.get('title')} ({m.get('duration_seconds')}s)")
        return

    print(f"Meeting {meeting_id}: {meeting.get('title')} ({meeting.get('duration_seconds')}s)")
    lines = meeting.get("transcript") or []
    print(f"Total transcript lines: {len(lines)}")
    for idx, l in enumerate(lines, 1):
        print(f"Line #{idx:02d} [{l.get('start_sec')}s - {l.get('end_sec')}s] {l.get('speaker')} ({l.get('speaker_label')}) [{l.get('language')}]:")
        print(f"  Text: {l.get('text')}")
        print("-" * 50)

    # Check raw file path
    audio_path = meeting.get("audio_file_path") or f"/home/stark/.gemini/antigravity/antigravity/storage/meetings/{meeting_id}.webm"
    print(f"\nAudio file path: {audio_path} (exists: {os.path.exists(audio_path)})")


if __name__ == "__main__":
    inspect()
