"""
One-shot import of the legacy JSON store into PostgreSQL.

Reads backend/storage/meetings.json and speakers.json and writes them through
the repository, so the import exercises exactly the same code path the running
application uses — an import that succeeded but the app couldn't read would be
worse than no import at all.

Idempotent: existing meeting ids are skipped unless --replace is passed. Safe
to run twice.

    python3 -m db.import_json            # import, skip anything already there
    python3 -m db.import_json --replace  # overwrite meetings that exist
    python3 -m db.import_json --verify   # compare row counts, write nothing

Audio files are untouched: they already live on disk under storage/audio/ and
the repository still reads them from there.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MEETINGS_DB_PATH, SPEAKERS_DB_PATH  # noqa: E402
from db import repository as repo  # noqa: E402
from db.models import Speaker  # noqa: E402
from db.session import session_scope  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("import")


def _load(path: str, default):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist — nothing to import")
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def import_meetings(replace: bool) -> tuple[int, int]:
    meetings = _load(MEETINGS_DB_PATH, [])
    imported = skipped = 0

    for record in meetings:
        meeting_id = record.get("id")
        if not meeting_id:
            logger.warning("record without an id, skipping")
            continue

        if repo.get_meeting(meeting_id) is not None:
            if not replace:
                logger.info(f"{meeting_id} already present, skipping")
                skipped += 1
                continue
            repo.delete_meeting_row_only(meeting_id)

        repo.add_meeting(record)
        lines = len(record.get("transcript") or [])
        logger.info(f"imported {meeting_id} ({lines} lines) — {record.get('title')}")
        imported += 1

    return imported, skipped


def import_speakers() -> int:
    enrolled = _load(SPEAKERS_DB_PATH, {})
    if not enrolled:
        return 0

    count = 0
    with session_scope() as session:
        for name, profile in enrolled.items():
            # Two on-disk formats existed: {"centroid": [...], "sample_count": n}
            # and, before that, a bare embedding list. Accept both.
            if isinstance(profile, dict) and "centroid" in profile:
                centroid = profile["centroid"]
                samples = int(profile.get("sample_count", 1) or 1)
            else:
                centroid = profile
                samples = 1

            existing = session.query(Speaker).filter_by(name=name).one_or_none()
            if existing:
                existing.centroid = json.dumps(centroid)
                existing.sample_count = samples
            else:
                session.add(
                    Speaker(name=name, centroid=json.dumps(centroid), sample_count=samples)
                )
            count += 1
    logger.info(f"imported {count} enrolled voice(s)")
    return count


def verify() -> bool:
    """Compare what is in the JSON file with what is in the database."""
    meetings = _load(MEETINGS_DB_PATH, [])
    ok = True

    for record in meetings:
        stored = repo.get_meeting(record["id"])
        if stored is None:
            logger.error(f"MISSING  {record['id']}")
            ok = False
            continue

        for field in ("transcript", "speakerStats", "languages"):
            expected = len(record.get(field) or [])
            actual = len(stored.get(field) or [])
            status = "ok" if expected == actual else "MISMATCH"
            if expected != actual:
                ok = False
            logger.info(f"{status:8} {record['id']} {field}: json={expected} db={actual}")

        for field in ("title", "status", "participants", "duration"):
            if record.get(field) != stored.get(field):
                logger.error(
                    f"MISMATCH {record['id']} {field}: "
                    f"json={record.get(field)!r} db={stored.get(field)!r}"
                )
                ok = False

    logger.info("verification passed" if ok else "verification FAILED")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="overwrite existing meetings")
    parser.add_argument("--verify", action="store_true", help="compare only, write nothing")
    args = parser.parse_args()

    if args.verify:
        return 0 if verify() else 1

    imported, skipped = import_meetings(args.replace)
    speakers = import_speakers()
    logger.info(f"done — {imported} imported, {skipped} skipped, {speakers} speaker(s)")

    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
