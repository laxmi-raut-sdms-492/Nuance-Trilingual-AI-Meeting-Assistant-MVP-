"""
Test fixtures for the repository layer.

The suite runs against SQLite, not PostgreSQL. That is a deliberate trade:

- It needs no running server, so `pytest` works on a clean checkout.
- `session.py` already turns on `PRAGMA foreign_keys=ON` for SQLite, so the
  cascade behaviour these tests assert is really enforced rather than assumed.

What SQLite does NOT cover is the full-text half of `search()` — that branch is
Postgres-only and falls back to ILIKE here. The substring behaviour is tested;
the `to_tsvector` path is not. Anything touching tsquery must be verified
against a real database.

DATABASE_URL is set before `config` is imported, because config reads it at
import time and `session.py` caches one engine per process.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

# Both must happen before anything imports config or db.session.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

# A file, not ":memory:". `session.py` passes pool_size/max_overflow to
# create_engine, and SQLAlchemy gives in-memory SQLite a SingletonThreadPool,
# which rejects max_overflow outright. File-backed SQLite gets a QueuePool and
# takes the same arguments Postgres does — so the engine under test is
# configured identically to the real one.
_TMPDIR = tempfile.mkdtemp(prefix="nuance-tests-")
os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{os.path.join(_TMPDIR, 'test.db')}"

import pytest  # noqa: E402

from db import repository as repo  # noqa: E402
from db.models import Base  # noqa: E402
from db.session import get_engine, get_session_factory  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    """
    Create the schema once, from the models rather than by running Alembic.

    That means these tests verify the ORM's view of the schema. Drift between
    the models and the migrations is caught separately by
    `alembic revision --autogenerate` producing an empty diff, not here.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    yield
    engine.dispose()
    shutil.rmtree(_TMPDIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_tables():
    """Empty every table between tests. Deleting meetings cascades to children."""
    yield
    factory = get_session_factory()
    with factory() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


@pytest.fixture
def audio_dir(tmp_path, monkeypatch):
    """
    Point the repository's audio storage at a temp directory.

    `repository` imports AUDIO_DIR by value, so patching config would not be
    seen. Patch the name the module actually reads.
    """
    directory = tmp_path / "audio"
    directory.mkdir()
    monkeypatch.setattr(repo, "AUDIO_DIR", str(directory))
    return directory


def make_record(meeting_id: str = "MTG-test000000", **overrides) -> dict:
    """A complete meeting record in the exact camelCase shape the API uses."""
    record = {
        "id": meeting_id,
        "title": "Quarterly planning",
        "agenda": "Budget and hiring",
        "fileName": "meeting.wav",
        "fileType": "audio/wav",
        "fileSizeBytes": 1024,
        "fileSizeLabel": "1.0 KB",
        "uploadedAtISO": "2026-07-01T10:00:00+00:00",
        "status": "Completed",
        "progress": 100,
        "duration": "00:01:13",
        "durationSeconds": 73.5,
        "participants": 2,
        "language": "English",
        "languages": [
            {"code": "en", "name": "English", "seconds": 40.0, "pct": 54.4},
            {"code": "mr", "name": "Marathi", "seconds": 33.5, "pct": 45.6},
        ],
        "speakerStats": [
            {"name": "Speaker 1", "seconds": 40.0, "time": "00:40", "pct": 54.4, "color": "#FC5100"},
            {"name": "Speaker 2", "seconds": 33.5, "time": "00:33", "pct": 45.6, "color": "#3B82F6"},
        ],
        "transcript": [
            {
                "start_sec": 0.0,
                "end_sec": 4.0,
                "time": "00:00",
                "speaker": "Speaker 1",
                "speaker_label": "Speaker 1",
                "identified_as": None,
                "confidence": 0.91,
                "color": "#FC5100",
                "language": "en",
                "language_name": "English",
                "language_prob": 0.98,
                "language_detected": "en",
                "language_fallback": False,
                "raw_text": "Let us begin the quarterly planning meeting.",
                "cleaned_text": "Let us begin the quarterly planning meeting.",
                "text": "Let us begin the quarterly planning meeting.",
            },
            {
                "start_sec": 4.0,
                "end_sec": 9.0,
                "time": "00:04",
                "speaker": "Speaker 2",
                "speaker_label": "Speaker 2",
                "identified_as": "Asha",
                "confidence": 0.77,
                "color": "#3B82F6",
                "language": "mr",
                "language_name": "Marathi",
                "language_prob": 0.62,
                "language_detected": "mr",
                "language_fallback": False,
                "raw_text": "मच्छरो मुळे तक्रारी वाढल्या आहेत.",
                "cleaned_text": "मच्छरो मुळे तक्रारी वाढल्या आहेत.",
                "text": "मच्छरो मुळे तक्रारी वाढल्या आहेत.",
            },
        ],
        "decisions": [],
        "actionItems": [],
        "keywords": [],
    }
    record.update(overrides)
    return record
