"""
Engine and session management.

One engine per process, sessions per unit of work. `pool_pre_ping` is on
because the transcription worker can hold a connection idle for minutes while
Whisper runs, and Postgres (or a network in between) may drop it; without the
ping the next statement fails with a stale-connection error instead of
transparently reconnecting.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import DATABASE_URL, SQL_ECHO

logger = logging.getLogger("db")

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _SessionFactory
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            echo=SQL_ECHO,
            pool_pre_ping=True,
            # Transcription is serialized and the API is light, so a large pool
            # buys nothing. Keep it small enough not to exhaust Postgres'
            # default 100 connections if several processes are ever run.
            pool_size=20,
            max_overflow=20,
            pool_recycle=1800,
            future=True,
        )
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        logger.info(f"database engine created for {_safe_url(DATABASE_URL)}")
    return _engine


def _safe_url(url: str) -> str:
    """Strip the password before anything touches a log file."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def get_session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Transactional scope. Commits on success, rolls back on any exception.

    Every repository function that writes uses this, so a partially-written
    meeting (rows in transcript_lines but no parent, or half a transcript) is
    not representable.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """
    SQLite would silently ignore foreign keys without this.

    Postgres needs none of it, and this is a no-op there. It exists so that
    running the test suite or a throwaway instance against SQLite enforces the
    same constraints the real database does, rather than passing tests that
    would fail in Postgres.
    """
    module = type(dbapi_connection).__module__
    if "sqlite" not in module:
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
