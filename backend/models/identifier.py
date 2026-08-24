"""
Speaker identification: matching an unknown voice against enrolled ones.

Profiles live in PostgreSQL (the `speakers` table), not speakers.json. The
behaviour is unchanged:

  - Enrolling the same name again BLENDS samples (running average, weighted by
    how many samples already exist) rather than overwriting — unless
    overwrite=True is passed explicitly.
  - Embeddings are assumed L2-normalised already (see models/embedding.py),
    which is what makes averaging multiple enrollment samples meaningful
    instead of being skewed by whichever sample was loudest.

Profiles are cached in memory and reloaded when the table changes, because
identify() runs once per diarized segment — hitting the database on every
segment of an hour-long meeting would be pointless traffic for data that
changes only when someone enrolls.

Cosine matching + ambiguity rejection live in models.speaker_matcher so they
can be unit-tested without a database.
"""

import json
import logging
import re
import threading

import numpy as np
from sqlalchemy import func, select

from config import IDENTIFICATION_SIMILARITY_THRESHOLD
from db.models import Speaker
from db.session import session_scope
from models.speaker_matcher import UNKNOWN, match_speaker

logger = logging.getLogger("identifier")


class SpeakerIdentifier:
    def __init__(self, threshold: float = IDENTIFICATION_SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self._lock = threading.Lock()
        # name -> {"centroid": np.ndarray, "sample_count": int}
        self._cache: dict[str, dict] = {}
        self._cache_stamp = None
        self.refresh()

    # ------------------------------------------------------------- caching

    def _current_stamp(self, session):
        """(row count, newest updated_at) — cheap change detector."""
        return session.execute(
            select(func.count(Speaker.id), func.max(Speaker.updated_at))
        ).one()

    def refresh(self, force: bool = True):
        """Reload profiles from the database if they may have changed."""
        with self._lock:
            with session_scope() as session:
                stamp = self._current_stamp(session)
                if not force and stamp == self._cache_stamp:
                    return
                rows = session.scalars(select(Speaker)).all()
                self._cache = {
                    row.name: {
                        "centroid": np.array(json.loads(row.centroid), dtype=np.float32),
                        "sample_count": row.sample_count,
                    }
                    for row in rows
                }
                self._cache_stamp = stamp
            logger.info(f"loaded {len(self._cache)} enrolled voice profile(s)")

    @property
    def enrolled(self) -> dict[str, dict]:
        """Kept for compatibility with anything that inspected this directly."""
        return self._cache

    # -------------------------------------------------------------- public

    def enroll(self, name: str, embedding: np.ndarray, *, overwrite: bool = False):
        """
        Add a sample for this speaker.

        Names are stored in Title Case for Latin text so 'anushka' and
        'Anushka' are one profile. If the name already has samples and
        overwrite is False, blend into the existing profile.
        """
        name = name.strip()
        if re.search(r"[A-Za-z]", name):
            name = name.title()

        with session_scope() as session:
            row = session.scalars(
                select(Speaker).where(func.lower(Speaker.name) == name.lower())
            ).one_or_none()

            if row is not None and overwrite:
                row.name = name
                row.centroid = json.dumps(np.asarray(embedding).tolist())
                row.sample_count = 1
                logger.info(f"enroll: overwrote profile for '{name}'")
            elif row is not None:
                existing = np.array(json.loads(row.centroid), dtype=np.float32)
                n = row.sample_count
                blended = (existing * n + embedding) / (n + 1)
                norm = np.linalg.norm(blended)
                blended = blended / norm if norm > 0 else blended
                row.name = name  # canonicalize casing
                row.centroid = json.dumps(blended.tolist())
                row.sample_count = n + 1
                logger.info(f"enroll: blended new sample into '{name}' (now {n + 1} samples)")
            else:
                session.add(
                    Speaker(
                        name=name,
                        centroid=json.dumps(np.asarray(embedding).tolist()),
                        sample_count=1,
                    )
                )
                logger.info(f"enroll: created new profile for '{name}'")

        self.refresh()

    def remove(self, name: str):
        with session_scope() as session:
            row = session.scalars(select(Speaker).where(Speaker.name == name)).one_or_none()
            if row is not None:
                session.delete(row)
        self.refresh()

    def list_speakers(self) -> list[str]:
        self.refresh(force=False)
        return sorted(self._cache.keys())

    def identify(self, embedding: np.ndarray) -> tuple[str, float]:
        """Returns (name, similarity_score). name is 'Unknown' below threshold."""
        # Pick up enrollments made in another worker / via import without a restart.
        self.refresh(force=False)
        if not self._cache:
            return UNKNOWN, 0.0
        return match_speaker(embedding, self._cache, threshold=self.threshold)
