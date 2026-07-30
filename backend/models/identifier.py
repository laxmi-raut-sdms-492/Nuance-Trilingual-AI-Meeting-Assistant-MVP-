"""
Speaker identification: matching an unknown voice against enrolled ones.

Profiles live in PostgreSQL (the `speakers` table), not speakers.json. The
behaviour is unchanged:

  - Enrolling the same name again BLENDS samples (running average, weighted by
    how many samples already exist) rather than overwriting. The UI's "record a
    6-second sample" button therefore makes an existing profile more robust
    each time it is used, with no UI change needed.
  - Embeddings are assumed L2-normalised already (see models/embedding.py),
    which is what makes averaging multiple enrollment samples meaningful
    instead of being skewed by whichever sample was loudest.

Profiles are cached in memory and reloaded when the table changes, because
identify() runs once per diarized segment — hitting the database on every
segment of an hour-long meeting would be pointless traffic for data that
changes only when someone enrolls.
"""

import json
import logging

import numpy as np
from scipy.spatial.distance import cosine
from sqlalchemy import func, select

from config import IDENTIFICATION_SIMILARITY_THRESHOLD
from db.models import Speaker
from db.session import session_scope

logger = logging.getLogger("identifier")


class SpeakerIdentifier:
    def __init__(self, threshold: float = IDENTIFICATION_SIMILARITY_THRESHOLD):
        self.threshold = threshold
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

    def enroll(self, name: str, embedding: np.ndarray):
        """
        Add a sample for this speaker.

        If the name already has samples, blend into the existing profile
        (weighted average) rather than overwriting — multiple enrollments make
        a profile more robust to distance, volume and mic-angle variation.

        The read-modify-write happens inside one transaction, so two
        simultaneous enrollments of the same name cannot lose a sample.
        """
        with session_scope() as session:
            row = session.scalars(select(Speaker).where(Speaker.name == name)).one_or_none()

            if row is not None:
                existing = np.array(json.loads(row.centroid), dtype=np.float32)
                n = row.sample_count
                blended = (existing * n + embedding) / (n + 1)
                norm = np.linalg.norm(blended)
                blended = blended / norm if norm > 0 else blended
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
        if not self._cache:
            return "Unknown", 0.0

        best_name, best_score = "Unknown", -1.0
        for name, profile in self._cache.items():
            similarity = 1 - cosine(embedding, profile["centroid"])
            if similarity > best_score:
                best_name, best_score = name, similarity

        if best_score < self.threshold:
            logger.info(
                f"identify: best match '{best_name}' at {best_score:.3f} "
                f"< threshold {self.threshold} -> Unknown"
            )
            return "Unknown", round(float(best_score), 3)

        logger.info(f"identify: matched '{best_name}' at {best_score:.3f}")
        return best_name, round(float(best_score), 3)
