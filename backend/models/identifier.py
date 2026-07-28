"""
Speaker identification: matching an unknown voice against enrolled ones.

v2 redesign:
  - Enrolling the same name multiple times BLENDS samples (running average,
    weighted by how many samples already exist) instead of overwriting.
    This is how the UI's "record 6s sample" button already works — calling
    it again for the same name now makes that profile more robust instead
    of just replacing it, with no UI changes needed.
  - Embeddings are assumed L2-normalized already (see models/embedding.py),
    which is what makes averaging multiple enrollment samples meaningful
    instead of being skewed by whichever sample was loudest.
"""

import json
import logging
import os

import numpy as np
from scipy.spatial.distance import cosine

from config import SPEAKERS_DB_PATH, IDENTIFICATION_SIMILARITY_THRESHOLD

logger = logging.getLogger("identifier")


class SpeakerIdentifier:
    def __init__(self, threshold: float = IDENTIFICATION_SIMILARITY_THRESHOLD):
        self.threshold = threshold
        # name -> {"centroid": [floats], "sample_count": int}
        self.enrolled: dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(SPEAKERS_DB_PATH):
            with open(SPEAKERS_DB_PATH, "r") as f:
                raw = json.load(f)
            # backward-compatible with the old format (name -> flat embedding list)
            self.enrolled = {
                name: (val if isinstance(val, dict) and "centroid" in val else {"centroid": val, "sample_count": 1})
                for name, val in raw.items()
            }

    def _save(self):
        with open(SPEAKERS_DB_PATH, "w") as f:
            json.dump(self.enrolled, f)

    def enroll(self, name: str, embedding: np.ndarray):
        """
        Add a sample for this speaker. If the name already has samples,
        blend into the existing profile (weighted average) rather than
        overwriting — multiple enrollments make the profile more robust
        to distance/volume/mic-angle variation.
        """
        if name in self.enrolled:
            existing = np.array(self.enrolled[name]["centroid"])
            n = self.enrolled[name]["sample_count"]
            blended = (existing * n + embedding) / (n + 1)
            norm = np.linalg.norm(blended)
            blended = blended / norm if norm > 0 else blended
            self.enrolled[name] = {"centroid": blended.tolist(), "sample_count": n + 1}
            logger.info(f"enroll: blended new sample into '{name}' (now {n + 1} samples)")
        else:
            self.enrolled[name] = {"centroid": embedding.tolist(), "sample_count": 1}
            logger.info(f"enroll: created new profile for '{name}'")
        self._save()

    def remove(self, name: str):
        self.enrolled.pop(name, None)
        self._save()

    def list_speakers(self) -> list[str]:
        return list(self.enrolled.keys())

    def identify(self, embedding: np.ndarray) -> tuple[str, float]:
        """Returns (name, similarity_score). name is 'Unknown' below threshold."""
        if not self.enrolled:
            return "Unknown", 0.0

        best_name, best_score = "Unknown", -1.0
        for name, profile in self.enrolled.items():
            ref_vec = np.array(profile["centroid"])
            similarity = 1 - cosine(embedding, ref_vec)
            if similarity > best_score:
                best_name, best_score = name, similarity

        if best_score < self.threshold:
            logger.info(f"identify: best match '{best_name}' at {best_score:.3f} < threshold {self.threshold} -> Unknown")
            return "Unknown", round(float(best_score), 3)

        logger.info(f"identify: matched '{best_name}' at {best_score:.3f}")
        return best_name, round(float(best_score), 3)
