"""
Voice -> vector.

Loads SpeechBrain's ECAPA-TDNN model once and reuses it for every request.
Loading it per-request would be extremely slow (multi-second model load).
"""

import numpy as np
import torch

from config import EMBEDDING_MODEL_SOURCE

_classifier = None  # lazy singleton


def _get_classifier():
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier

        _classifier = EncoderClassifier.from_hparams(
            source=EMBEDDING_MODEL_SOURCE,
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
    return _classifier


def get_embedding(audio: np.ndarray) -> np.ndarray:
    """
    audio: 1-D float32 numpy array, 16kHz mono.
    Returns a 192-dimensional, L2-NORMALIZED embedding vector.

    Normalization matters: raw ECAPA-TDNN embedding magnitude correlates
    with signal energy (volume/distance from mic), not just voice identity.
    Without normalizing, averaging embeddings (enrollment, centroids) lets
    loud/close segments dominate the profile, and a quiet/distant segment
    then compares poorly even for the same person. L2-normalizing puts
    every embedding on the unit hypersphere so only direction (voice
    characteristics) matters, not magnitude (loudness/distance).
    """
    classifier = _get_classifier()
    tensor = torch.from_numpy(audio).unsqueeze(0).float()
    with torch.no_grad():
        embedding = classifier.encode_batch(tensor)
    vec = embedding.squeeze().cpu().numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
