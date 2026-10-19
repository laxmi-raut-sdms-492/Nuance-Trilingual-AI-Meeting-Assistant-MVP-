"""
Local Hindi/Marathi ASR via AI4Bharat IndicConformer-600M.

Model: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual

First run downloads weights from HuggingFace (~600M params). You must accept
the model license on HuggingFace and set HF_TOKEN if the repo requires auth.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from config import (
    INDIC_CONFORMER_DECODE,
    INDIC_CONFORMER_MODEL,
    INDIC_DECODE_LANGUAGES,
    SAMPLE_RATE,
)

logger = logging.getLogger("indic_conformer")

_model = None
_load_error: Exception | None = None


def _get_model():
    global _model, _load_error

    if _load_error is not None:
        raise _load_error

    if _model is None:
        from transformers import AutoModel

        logger.info(
            f"loading IndicConformer '{INDIC_CONFORMER_MODEL}' "
            f"(first run downloads from HuggingFace — accept the license on the model page)"
        )
        try:
            _model = _load_from_pretrained(AutoModel)
            _model.eval()
        except Exception as exc:
            _load_error = exc
            logger.error(f"IndicConformer failed to load: {exc}")
            raise
    return _model


def _load_from_pretrained(auto_model):
    """
    Load the model, preferring a complete snapshot already in the local cache.

    The AI4Bharat repo is gated. Without an HF_TOKEN that has been granted
    access, resolving it raises GatedRepoError (401) even when an earlier
    authenticated download left a complete snapshot on disk — and the failed
    attempt also rewrites the repo's refs/main to the empty snapshot it just
    created, so the cached copy stops being reachable and every later start
    fails the same way. One unauthenticated start is enough to turn a working
    install into one that silently decodes all Hindi and Marathi with Whisper.

    Hence cache first. Two details make it work:

      - The pointer is repaired before anything is resolved. Doing it after a
        failed attempt is too late: the hub has by then resolved main to the
        revision it could not fetch and holds that in process, so the retry
        looks for the wrong commit and misses the cache.
      - Offline is forced via constants.HF_HUB_OFFLINE, which the hub consults
        per request (utils/_http.py), rather than the environment variable it
        only reads at import. The model also ships custom code that calls
        snapshot_download() itself and passes no local_files_only, so the flag
        has to be global for the duration of the load.

    A machine with no cached snapshot goes straight to the network, which is
    what a fresh authenticated install needs.
    """
    import os

    if _cached_snapshot_dir() is not None:
        _repair_cache_ref()
        from huggingface_hub import constants as hf_constants

        previous_env = os.environ.get("HF_HUB_OFFLINE")
        previous_const = hf_constants.HF_HUB_OFFLINE
        os.environ["HF_HUB_OFFLINE"] = "1"
        hf_constants.HF_HUB_OFFLINE = True
        try:
            model = auto_model.from_pretrained(
                INDIC_CONFORMER_MODEL,
                trust_remote_code=True,
                local_files_only=True,
            )
            logger.info("IndicConformer loaded from the local HuggingFace cache")
            return model
        except Exception as exc:
            logger.warning(
                f"IndicConformer not usable from the local cache "
                f"({exc.__class__.__name__}); fetching from HuggingFace"
            )
        finally:
            hf_constants.HF_HUB_OFFLINE = previous_const
            if previous_env is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_env

    return auto_model.from_pretrained(
        INDIC_CONFORMER_MODEL,
        trust_remote_code=True,
    )


def _cached_snapshot_dir():
    """The newest cached snapshot of the model that actually holds a config."""
    import os
    from pathlib import Path

    repo_id = str(INDIC_CONFORMER_MODEL)
    if "/" not in repo_id or os.path.isdir(repo_id):
        return None

    cache_root = Path(
        os.environ.get("HF_HUB_CACHE")
        or os.path.join(
            os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
            "hub",
        )
    )
    snapshots = cache_root / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not snapshots.is_dir():
        return None

    complete = [
        d for d in snapshots.iterdir() if d.is_dir() and (d / "config.json").exists()
    ]
    if not complete:
        return None
    return max(complete, key=lambda d: d.stat().st_mtime)


def _repair_cache_ref() -> None:
    """
    Point the cached repo's refs/main at a snapshot that actually has weights.

    Offline resolution follows refs/main, and a failed gated fetch leaves it
    aimed at the empty snapshot that fetch created. Nothing here downloads,
    deletes or overwrites anything; only the pointer is corrected, and only
    when a usable snapshot is already sitting beside it.
    """
    newest = _cached_snapshot_dir()
    if newest is None:
        return

    ref = newest.parent.parent / "refs" / "main"
    try:
        if ref.exists() and ref.read_text().strip() == newest.name:
            return
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(newest.name)
        logger.info(
            f"repaired HuggingFace cache ref for {INDIC_CONFORMER_MODEL} -> {newest.name}"
        )
    except OSError as exc:
        logger.warning(f"could not repair HuggingFace cache ref: {exc}")

def transcribe_indic(audio: np.ndarray, language: str) -> str:
    """
    Transcribe one float32 mono segment at SAMPLE_RATE.

    language must be 'hi' or 'mr' — Indic Conformer does not support English.
    """
    if language not in INDIC_DECODE_LANGUAGES:
        raise ValueError(
            f"IndicConformer supports {INDIC_DECODE_LANGUAGES}, got {language!r}"
        )

    if len(audio) == 0:
        return ""

    decode = INDIC_CONFORMER_DECODE
    if decode not in ("ctc", "rnnt"):
        logger.warning(f"unknown INDIC_CONFORMER_DECODE={decode!r}, using ctc")
        decode = "ctc"

    model = _get_model()
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)

    with torch.no_grad():
        out = model(wav, language, decode)

    if isinstance(out, (list, tuple)):
        text = out[0] if out else ""
    else:
        text = out

    return str(text or "").strip()
