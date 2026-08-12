"""Factory resolver for STT Adapters."""

from __future__ import annotations

import logging
from config import CLOUD_STT_PROVIDER, STT_ADAPTER
from stt.base import STTAdapter
from stt.cloud.cloud_adapter import CloudSTTAdapter
from stt.local.local_adapter import LocalSTTAdapter

logger = logging.getLogger("stt_factory")


def get_stt_adapter(
    stt_adapter_choice: str | None = None,
    cloud_provider_choice: str | None = None,
) -> STTAdapter:
    """
    STT Factory Service Resolver.
    
    Returns LocalSTTAdapter or CloudSTTAdapter based on configuration / request option.
    """
    choice = (stt_adapter_choice or STT_ADAPTER or "local").lower()

    if choice == "cloud":
        logger.info(f"Instantiating CloudSTTAdapter (provider={cloud_provider_choice or CLOUD_STT_PROVIDER})")
        return CloudSTTAdapter(provider_choice=cloud_provider_choice)

    logger.info("Instantiating LocalSTTAdapter (Whisper Medium + IndicConformer)")
    return LocalSTTAdapter()
