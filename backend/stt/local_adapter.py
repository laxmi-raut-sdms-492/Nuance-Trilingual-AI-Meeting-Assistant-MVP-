"""
Compatibility import path for LocalSTTAdapter.

The adapter layer landed twice under two naming schemes — stt/local_adapter.py
and stt/local/local_adapter.py — and both are imported by live code:
stt/resolver.py and pipeline.py reach for this path, stt/factory.py and the
comparison tool reach for the package one.

This file used to hold a SECOND, older implementation of the class. It predated
the adapter_type / provider_name / model_name properties becoming abstract in
stt/base.py, so it could no longer be instantiated at all — and since the
resolver imports from here, and local is the default processing mode, that made
every meeting fail on construction.

It is now a re-export. One implementation, two spellings of its address, so the
two halves of the codebase cannot drift apart again and `isinstance` agrees
whichever path a caller imported.
"""

from __future__ import annotations

from stt.local.local_adapter import LocalSTTAdapter

__all__ = ["LocalSTTAdapter"]
