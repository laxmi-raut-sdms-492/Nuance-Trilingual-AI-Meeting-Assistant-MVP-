"""
A shared password in front of the API.

The server listens on every interface (startup.sh), so without this anyone
on the network could read every transcript, play every recording and purge
the trash. One password, set as APP_PASSWORD, is the smallest thing that
closes that: the app asks for it once, keeps the token it gets back, and
sends it with every request. With APP_PASSWORD unset nothing is required,
which keeps a solo development setup as it was.

The token is an HMAC of a fixed label under a key derived from the
password, so it is stable across restarts and every token dies the moment
the password changes. There are no per-user accounts: one password, one
role, which is what this deployment has.
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from config import APP_PASSWORD

# Reachable without a token: the health check, the docs, and login itself.
PUBLIC_PATHS = {"/", "/health", "/favicon.ico", "/docs", "/openapi.json", "/redoc", "/auth/login", "/auth/status"}


def auth_required() -> bool:
    return bool(APP_PASSWORD)


def _key() -> bytes:
    return hashlib.sha256(APP_PASSWORD.encode("utf-8")).digest()


def make_token() -> str:
    return hmac.new(_key(), b"nuance-session-v1", hashlib.sha256).hexdigest()


def check_password(password: str) -> bool:
    return hmac.compare_digest((password or "").encode("utf-8"), APP_PASSWORD.encode("utf-8"))


def token_is_valid(token: str | None) -> bool:
    if not auth_required():
        return True
    return bool(token) and hmac.compare_digest(token, make_token())


def token_from_request(request: Request) -> str | None:
    """Bearer header first; `?token=` for the <audio> element, which cannot set headers."""
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.query_params.get("token")


async def require_token(request: Request, call_next):
    """HTTP middleware: every non-public route needs a valid token."""
    if (
        not auth_required()
        or request.method == "OPTIONS"
        or request.url.path in PUBLIC_PATHS
    ):
        return await call_next(request)
    if not token_is_valid(token_from_request(request)):
        return JSONResponse({"detail": "Sign in to use this app."}, status_code=401)
    return await call_next(request)


def login(password: str) -> dict:
    if not auth_required():
        return {"token": "", "required": False}
    if not check_password(password):
        raise HTTPException(401, "Wrong password.")
    return {"token": make_token(), "required": True}
