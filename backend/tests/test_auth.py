"""One shared password in front of the API (auth.py)."""

from __future__ import annotations

import importlib


def _client(monkeypatch, password):
    monkeypatch.setenv("APP_PASSWORD", password)
    import config
    importlib.reload(config)
    import auth
    importlib.reload(auth)
    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    return TestClient(main.app), auth


def test_without_a_password_nothing_is_required(monkeypatch):
    client, auth = _client(monkeypatch, "")
    assert client.get("/auth/status").json() == {"required": False}
    assert client.get("/speakers").status_code == 200


def test_with_a_password_every_route_needs_the_token(monkeypatch):
    client, auth = _client(monkeypatch, "s3cret")
    assert client.get("/auth/status").json() == {"required": True}
    assert client.get("/health").status_code == 200
    assert client.get("/speakers").status_code == 401
    assert client.get("/api/meetings").status_code == 401

    assert client.post("/auth/login", json={"password": "wrong"}).status_code == 401
    token = client.post("/auth/login", json={"password": "s3cret"}).json()["token"]

    assert client.get("/speakers", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    # The <audio> element cannot set a header; it passes the token in the URL.
    assert client.get(f"/api/meetings?token={token}").status_code == 200
    assert client.get("/speakers", headers={"Authorization": "Bearer nope"}).status_code == 401
