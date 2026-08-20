"""Capability-boundary tests for the morning digest API."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="openjarvis[server] not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from openjarvis.agents.digest_store import DigestArtifact, DigestStore
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.server.auth_middleware import AuthMiddleware
from openjarvis.server.digest_routes import create_digest_router

_API_KEY = "oj_sk_digest_test"
_PRINCIPAL = "api:digest-test"
_AUTH_HEADERS = {"Authorization": f"Bearer {_API_KEY}"}
_DIGEST_READ_CAPABILITIES = {
    "/api/digest": "memory:read",
    "/api/digest/audio": "memory:read",
    "/api/digest/history": "memory:read",
    "/api/digest/schedule": "schedule:create",
}


@pytest.fixture()
def digest_db(tmp_path):
    audio_path = tmp_path / "digest.mp3"
    audio_path.write_bytes(b"private-digest-audio")
    store = DigestStore(db_path=str(tmp_path / "digest.db"))
    store.save(
        DigestArtifact(
            text="Private morning digest.",
            audio_path=audio_path,
            sections={"messages": "Private message summary"},
            sources_used=["private-source"],
            generated_at=datetime.now(timezone.utc),
            model_used="test",
            voice_used="test",
        )
    )
    store.close()
    return tmp_path / "digest.db"


def _make_secured_app(
    db_path,
    *,
    grants: tuple[tuple[str, str], ...] = (),
) -> FastAPI:
    policy = CapabilityPolicy()
    for capability, pattern in grants:
        policy.grant(_PRINCIPAL, capability, pattern)

    app = FastAPI()
    app.include_router(create_digest_router(db_path=str(db_path)))
    app.add_middleware(
        AuthMiddleware,
        api_key=_API_KEY,
        principal=_PRINCIPAL,
        allowed_principals={_PRINCIPAL},
        capability_policy=policy,
    )
    return app


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize(
    ("path", "capability"),
    list(_DIGEST_READ_CAPABILITIES.items()),
)
def test_digest_read_capability_matrix(method, path, capability):
    assert AuthMiddleware._required_capability(method, path) == capability


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_future_digest_reads_fail_closed(method):
    assert (
        AuthMiddleware._required_capability(method, "/api/digest/future-sensitive")
        == "memory:read"
    )


def test_digest_get_route_inventory_requires_capabilities(tmp_path):
    router = create_digest_router(db_path=str(tmp_path / "inventory.db"))
    get_paths = {
        route.path
        for route in router.routes
        if "GET" in (getattr(route, "methods", None) or set())
    }

    assert get_paths == set(_DIGEST_READ_CAPABILITIES)
    for path in get_paths:
        assert (
            AuthMiddleware._required_capability("GET", path)
            == _DIGEST_READ_CAPABILITIES[path]
        )
        assert (
            AuthMiddleware._required_capability("HEAD", path)
            == _DIGEST_READ_CAPABILITIES[path]
        )


@pytest.mark.parametrize(
    "path",
    [
        "/api/digest",
        "/api/digest/audio",
        "/api/digest/history",
        "/api/digest/schedule",
    ],
)
def test_digest_reads_deny_valid_bearer_without_capability(digest_db, path):
    client = TestClient(_make_secured_app(digest_db))

    response = client.get(path, headers=_AUTH_HEADERS)

    assert response.status_code == 403
    assert response.json() == {"detail": "API principal is not authorized"}
    assert b"private-digest-audio" not in response.content


def test_memory_read_does_not_authorize_schedule(digest_db):
    app = _make_secured_app(
        digest_db,
        grants=(("memory:read", "/api/digest*"),),
    )
    client = TestClient(app)

    digest_response = client.get("/api/digest", headers=_AUTH_HEADERS)
    audio_response = client.get("/api/digest/audio", headers=_AUTH_HEADERS)
    history_response = client.get("/api/digest/history", headers=_AUTH_HEADERS)
    schedule_response = client.get("/api/digest/schedule", headers=_AUTH_HEADERS)

    assert digest_response.status_code == 200
    assert audio_response.status_code == 200
    assert audio_response.content == b"private-digest-audio"
    assert history_response.status_code == 200
    assert schedule_response.status_code == 403


def test_schedule_grant_does_not_authorize_digest_content(
    digest_db,
    monkeypatch,
):
    monkeypatch.setattr(
        "openjarvis.server.digest_routes.load_config",
        lambda: SimpleNamespace(
            digest=SimpleNamespace(enabled=False, schedule="0 6 * * *")
        ),
    )
    app = _make_secured_app(
        digest_db,
        grants=(("schedule:create", "/api/digest/schedule"),),
    )
    client = TestClient(app)

    schedule_response = client.get("/api/digest/schedule", headers=_AUTH_HEADERS)
    digest_response = client.get("/api/digest", headers=_AUTH_HEADERS)
    audio_response = client.get("/api/digest/audio", headers=_AUTH_HEADERS)

    assert schedule_response.status_code == 200
    assert digest_response.status_code == 403
    assert audio_response.status_code == 403
    assert b"private-digest-audio" not in audio_response.content
