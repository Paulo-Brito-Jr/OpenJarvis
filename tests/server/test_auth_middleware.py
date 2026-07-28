"""Tests for API key authentication middleware."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="openjarvis[server] not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from openjarvis.server.auth_middleware import AuthMiddleware


def _make_app(api_key: str, capability_policy=None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        AuthMiddleware,
        api_key=api_key,
        principal="api:test",
        allowed_principals={"api:test"},
        capability_policy=capability_policy,
    )

    @app.get("/v1/models")
    async def models():
        return {"models": []}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/webhooks/twilio")
    async def twilio_webhook():
        return {"status": "received"}

    @app.get("/metrics")
    async def metrics():
        return {"requests": 0}

    @app.post("/v1/models/pull")
    async def model_pull():
        return {"status": "accepted"}

    return app


@pytest.fixture
def client():
    return TestClient(_make_app("oj_sk_test123"))


class TestAuthMiddleware:
    def test_rejects_missing_auth_header(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 401
        assert "missing" in resp.json()["detail"].lower()

    def test_rejects_wrong_key(self, client):
        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401
        assert "invalid" in resp.json()["detail"].lower()

    def test_accepts_valid_key(self, client):
        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer oj_sk_test123"},
        )
        assert resp.status_code == 200

    def test_health_exempt(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_webhooks_exempt(self, client):
        resp = client.post("/webhooks/twilio")
        assert resp.status_code == 200

    def test_metrics_requires_auth(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 401

    def test_metrics_accepts_valid_key(self, client):
        resp = client.get("/metrics", headers={"Authorization": "Bearer oj_sk_test123"})
        assert resp.status_code == 200

    def test_no_key_configured_denies_protected_routes(self):
        client = TestClient(_make_app(""))
        resp = client.get("/v1/models")
        assert resp.status_code == 503
        assert client.get("/metrics").status_code == 503

    def test_mutation_without_capability_policy_fails_closed(self):
        client = TestClient(_make_app("oj_sk_test123"))
        resp = client.post(
            "/v1/models/pull",
            headers={"Authorization": "Bearer oj_sk_test123"},
        )
        assert resp.status_code == 503
        assert "policy" in resp.json()["detail"].lower()

    def test_unknown_mutation_requires_system_admin(self):
        from openjarvis.security.capabilities import CapabilityPolicy

        denied_policy = CapabilityPolicy()
        denied = TestClient(
            _make_app("oj_sk_test123", capability_policy=denied_policy)
        )
        denied_response = denied.post(
            "/v1/models/pull",
            headers={"Authorization": "Bearer oj_sk_test123"},
        )
        assert denied_response.status_code == 403

        allowed_policy = CapabilityPolicy()
        allowed_policy.grant("api:test", "system:admin", "/v1/models/pull")
        allowed = TestClient(
            _make_app("oj_sk_test123", capability_policy=allowed_policy)
        )
        allowed_response = allowed.post(
            "/v1/models/pull",
            headers={"Authorization": "Bearer oj_sk_test123"},
        )
        assert allowed_response.status_code == 200

    @pytest.mark.parametrize(
        ("method", "path", "capability"),
        [
            ("POST", "/v1/memory/store", "memory:write"),
            ("POST", "/v1/memory/index", "memory:write"),
            ("POST", "/v1/memory/search", "memory:read"),
            ("GET", "/v1/memory/stats", "memory:read"),
            ("GET", "/v1/traces/trace-1", "memory:read"),
            ("GET", "/v1/sessions/session-1", "memory:read"),
            ("GET", "/v1/approvals/pending", "approval:decide"),
            ("GET", "/v1/managed-agents", "system:admin"),
            (
                "GET",
                "/v1/managed-agents/agent-1/channels",
                "system:admin",
            ),
            ("GET", "/v1/agents", "system:admin"),
            (
                "GET",
                "/v1/connectors/gdrive/oauth/start",
                "system:admin",
            ),
            (
                "POST",
                "/v1/approvals/action-1/approve",
                "approval:decide",
            ),
            ("POST", "/v1/channels/send", "channel:send"),
            (
                "POST",
                "/v1/managed-agents/agent-1/messages",
                "message:send",
            ),
            ("POST", "/api/digest/schedule", "schedule:create"),
            ("POST", "/api/research", "tool:invoke"),
            ("POST", "/v1/models/pull", "system:admin"),
            ("DELETE", "/v1/models/model-a", "system:admin"),
            ("POST", "/v1/connectors/upload/ingest", "system:admin"),
            ("POST", "/v1/telemetry/reset", "system:admin"),
            ("POST", "/v1/future-mutating-route", "system:admin"),
        ],
    )
    def test_route_capability_matrix(self, method, path, capability):
        assert AuthMiddleware._required_capability(method, path) == capability

    def test_principal_must_be_explicitly_allowlisted(self):
        app = FastAPI()
        app.add_middleware(
            AuthMiddleware,
            api_key="key",
            principal="api:test",
            allowed_principals={"api:other"},
        )

        @app.get("/v1/models")
        async def models():
            return {}

        with pytest.raises(RuntimeError, match="allowlist"):
            with TestClient(app):
                pass
