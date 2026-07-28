"""WebSocket authentication tests (issue #217).

The HTTP ``AuthMiddleware`` never sees WebSocket upgrade requests, so the
streaming endpoints (`/v1/chat/stream`, `/v1/agents/events`) must validate the
token themselves in the handshake before accepting the connection.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from openjarvis.core.events import EventBus, EventType  # noqa: E402
from openjarvis.security.capabilities import CapabilityPolicy  # noqa: E402
from openjarvis.server.api_routes import include_all_routes  # noqa: E402
from openjarvis.server.auth_middleware import websocket_authorized  # noqa: E402
from openjarvis.server.ws_bridge import create_ws_router  # noqa: E402


def _ws(query=None, headers=None):
    stub = MagicMock()
    stub.query_params = query or {}
    stub.headers = headers or {}
    return stub


class TestWebsocketAuthorizedHelper:
    def test_empty_key_denies(self):
        assert websocket_authorized(_ws(), "") is False

    def test_token_via_query(self):
        assert (
            websocket_authorized(
                _ws(query={"token": "sek"}),
                "sek",
                principal="api:test",
                allowed_principals={"api:test"},
            )
            is True
        )

    def test_token_via_bearer_header(self):
        ws = _ws(headers={"authorization": "Bearer sek"})
        assert (
            websocket_authorized(
                ws,
                "sek",
                principal="api:test",
                allowed_principals={"api:test"},
            )
            is True
        )

    def test_wrong_token_rejected(self):
        assert (
            websocket_authorized(
                _ws(query={"token": "nope"}),
                "sek",
                principal="api:test",
                allowed_principals={"api:test"},
            )
            is False
        )

    def test_missing_token_rejected_when_required(self):
        assert (
            websocket_authorized(
                _ws(),
                "sek",
                principal="api:test",
                allowed_principals={"api:test"},
            )
            is False
        )


def _make_app(api_key=""):
    app = FastAPI()
    principal = "api:test"
    policy = CapabilityPolicy()
    policy.grant(principal, "tool:invoke", "/v1/chat/stream")
    engine = MagicMock()
    engine.engine_id = "mock"

    async def mock_stream(messages, *, model="test-model", **kwargs):
        for tok in ["hi"]:
            yield tok

    engine.stream = mock_stream
    app.state.engine = engine
    app.state.model = "test-model"
    app.state.api_key = api_key
    app.state.api_principal = principal
    app.state.api_principal_allowlist = frozenset({principal})
    app.state.capability_policy = policy
    include_all_routes(app)
    return app


class TestChatStreamAuth:
    def test_rejected_without_token_when_key_set(self):
        client = TestClient(_make_app(api_key="secret"))
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.receive_text()

    def test_rejected_with_wrong_token(self):
        client = TestClient(_make_app(api_key="secret"))
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/chat/stream?token=wrong") as ws:
                ws.receive_text()

    def test_accepted_with_correct_token(self):
        client = TestClient(_make_app(api_key="secret"))
        with client.websocket_connect("/v1/chat/stream?token=secret") as ws:
            ws.send_text(json.dumps({"message": "hi"}))
            assert ws.receive_json()["type"] in ("chunk", "done", "error")

    def test_rejected_when_no_key_configured(self):
        client = TestClient(_make_app(api_key=""))
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.receive_text()

    def test_rejected_without_explicit_capability(self):
        app = _make_app(api_key="secret")
        app.state.capability_policy = CapabilityPolicy()
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/chat/stream?token=secret") as ws:
                ws.receive_text()

    def test_oversized_frame_closes_with_1009(self):
        from openjarvis.server.input_limits import MAX_WS_FRAME_BYTES

        client = TestClient(_make_app(api_key="secret"))
        with client.websocket_connect("/v1/chat/stream?token=secret") as ws:
            ws.send_text("x" * (MAX_WS_FRAME_BYTES + 1))
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_text()
        assert exc_info.value.code == 1009


class TestAgentEventsAuth:
    def _app(self, api_key=""):
        app = FastAPI()
        principal = "api:test"
        policy = CapabilityPolicy()
        policy.grant(principal, "system:admin", "/v1/agents/events")
        app.state.api_key = api_key
        app.state.api_principal = principal
        app.state.api_principal_allowlist = frozenset({principal})
        app.state.capability_policy = policy
        app.include_router(create_ws_router(EventBus()))
        return app

    def test_rejected_without_token_when_key_set(self):
        client = TestClient(self._app(api_key="secret"))
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/agents/events") as ws:
                ws.receive_text()

    def test_accepted_with_correct_token(self):
        bus = EventBus()
        app = FastAPI()
        principal = "api:test"
        policy = CapabilityPolicy()
        policy.grant(principal, "system:admin", "/v1/agents/events")
        app.state.api_key = "secret"
        app.state.api_principal = principal
        app.state.api_principal_allowlist = frozenset({principal})
        app.state.capability_policy = policy
        app.include_router(create_ws_router(bus))
        client = TestClient(app)
        with client.websocket_connect("/v1/agents/events?token=secret") as ws:
            bus.publish(EventType.AGENT_TICK_START, {"agent_id": "a"})
            assert ws.receive_json()["data"]["agent_id"] == "a"

    def test_rejected_without_explicit_capability(self):
        app = self._app(api_key="secret")
        app.state.capability_policy = CapabilityPolicy()
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/agents/events?token=secret") as ws:
                ws.receive_text()

    def test_oversized_agent_filter_closes_with_1009(self):
        from openjarvis.server.input_limits import MAX_WS_FILTER_BYTES

        client = TestClient(self._app(api_key="secret"))
        oversized = "a" * (MAX_WS_FILTER_BYTES + 1)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/v1/agents/events?token=secret&agent_id={oversized}"
            ) as ws:
                ws.receive_text()
        assert exc_info.value.code == 1009
