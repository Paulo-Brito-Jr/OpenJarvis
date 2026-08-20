"""Fail-closed WebSocket capability and event projection tests."""

from __future__ import annotations

from types import SimpleNamespace

from openjarvis.core.events import Event, EventType
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.server.auth_middleware import (
    websocket_authorized,
    websocket_capability_authorized,
)
from openjarvis.server.ws_bridge import _project_event


def _websocket_stub(
    *,
    principal: str = "api:test",
    policy: CapabilityPolicy | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(api_principal=principal),
        app=SimpleNamespace(
            state=SimpleNamespace(
                api_principal=principal,
                capability_policy=policy,
            )
        ),
    )


def test_websocket_requires_explicit_capability_grant() -> None:
    policy = CapabilityPolicy()
    websocket = _websocket_stub(policy=policy)

    assert not websocket_capability_authorized(
        websocket,
        "system:admin",
        "/v1/agents/events",
    )

    policy.grant("api:test", "system:admin", "/v1/agents/events")

    assert websocket_capability_authorized(
        websocket,
        "system:admin",
        "/v1/agents/events",
    )


def test_valid_token_without_grant_is_still_denied() -> None:
    policy = CapabilityPolicy(default_deny=False)
    websocket = _websocket_stub(principal="", policy=policy)
    websocket.query_params = {"token": "secret"}
    websocket.headers = {}

    assert websocket_authorized(
        websocket,
        "secret",
        principal="api:test",
        allowed_principals={"api:test"},
    )
    assert not websocket_capability_authorized(
        websocket,
        "system:admin",
        "/v1/agents/events",
    )


def test_websocket_denies_missing_policy_or_principal() -> None:
    assert not websocket_capability_authorized(
        _websocket_stub(policy=None),
        "system:admin",
        "/v1/agents/events",
    )
    policy = CapabilityPolicy()
    policy.grant("api:test", "system:admin", "/v1/agents/events")
    assert not websocket_capability_authorized(
        _websocket_stub(principal="", policy=policy),
        "system:admin",
        "/v1/agents/events",
    )


def test_tool_event_projection_never_forwards_args_results_or_metadata() -> None:
    event = Event(
        event_type=EventType.TOOL_CALL_END,
        timestamp=123.0,
        data={
            "tool": "http_request",
            "agent": "agent-1",
            "success": True,
            "latency": 0.2,
            "arguments": {"authorization": "Bearer raw-secret"},
            "args": {"password": "raw-secret"},
            "result": "raw-secret-result",
            "metadata": {"token": "raw-secret-token"},
            "payload": "raw-secret-payload",
        },
    )

    projected = _project_event(event)

    assert projected == {
        "type": "tool_call_end",
        "timestamp": 123.0,
        "data": {
            "tool": "http_request",
            "agent": "agent-1",
            "success": True,
            "latency": 0.2,
        },
    }
    serialized = repr(projected)
    assert "raw-secret" not in serialized
    assert "arguments" not in serialized
    assert "result" not in serialized
    assert "metadata" not in serialized
    assert "payload" not in serialized


def test_inference_projection_uses_nested_usage_allowlist() -> None:
    event = Event(
        event_type=EventType.INFERENCE_END,
        timestamp=456.0,
        data={
            "model": "safe-model",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
                "api_key": "sk-raw-secret-value-123456789",
            },
            "content": "private response",
            "tool_calls": [{"arguments": "private args"}],
            "tool_results": ["private result"],
            "content_blocks": ["private block"],
        },
    )

    projected = _project_event(event)

    assert projected["data"] == {
        "model": "safe-model",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    }
    serialized = repr(projected)
    assert "raw-secret" not in serialized
    assert "private" not in serialized


def test_agent_error_projection_drops_raw_error() -> None:
    event = Event(
        event_type=EventType.AGENT_TICK_ERROR,
        timestamp=789.0,
        data={
            "agent_id": "agent-1",
            "error_type": "fatal",
            "duration": 1.5,
            "error": "password='raw-secret'",
        },
    )

    projected = _project_event(event)

    assert projected["data"] == {
        "agent_id": "agent-1",
        "error_type": "fatal",
        "duration": 1.5,
    }
    assert "raw-secret" not in repr(projected)


def test_projected_identifiers_are_redacted() -> None:
    event = Event(
        event_type=EventType.AGENT_TICK_START,
        timestamp=1.0,
        data={
            "agent_id": "agent-1",
            "agent_name": "owner@example.com",
        },
    )

    projected = _project_event(event)

    assert "owner@example.com" not in repr(projected)
    assert "REDACTED" in projected["data"]["agent_name"]


def test_unapproved_event_type_is_not_forwarded() -> None:
    event = Event(
        event_type=EventType.MEMORY_STORE,
        timestamp=1.0,
        data={"content": "private memory"},
    )

    assert _project_event(event) is None


def test_malformed_event_data_is_withheld() -> None:
    event = Event(
        event_type=EventType.TOOL_CALL_END,
        timestamp=1.0,
        data="raw-secret-payload",  # type: ignore[arg-type]
    )

    projected = _project_event(event)

    assert projected["data"] == {}
    assert "raw-secret" not in repr(projected)
