"""Pure security regressions for sensitive server boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.server.auth_middleware import (
    AuthMiddleware,
    explicitly_authorized,
)
from openjarvis.server.input_limits import (
    MAX_WS_FRAME_BYTES,
    utf8_size_exceeds,
)
from openjarvis.server.managed_channel_security import (
    quarantine_legacy_sendblue_bindings,
)
from openjarvis.server.oauth_state import OAuthStateStore
from openjarvis.server.response_security import (
    project_channel_binding,
    project_managed_agent,
)


def test_managed_agent_projection_drops_secrets_and_prompt() -> None:
    projected = project_managed_agent(
        {
            "id": "agent-1",
            "name": "owner@example.com",
            "config": {
                "model": "safe-model",
                "max_turns": 8,
                "tools": ["web_search"],
                "system_prompt": "private household instructions",
                "api_secret_key": "raw-secret",
                "bot_token": "raw-token",
                "custom_credential": "raw-credential",
            },
        }
    )

    assert projected["config"] == {
        "model": "safe-model",
        "max_turns": 8,
        "tools": ["web_search"],
        "system_prompt_configured": True,
    }
    serialized = repr(projected)
    assert "raw-secret" not in serialized
    assert "raw-token" not in serialized
    assert "raw-credential" not in serialized
    assert "private household" not in serialized
    assert "owner@example.com" not in serialized


def test_channel_binding_projection_drops_sender_ids_tokens_and_session() -> None:
    projected = project_channel_binding(
        {
            "id": "binding-1",
            "agent_id": "agent-1",
            "channel_type": "slack",
            "routing_mode": "dedicated",
            "session_id": "private-session",
            "config": {
                "bot_token": "xoxb-raw",
                "app_token": "xapp-raw",
                "allowed_senders": ["private-sender", "another-sender"],
                "channel": "#private",
            },
        }
    )

    assert projected["config"] == {
        "configured": True,
        "credentials_configured": True,
        "allowed_senders_configured": True,
        "allowed_sender_count": 2,
    }
    serialized = repr(projected)
    assert "xoxb-raw" not in serialized
    assert "xapp-raw" not in serialized
    assert "private-sender" not in serialized
    assert "private-session" not in serialized
    assert "#private" not in serialized


def test_sensitive_agent_reads_require_system_admin() -> None:
    for path in (
        "/v1/managed-agents",
        "/v1/managed-agents/agent-1/channels",
        "/v1/managed-agents/agent-1/state",
        "/v1/agents",
        "/v1/agents/errors",
        "/v1/connectors/gdrive/oauth/start",
    ):
        assert AuthMiddleware._required_capability("GET", path) == "system:admin"


def test_oauth_callback_uses_state_instead_of_bearer_auth() -> None:
    assert not AuthMiddleware._requires_auth("/v1/connectors/gdrive/oauth/callback")
    assert AuthMiddleware._requires_auth("/v1/connectors/gdrive/oauth/start")
    assert AuthMiddleware._requires_auth("/v1/connectors/gdrive/oauth/callback/extra")


def test_default_allow_is_not_an_explicit_external_grant() -> None:
    policy = CapabilityPolicy(default_deny=False)
    assert not explicitly_authorized(
        policy,
        "api:test",
        "tool:invoke",
        "agent:one",
    )

    policy.grant("api:test", "tool:invoke", "agent:one")
    assert explicitly_authorized(
        policy,
        "api:test",
        "tool:invoke",
        "agent:one",
    )


def test_oauth_state_is_one_time_scoped_and_expires() -> None:
    store = OAuthStateStore(ttl_seconds=10, max_pending=3)
    state = store.issue("gdrive", "api:test", now=100)

    assert store.consume(state, "gmail", now=101) is None
    assert store.consume(state, "gdrive", now=101) is None

    expiring = store.issue("gdrive", "api:test", now=200)
    assert store.consume(expiring, "gdrive", now=210) is None

    valid = store.issue("gdrive", "api:test", now=300)
    authorization = store.consume(valid, "gdrive", now=301)
    assert authorization is not None
    assert authorization.principal == "api:test"
    assert store.consume(valid, "gdrive", now=302) is None


def test_websocket_utf8_limit_is_byte_accurate() -> None:
    assert not utf8_size_exceeds("a" * MAX_WS_FRAME_BYTES, MAX_WS_FRAME_BYTES)
    assert utf8_size_exceeds(
        "é" * (MAX_WS_FRAME_BYTES // 2 + 1),
        MAX_WS_FRAME_BYTES,
    )
    assert utf8_size_exceeds(object(), MAX_WS_FRAME_BYTES)


def test_legacy_sendblue_restore_is_quarantined_without_db_reads() -> None:
    manager = MagicMock()
    app = SimpleNamespace(
        state=SimpleNamespace(
            agent_manager=manager,
            sendblue_channel=None,
        )
    )

    quarantine_legacy_sendblue_bindings(app)

    manager.list_agents.assert_not_called()
    manager.list_channel_bindings.assert_not_called()
    assert app.state.sendblue_channel is None
