"""Security tests for ChannelBridge research and system routing."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_shared_deep_research_agent_is_disabled_fail_closed() -> None:
    """A channel grant cannot turn a shared agent into a confused deputy."""
    from openjarvis.server.channel_bridge import ChannelBridge
    from openjarvis.server.session_store import SessionStore

    mock_agent = MagicMock()
    from openjarvis.security.capabilities import CapabilityPolicy

    policy = CapabilityPolicy()
    principal = ChannelBridge.principal_for("twilio", "+15551234567")
    policy.grant(principal, "tool:invoke", "agent:deep_research")

    bridge = ChannelBridge(
        channels={},
        session_store=SessionStore(db_path=":memory:"),
        bus=MagicMock(),
        deep_research_agent=mock_agent,
        sender_allowlist={"twilio": ["+15551234567"]},
        capability_policy=policy,
    )

    result = bridge.handle_incoming(
        sender_id="+15551234567",
        content="When was my last trip to Spain?",
        channel_type="twilio",
    )

    assert "unavailable" in result.lower()
    mock_agent.run.assert_not_called()


def test_shared_deep_research_never_preempts_sender_scoped_system() -> None:
    """The sender-scoped system path remains usable without invoking research."""
    from openjarvis.security.capabilities import CapabilityPolicy
    from openjarvis.server.channel_bridge import ChannelBridge
    from openjarvis.server.session_store import SessionStore

    mock_agent = MagicMock()
    mock_system = MagicMock()
    mock_system.ask.return_value = {"content": "Scoped response"}
    policy = CapabilityPolicy()
    principal = ChannelBridge.principal_for("twilio", "+15551234567")
    policy.grant(principal, "tool:invoke", "agent:deep_research")
    policy.grant(principal, "tool:invoke", "agent:system")

    bridge = ChannelBridge(
        channels={},
        session_store=SessionStore(db_path=":memory:"),
        bus=MagicMock(),
        system=mock_system,
        deep_research_agent=mock_agent,
        sender_allowlist={"twilio": ["+15551234567"]},
        capability_policy=policy,
    )

    result = bridge.handle_incoming(
        sender_id="+15551234567",
        content="Research this safely",
        channel_type="twilio",
    )

    assert result == "Scoped response"
    mock_agent.run.assert_not_called()
    mock_system.ask.assert_called_once_with(
        "Research this safely",
        operator_id=principal,
    )


def test_handle_chat_falls_back_to_system() -> None:
    """When no DeepResearch agent, fall back to system.ask()."""
    from openjarvis.security.capabilities import CapabilityPolicy
    from openjarvis.server.channel_bridge import ChannelBridge
    from openjarvis.server.session_store import SessionStore

    mock_system = MagicMock()
    mock_system.ask.return_value = {"content": "Generic response"}
    policy = CapabilityPolicy()
    principal = ChannelBridge.principal_for("twilio", "+15551234567")
    policy.grant(principal, "tool:invoke", "agent:system")

    bridge = ChannelBridge(
        channels={},
        session_store=SessionStore(db_path=":memory:"),
        bus=MagicMock(),
        system=mock_system,
        sender_allowlist={"twilio": ["+15551234567"]},
        capability_policy=policy,
    )

    result = bridge.handle_incoming(
        sender_id="+15551234567",
        content="Hello",
        channel_type="twilio",
    )

    assert result == "Generic response"
    mock_system.ask.assert_called_once()


def test_authenticated_sender_without_grant_never_reaches_system() -> None:
    """An allowlisted sender still needs an explicit system-chat grant."""
    from openjarvis.security.capabilities import CapabilityPolicy
    from openjarvis.server.channel_bridge import ChannelBridge
    from openjarvis.server.session_store import SessionStore

    mock_system = MagicMock()
    bridge = ChannelBridge(
        channels={},
        session_store=SessionStore(db_path=":memory:"),
        bus=MagicMock(),
        system=mock_system,
        sender_allowlist={"twilio": ["+15551234567"]},
        capability_policy=CapabilityPolicy(),
    )

    result = bridge.handle_incoming(
        sender_id="+15551234567",
        content="Use any available tool",
        channel_type="twilio",
    )

    assert result == "Not authorized."
    mock_system.ask.assert_not_called()


def test_default_allow_policy_cannot_replace_an_explicit_channel_grant() -> None:
    """Channel execution requires a real grant even under permissive defaults."""
    from openjarvis.security.capabilities import CapabilityPolicy
    from openjarvis.server.channel_bridge import ChannelBridge
    from openjarvis.server.session_store import SessionStore

    mock_system = MagicMock()
    bridge = ChannelBridge(
        channels={},
        session_store=SessionStore(db_path=":memory:"),
        bus=MagicMock(),
        system=mock_system,
        sender_allowlist={"twilio": ["+15551234567"]},
        capability_policy=CapabilityPolicy(default_deny=False),
    )

    result = bridge.handle_incoming(
        sender_id="+15551234567",
        content="Use any available tool",
        channel_type="twilio",
    )

    assert result == "Not authorized."
    mock_system.ask.assert_not_called()


def test_agent_command_never_enqueues_without_operator_context() -> None:
    """A channel principal must not create a service-agent pending message."""
    from openjarvis.security.capabilities import CapabilityPolicy
    from openjarvis.server.channel_bridge import ChannelBridge
    from openjarvis.server.session_store import SessionStore

    manager = MagicMock()
    policy = CapabilityPolicy()
    principal = ChannelBridge.principal_for("twilio", "+15551234567")
    policy.grant(principal, "system:admin", "agent:agent-1")
    bridge = ChannelBridge(
        channels={},
        session_store=SessionStore(db_path=":memory:"),
        bus=MagicMock(),
        agent_manager=manager,
        sender_allowlist={"twilio": ["+15551234567"]},
        capability_policy=policy,
    )

    result = bridge.handle_incoming(
        sender_id="+15551234567",
        content="/agent agent-1 run the private tool",
        channel_type="twilio",
    )

    assert "disabled" in result.lower()
    manager.send_message.assert_not_called()
