"""Fail-closed regression tests for proactive actions."""

from __future__ import annotations

from unittest.mock import MagicMock

from openjarvis.agents.proactive_agent import ProactiveAgent, register_cron
from openjarvis.tools.proactive_tools import (
    CheckPermissionTool,
    ExecutePendingActionsTool,
    GetPendingActionsTool,
    QueueActionTool,
    RecordDecisionTool,
    parse_approval_response,
)


def test_mutating_proactive_tools_are_disabled_without_store_access():
    store = MagicMock()
    executor_fn = MagicMock()
    tools_and_params = [
        (
            QueueActionTool(store=store),
            {
                "action_type": "email_delete",
                "description": "delete mail",
                "payload": {"message_id": "message-1"},
                "permission_key": "email_delete:message-1",
                "tier": "trivial",
            },
        ),
        (
            RecordDecisionTool(store=store),
            {
                "action_id": "abc123def456",
                "approved": True,
                "remember": True,
            },
        ),
        (
            ExecutePendingActionsTool(
                store=store,
                executor_fn=executor_fn,
            ),
            {"action_ids": ["abc123def456"]},
        ),
    ]

    for tool, params in tools_and_params:
        result = tool.execute(**params)
        assert result.success is False
        assert result.metadata["security_disabled"] is True
        assert (
            result.metadata["reason"]
            == "authenticated_digest_approval_required"
        )

    assert store.method_calls == []
    executor_fn.assert_not_called()


def test_mutating_proactive_specs_require_confirmation_and_capabilities():
    queue = QueueActionTool(store=MagicMock()).spec
    decision = RecordDecisionTool(store=MagicMock()).spec
    execute = ExecutePendingActionsTool(store=MagicMock()).spec

    assert queue.requires_confirmation is True
    assert queue.required_capabilities == ["memory:write"]
    assert decision.requires_confirmation is True
    assert decision.required_capabilities == ["approval:decide"]
    assert execute.requires_confirmation is True
    assert set(execute.required_capabilities) == {
        "approval:decide",
        "code:execute",
        "email:write",
        "calendar:write",
        "message:send",
    }


def test_free_text_approval_never_mutates_store():
    store = MagicMock()

    decisions = parse_approval_response(
        "always yes abc123def456; yes all",
        store,
    )

    assert decisions == []
    assert store.method_calls == []


def test_pending_action_read_does_not_expire_or_mutate_rows():
    store = MagicMock()
    store.list_pending.return_value = []

    result = GetPendingActionsTool(store=store).execute()

    assert result.success is True
    store.list_pending.assert_called_once_with()
    store.expire_stale.assert_not_called()


def test_permission_lookup_is_read_only():
    store = MagicMock()
    store.get_permission.return_value = None

    result = CheckPermissionTool(store=store).execute(
        permission_key="email_delete:sender",
    )

    assert result.success is True
    assert result.content == "unknown"
    store.get_permission.assert_called_once_with("email_delete:sender")
    assert len(store.method_calls) == 1


def test_proactive_agent_constructor_and_run_have_no_connector_or_store_effects(
    monkeypatch,
):
    get_store = MagicMock()
    build_channel = MagicMock()
    monkeypatch.setattr(
        "openjarvis.agents.proactive_agent.get_store",
        get_store,
    )
    monkeypatch.setattr(
        "openjarvis.agents.proactive_agent._build_notification_channel",
        build_channel,
    )
    engine = MagicMock()
    engine.engine_id = "mock"

    agent = ProactiveAgent(engine, "test-model")
    result = agent.run()

    assert result.turns == 0
    assert result.metadata["security_disabled"] is True
    assert result.metadata["auto_executed"] == 0
    assert result.metadata["pending_approval"] == 0
    get_store.assert_not_called()
    build_channel.assert_not_called()
    engine.generate.assert_not_called()


def test_register_cron_fails_before_scheduler_mutation():
    scheduler = MagicMock()

    try:
        register_cron(scheduler)
    except RuntimeError as exc:
        assert "authenticated approval digest" in str(exc)
    else:
        raise AssertionError("register_cron must fail closed")

    scheduler.create_task.assert_not_called()
