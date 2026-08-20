"""Fail-closed tool execution tests for server-managed agents."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import BaseTool, ToolSpec

pytest.importorskip("fastapi", reason="openjarvis[server] not installed")

from openjarvis.server.agent_manager_routes import (  # noqa: E402
    _build_server_tool_executor,
)


class _SensitiveTool(BaseTool):
    tool_id = "sensitive"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="sensitive",
            description="A server-side sensitive tool.",
            requires_confirmation=True,
        )

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(tool_name="sensitive", content="executed", success=True)


def test_server_denies_when_policy_is_missing() -> None:
    executor = _build_server_tool_executor(
        tools=[_SensitiveTool()],
        bus=None,
        app_state=SimpleNamespace(),
        agent_id="managed-agent",
    )

    result = executor.execute(
        ToolCall(id="1", name="sensitive", arguments="{}"),
    )

    assert result.success is False
    assert "policy unavailable" in result.content


def test_server_never_auto_confirms_sensitive_tool() -> None:
    policy = CapabilityPolicy()
    policy.grant("managed-agent", "tool:invoke")
    executor = _build_server_tool_executor(
        tools=[_SensitiveTool()],
        bus=None,
        app_state=SimpleNamespace(capability_policy=policy),
        agent_id="managed-agent",
    )

    result = executor.execute(
        ToolCall(id="1", name="sensitive", arguments="{}"),
    )

    assert result.success is False
    assert "requires confirmation" in result.content
