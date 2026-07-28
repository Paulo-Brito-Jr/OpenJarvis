"""Tests for tool confirmation enforcement in ToolExecutor."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec

# ---------------------------------------------------------------------------
# Test tool helpers
# ---------------------------------------------------------------------------


class _SafeTool(BaseTool):
    """Tool that does NOT require confirmation."""

    tool_id = "safe"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="safe",
            description="A safe tool.",
            requires_confirmation=False,
        )

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(tool_name="safe", content="safe result", success=True)


class _DangerousTool(BaseTool):
    """Tool that REQUIRES confirmation."""

    tool_id = "dangerous"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="dangerous",
            description="A dangerous tool.",
            requires_confirmation=True,
        )

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(tool_name="dangerous", content="executed!", success=True)


class _PrivilegedTool(BaseTool):
    tool_id = "privileged"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="privileged",
            description="A privileged tool.",
            required_capabilities=["file:write"],
        )

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(tool_name="privileged", content="written", success=True)


class _LegacyFileReadTool(BaseTool):
    """Legacy spec relies on the central default-capability mapping."""

    tool_id = "file_read"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_read",
            description="Legacy file reader.",
        )

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(tool_name="file_read", content="read", success=True)


def _make_policy(agent_id: str = "test-agent") -> CapabilityPolicy:
    policy = CapabilityPolicy()
    policy.grant(agent_id, "tool:invoke")
    return policy


def _make_executor(tools, **kwargs) -> ToolExecutor:
    return ToolExecutor(
        tools,
        capability_policy=_make_policy(),
        agent_id="test-agent",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolConfirmation:
    def test_requires_confirmation_no_callback(self) -> None:
        """Tool requiring confirmation but no callback → blocked."""
        executor = _make_executor([_DangerousTool()])
        call = ToolCall(id="1", name="dangerous", arguments="{}")
        result = executor.execute(call)
        assert result.success is False
        assert "requires confirmation" in result.content

    def test_requires_confirmation_not_interactive(self) -> None:
        """Tool requiring confirmation but interactive=False → blocked."""
        executor = _make_executor(
            [_DangerousTool()],
            interactive=False,
            confirm_callback=lambda _: True,
        )
        call = ToolCall(id="1", name="dangerous", arguments="{}")
        result = executor.execute(call)
        assert result.success is False
        assert "requires confirmation" in result.content

    def test_requires_confirmation_denied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tool requiring confirmation, callback returns False → denied."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        executor = _make_executor(
            [_DangerousTool()],
            interactive=True,
            confirm_callback=lambda _: False,
        )
        call = ToolCall(id="1", name="dangerous", arguments="{}")
        result = executor.execute(call)
        assert result.success is False
        assert "denied by user" in result.content

    def test_requires_confirmation_approved(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tool requiring confirmation, callback returns True → executes."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        executor = _make_executor(
            [_DangerousTool()],
            interactive=True,
            confirm_callback=lambda _: True,
        )
        call = ToolCall(id="1", name="dangerous", arguments="{}")
        result = executor.execute(call)
        assert result.success is True
        assert result.content == "executed!"

    def test_no_confirmation_needed(self) -> None:
        """Tool without requires_confirmation works normally."""
        executor = _make_executor([_SafeTool()])
        call = ToolCall(id="1", name="safe", arguments="{}")
        result = executor.execute(call)
        assert result.success is True
        assert result.content == "safe result"

    def test_no_confirmation_needed_with_callback(self) -> None:
        """Tool without requires_confirmation ignores callback."""
        calls = []
        executor = _make_executor(
            [_SafeTool()],
            interactive=True,
            confirm_callback=lambda msg: calls.append(msg) or True,
        )
        call = ToolCall(id="1", name="safe", arguments="{}")
        result = executor.execute(call)
        assert result.success is True
        # Callback should NOT have been called
        assert len(calls) == 0

    def test_confirmation_callback_receives_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Confirm callback receives a descriptive message."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        received = []

        def capture(msg: str) -> bool:
            received.append(msg)
            return True

        executor = _make_executor(
            [_DangerousTool()],
            interactive=True,
            confirm_callback=capture,
        )
        call = ToolCall(id="1", name="dangerous", arguments='{"action": "delete"}')
        executor.execute(call)

        assert len(received) == 1
        assert "dangerous" in received[0]
        assert "action" in received[0]

    def test_confirmation_callback_exception_denies(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        def broken_callback(_prompt: str) -> bool:
            raise RuntimeError("approval service unavailable")

        executor = _make_executor(
            [_DangerousTool()],
            interactive=True,
            confirm_callback=broken_callback,
        )
        result = executor.execute(
            ToolCall(id="1", name="dangerous", arguments="{}"),
        )

        assert result.success is False
        assert "confirmation failed" in result.content

    def test_confirmation_requires_literal_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        executor = _make_executor(
            [_DangerousTool()],
            interactive=True,
            confirm_callback=lambda _: "yes",
        )
        result = executor.execute(
            ToolCall(id="1", name="dangerous", arguments="{}"),
        )

        assert result.success is False
        assert "denied by user" in result.content

    def test_callback_cannot_approve_without_live_tty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        callback_calls = []
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        executor = _make_executor(
            [_DangerousTool()],
            interactive=True,
            confirm_callback=lambda prompt: callback_calls.append(prompt) or True,
        )

        result = executor.execute(
            ToolCall(id="1", name="dangerous", arguments="{}"),
        )

        assert result.success is False
        assert "live TTY" in result.content
        assert callback_calls == []

    def test_missing_policy_denies_every_tool(self) -> None:
        executor = ToolExecutor([_SafeTool()], agent_id="test-agent")
        result = executor.execute(ToolCall(id="1", name="safe", arguments="{}"))

        assert result.success is False
        assert "policy unavailable" in result.content

    def test_missing_identity_denies_every_tool(self) -> None:
        executor = ToolExecutor(
            [_SafeTool()],
            capability_policy=_make_policy(),
        )
        result = executor.execute(ToolCall(id="1", name="safe", arguments="{}"))

        assert result.success is False
        assert "identity unavailable" in result.content

    def test_tool_invoke_grant_is_mandatory(self) -> None:
        policy = CapabilityPolicy()
        policy.grant("test-agent", "file:read")
        executor = ToolExecutor(
            [_SafeTool()],
            capability_policy=policy,
            agent_id="test-agent",
        )
        result = executor.execute(ToolCall(id="1", name="safe", arguments="{}"))

        assert result.success is False
        assert "tool:invoke" in result.content

    def test_declared_capability_is_required_after_tool_invoke(self) -> None:
        policy = _make_policy()
        executor = ToolExecutor(
            [_PrivilegedTool()],
            capability_policy=policy,
            agent_id="test-agent",
        )
        denied = executor.execute(
            ToolCall(id="1", name="privileged", arguments="{}"),
        )
        policy.grant("test-agent", "file:write")
        allowed = executor.execute(
            ToolCall(id="2", name="privileged", arguments="{}"),
        )

        assert denied.success is False
        assert "file:write" in denied.content
        assert allowed.success is True

    def test_capability_is_checked_against_concrete_path_resource(self) -> None:
        policy = CapabilityPolicy()
        policy.grant("test-agent", "tool:invoke")
        policy.grant("test-agent", "file:write", "/safe/*")
        executor = ToolExecutor(
            [_PrivilegedTool()],
            capability_policy=policy,
            agent_id="test-agent",
        )

        allowed = executor.execute(
            ToolCall(
                id="1",
                name="privileged",
                arguments='{"path":"/safe/result.txt"}',
            )
        )
        denied = executor.execute(
            ToolCall(
                id="2",
                name="privileged",
                arguments='{"path":"/etc/passwd"}',
            )
        )

        assert allowed.success is True
        assert denied.success is False
        assert "/etc/passwd" in denied.content

    def test_path_resource_is_canonicalized_before_authorization(self) -> None:
        policy = CapabilityPolicy()
        policy.grant("test-agent", "tool:invoke")
        policy.grant("test-agent", "file:write", "/safe/*")
        executor = ToolExecutor(
            [_PrivilegedTool()],
            capability_policy=policy,
            agent_id="test-agent",
        )

        result = executor.execute(
            ToolCall(
                id="1",
                name="privileged",
                arguments='{"path":"/safe/../etc/passwd"}',
            )
        )

        assert result.success is False
        assert "/etc/passwd" in result.content

    def test_legacy_default_capability_mapping_is_enforced(self) -> None:
        policy = CapabilityPolicy()
        policy.grant("test-agent", "tool:invoke")
        executor = ToolExecutor(
            [_LegacyFileReadTool()],
            capability_policy=policy,
            agent_id="test-agent",
        )

        denied = executor.execute(
            ToolCall(
                id="1",
                name="file_read",
                arguments='{"path":"/safe/input.txt"}',
            )
        )
        policy.grant("test-agent", "file:read", "/safe/*")
        allowed = executor.execute(
            ToolCall(
                id="2",
                name="file_read",
                arguments='{"path":"/safe/input.txt"}',
            )
        )

        assert denied.success is False
        assert "file:read" in denied.content
        assert allowed.success is True

    def test_policy_exception_denies(self) -> None:
        class BrokenPolicy:
            enabled = True
            enforce_tool_confirmation = True

            def check(self, *_args):
                raise RuntimeError("policy backend unavailable")

        executor = ToolExecutor(
            [_SafeTool()],
            capability_policy=BrokenPolicy(),
            agent_id="test-agent",
        )
        result = executor.execute(ToolCall(id="1", name="safe", arguments="{}"))

        assert result.success is False
        assert "policy check failed" in result.content

    def test_policy_cannot_disable_declared_confirmation(self) -> None:
        policy = CapabilityPolicy(enforce_tool_confirmation=False)
        policy.grant("test-agent", "tool:invoke")
        executor = ToolExecutor(
            [_DangerousTool()],
            capability_policy=policy,
            agent_id="test-agent",
            interactive=False,
        )
        result = executor.execute(
            ToolCall(id="1", name="dangerous", arguments="{}"),
        )

        assert result.success is False
        assert "requires confirmation" in result.content

    def test_explicit_security_disable_still_requires_identity(self) -> None:
        disabled_policy = CapabilityPolicy(enabled=False)
        executor = ToolExecutor(
            [_DangerousTool()],
            capability_policy=disabled_policy,
        )
        result = executor.execute(
            ToolCall(id="1", name="dangerous", arguments="{}"),
        )

        assert result.success is False
        assert "identity unavailable" in result.content

    def test_explicit_security_disable_denies_identified_execution(self) -> None:
        disabled_policy = CapabilityPolicy(enabled=False)
        executor = ToolExecutor(
            [_DangerousTool()],
            capability_policy=disabled_policy,
            agent_id="explicitly-disabled-agent",
        )
        result = executor.execute(
            ToolCall(id="1", name="dangerous", arguments="{}"),
        )

        assert result.success is False
        assert "policy is disabled" in result.content
