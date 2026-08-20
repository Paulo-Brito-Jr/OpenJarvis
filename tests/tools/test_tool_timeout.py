"""Tests for tool execution timeout (Phase 14.1)."""

from __future__ import annotations

import time

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec


class SlowTool(BaseTool):
    """A tool that sleeps for a configurable duration."""

    tool_id = "slow_tool"

    def __init__(self, delay: float = 0.2):
        self._delay = delay

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slow_tool",
            description="A tool that takes a long time.",
            timeout_seconds=0.05,
        )

    def execute(self, **params) -> ToolResult:
        time.sleep(self._delay)
        return ToolResult(tool_name="slow_tool", content="Done", success=True)


class FastTool(BaseTool):
    """A tool that returns immediately."""

    tool_id = "fast_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fast_tool",
            description="A fast tool.",
            timeout_seconds=10.0,
        )

    def execute(self, **params) -> ToolResult:
        return ToolResult(
            tool_name="fast_tool",
            content=f"Result: {params.get('input', '')}",
            success=True,
        )


def _permitted_executor(tools, **kwargs) -> ToolExecutor:
    """Build an executor with the explicit baseline capability under test."""
    policy = CapabilityPolicy()
    policy.grant("timeout-test-agent", "tool:invoke")
    return ToolExecutor(
        tools,
        capability_policy=policy,
        agent_id="timeout-test-agent",
        **kwargs,
    )


class TestToolTimeout:
    def test_fast_tool_succeeds(self):
        executor = _permitted_executor([FastTool()])
        call = ToolCall(id="1", name="fast_tool", arguments='{"input": "hello"}')
        result = executor.execute(call)
        assert result.success
        assert "hello" in result.content

    def test_slow_tool_times_out(self):
        executor = _permitted_executor([SlowTool()])
        call = ToolCall(id="1", name="slow_tool", arguments="{}")
        started = time.monotonic()
        result = executor.execute(call)
        elapsed = time.monotonic() - started
        assert not result.success
        assert elapsed < 0.15
        assert "Outcome is unknown" in result.content
        assert result.metadata["outcome"] == "unknown"
        assert result.metadata["reconcile_required"] is True

    def test_timeout_event_emitted(self):
        bus = EventBus(record_history=True)
        executor = _permitted_executor([SlowTool()], bus=bus)
        call = ToolCall(id="1", name="slow_tool", arguments="{}")
        executor.execute(call)

        timeout_events = [
            e for e in bus.history if e.event_type == EventType.TOOL_TIMEOUT
        ]
        assert len(timeout_events) == 1
        assert timeout_events[0].data["tool"] == "slow_tool"
        assert timeout_events[0].data["outcome"] == "unknown"
        assert timeout_events[0].data["reconcile_required"] is True

    def test_timeout_does_not_claim_a_delayed_effect_was_cancelled(self):
        effects: list[str] = []

        class DelayedEffectTool(SlowTool):
            tool_id = "delayed_effect"

            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="delayed_effect",
                    description="Completes an effect after its deadline.",
                    timeout_seconds=0.05,
                )

            def execute(self, **params) -> ToolResult:
                time.sleep(self._delay)
                effects.append("committed")
                return ToolResult(
                    tool_name="delayed_effect",
                    content="committed",
                    success=True,
                )

        executor = _permitted_executor([DelayedEffectTool()])
        result = executor.execute(
            ToolCall(id="1", name="delayed_effect", arguments="{}")
        )

        assert result.success is False
        assert result.metadata["reconcile_required"] is True
        time.sleep(0.25)
        assert effects == ["committed"]

    def test_default_timeout_used(self):
        """When ToolSpec has no timeout, the executor default is used."""

        class NoTimeoutTool(BaseTool):
            tool_id = "no_timeout"

            @property
            def spec(self):
                return ToolSpec(
                    name="no_timeout",
                    description="test",
                    timeout_seconds=0,
                )

            def execute(self, **params):
                return ToolResult(tool_name="no_timeout", content="ok")

        executor = _permitted_executor(
            [NoTimeoutTool()],
            default_timeout=60.0,
        )
        call = ToolCall(id="1", name="no_timeout", arguments="{}")
        result = executor.execute(call)
        assert result.success

    def test_timeout_seconds_on_toolspec(self):
        spec = ToolSpec(name="test", description="test", timeout_seconds=42.0)
        assert spec.timeout_seconds == 42.0

    def test_unknown_tool(self):
        executor = _permitted_executor([FastTool()])
        call = ToolCall(id="1", name="nonexistent", arguments="{}")
        result = executor.execute(call)
        assert not result.success
        assert "Unknown tool" in result.content
