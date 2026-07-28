"""Tests for tool wiring in AgentExecutor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openjarvis.agents._stubs import AgentResult
from openjarvis.agents.errors import FatalError
from openjarvis.agents.executor import AgentExecutor, _bind_agent_security
from openjarvis.agents.manager import AgentManager
from openjarvis.core.events import EventBus
from openjarvis.security.capabilities import CapabilityPolicy
from tests.agents.fake_engine import FakeEngine
from tests.agents.scenario_harness import FakeSystem


def _register_agent():
    """Re-register MonitorOperativeAgent (cleared by autouse fixture)."""
    from openjarvis.agents.monitor_operative import MonitorOperativeAgent
    from openjarvis.core.registry import AgentRegistry

    if not AgentRegistry.contains("monitor_operative"):
        AgentRegistry.register("monitor_operative")(MonitorOperativeAgent)


def test_executor_runs_with_tools_from_config(tmp_path):
    """Executor should resolve tool names from config and complete tick."""
    _register_agent()

    engine = FakeEngine([{"content": "test response"}])
    system = FakeSystem(engine=engine)

    mgr = AgentManager(db_path=str(tmp_path / "test.db"))
    agent = mgr.create_agent(
        "test",
        agent_type="monitor_operative",
        config={
            "system_prompt": "You are a test agent.",
            "tools": ["think"],
            "instruction": "test",
        },
    )
    mgr.send_message(agent["id"], "hello", mode="immediate")

    executor = AgentExecutor(manager=mgr, event_bus=EventBus())
    executor.set_system(system)

    executor.execute_tick(agent["id"])
    result_agent = mgr.get_agent(agent["id"])
    assert result_agent["status"] == "idle"
    assert result_agent["total_runs"] == 1
    mgr.close()


def test_executor_handles_missing_tools(tmp_path):
    """Executor should not crash if tool names don't exist in registry."""
    _register_agent()

    engine = FakeEngine([{"content": "test response"}])
    system = FakeSystem(engine=engine)

    mgr = AgentManager(db_path=str(tmp_path / "test.db"))
    agent = mgr.create_agent(
        "test",
        agent_type="monitor_operative",
        config={
            "system_prompt": "You are a test agent.",
            "tools": ["nonexistent_tool_xyz"],
            "instruction": "test",
        },
    )
    mgr.send_message(agent["id"], "hello", mode="immediate")

    executor = AgentExecutor(manager=mgr, event_bus=EventBus())
    executor.set_system(system)

    executor.execute_tick(agent["id"])
    result_agent = mgr.get_agent(agent["id"])
    assert result_agent["status"] == "idle"
    assert result_agent["total_runs"] == 1
    mgr.close()


def test_executor_handles_string_tools(tmp_path):
    """Executor should handle comma-separated tool string as well as list."""
    _register_agent()

    engine = FakeEngine([{"content": "test response"}])
    system = FakeSystem(engine=engine)

    mgr = AgentManager(db_path=str(tmp_path / "test.db"))
    agent = mgr.create_agent(
        "test",
        agent_type="monitor_operative",
        config={
            "system_prompt": "You are a test agent.",
            "tools": "think,calculator",
            "instruction": "test",
        },
    )
    mgr.send_message(agent["id"], "hello", mode="immediate")

    executor = AgentExecutor(manager=mgr, event_bus=EventBus())
    executor.set_system(system)

    executor.execute_tick(agent["id"])
    result_agent = mgr.get_agent(agent["id"])
    assert result_agent["status"] == "idle"
    mgr.close()


def test_executor_post_binds_policy_and_managed_identity(tmp_path):
    from openjarvis.core.registry import AgentRegistry

    captured = {}

    class StrictToolAgent:
        accepts_tools = True

        def __init__(self, engine, model, *, bus=None):
            captured["constructor"] = (engine, model, bus)

        def bind_security(self, policy, agent_id):
            captured["security"] = (policy, agent_id)

        def run(self, input, context=None):
            return AgentResult(content="ok")

    AgentRegistry.register_value("strict_tool_agent", StrictToolAgent)
    engine = FakeEngine([{"content": "unused"}])
    policy = CapabilityPolicy()
    system = SimpleNamespace(
        engine=engine,
        model="fake-model",
        capability_policy=policy,
        config=None,
        memory_backend=None,
        session_store=None,
        tool_executor=None,
    )
    mgr = AgentManager(db_path=str(tmp_path / "test.db"))
    agent = mgr.create_agent("test", agent_type="strict_tool_agent")
    executor = AgentExecutor(manager=mgr, event_bus=EventBus(), system=system)

    result = executor._invoke_agent(agent)

    assert result.content == "ok"
    assert captured["security"] == (policy, agent["id"])
    mgr.close()


def test_executor_does_not_retry_constructor_without_kwargs(tmp_path):
    from openjarvis.core.registry import AgentRegistry

    class BrokenAgent:
        accepts_tools = False
        attempts = 0

        def __init__(self, engine, model):
            BrokenAgent.attempts += 1
            raise TypeError("constructor failure")

    AgentRegistry.register_value("broken_agent", BrokenAgent)
    system = SimpleNamespace(engine=FakeEngine([]), model="fake-model")
    mgr = AgentManager(db_path=str(tmp_path / "test.db"))
    agent = mgr.create_agent("test", agent_type="broken_agent")
    executor = AgentExecutor(manager=mgr, event_bus=EventBus(), system=system)

    with pytest.raises(FatalError, match="Failed to initialize agent"):
        executor._invoke_agent(agent)

    assert BrokenAgent.attempts == 1
    mgr.close()


def test_executor_does_not_replay_empty_agent_result(tmp_path):
    from openjarvis.core.registry import AgentRegistry

    class EmptyAgent:
        accepts_tools = False
        runs = 0

        def __init__(self, engine, model):
            del engine, model

        def run(self, input, context=None):
            del input, context
            EmptyAgent.runs += 1
            return AgentResult(content="")

    AgentRegistry.register_value("empty_agent", EmptyAgent)
    system = SimpleNamespace(
        engine=FakeEngine([]),
        model="fake-model",
        capability_policy=None,
        config=None,
    )
    mgr = AgentManager(db_path=str(tmp_path / "test.db"))
    agent = mgr.create_agent("test", agent_type="empty_agent")
    executor = AgentExecutor(manager=mgr, event_bus=EventBus(), system=system)

    with pytest.raises(FatalError, match="automatic replay is disabled"):
        executor._invoke_agent(agent)

    assert EmptyAgent.runs == 1
    mgr.close()


def test_legacy_security_hook_receives_boundary_through_separate_method():
    captured = {}

    class LegacyAgent:
        def bind_security(self, policy, agent_id):
            captured["security"] = (policy, agent_id)

        def bind_boundary_guard(self, guard):
            captured["guard"] = guard

    policy = CapabilityPolicy()
    guard = object()

    _bind_agent_security(
        LegacyAgent(),
        policy,
        "legacy-agent",
        guard,
        agent_type="legacy",
    )

    assert captured["security"] == (policy, "legacy-agent")
    assert captured["guard"] is guard


def test_legacy_security_hook_without_boundary_support_fails_closed():
    class UnsafeLegacyAgent:
        def bind_security(self, policy, agent_id):
            del policy, agent_id

    with pytest.raises(FatalError, match="mandatory outbound boundary guard"):
        _bind_agent_security(
            UnsafeLegacyAgent(),
            CapabilityPolicy(),
            "legacy-agent",
            object(),
            agent_type="legacy",
        )
