"""Verify security wiring reaches agents and ToolExecutor."""

from __future__ import annotations

from unittest.mock import MagicMock

from openjarvis.agents._stubs import AgentResult, ToolUsingAgent
from openjarvis.core.config import (
    CapabilitiesConfig,
    JarvisConfig,
    SecurityConfig,
)
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security import setup_security
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.system.orchestrator import QueryOrchestrator
from openjarvis.tools._stubs import BaseTool, ToolSpec


class _ConcreteAgent(ToolUsingAgent):
    """Minimal concrete subclass — ToolUsingAgent is abstract."""

    agent_id = "test"

    def run(self, input, context=None, **kwargs):
        return AgentResult(content="ok")


class _LegacyConstructorAgent(ToolUsingAgent):
    """Tool agent whose constructor does not accept security kwargs."""

    agent_id = "legacy"
    last_instance = None

    def __init__(
        self,
        engine,
        model,
        *,
        tools=None,
        bus=None,
        max_turns=None,
        temperature=None,
        max_tokens=None,
    ):
        super().__init__(
            engine,
            model,
            tools=tools,
            bus=bus,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        type(self).last_instance = self

    def run(self, input, context=None, **kwargs):
        return AgentResult(content="secure")


class _PrincipalProbeTool(BaseTool):
    tool_id = "principal_probe"
    calls = 0

    @property
    def spec(self):
        return ToolSpec(
            name=self.tool_id,
            description="Prove which runtime principal was authorized.",
        )

    def execute(self, **params):
        del params
        type(self).calls += 1
        return ToolResult(
            tool_name=self.tool_id,
            content="executed",
            success=True,
        )


class _OperatorPrincipalAgent(ToolUsingAgent):
    agent_id = "operator-principal-probe"

    def run(self, input, context=None, **kwargs):
        del input, context, kwargs
        result = self._executor.execute(
            ToolCall(
                id="principal-probe",
                name="principal_probe",
                arguments="{}",
            )
        )
        return AgentResult(
            content=result.content,
            tool_results=[result],
        )


def _make_mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {
        "content": "ok",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "model": "m",
        "finish_reason": "stop",
    }
    engine.list_models.return_value = ["m"]
    engine.health.return_value = True
    return engine


def _has_rust() -> bool:
    try:
        import openjarvis_rust  # noqa: F401

        return True
    except ImportError:
        return False


class TestCapabilityPolicyReachesExecutor:
    def test_policy_remains_fail_closed_when_caps_disabled(self) -> None:
        cfg = JarvisConfig()
        cfg.security = SecurityConfig(
            enabled=True,
            capabilities=CapabilitiesConfig(enabled=False),
        )
        bus = EventBus()
        engine = _make_mock_engine()
        sec = setup_security(cfg, engine, bus)

        agent = _ConcreteAgent(
            sec.engine,
            "m",
            tools=[],
            capability_policy=sec.capability_policy,
        )
        assert agent._executor._capability_policy is sec.capability_policy
        assert agent._executor._security_enabled is True
        assert sec.capability_policy.check("test", "tool:invoke") is False

    def test_no_policy_when_security_disabled(self) -> None:
        cfg = JarvisConfig()
        cfg.security = SecurityConfig(enabled=False)
        engine = _make_mock_engine()
        sec = setup_security(cfg, engine)

        agent = _ConcreteAgent(
            sec.engine,
            "m",
            tools=[],
            capability_policy=sec.capability_policy,
        )
        assert agent._executor._capability_policy is sec.capability_policy
        assert agent._executor._security_enabled is False
        # Engine should be the original, unwrapped
        assert sec.engine is engine

    def test_post_binding_updates_policy_and_identity(self) -> None:
        agent = _ConcreteAgent(_make_mock_engine(), "m", tools=[])
        policy = CapabilityPolicy()

        agent.bind_security(policy, "managed-agent")

        assert agent._executor._capability_policy is policy
        assert agent._executor._agent_id == "managed-agent"
        assert agent._executor._security_enabled is True

    def test_query_orchestrator_securely_binds_legacy_constructor(self) -> None:
        from types import SimpleNamespace

        AgentRegistry.register_value("legacy-secure", _LegacyConstructorAgent)
        policy = CapabilityPolicy()
        system = SimpleNamespace(
            engine=_make_mock_engine(),
            model="m",
            tools=[],
            bus=EventBus(),
            config=SimpleNamespace(agent=SimpleNamespace(max_turns=3)),
            capability_policy=policy,
            session_store=None,
            memory_backend=None,
            trace_store=None,
            trace_collector=None,
            engine_key="mock",
        )

        result = QueryOrchestrator(system)._run_agent(
            "hello",
            [],
            "legacy-secure",
            [],
            0.2,
            128,
        )

        assert result["content"] == "secure"
        instance = _LegacyConstructorAgent.last_instance
        assert instance is not None
        assert instance._executor._capability_policy is policy
        assert instance._executor._agent_id == "legacy-secure"

    def test_channel_operator_does_not_inherit_agent_service_grants(self) -> None:
        from types import SimpleNamespace

        AgentRegistry.register_value(
            "operator-principal-probe",
            _OperatorPrincipalAgent,
        )
        _PrincipalProbeTool.calls = 0
        policy = CapabilityPolicy()
        policy.grant(
            "operator-principal-probe",
            "tool:invoke",
            "tool:principal_probe",
        )
        system = SimpleNamespace(
            engine=_make_mock_engine(),
            model="m",
            tools=[_PrincipalProbeTool()],
            bus=EventBus(),
            config=SimpleNamespace(agent=SimpleNamespace(max_turns=3)),
            capability_policy=policy,
            boundary_guard=None,
            session_store=None,
            memory_backend=None,
            trace_store=None,
            trace_collector=None,
            engine_key="mock",
        )

        result = QueryOrchestrator(system)._run_agent(
            "attempt tool call",
            [],
            "operator-principal-probe",
            None,
            0.2,
            128,
            operator_id="channel:sender-without-grants",
        )

        assert result["tool_results"][0]["success"] is False
        assert "denied" in result["content"].lower()
        assert _PrincipalProbeTool.calls == 0

    def test_channel_operator_without_memory_grant_gets_no_global_context(
        self,
        monkeypatch,
    ) -> None:
        from types import SimpleNamespace

        inject_context = MagicMock(side_effect=AssertionError("memory leaked"))
        monkeypatch.setattr(
            "openjarvis.tools.storage.context.inject_context",
            inject_context,
        )
        policy = CapabilityPolicy()
        system = SimpleNamespace(
            engine=_make_mock_engine(),
            model="m",
            agent_name="none",
            tools=[],
            bus=EventBus(),
            config=JarvisConfig(),
            capability_policy=policy,
            memory_backend=MagicMock(),
            engine_key="mock",
        )

        result = QueryOrchestrator(system).ask(
            "ordinary channel message",
            operator_id="channel:sender-without-memory-grant",
        )

        assert result["content"] == "ok"
        inject_context.assert_not_called()
