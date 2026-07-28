from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REGISTRY_PATH = "openjarvis.core.registry.AgentRegistry.get"


def test_run_ephemeral_creates_and_runs_agent():
    from openjarvis.agents.executor import AgentExecutor

    manager = MagicMock()
    executor = AgentExecutor(manager=manager, event_bus=MagicMock())

    mock_agent_cls = MagicMock()
    mock_agent_instance = MagicMock()
    mock_agent_instance.run.return_value = MagicMock(content="Flushed.")
    mock_agent_cls.return_value = mock_agent_instance

    with patch(REGISTRY_PATH, return_value=mock_agent_cls):
        executor.run_ephemeral(
            agent_type="simple",
            system_prompt="Save important context.",
            input_text="Review and flush.",
        )
    assert mock_agent_instance.run.called


def test_run_ephemeral_passes_input():
    from openjarvis.agents.executor import AgentExecutor

    manager = MagicMock()
    executor = AgentExecutor(manager=manager, event_bus=MagicMock())

    mock_agent_cls = MagicMock()
    mock_agent_instance = MagicMock()
    mock_agent_instance.run.return_value = MagicMock(content="Done.")
    mock_agent_cls.return_value = mock_agent_instance

    with patch(REGISTRY_PATH, return_value=mock_agent_cls):
        executor.run_ephemeral(
            agent_type="simple",
            system_prompt="Test prompt.",
            input_text="Hello world",
        )
    mock_agent_instance.run.assert_called_once_with("Hello world")


def test_run_ephemeral_binds_tool_agent_security():
    from openjarvis.agents.executor import AgentExecutor
    from openjarvis.security.capabilities import CapabilityPolicy

    policy = CapabilityPolicy()
    system = SimpleNamespace(capability_policy=policy)
    executor = AgentExecutor(
        manager=MagicMock(),
        event_bus=MagicMock(),
        system=system,
    )
    mock_agent_cls = MagicMock()
    mock_agent_cls.accepts_tools = True
    mock_agent_instance = MagicMock()
    mock_agent_cls.return_value = mock_agent_instance

    with patch(REGISTRY_PATH, return_value=mock_agent_cls):
        executor.run_ephemeral(
            agent_type="native_react",
            system_prompt="Use tools.",
            input_text="Calculate.",
            agent_id="ephemeral-test-agent",
        )

    mock_agent_instance.bind_security.assert_called_once_with(
        policy,
        "ephemeral-test-agent",
    )


def test_run_ephemeral_binds_provider_action_security():
    from openjarvis.agents.executor import AgentExecutor
    from openjarvis.security.capabilities import CapabilityPolicy

    policy = CapabilityPolicy()
    system = SimpleNamespace(capability_policy=policy)
    executor = AgentExecutor(
        manager=MagicMock(),
        event_bus=MagicMock(),
        system=system,
    )
    mock_agent_cls = MagicMock()
    mock_agent_cls.accepts_tools = False
    mock_agent_cls.requires_security_context = True
    mock_agent_instance = MagicMock()
    mock_agent_cls.return_value = mock_agent_instance

    with patch(REGISTRY_PATH, return_value=mock_agent_cls):
        executor.run_ephemeral(
            agent_type="hybrid-provider",
            system_prompt="Use provider actions.",
            input_text="Research.",
            agent_id="hybrid-ephemeral-agent",
        )

    mock_agent_instance.bind_security.assert_called_once_with(
        policy,
        "hybrid-ephemeral-agent",
    )
