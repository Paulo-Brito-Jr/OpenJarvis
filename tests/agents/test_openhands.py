"""Tests for OpenHandsAgent (real openhands-sdk wrapper)."""

from __future__ import annotations

from unittest.mock import MagicMock

from openjarvis.agents._stubs import BaseAgent
from openjarvis.agents.openhands import OpenHandsAgent
from openjarvis.core.registry import AgentRegistry


class TestOpenHandsAgentRegistration:
    def test_registered(self):
        AgentRegistry.register_value("openhands", OpenHandsAgent)
        assert AgentRegistry.contains("openhands")

    def test_agent_id(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        agent = OpenHandsAgent(engine, "test-model")
        assert agent.agent_id == "openhands"

    def test_does_not_accept_tools(self):
        """Real OpenHandsAgent doesn't use ToolUsingAgent base."""
        assert OpenHandsAgent.accepts_tools is False

    def test_is_base_agent(self):
        assert issubclass(OpenHandsAgent, BaseAgent)


class TestOpenHandsAgentFailClosed:
    def test_run_returns_security_disabled_without_loading_sdk(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        agent = OpenHandsAgent(engine, "test-model")
        result = agent.run("Hello")

        assert result.turns == 0
        assert result.metadata["error"] is True
        assert result.metadata["security_disabled"] is True
        assert result.metadata["reason"] == "unverified_external_sandbox"
        assert "disabled" in result.content.lower()

    def test_declares_security_context_requirement(self):
        assert OpenHandsAgent.requires_security_context is True


class TestOpenHandsAgentConstructor:
    def test_default_workspace(self):
        engine = MagicMock()
        agent = OpenHandsAgent(engine, "test-model")
        assert agent._workspace  # should be cwd

    def test_custom_workspace(self):
        engine = MagicMock()
        agent = OpenHandsAgent(engine, "test-model", workspace="/tmp/test")
        assert agent._workspace == "/tmp/test"

    def test_custom_api_key_is_not_retained_while_disabled(self):
        engine = MagicMock()
        agent = OpenHandsAgent(engine, "test-model", api_key="sk-test")
        assert agent._api_key == ""
