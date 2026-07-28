"""Tests for the fail-closed ClaudeCodeAgent adapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import openjarvis.agents  # noqa: F401 -- trigger registration
from openjarvis.agents._stubs import AgentResult
from openjarvis.agents.claude_code import (
    _OUTPUT_END,
    _OUTPUT_START,
    ClaudeCodeAgent,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import AgentRegistry

_SENTINEL_WRAP = "{start}\n{payload}\n{end}"


def _wrap_output(payload: dict) -> str:
    return _SENTINEL_WRAP.format(
        start=_OUTPUT_START,
        payload=json.dumps(payload),
        end=_OUTPUT_END,
    )


def _make_agent(**kwargs) -> ClaudeCodeAgent:
    engine = MagicMock()
    engine.engine_id = "mock"
    defaults = {
        "api_key": "test-key",
        "workspace": "/tmp/test",
    }
    defaults.update(kwargs)
    return ClaudeCodeAgent(engine, "test-model", **defaults)


class TestClaudeCodeRegistration:
    def test_agent_id(self):
        assert _make_agent().agent_id == "claude_code"

    def test_accepts_tools_false(self):
        assert ClaudeCodeAgent.accepts_tools is False

    def test_requires_security_context(self):
        assert ClaudeCodeAgent.requires_security_context is True

    def test_registry_key(self):
        AgentRegistry.register_value("claude_code", ClaudeCodeAgent)
        assert AgentRegistry.contains("claude_code")
        assert AgentRegistry.get("claude_code") is ClaudeCodeAgent


class TestEnsureRunnerFailClosed:
    def test_runner_is_refused_without_filesystem_or_package_side_effects(
        self,
        tmp_path,
        monkeypatch,
    ):
        agent = _make_agent(workspace=str(tmp_path))
        monkeypatch.setattr(
            "openjarvis.agents.claude_code._RUNNER_SRC",
            tmp_path / "missing-runner-source",
        )

        with pytest.raises(RuntimeError, match="disabled"):
            agent._ensure_runner()

        assert list(tmp_path.iterdir()) == []


class TestClaudeCodeRunFailClosed:
    def test_run_returns_security_disabled_without_runner(self, monkeypatch):
        agent = _make_agent()
        ensure_runner = MagicMock()
        monkeypatch.setattr(agent, "_ensure_runner", ensure_runner)

        result = agent.run("Say hello")

        assert isinstance(result, AgentResult)
        assert result.turns == 0
        assert result.tool_results == []
        assert result.metadata == {
            "error": True,
            "security_disabled": True,
            "reason": "unverified_external_sandbox",
        }
        assert "disabled" in result.content.lower()
        ensure_runner.assert_not_called()

    def test_no_bus_works(self):
        result = _make_agent().run("Hello")
        assert result.metadata["security_disabled"] is True


class TestClaudeCodeEvents:
    def test_emits_turn_start_and_error_end(self):
        bus = EventBus(record_history=True)
        agent = _make_agent(bus=bus)

        agent.run("Hello")

        types = [event.event_type for event in bus.history]
        assert EventType.AGENT_TURN_START in types
        assert EventType.AGENT_TURN_END in types
        start = next(
            event
            for event in bus.history
            if event.event_type == EventType.AGENT_TURN_START
        )
        assert start.data["agent"] == "claude_code"
        assert start.data["input"] == "Hello"


class TestParseOutput:
    def test_parses_valid_sentinels(self):
        stdout = _wrap_output(
            {
                "content": "hello",
                "tool_results": [],
                "metadata": {"k": "v"},
            }
        )
        content, tools, meta = ClaudeCodeAgent._parse_output(stdout)
        assert content == "hello"
        assert tools == []
        assert meta == {"k": "v"}

    def test_no_sentinels(self):
        content, tools, meta = ClaudeCodeAgent._parse_output("plain text")
        assert content == "plain text"
        assert tools == []
        assert meta == {}

    def test_tool_results_parsed(self):
        stdout = _wrap_output(
            {
                "content": "done",
                "tool_results": [
                    {
                        "tool_name": "Bash",
                        "content": "output",
                        "success": True,
                    },
                    {
                        "tool_name": "Write",
                        "content": "wrote file",
                        "success": False,
                    },
                ],
                "metadata": {},
            }
        )
        _, tools, _ = ClaudeCodeAgent._parse_output(stdout)
        assert len(tools) == 2
        assert tools[0].tool_name == "Bash"
        assert tools[0].success is True
        assert tools[1].tool_name == "Write"
        assert tools[1].success is False

    def test_extra_stdout_before_sentinels(self):
        payload = {
            "content": "result",
            "tool_results": [],
            "metadata": {},
        }
        stdout = "debug\n" + _wrap_output(payload) + "\nmore output"
        content, _, _ = ClaudeCodeAgent._parse_output(stdout)
        assert content == "result"

    def test_invalid_json(self):
        stdout = f"{_OUTPUT_START}\n{{broken\n{_OUTPUT_END}"
        _, _, meta = ClaudeCodeAgent._parse_output(stdout)
        assert meta["parse_error"] is True


class TestClaudeCodeDefaults:
    def test_api_key_is_not_read_from_env_while_disabled(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
        assert _make_agent(api_key="")._api_key == ""

    def test_explicit_api_key_is_not_retained_while_disabled(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        assert _make_agent(api_key="explicit-key")._api_key == ""

    def test_default_and_custom_timeout(self):
        assert _make_agent()._timeout == 300
        assert _make_agent(timeout=60)._timeout == 60
