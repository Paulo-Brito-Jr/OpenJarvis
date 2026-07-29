"""Tests for the RLM agent."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.rlm import RLMAgent
from openjarvis.agents.rlm_repl import RLMRepl
from openjarvis.core.cancellation import (
    AgentCancelledError,
    CancellationToken,
    cancellation_scope,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolResult
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import BaseTool, ToolSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _isolated_test_executor(code, namespace, max_output_chars):
    """Test double for a sandbox transport; production never uses host exec."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, namespace)  # noqa: S102
    except AgentCancelledError:
        raise
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    output = stdout_buf.getvalue()
    err_output = stderr_buf.getvalue()
    if err_output:
        output += ("\n" if output else "") + err_output
    if len(output) > max_output_chars:
        output = output[:max_output_chars] + "\n... (output truncated)"
    return output


@pytest.fixture(autouse=True)
def _inject_isolated_test_executor(monkeypatch):
    original = RLMAgent.__init__

    def secured_init(self, *args, **kwargs):
        kwargs.setdefault("repl_sandbox_executor", _isolated_test_executor)
        original(self, *args, **kwargs)

    monkeypatch.setattr(RLMAgent, "__init__", secured_init)


class _CalcStub(BaseTool):
    tool_id = "calculator"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Math calculator.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )

    def execute(self, **params) -> ToolResult:
        expr = params.get("expression", "0")
        try:
            val = eval(expr)  # noqa: S307
        except Exception as e:
            return ToolResult(tool_name="calculator", content=str(e), success=False)
        return ToolResult(tool_name="calculator", content=str(val), success=True)


class _FileReadStub(BaseTool):
    tool_id = "file_read"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_read",
            description="Read a file.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

    def execute(self, **params) -> ToolResult:
        path = params.get("path", "")
        max_lines = params.get("max_lines")
        if path == "rust/Cargo.toml":
            content = '[workspace]\nmembers = ["a", "b", "c"]\n'
            if max_lines is not None:
                lines = content.splitlines(keepends=True)
                content = "".join(lines[: int(max_lines)])
            return ToolResult(
                tool_name="file_read",
                content=content,
                success=True,
            )
        if path == "long.txt":
            content = "line1\nline2\nline3\nline4\nline5\n"
            if max_lines is not None:
                lines = content.splitlines(keepends=True)
                content = "".join(lines[: int(max_lines)])
            return ToolResult(
                tool_name="file_read",
                content=content,
                success=True,
            )
        return ToolResult(tool_name="file_read", content="", success=False)


def _make_engine(content: str = "Final answer.") -> MagicMock:
    """Engine that returns plain content (no code block)."""
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {
        "content": content,
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "test-model",
        "finish_reason": "stop",
    }
    return engine


def _allow_test_tools(agent):
    policy = CapabilityPolicy()
    policy.grant("rlm-test-agent", "*")
    agent.bind_security(policy, "rlm-test-agent")
    return agent


def _allow_repl(agent):
    policy = CapabilityPolicy()
    policy.grant("rlm-test-agent", "tool:invoke", "code:rlm-repl")
    policy.grant("rlm-test-agent", "code:execute", "code:rlm-repl")
    agent.bind_security(policy, "rlm-test-agent")
    return agent


def _make_engine_with_code(
    code: str,
    final_content: str = "Done.",
) -> MagicMock:
    """Engine that returns a python code block, then a final answer."""
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        {
            "content": f"```python\n{code}\n```",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        },
        {
            "content": final_content,
            "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            "model": "test-model",
            "finish_reason": "stop",
        },
    ]
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRLMAgentRegistration:
    def test_registered(self):
        # Re-register after conftest clears all registries
        AgentRegistry.register_value("rlm", RLMAgent)
        assert AgentRegistry.contains("rlm")

    def test_agent_id(self):
        engine = _make_engine()
        agent = RLMAgent(engine, "test-model")
        assert agent.agent_id == "rlm"


class TestRLMCodeExtraction:
    def test_extract_python_block(self):
        text = "Here is code:\n```python\nx = 1\n```\nDone."
        code = RLMAgent._extract_code(text)
        assert code == "x = 1"

    def test_extract_bare_block(self):
        text = "Here is code:\n```\nx = 1\n```\nDone."
        code = RLMAgent._extract_code(text)
        assert code == "x = 1"

    def test_no_block(self):
        text = "No code here, just text."
        code = RLMAgent._extract_code(text)
        assert code is None

    def test_python_preferred_over_bare(self):
        text = "```python\nx = 1\n```\n\n```\ny = 2\n```"
        code = RLMAgent._extract_code(text)
        assert code == "x = 1"


class TestRLMStripThink:
    def test_strip_think(self):
        text = "<think>thinking...</think>Answer here."
        result = RLMAgent._strip_think_tags(text)
        assert result == "Answer here."

    def test_no_think_tags(self):
        text = "Just text."
        result = RLMAgent._strip_think_tags(text)
        assert result == "Just text."


class TestRLMDirectAnswer:
    def test_no_code_block_returns_content(self):
        """When model returns no code block, treat content as final answer."""
        engine = _make_engine("The answer is 42.")
        agent = RLMAgent(engine, "test-model")
        result = agent.run("What is the answer?")
        assert result.content == "The answer is 42."
        assert result.turns == 1
        assert result.tool_results == []


class TestRLMFinalTermination:
    def test_final_terminates(self):
        """FINAL() in code should terminate the agent."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nFINAL('hello world')\n```",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        }
        agent = _allow_repl(RLMAgent(engine, "test-model"))
        result = agent.run("Test")
        assert result.content == "hello world"
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "rlm_repl"

    def test_final_var_terminates(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nresult = 42\nFINAL_VAR('result')\n```",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        }
        agent = _allow_repl(RLMAgent(engine, "test-model"))
        result = agent.run("Test")
        assert result.content == "42"


class TestRLMContextInjection:
    def test_context_from_metadata(self):
        engine = _make_engine("The answer is 42.")
        agent = RLMAgent(engine, "test-model")
        ctx = AgentContext(metadata={"context": "Some long document text."})
        result = agent.run("Summarize", context=ctx)
        assert result.content == "The answer is 42."

    def test_context_from_memory_results(self):
        engine = _make_engine("Summary.")
        agent = RLMAgent(engine, "test-model")
        ctx = AgentContext(memory_results=["chunk1", "chunk2"])
        result = agent.run("Summarize", context=ctx)
        assert result.content == "Summary."


class TestRLMSubLMCalls:
    def test_sub_lm_called_from_repl(self):
        """Verify that llm_query() inside REPL code calls engine.generate."""
        engine = MagicMock()
        engine.engine_id = "mock"
        # First call: root LM generates code that calls llm_query
        # Second call: sub-LM responds to llm_query
        # Third call: root LM gets REPL output, returns final (no code)
        engine.generate.side_effect = [
            {
                "content": (
                    "```python\nresult = llm_query('What is 2+2?')\nFINAL(result)\n```"
                ),
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
            # Sub-LM response for llm_query
            {
                "content": "4",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]
        agent = _allow_repl(RLMAgent(engine, "test-model"))
        result = agent.run("Calculate")
        assert result.content == "4"
        # engine.generate should be called at least twice (root + sub)
        assert engine.generate.call_count >= 2


class TestRLMCancellation:
    def test_repl_re_raises_cancellation_from_real_callback(self):
        token = CancellationToken()

        def cancel_callback(_prompt):
            token.cancel()
            token.raise_if_cancelled()

        repl = RLMRepl(
            llm_query_fn=cancel_callback,
            sandbox_executor=_isolated_test_executor,
        )

        with pytest.raises(AgentCancelledError):
            repl.execute("llm_query('cancel now')")

    def test_cancelled_real_repl_callback_cannot_return_final_answer(self):
        token = CancellationToken()
        engine = _make_engine(
            "```python\nFINAL(llm_query('cancel during callback'))\n```"
        )
        agent = _allow_repl(RLMAgent(engine, "test-model"))

        def cancel_callback(_prompt):
            token.cancel()
            return "must not become final"

        agent._make_sub_query = MagicMock(side_effect=cancel_callback)

        with cancellation_scope(token), pytest.raises(AgentCancelledError):
            agent.run("question")

        agent._make_sub_query.assert_called_once_with("cancel during callback")
        engine.generate.assert_called_once()

    def test_cancelled_context_prevents_root_inference(self):
        token = CancellationToken()
        token.cancel()
        engine = _make_engine("must not run")
        agent = RLMAgent(engine, "test-model")

        with pytest.raises(AgentCancelledError):
            agent.run(
                "question",
                context=AgentContext(cancellation_token=token),
            )

        engine.generate.assert_not_called()

    def test_cancelled_root_result_never_reaches_repl(self):
        token = CancellationToken()
        engine = _make_engine()
        agent = _allow_repl(RLMAgent(engine, "test-model"))

        def complete_root_inference(messages):
            token.cancel()
            return {
                "content": "```python\nFINAL('must not execute')\n```",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }

        agent._generate = MagicMock(side_effect=complete_root_inference)

        with (
            patch("openjarvis.agents.rlm.RLMRepl.execute", autospec=True) as execute,
            cancellation_scope(token),
            pytest.raises(AgentCancelledError),
        ):
            agent.run("question")

        execute.assert_not_called()
        agent._generate.assert_called_once()

    def test_cancelled_subquery_result_prevents_tool_and_followup(self):
        token = CancellationToken()
        engine = MagicMock(engine_id="mock")

        def complete_subquery(*args, **kwargs):
            token.cancel()
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "sub_0",
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    }
                ],
            }

        engine.generate.side_effect = complete_subquery
        agent = _allow_test_tools(RLMAgent(engine, "test-model", tools=[_CalcStub()]))
        execute = MagicMock()
        agent._executor.execute = execute

        with cancellation_scope(token), pytest.raises(AgentCancelledError):
            agent._make_sub_query("calculate")

        assert engine.generate.call_count == 1
        execute.assert_not_called()

    def test_cancelled_batch_query_never_starts_second_subquery(self):
        token = CancellationToken()
        engine = MagicMock(engine_id="mock")

        def complete_first(*args, **kwargs):
            token.cancel()
            return {"content": "first"}

        engine.generate.side_effect = complete_first
        agent = RLMAgent(engine, "test-model")

        with cancellation_scope(token), pytest.raises(AgentCancelledError):
            agent._make_batch_query(["first", "second"])

        assert engine.generate.call_count == 1


class TestRLMMultiTurn:
    def test_multi_turn_loop(self):
        """Agent should loop: generate code → execute → feed output → generate again."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            # Turn 1: code that sets a variable
            {
                "content": ("```python\nx = 10\nprint(f'x = {x}')\n```"),
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
            # Turn 2: code that uses the variable and terminates
            {
                "content": ("```python\ny = x * 2\nFINAL(y)\n```"),
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]
        agent = _allow_repl(RLMAgent(engine, "test-model"))
        result = agent.run("Calculate")
        assert result.content == "20"
        assert result.turns == 2
        assert len(result.tool_results) == 2

    def test_max_turns_exceeded(self):
        """Agent should stop after max_turns."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nprint('looping')\n```",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        }
        agent = _allow_repl(RLMAgent(engine, "test-model", max_turns=3))
        result = agent.run("Loop")
        assert result.turns == 3
        assert result.metadata.get("max_turns_exceeded") is True

    def test_max_turns_with_partial_answer(self):
        """When max turns exceeded but answer dict has value, use it."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nanswer['value'] = 'partial'\nprint('working')\n```",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        }
        agent = _allow_repl(RLMAgent(engine, "test-model", max_turns=2))
        result = agent.run("Work")
        assert result.content == "partial"
        assert result.metadata.get("max_turns_exceeded") is True


class TestRLMEventBus:
    def test_agent_events(self):
        bus = EventBus(record_history=True)
        engine = _make_engine("Direct answer.")
        agent = RLMAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types

    def test_agent_events_with_code(self):
        bus = EventBus(record_history=True)
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nFINAL('done')\n```",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        }
        agent = _allow_repl(RLMAgent(engine, "test-model", bus=bus))
        agent.run("Test")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types


class TestRLMSubLMWithTools:
    def test_sub_lm_tool_resolution(self):
        """When sub-LM returns tool_calls, agent resolves them."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            # Root LM: code that calls llm_query
            {
                "content": (
                    "```python\nresult = llm_query('Calculate 2+2')\nFINAL(result)\n```"
                ),
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
            # Sub-LM: returns tool call
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "sub_0",
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
                "model": "test-model",
                "finish_reason": "tool_calls",
            },
            # Sub-LM follow-up after tool result
            {
                "content": "The answer is 4.",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]
        agent = _allow_test_tools(RLMAgent(engine, "test-model", tools=[_CalcStub()]))
        result = agent.run("Calculate")
        assert result.content == "The answer is 4."


class TestRLMDirectToolBridge:
    def test_root_repl_can_use_file_read_tool_directly(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            {
                "content": (
                    "```python\n"
                    'content = file_read("rust/Cargo.toml")\n'
                    "print(content)\n"
                    "FINAL('read ok')\n"
                    "```"
                ),
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 20,
                    "total_tokens": 25,
                },
                "model": "test-model",
                "finish_reason": "stop",
            }
        ]
        agent = RLMAgent(engine, "test-model", tools=[_FileReadStub()])
        agent = _allow_test_tools(agent)
        result = agent.run("Read Cargo")
        assert result.content == "read ok"
        assert any(tr.tool_name == "file_read" for tr in result.tool_results)
        assert any(
            tr.tool_name == "rlm_repl" and "members" in tr.content
            for tr in result.tool_results
        )

    def test_root_repl_can_use_bounded_read_helper(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            {
                "content": (
                    "```python\n"
                    'snippet = read_file("long.txt", max_lines=2)\n'
                    "FINAL(snippet)\n"
                    "```"
                ),
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 20,
                    "total_tokens": 25,
                },
                "model": "test-model",
                "finish_reason": "stop",
            }
        ]
        agent = RLMAgent(engine, "test-model", tools=[_FileReadStub()])
        agent = _allow_test_tools(agent)
        result = agent.run("Read file head")
        assert result.content == "line1\nline2\n"
        assert any(tr.tool_name == "file_read" for tr in result.tool_results)

    def test_root_repl_can_use_file_chunk_helper(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            {
                "content": (
                    "```python\n"
                    'snippet = read_file_chunk("long.txt", 2, 4)\n'
                    "FINAL(snippet)\n"
                    "```"
                ),
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 20,
                    "total_tokens": 25,
                },
                "model": "test-model",
                "finish_reason": "stop",
            }
        ]
        agent = RLMAgent(engine, "test-model", tools=[_FileReadStub()])
        agent = _allow_test_tools(agent)
        result = agent.run("Read file chunk")
        assert result.content == "line2\nline3\nline4\n"
        assert any(tr.tool_name == "file_read" for tr in result.tool_results)


class TestRLMBlockedCode:
    def test_blocked_code_returns_error(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            # Code with blocked pattern
            {
                "content": "```python\nos.system('ls')\n```",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
            # After error feedback, model gives direct answer
            {
                "content": "I apologize, let me answer directly.",
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 5,
                    "total_tokens": 20,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]
        agent = _allow_repl(RLMAgent(engine, "test-model"))
        result = agent.run("Test")
        assert result.content == "I apologize, let me answer directly."
        # The blocked code should produce a failed tool result
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        assert "Blocked" in result.tool_results[0].content


class TestRLMToolSectionInjection:
    """Verify that tool descriptions are injected into the RLM system prompt."""

    def test_system_prompt_includes_tool_section(self):
        """Tools provided -> system prompt includes descriptions."""
        engine = _make_engine("Direct answer.")
        agent = RLMAgent(engine, "test-model", tools=[_CalcStub()])
        agent.run("Hello")
        call_args = engine.generate.call_args
        messages = call_args[0][0]
        system_msg = messages[0].content
        assert "## Available Tools" in system_msg
        assert "### calculator" in system_msg
        assert "expression" in system_msg

    def test_system_prompt_no_tool_section_without_tools(self):
        """No tools -> system prompt has no tool section."""
        engine = _make_engine("Direct answer.")
        agent = RLMAgent(engine, "test-model")
        agent.run("Hello")
        call_args = engine.generate.call_args
        messages = call_args[0][0]
        system_msg = messages[0].content
        assert "## Available Tools" not in system_msg


class TestRLMReplResults:
    def test_repl_results_in_tool_results(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nprint('hello')\nFINAL('done')\n```",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        }
        agent = _allow_repl(RLMAgent(engine, "test-model"))
        result = agent.run("Test")
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "rlm_repl"
        assert "hello" in result.tool_results[0].content


class TestRLMSecurityBoundary:
    def test_missing_isolated_sandbox_denies_before_host_execution(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nFINAL('must not execute')\n```",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        agent = RLMAgent(
            engine,
            "test-model",
            repl_sandbox_executor=None,
        )
        agent = _allow_repl(agent)

        with patch(
            "openjarvis.agents.rlm.RLMRepl.execute",
            autospec=True,
        ) as execute:
            result = agent.run("Test")

        execute.assert_not_called()
        assert result.metadata["security_disabled"] is True
        assert "isolated sandbox" in result.content

    def test_missing_policy_denies_before_repl_execution(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nFINAL('must not execute')\n```",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        agent = RLMAgent(engine, "test-model")

        with patch(
            "openjarvis.agents.rlm.RLMRepl.execute",
            autospec=True,
        ) as execute:
            result = agent.run("Test")

        execute.assert_not_called()
        assert result.metadata["security_denied"] is True
        assert result.tool_results[0].tool_name == "repl"
        assert result.tool_results[0].success is False
        assert "policy unavailable" in result.content.lower()

    def test_missing_code_capability_denies_before_repl_execution(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nFINAL('must not execute')\n```",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        policy = CapabilityPolicy()
        policy.grant("rlm-test-agent", "tool:invoke", "code:rlm-repl")
        agent = RLMAgent(engine, "test-model")
        agent.bind_security(policy, "rlm-test-agent")

        with patch(
            "openjarvis.agents.rlm.RLMRepl.execute",
            autospec=True,
        ) as execute:
            result = agent.run("Test")

        execute.assert_not_called()
        assert result.metadata["security_denied"] is True
        assert "code:execute" in result.content

    def test_minimal_repl_grants_allow_execution(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "```python\nFINAL('authorized')\n```",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        agent = _allow_repl(RLMAgent(engine, "test-model"))

        result = agent.run("Test")

        assert result.content == "authorized"
        assert result.tool_results[0].tool_name == "rlm_repl"
        assert result.tool_results[0].success is True
