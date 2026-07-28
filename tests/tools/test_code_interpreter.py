"""Tests for the code interpreter tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolCall
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import ToolExecutor
from openjarvis.tools.code_interpreter import CodeInterpreterTool


class TestCodeInterpreterTool:
    def test_spec_name_and_category(self):
        tool = CodeInterpreterTool()
        assert tool.spec.name == "code_interpreter"
        assert tool.spec.category == "code"

    def test_spec_parameters_require_code(self):
        tool = CodeInterpreterTool()
        assert "code" in tool.spec.parameters["properties"]
        assert "code" in tool.spec.parameters["required"]

    def test_spec_requires_confirmation_and_code_capability(self):
        tool = CodeInterpreterTool()
        assert tool.spec.requires_confirmation is True
        assert tool.spec.required_capabilities == ["code:execute"]
        assert tool.spec.metadata["sandbox_required"] is True
        assert tool.spec.metadata["host_execution"] is False

    @pytest.mark.parametrize(
        "code",
        [
            "print(2 + 2)",
            "import math; print(math.pi)",
            "import os; os.system('id')",
            "import subprocess; subprocess.run(['id'])",
            "open('/etc/passwd').read()",
            "eval('2 + 2')",
            "while True: pass",
        ],
    )
    def test_host_python_execution_is_always_disabled(self, code):
        tool = CodeInterpreterTool()
        result = tool.execute(code=code)
        assert result.success is False
        assert "no verified isolated sandbox" in result.content
        assert result.metadata["security_disabled"] is True
        assert result.metadata["reason"] == "isolated_sandbox_required"

    def test_no_code_provided(self):
        tool = CodeInterpreterTool()
        result = tool.execute(code="")
        assert result.success is False
        assert "No code" in result.content

    def test_no_code_param(self):
        tool = CodeInterpreterTool()
        result = tool.execute()
        assert result.success is False
        assert "No code" in result.content

    def test_to_openai_function(self):
        tool = CodeInterpreterTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "code_interpreter"
        assert "code" in fn["function"]["parameters"]["properties"]

    def test_tool_id(self):
        tool = CodeInterpreterTool()
        assert tool.tool_id == "code_interpreter"

    def test_registry_registration(self):
        ToolRegistry.register_value("code_interpreter", CodeInterpreterTool)
        assert ToolRegistry.contains("code_interpreter")

    def test_authorization_uses_non_secret_code_resource(self):
        tool = CodeInterpreterTool()
        assert tool.authorization_resource({"code": "secret source"}) == "code:python"

    def test_executor_denies_without_security_context(self):
        tool = CodeInterpreterTool()
        result = ToolExecutor([tool]).execute(
            ToolCall(
                id="1",
                name="code_interpreter",
                arguments='{"code":"print(1)"}',
            )
        )
        assert result.success is False
        assert "policy unavailable" in result.content.lower()

    def test_executor_requires_live_confirmation_even_with_capabilities(
        self,
        monkeypatch,
    ):
        tool = CodeInterpreterTool()
        policy = CapabilityPolicy()
        policy.grant("agent-1", "tool:invoke", "code:python")
        policy.grant("agent-1", "code:execute", "code:python")
        confirmation = MagicMock(return_value=True)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        executor = ToolExecutor(
            [tool],
            capability_policy=policy,
            agent_id="agent-1",
            interactive=True,
            confirm_callback=confirmation,
        )

        result = executor.execute(
            ToolCall(
                id="1",
                name="code_interpreter",
                arguments='{"code":"print(1)"}',
            )
        )

        assert result.success is False
        assert "live TTY" in result.content
        confirmation.assert_not_called()
