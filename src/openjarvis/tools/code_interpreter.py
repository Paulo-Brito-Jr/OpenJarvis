"""Code interpreter tool.

The legacy host subprocess implementation is intentionally disabled.  A
process running as the same user is not an isolation boundary, and string
blacklists cannot make arbitrary Python safe.
"""

from __future__ import annotations

from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


@ToolRegistry.register("code_interpreter")
class CodeInterpreterTool(BaseTool):
    """Fail-closed placeholder for isolated Python execution.

    Use ``code_interpreter_docker`` when a verified container sandbox is
    available.  This tool never falls back to host execution.
    """

    tool_id = "code_interpreter"

    def __init__(self, timeout: int = 30, max_output: int = 10000):
        self._timeout = timeout
        self._max_output = max_output

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_interpreter",
            description=(
                "Execute Python code only when an isolated sandbox backend "
                "is configured. Host execution is disabled."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                },
                "required": ["code"],
            },
            category="code",
            requires_confirmation=True,
            required_capabilities=["code:execute"],
            metadata={"sandbox_required": True, "host_execution": False},
        )

    def authorization_resource(self, params: dict[str, Any]) -> str:
        """Authorize the execution class without exposing source code."""
        del params
        return "code:python"

    def execute(self, **params: Any) -> ToolResult:
        code = params.get("code", "")
        if not code:
            return ToolResult(
                tool_name="code_interpreter",
                content="No code provided.",
                success=False,
            )

        del code
        return ToolResult(
            tool_name="code_interpreter",
            content=(
                "Python execution disabled: no verified isolated sandbox "
                "backend is configured. Use code_interpreter_docker."
            ),
            success=False,
            metadata={
                "security_disabled": True,
                "reason": "isolated_sandbox_required",
            },
        )


__all__ = ["CodeInterpreterTool"]
