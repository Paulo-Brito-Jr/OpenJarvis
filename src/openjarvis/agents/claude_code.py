"""Security-disabled Claude Code subprocess adapter.

The legacy runner delegated inference and tools to an external Node process
outside OpenJarvis's enforceable capability boundary. Construction and
``run`` now avoid credentials and processes until a verified isolated bridge
exists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolResult
from openjarvis.engine._stubs import InferenceEngine

# Sentinel markers for parsing subprocess output
_OUTPUT_START = "---OPENJARVIS_OUTPUT_START---"
_OUTPUT_END = "---OPENJARVIS_OUTPUT_END---"

# Path to the bundled runner source (relative to this module).
# In editable installs this lives next to this file; in wheel installs
# it is placed under _node_modules/ to avoid namespace package conflicts.
_RUNNER_SRC = Path(__file__).resolve().parent / "claude_code_runner"
if not _RUNNER_SRC.exists():
    _RUNNER_SRC = (
        Path(__file__).resolve().parents[2] / "_node_modules" / "claude_code_runner"
    )


@AgentRegistry.register("claude_code")
class ClaudeCodeAgent(BaseAgent):
    """Agent that wraps the Claude Agent SDK via a Node.js subprocess.

    Spawns a Node.js process running ``dist/index.js`` which imports
    ``@anthropic-ai/claude-code`` and streams agentic responses.  Results
    are communicated back via sentinel-delimited JSON on stdout.

    The ``engine`` parameter is accepted for BaseAgent interface conformance
    but is not used -- all inference is handled by the Claude Agent SDK.
    """

    agent_id = "claude_code"
    accepts_tools = False
    requires_security_context = True
    _default_temperature = 0.7
    _default_max_tokens = 1024

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        bus: Optional[EventBus] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: str = "",
        workspace: str = "",
        session_id: str = "",
        allowed_tools: Optional[List[str]] = None,
        system_prompt: str = "",
        timeout: int = 300,
    ) -> None:
        super().__init__(
            engine,
            model,
            bus=bus,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        del api_key
        self._api_key = ""
        self._workspace = workspace or os.getcwd()
        self._session_id = session_id
        self._allowed_tools = allowed_tools
        self._system_prompt = system_prompt
        self._timeout = timeout
        self._capability_policy = None
        self._security_agent_id = ""

    def bind_security(
        self,
        capability_policy: Optional[Any],
        agent_id: Optional[str] = None,
        boundary_guard: Optional[Any] = None,
    ) -> None:
        self._capability_policy = capability_policy
        self._security_agent_id = agent_id if isinstance(agent_id, str) else ""
        self._boundary_guard = boundary_guard

    def bind_boundary_guard(self, boundary_guard: Optional[Any]) -> None:
        self._boundary_guard = boundary_guard

    # ------------------------------------------------------------------
    # Runner management
    # ------------------------------------------------------------------

    def _ensure_runner(self) -> Path:
        """Refuse the unaudited external runner without side effects."""
        raise RuntimeError(
            "Claude Code adapter disabled: the bundled SDK runner has no "
            "verified OpenJarvis capability and isolated-sandbox boundary. "
            "Automatic npm installation is disabled."
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute a query via the Claude Agent SDK subprocess.

        Spawns ``node dist/index.js``, writes a JSON request to stdin, and
        reads sentinel-delimited JSON output from stdout.
        """
        self._emit_turn_start(input)
        self._emit_turn_end(turns=0, error=True)
        return AgentResult(
            content=(
                "Claude Code adapter disabled: no verified isolated sandbox "
                "and OpenJarvis capability bridge is available."
            ),
            turns=0,
            metadata={
                "error": True,
                "security_disabled": True,
                "reason": "unverified_external_sandbox",
            },
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(
        stdout: str,
    ) -> tuple[str, list[ToolResult], dict[str, Any]]:
        """Extract the sentinel-wrapped JSON from subprocess stdout.

        Returns ``(content, tool_results, metadata)``.
        """
        start = stdout.find(_OUTPUT_START)
        end = stdout.find(_OUTPUT_END)

        if start == -1 or end == -1:
            # No sentinels -- treat entire stdout as plain content
            return stdout.strip(), [], {}

        json_str = stdout[start + len(_OUTPUT_START) : end].strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return stdout.strip(), [], {"parse_error": True}

        content = data.get("content", "")
        raw_tools = data.get("tool_results", [])
        metadata = data.get("metadata", {})

        tool_results = [
            ToolResult(
                tool_name=tr.get("tool_name", "unknown"),
                content=tr.get("content", ""),
                success=tr.get("success", True),
            )
            for tr in raw_tools
        ]

        return content, tool_results, metadata


__all__ = ["ClaudeCodeAgent"]
